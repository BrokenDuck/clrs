# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Published CLRS records and generated trajectories as PyTorch datasets.

No TensorFlow runtime is needed: TFRecord framing and Example protobufs are
read using the standalone tfrecord package and the archive's feature metadata.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import struct
import tarfile
import tempfile
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tfrecord import example_pb2
from clrs._src import probing, samplers, specs, tree

VERSION = '1.0.0'
CACHE_VERSION = 1


def get_clrs_folder():
    return f'CLRS30_v{VERSION}'


def get_dataset_gcp_url():
    return f'https://storage.googleapis.com/dm-clrs/{get_clrs_folder()}.tar.gz'


def _numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def to_torch(value, device=None):
    """Convert CLRS containers, preserving metadata and numeric precision."""
    def convert(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device)
        x = np.asarray(x)
        if not x.flags.writeable:
            x = x.copy()
        return torch.as_tensor(x, device=device)
    return tree.map_arrays(convert, value)


def _record_feedback(record, spec):
    groups = {stage: [] for stage in (specs.Stage.INPUT, specs.Stage.HINT, specs.Stage.OUTPUT)}
    for name, (stage, loc, typ) in spec.items():
        key = f'{stage}_{name}'
        if key not in record:
            raise ValueError(f'Record is missing {key}')
        value = record[key]
        # Store one example in the sampler's conventional singleton-batch shape.
        value = np.expand_dims(value, 1 if stage == specs.Stage.HINT else 0)
        groups[stage].append(probing.DataPoint(name, loc, typ, value))
    return samplers.Feedback(samplers.Features(groups['input'], groups['hint'],
                                              np.asarray(record['lengths']).reshape(1)), groups['output'])


def collate_feedback(examples):
    """Pad time to the longest trajectory in this batch, retaining CLRS axes."""
    examples = [tree.map_arrays(_numpy, e) for e in examples]
    inputs = samplers._batch_io([e.features.inputs for e in examples])
    outputs = samplers._batch_io([e.outputs for e in examples])
    hints, _ = samplers._batch_hints([e.features.hints for e in examples], 0)
    lengths = np.concatenate([e.features.lengths for e in examples]).astype(np.int64)
    return to_torch(samplers.Feedback(samplers.Features(inputs, hints, lengths), outputs))


def _find_published(folder, algorithm, split):
    for path in (folder/'clrs_dataset'/f'{algorithm}_{split}'/VERSION,
                 folder/f'{algorithm}_{split}'/VERSION,
                 folder/get_clrs_folder()/'clrs_dataset'/f'{algorithm}_{split}'/VERSION):
        if (path/'features.json').exists():
            return path
    return None


def _download_published(folder, algorithm, split):
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f'{get_clrs_folder()}.tar.gz'
    if not archive.exists():
        fd, temporary = tempfile.mkstemp(dir=folder, suffix='.download')
        os.close(fd)
        try:
            urllib.request.urlretrieve(get_dataset_gcp_url(), temporary)
            os.replace(temporary, archive)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    prefix = f'clrs_dataset/{algorithm}_{split}/'
    with tarfile.open(archive, 'r:gz') as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(prefix)]
        if not members:
            raise ValueError(f'Archive has no {algorithm}/{split}')
        tar.extractall(folder, members=members, filter='data')
    return _find_published(folder, algorithm, split)


class CLRSDataset(Dataset):
    """Map-style dataset: published records by default, or seeded generation.

    Each item is a Feedback with a singleton batch axis. Use collate_feedback
    (or create_dataset) to build batches. A dataset has a fixed problem size.
    """
    VERSION = VERSION

    def __init__(self, folder, algorithm, split='train', *, source='published',
                 num_samples=None, length=None, seed=None, download=True,
                 cache=True, **sampler_kwargs):
        if algorithm not in specs.SPECS:
            raise ValueError(f'Unknown algorithm {algorithm}')
        if split not in samplers.CLRS30:
            raise ValueError(f'Unknown split {split}')
        if source not in ('published', 'generated'):
            raise ValueError('source must be published or generated')
        self.spec, self.algorithm, self.source = specs.SPECS[algorithm], algorithm, source
        folder = Path(folder).expanduser()
        self.feedback = None
        if source == 'published':
            if algorithm not in specs.CLRS_30_ALGS:
                raise ValueError(f'{algorithm} has no published CLRS-30 records; use source="generated"')
            if any(x is not None for x in (num_samples, length, seed)) or sampler_kwargs:
                raise ValueError('Sampling options only apply to generated data')
            path = _find_published(folder, algorithm, split)
            if path is None and download:
                path = _download_published(folder, algorithm, split)
            if path is None:
                raise FileNotFoundError(f'No published {algorithm}/{split} in {folder}')
            metadata = json.loads((path/'features.json').read_text())
            self.fields = {k: v['tensor'] for k, v in metadata['featuresDict']['features'].items()}
            # The 2022 archive predates the pivot hint added to quickselect.
            # Describe the published trace faithfully rather than synthesizing
            # hints or requiring a field absent from the official benchmark.
            if algorithm == 'quickselect' and 'hint_pivot' not in self.fields:
                self.spec = {k: v for k, v in self.spec.items() if k != 'pivot'}
            expected_fields = {f'{stage}_{name}' for name, (stage, _, _) in self.spec.items()} | {'lengths'}
            if set(self.fields) != expected_fields:
                raise ValueError(f'Archive feature schema mismatch: {set(self.fields) ^ expected_fields}')
            self.records = []
            for shard in sorted(path.glob(f'*-{split}.tfrecord-*')):
                with shard.open('rb') as stream:
                    while header := stream.read(12):
                        if len(header) != 12:
                            raise ValueError(f'Truncated TFRecord header in {shard}')
                        size = struct.unpack('<Q', header[:8])[0]
                        offset = stream.tell()
                        if offset + size + 4 > shard.stat().st_size:
                            raise ValueError(f'Truncated TFRecord payload in {shard}')
                        self.records.append((str(shard), offset, size))
                        stream.seek(size+4, 1)
            if not self.records:
                raise ValueError(f'No TFRecords in {path}')
            info = json.loads((path/'dataset_info.json').read_text())
            expected = sum(int(n) for part in info['splits'] if part['name'] == split for n in part['shardLengths'])
            if len(self.records) != expected:
                raise ValueError(f'Expected {expected} records, found {len(self.records)}')
            self.provenance = {'source': source, 'version': VERSION, 'algorithm': algorithm, 'split': split}
        else:
            defaults = samplers.CLRS30[split]
            multiplier = specs.CLRS_30_ALGS_SETTINGS.get(algorithm, {}).get('num_samples_multiplier', 1)
            count = num_samples if num_samples is not None else defaults['num_samples'] * (1 if split == 'train' else multiplier)
            length = defaults['length'] if length is None else length
            seed = defaults['seed'] if seed is None else seed
            if count < 1 or length < 1:
                raise ValueError('num_samples and length must be positive')
            revision_file = Path(__file__).parents[1] / 'UPSTREAM_COMMIT'
            revision = revision_file.read_text().strip()
            self.provenance = {'source': source, 'format': CACHE_VERSION, 'upstream': revision,
                               'algorithm': algorithm, 'split': split, 'num_samples': count,
                               'length': length, 'seed': seed, 'sampler': sampler_kwargs}
            identity = json.dumps(self.provenance, sort_keys=True)
            cache_path = folder/'generated'/f'{algorithm}-{hashlib.sha256(identity.encode()).hexdigest()[:20]}.npz'
            if cache and cache_path.exists():
                with np.load(cache_path, allow_pickle=False) as arrays:
                    if str(arrays['metadata']) != identity:
                        raise ValueError('Cache provenance mismatch')
                    self.feedback = self._unpack(arrays)
            else:
                sampler, _ = samplers.build_sampler(algorithm, count, seed=seed, length=length, **sampler_kwargs)
                self.feedback = sampler.next()
                if cache:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    data = self._pack(self.feedback)
                    with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix='.npz', delete=False) as f:
                        temporary = Path(f.name)
                    try:
                        np.savez_compressed(temporary, metadata=np.array(identity), **data)
                        os.replace(temporary, cache_path)
                    finally:
                        temporary.unlink(missing_ok=True)

    def _pack(self, feedback):
        data = {'lengths': feedback.features.lengths}
        for stage, points in (('input', feedback.features.inputs), ('hint', feedback.features.hints), ('output', feedback.outputs)):
            data.update({f'{stage}_{p.name}': p.data for p in points})
        return data

    def _unpack(self, arrays):
        groups = {stage: [] for stage in ('input', 'hint', 'output')}
        for name, (stage, loc, typ) in self.spec.items():
            groups[stage].append(probing.DataPoint(name, loc, typ, arrays[f'{stage}_{name}']))
        return samplers.Feedback(samplers.Features(groups['input'], groups['hint'], arrays['lengths']), groups['output'])

    def __len__(self):
        return len(self.feedback.features.lengths) if self.feedback is not None else len(self.records)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.feedback is not None:
            f = self.feedback
            length = int(f.features.lengths[index])
            return samplers.Feedback(samplers.Features(
                tree.map_arrays(lambda x: x[index:index+1], f.features.inputs),
                tree.map_arrays(lambda x: x[:length, index:index+1], f.features.hints),
                f.features.lengths[index:index+1]),
                tree.map_arrays(lambda x: x[index:index+1], f.outputs))
        shard, offset, size = self.records[index]
        with open(shard, 'rb') as stream:
            stream.seek(offset)
            example = example_pb2.Example.FromString(stream.read(size))
        record = {}
        for name, tensor in self.fields.items():
            feature = example.features.feature[name]
            kind = feature.WhichOneof('kind')
            dtype = np.dtype(tensor['dtype'])
            shape = tuple(int(x) for x in tensor.get('shape', {}).get('dimensions', []))
            if kind == 'bytes_list':
                value = np.frombuffer(feature.bytes_list.value[0], dtype=dtype)
            elif kind in ('float_list', 'int64_list'):
                value = np.asarray(getattr(feature, kind).value, dtype=dtype)
            else:
                raise ValueError(f'Missing or unsupported TFRecord field {name}')
            record[name] = value.reshape(shape)
        result = _record_feedback(record, self.spec)
        length = int(result.features.lengths[0])
        return result._replace(features=result.features._replace(
            hints=tree.map_arrays(lambda x: x[:length], result.features.hints)))


def create_dataset(folder, algorithm, split, batch_size, *, source='published',
                   shuffle=None, num_workers=0, loader_seed=0, **kwargs):
    """Return (DataLoader, number of examples, spec); iteration is finite."""
    dataset = CLRSDataset(folder, algorithm, split, source=source, **kwargs)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=(split == 'train') if shuffle is None else shuffle,
                        num_workers=num_workers, collate_fn=collate_feedback,
                        generator=torch.Generator().manual_seed(loader_seed))
    return loader, len(dataset), dataset.spec


def create_chunked_dataset(folder, algorithm, split, batch_size, chunk_length, **kwargs):
    loader, _, spec = create_dataset(folder, algorithm, split, batch_size, **kwargs)
    # Repeating full batches maintains stable lane counts across chunk boundaries.
    def repeat():
        while True:
            for batch in loader:
                if len(batch.features.lengths) == batch_size:
                    yield batch
    if len(loader.dataset) < batch_size:
        raise ValueError('Chunked datasets require at least batch_size examples')
    return chunkify(repeat(), chunk_length), spec


def _copy_hint(source, dest, i, start_source, start_dest, to_add):
  """Copy from full-sample hint to a hint chunk."""
  assert np.all(dest[start_dest:, i:] == 0)
  assert start_dest < dest.shape[0]
  assert start_dest + to_add <= dest.shape[0]
  assert start_source < source.shape[0]
  assert start_source + to_add <= source.shape[0]
  dest[start_dest:start_dest+to_add, i] = source[
      start_source:start_source+to_add, i]
  return dest


def _copy_io(source, dest, i, start_dest, to_add):
  """Copy from an input or output to an input or output chunk."""
  assert np.all(dest[start_dest:, i:] == 0)
  dest[start_dest:start_dest+to_add, i] = source[i]
  return dest


def chunkify(dataset: Iterator[samplers.Feedback], chunk_length: int):
  """Generator of fixed-length chunks from full-trajectory samples.

  Args:
    dataset: full-sample dataset as numpy iterator.
    chunk_length: time length of chunks.
  Yields:
    Fixed-timelength chunks of data. Each tensor of inputs, hints and outputs
    has dimensions chunk_length x batch_size x ... Samples are not time-padded,
    after the end of one sample immediately comes the next. Since different
    samples can have different time lengths, the beginnings and ends of samples
    within a batch do not need to coincide. For this reason, the chunked
    dataset features include two chunk_length x batch_size int tensors,
    `is_first` and `is_last`, that mark the beginning and end of each sample.
    For example, if `chunk_legnth`==6 and `batch_size`==2 and the first
    full-sample batch had one sample of length 3 and one of length 5,
    we would have a first chunked batch with the following `is_first` and
    `is_last` tensors:

    is_first = [[1, 1]    is_last = [[0, 0]     ( sample id [[0 1]
                [0, 0]               [0, 0]                  [0 1]
                [0, 0]               [1, 0]                  [0 1]
                [1, 0]               [0, 0]                  [2 1]
                [0, 0]               [0, 1]                  [2 1]
                [0, 1]]              [0, 0]]                 [2 3]] )

    while the data in the inputs, outputs and hints tensors would correspond
    to samples as identified by the sample_id indicated above for reference.
    Notice that, while in the full-sample dataset inputs and outputs have
    no time dimension, here they do; the input and output tensors are simply
    repeated along each sample's time length.
  """
  if chunk_length < 1:
    raise ValueError("chunk_length must be positive")
  dataset = iter(dataset)
  def _get_batch():
    d = tree.map_arrays(_numpy, next(dataset))
    return (d.features.inputs, d.features.hints, d.outputs,
            d.features.lengths.astype(int))

  try:
    inputs, hints, outputs, lengths = _get_batch()
  except StopIteration:
    return
  for inp in inputs:
    if inp.location in [specs.Location.NODE, specs.Location.EDGE]:
      batch_size = inp.data.shape[0]
      break

  io_chunk = lambda x: np.zeros((chunk_length,) + x.shape, dtype=x.dtype)
  chunk_inputs = tree.map_arrays(io_chunk, inputs)
  chunk_outputs = tree.map_arrays(io_chunk, outputs)

  hint_chunk = lambda x: np.zeros((chunk_length,) + x.shape[1:], dtype=x.dtype)
  chunk_hints = tree.map_arrays(hint_chunk, hints)

  inputs = [inputs]
  hints = [hints]
  outputs = [outputs]
  left = [lengths.copy()]
  lengths = [lengths.copy()]

  while True:
    # Create a new empty chunk
    chunk_inputs = tree.map_arrays(np.zeros_like, chunk_inputs)
    chunk_hints = tree.map_arrays(np.zeros_like, chunk_hints)
    chunk_outputs = tree.map_arrays(np.zeros_like, chunk_outputs)
    start_mark = np.zeros((chunk_length, batch_size), dtype=int)  # pyrefly: ignore[unbound-name]
    end_mark = np.zeros((chunk_length, batch_size), dtype=int)

    # Get enough data batches to fill the new chunk
    while np.any(np.sum(left, axis=0) < chunk_length):
      try:
        inp, hh, out, ll = _get_batch()
      except StopIteration:
        return
      inputs.append(inp)
      hints.append(hh)
      outputs.append(out)
      left.append(ll.copy())
      lengths.append(ll.copy())

    # Fill the chunk, one batch element at a time
    for i in range(batch_size):
      total, idx = 0, 0
      while total < chunk_length:
        to_add = min(left[idx][i], chunk_length - total)
        if to_add:
          start = lengths[idx][i] - left[idx][i]
          assert start >= 0
          f_io = functools.partial(_copy_io, i=i, start_dest=total,
                                   to_add=to_add)
          chunk_inputs = tree.map_arrays(f_io, inputs[idx], chunk_inputs)
          chunk_outputs = tree.map_arrays(f_io, outputs[idx],
                                                 chunk_outputs)
          f_hint = functools.partial(_copy_hint, i=i, start_source=start,
                                     start_dest=total, to_add=to_add)
          chunk_hints = tree.map_arrays(f_hint, hints[idx], chunk_hints)
          if start == 0:
            start_mark[total, i] = 1
          total += to_add
          left[idx][i] -= to_add
          assert left[idx][i] >= 0
          if left[idx][i] == 0:
            end_mark[total - 1, i] = 1
        idx += 1
      assert total == chunk_length

    while left and np.all(left[0] == 0):
      inputs.pop(0)
      hints.pop(0)
      outputs.pop(0)
      left.pop(0)
      lengths.pop(0)

    yield to_torch(samplers.Feedback(
        samplers.FeaturesChunked(chunk_inputs, chunk_hints,
                                 start_mark, end_mark),
        chunk_outputs))
