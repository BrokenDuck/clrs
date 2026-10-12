"""Generated caches and published TFRecords preserve data and batching."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from tfrecord.writer import TFRecordWriter
from clrs._src import dataset, samplers, specs, tree


def test_generated_cache_and_workers(tmp_path):
    args = dict(source='generated', num_samples=5, length=5, seed=3, shuffle=False)
    one, count, spec = dataset.create_dataset(tmp_path, 'insertion_sort', 'train', 2, **args)
    two, _, _ = dataset.create_dataset(tmp_path, 'insertion_sort', 'train', 2, num_workers=2, **args)
    assert count == 5
    for a, b in zip(one, two):
        tree.map_arrays(torch.testing.assert_close, a, b)
    sampler, _ = samplers.build_sampler('insertion_sort', 5, seed=3, length=5)
    original = sampler.next()
    cached = one.dataset.feedback
    tree.map_arrays(np.testing.assert_array_equal, cached, original)
    assert all(x not in sys.modules for x in ('jax', 'tensorflow', 'haiku', 'optax', 'chex'))
    assert len(list((tmp_path/'generated').glob('*.npz'))) == 1


def write_fixture(folder, name='floyd_warshall'):
    split = 'val'
    path = folder/'clrs_dataset'/f'{name}_{split}'/'1.0.0'
    path.mkdir(parents=True)
    sampler, spec = samplers.build_sampler(name, 2, seed=1, length=4)
    f = sampler.next()
    flat = {'lengths': f.features.lengths}
    for stage, points in [('input', f.features.inputs), ('hint', f.features.hints), ('output', f.outputs)]:
        flat.update({stage+'_'+p.name: p.data.swapaxes(0, 1) if stage == 'hint' else p.data for p in points})
    if name == 'quickselect':
        flat.pop('hint_pivot')
    metadata = {'featuresDict': {'features': {k: {'tensor': {'dtype': str(v.dtype), 'shape': {'dimensions': list(v.shape[1:])}, 'encoding': 'none'}} for k,v in flat.items()}}}
    (path/'features.json').write_text(json.dumps(metadata))
    (path/'dataset_info.json').write_text(json.dumps({'splits': [{'name': split, 'shardLengths': ['2']}]}))
    writer = TFRecordWriter(str(path/'clrs_dataset-val.tfrecord-00000-of-00001'))
    for i in range(2):
        writer.write({k: (np.asarray(v[i]).reshape(-1).astype(np.float32), 'float') for k,v in flat.items()})
    writer.close()
    return f


def test_published_fixture(tmp_path):
    original = write_fixture(tmp_path)
    loader, count, _ = dataset.create_dataset(tmp_path, 'floyd_warshall', 'val', 2, download=False)
    actual = next(iter(loader))
    assert count == 2
    tree.map_arrays(lambda a, b: np.testing.assert_array_equal(a, np.asarray(b).astype(np.float32).astype(np.asarray(b).dtype)), actual, original)
    assert actual.features.hints[0].data.dtype == torch.float64
    assert actual.features.lengths.dtype == torch.int64


def test_new_task_has_no_published_data(tmp_path):
    with pytest.raises(ValueError, match='generated'):
        dataset.CLRSDataset(tmp_path, 'min_plus_apsp')


@pytest.mark.skipif(not os.environ.get('CLRS_PUBLISHED_TEST_DIR'), reason='Set CLRS_PUBLISHED_TEST_DIR to test the official archive')
def test_official_records():
    from tfrecord.reader import tfrecord_loader
    folder = Path(os.environ['CLRS_PUBLISHED_TEST_DIR'])
    ds = dataset.CLRSDataset(folder, 'floyd_warshall', 'train', download=False)
    raw = next(tfrecord_loader(ds.records[0][0], None))
    example = ds[0]
    for stage, points in [('input', example.features.inputs), ('hint', example.features.hints), ('output', example.outputs)]:
        for dp in points:
            np.testing.assert_array_equal(dp.data.reshape(-1), raw[stage+'_'+dp.name])
    assert len(ds) == 1000


def test_published_quickselect_legacy_spec(tmp_path):
    write_fixture(tmp_path, 'quickselect')
    loader, _, spec = dataset.create_dataset(tmp_path, 'quickselect', 'val', 2, download=False)
    assert 'pivot' not in spec
    assert 'pivot' in specs.SPECS['quickselect']
    assert all(p.name != 'pivot' for p in next(iter(loader)).features.hints)


@pytest.mark.skipif(not os.environ.get('CLRS_PUBLISHED_TEST_DIR'), reason='Set CLRS_PUBLISHED_TEST_DIR for archive integration')
@pytest.mark.parametrize('algorithm', specs.CLRS_30_ALGS)
def test_all_published_validation_specs(algorithm):
    folder = Path(os.environ['CLRS_PUBLISHED_TEST_DIR'])
    if dataset._find_published(folder, algorithm, 'val') is None:
        pytest.skip('Validation split not extracted')
    loader, count, spec = dataset.create_dataset(folder, algorithm, 'val', 2, download=False)
    feedback = next(iter(loader))
    assert len(feedback.features.lengths) == min(count, 2)
    assert {p.name for p in feedback.features.hints} == {k for k,v in spec.items() if v[0] == specs.Stage.HINT}
