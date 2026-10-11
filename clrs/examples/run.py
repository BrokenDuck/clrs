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

"""Train and evaluate PyTorch CLRS models on published or generated data.

Example: python -m clrs.examples.run --algorithms min_plus_apsp --steps 100
"""
import argparse
import json
from pathlib import Path
import torch
import clrs
from clrs._src import dataset, tree


def repeat(loader):
    while True:
        yield from loader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithms', default='min_plus_apsp', help='Comma-separated tasks sharing a processor')
    parser.add_argument('--processor-type', '--processor_type', default='pgn')
    parser.add_argument('--source', choices=['generated', 'published'], default='generated')
    parser.add_argument('--dataset-path', '--dataset_path', default='./data/clrs')
    parser.add_argument('--checkpoint-path', default='./checkpoints/clrs')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=.005)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-length', type=int, default=16)
    parser.add_argument('--test-length', type=int, default=64)
    parser.add_argument('--num-samples', type=int, default=None, help='Override generated training count')
    parser.add_argument('--eval-samples', type=int, default=None, help='Override generated val/test counts')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--chunk-length', type=int, default=0)
    parser.add_argument('--teacher-forcing', type=float, default=.5)
    parser.add_argument('--encode-hints', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--decode-hints', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use-lstm', action='store_true')
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args(argv)
    if args.steps < 0 or args.batch_size < 1 or args.chunk_length < 0:
        parser.error('steps/chunk-length must be nonnegative and batch-size positive')
    torch.set_num_threads(args.threads)
    names = args.algorithms.split(',')
    loaders, specs, dummy, streams = [], [], [], []
    for name in names:
        options = {'source': args.source}
        if args.source == 'generated':
            options.update(length=args.train_length, num_samples=args.num_samples)
        loader, _, spec = clrs.create_dataset(args.dataset_path, name, 'train', args.batch_size,
                                              loader_seed=args.seed, **options)
        loaders.append(loader)
        specs.append(spec)
        dummy.append(next(iter(loader)))
        if args.chunk_length:
            stream, _ = clrs.create_chunked_dataset(args.dataset_path, name, 'train', args.batch_size,
                                                    args.chunk_length, loader_seed=args.seed, **options)
            streams.append(stream)
        else:
            streams.append(repeat(loader))
    factory = clrs.get_processor_factory(args.processor_type, use_ln=True, nb_triplet_fts=8, nb_heads=1)
    options = dict(hidden_dim=args.hidden_dim, encode_hints=args.encode_hints, decode_hints=args.decode_hints,
                   use_lstm=args.use_lstm, learning_rate=args.learning_rate, hint_teacher_forcing=args.teacher_forcing,
                   checkpoint_path=args.checkpoint_path, device=args.device, seed=args.seed)
    if args.chunk_length:
        chunk_dummy = [next(stream) for stream in streams]
        learner = clrs.models.BaselineModelChunked(specs, chunk_dummy, factory, **options)
        # Feed the consumed first chunk before advancing each stream.
        import itertools
        streams = [itertools.chain([first], stream) for first, stream in zip(chunk_dummy, streams)]
    else:
        learner = clrs.models.BaselineModel(specs, dummy, factory, **options)
    if args.restore:
        learner.restore_model()
    for step in range(args.steps):
        index = step % len(names)
        loss = learner.feedback(next(streams[index]), algorithm_index=index)
        if step == 0 or (step+1) % max(1, args.steps//10) == 0:
            print(json.dumps({'step': step+1, 'algorithm': names[index], 'loss': float(loss)}), flush=True)
    checkpoint = learner.save_model()
    evaluator = learner
    if args.chunk_length:
        evaluator = clrs.models.BaselineModel(specs, dummy, factory, **options)
        evaluator.net.load_state_dict(learner.net.state_dict())
    for index, name in enumerate(names):
        for split in ('val', 'test'):
            options = {'source': args.source}
            if args.source == 'generated':
                options.update(length=args.train_length if split == 'val' else args.test_length,
                               num_samples=args.eval_samples)
            loader, _, _ = clrs.create_dataset(args.dataset_path, name, split, args.batch_size, **options)
            truths, predictions = [], []
            for feedback in loader:
                out, _ = evaluator.predict(feedback.features, algorithm_index=index, return_hints=False)
                truths.append(dataset.to_torch(feedback.outputs, 'cpu'))
                predictions.append(dataset.to_torch(out, 'cpu'))
            truth = tree.map_arrays(lambda *xs: torch.cat(xs), *truths)
            pred = tree.map_arrays(lambda *xs: torch.cat(xs), *predictions)
            metrics = clrs.evaluate(truth, pred)
            print(json.dumps({'algorithm': name, 'split': split, **{k: float(v) for k,v in metrics.items()}}), flush=True)
    print(f'Checkpoint: {checkpoint}')


if __name__ == '__main__':
    main()
