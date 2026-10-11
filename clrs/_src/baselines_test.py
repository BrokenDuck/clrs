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

"""End-to-end coverage for all tasks, model state, and masked trajectories."""
import itertools
import pytest
import torch
import clrs
from clrs._src import dataset, specs, tree
from clrs._src.processors_test import KINDS


def sample(name='min_plus_apsp', n=5, count=2):
    sampler, spec = clrs.build_sampler(name, count, seed=7, length=n)
    return dataset.to_torch(sampler.next()), spec


def model(feedback, spec, **kwargs):
    return clrs.models.BaselineModel(spec, feedback, clrs.get_processor_factory(kwargs.pop('processor', 'pgn')),
                                    hidden_dim=8, encode_hints=True, **kwargs)


@pytest.mark.parametrize('name', list(specs.CLRS_30_ALGS) + ['min_plus_apsp'])
def test_all_algorithms_train(name):
    f, spec = sample(name)
    m = model(f, spec)
    assert torch.isfinite(m.feedback(f))
    pred, hints = m.predict(f.features)
    assert set(pred) == {p.name for p in f.outputs}
    assert all(torch.isfinite(p.data).all() for p in pred.values())
    assert 'score' in clrs.evaluate(f.outputs, pred)


@pytest.mark.parametrize('kind', KINDS)
def test_all_processors_train(kind):
    f, spec = sample()
    m = model(f, spec, processor=kind)
    assert torch.isfinite(m.feedback(f))
    m.predict(f.features)


def test_checkpoint_and_frozen_processor(tmp_path):
    f, spec = sample()
    m = model(f, spec, freeze_processor=True, checkpoint_path=tmp_path, use_lstm=True)
    before = {k: p.detach().clone() for k, p in m.net.processor.named_parameters()}
    m.feedback(f)
    for k, p in m.net.processor.named_parameters():
        torch.testing.assert_close(p, before[k], rtol=0, atol=0)
    predicted = m.predict(f.features)[0]
    m.save_model()
    other = model(f, spec, freeze_processor=True, checkpoint_path=tmp_path, use_lstm=True)
    other.restore_model()
    tree.map_arrays(torch.testing.assert_close, predicted, other.predict(f.features)[0])
    torch.testing.assert_close(m.feedback(f), other.feedback(f), rtol=0, atol=0)


def test_multitask_unused_heads():
    a, sa = sample('bfs')
    b, sb = sample()
    m = model([a, b], [sa, sb])
    m.feedback(b, algorithm_index=1)
    before = {k: v.detach().clone() for k, v in m.net.decoders[1].named_parameters()}
    m.feedback(a, algorithm_index=0)
    for k, v in m.net.decoders[1].named_parameters():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        assert v.grad is None


def test_evaluation_does_not_read_future_hints():
    f, spec = sample()
    m = model(f, spec, hint_teacher_forcing=1.)
    expected = m.predict(f.features)[0]
    def corrupt(x):
        x = x.clone()
        x[1:] = 0
        return x
    altered = f.features._replace(hints=tree.map_arrays(corrupt, f.features.hints))
    tree.map_arrays(torch.testing.assert_close, expected, m.predict(altered)[0])


def test_padding_does_not_change_loss():
    f, spec = sample('insertion_sort', count=4)
    m = model(f, spec, hint_teacher_forcing=1.)
    first, _ = m.loss(f)
    def alter(x):
        x = x.clone()
        for b, length in enumerate(f.features.lengths):
            x[int(length):, b] = 0
        return x
    changed = f._replace(features=f.features._replace(hints=tree.map_arrays(alter, f.features.hints)))
    second, _ = m.loss(changed)
    torch.testing.assert_close(first, second)


def test_full_chunked_equivalence_and_continuation():
    f, spec = sample()
    full = model(f, spec, hint_teacher_forcing=1.)
    chunks = dataset.chunkify(itertools.repeat(f), 2)
    first = next(chunks)
    chunked = clrs.models.BaselineModelChunked(spec, first, clrs.get_processor_factory('pgn'),
                                            hidden_dim=8, encode_hints=True, hint_teacher_forcing=1.)
    chunked.net.load_state_dict(full.net.state_dict())
    expected, _ = full.net(f.features, repred=True)
    out1, _ = chunked.forward(first.features, repred=True)
    second = next(chunks)
    out2, _ = chunked.forward(second.features, repred=True)
    combined = {k: torch.cat((out1[k], out2[k])) for k in out1}
    for k, v in expected.items():
        torch.testing.assert_close(combined[k][int(f.features.lengths[0])-1], v)
    assert torch.isfinite(chunked.feedback(first))
    assert torch.isfinite(chunked.feedback(second))


def test_tiny_overfit():
    f, spec = sample(n=4, count=2)
    m = model(f, spec, hint_teacher_forcing=1., learning_rate=.01)
    first = float(m.feedback(f))
    for _ in range(299):
        last = float(m.feedback(f))
    assert last < first * .3
    assert clrs.evaluate(f.outputs, m.predict(f.features)[0])['score'] >= .8


@pytest.mark.parametrize('chunk_length', [1, 2, 4])
def test_chunk_boundaries_have_gradients(chunk_length):
    f, spec = sample()
    chunks = dataset.chunkify(itertools.repeat(f), chunk_length)
    first = next(chunks)
    m = clrs.models.BaselineModelChunked(spec, first, clrs.get_processor_factory('pgn'),
                                       hidden_dim=8, encode_hints=True, hint_teacher_forcing=1.)
    stream = itertools.chain([first], chunks)
    for _ in range(6):
        chunk = next(stream)
        loss, gradients = m.compute_grad(chunk)
        assert torch.isfinite(loss)
        if chunk.features.is_last.any():
            assert sum(g.abs().sum() for k,g in gradients.items() if g is not None and 'decoders.0.Pi.' in k) > 0


def test_permutation_model():
    f, spec = sample('insertion_sort')
    spec, iterator = clrs.process_permutations(spec, itertools.repeat(f), enforce_permutations=True)
    feedback = next(iterator)
    m = model(feedback, spec, hint_teacher_forcing=.5)
    assert torch.isfinite(m.feedback(feedback))
    assert 'score' in clrs.evaluate(feedback.outputs, m.predict(feedback.features)[0])


def test_chunk_singleton():
    f, spec = sample(n=1)
    chunk = next(dataset.chunkify(itertools.repeat(f), 1))
    m = clrs.models.BaselineModelChunked(spec, chunk, clrs.get_processor_factory('pgn'),
                                       hidden_dim=8, encode_hints=True)
    assert torch.isfinite(m.feedback(chunk))
