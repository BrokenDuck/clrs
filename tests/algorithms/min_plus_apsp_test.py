"""Min-plus traces must reflect bounded paths, not in-place Floyd-Warshall."""
import math
import numpy as np
import pytest
from clrs._src import algorithms, probing, samplers, specs


def oracle(A, bound):
    n = len(A)
    edges = np.where(A != 0, A, np.inf)
    np.fill_diagonal(edges, 0)
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)
    for _ in range(bound):
        D = np.min(D[:, :, None] + edges[None, :, :], axis=1)
    return D


def check(A):
    pi, probes = algorithms.min_plus_apsp(A)
    _, outputs, hints = probing.split_stages(probes, specs.SPECS['min_plus_apsp'])
    hints = {p.name: p.data[:, 0] for p in hints}
    n = len(A)
    count = 1 + math.ceil(math.log2(max(1, n-1)))
    assert len(hints['D']) == count
    assert np.isfinite(hints['D']).all()
    for t, bound in enumerate(hints['m']):
        expected = oracle(A, int(bound))
        np.testing.assert_array_equal(hints['msk'][t], np.isfinite(expected))
        np.testing.assert_allclose(np.where(hints['msk'][t], hints['D'][t], np.inf), expected)
    np.testing.assert_array_equal(pi, outputs[0].data[0])
    expected = oracle(A, max(1, n-1))
    for i in range(n):
        for j in range(n):
            if not np.isfinite(expected[i, j]) or i == j:
                assert pi[i, j] == i
                continue
            vertex, total, seen = j, 0., set()
            while vertex != i:
                assert vertex not in seen, (i, j, pi)
                seen.add(vertex)
                parent = int(pi[i, vertex])
                assert A[parent, vertex] != 0
                total += A[parent, vertex]
                vertex = parent
            np.testing.assert_allclose(total, expected[i, j])
    return pi, hints


@pytest.mark.parametrize('n', [1, 2, 3, 5, 9, 16])
def test_directed_chain(n):
    A = np.diag(np.ones(n-1), 1) if n > 1 else np.zeros((1,1))
    _, hints = check(A)
    if n > 3:
        assert hints['msk'][1, 0, 3] == 0  # One square cannot traverse three edges.


def test_disconnected_ties_and_negative_edges():
    check(np.zeros((5, 5)))
    pi, _ = check(np.array([[0, 1, 1, 2], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0.]]))
    assert pi[0, 3] == 0  # Preserve the equally short original edge.
    check(np.array([[0, -2, 5, 0], [0, 0, 1, 4], [0, 0, 0, -1], [0, 0, 0, 0.]]))


@pytest.mark.parametrize('seed', range(10))
def test_random_graphs(seed):
    rng = np.random.RandomState(seed)
    A = rng.uniform(.1, 2, (9, 9)) * (rng.rand(9, 9) < .3)
    np.fill_diagonal(A, 0)
    check(A)
    # Reweight positive edges by vertex potentials: negative edges are possible,
    # but cycle costs stay positive and the APSP precondition is guaranteed.
    potentials = rng.normal(size=9)
    signed = np.where(A != 0, A + potentials[:, None] - potentials[None, :], 0)
    check(signed)


@pytest.mark.parametrize('A', [[], [1, 2], np.zeros((2, 3)), [[np.nan]], [[np.inf]], [[-1]]])
def test_invalid(A):
    with pytest.raises(ValueError):
        algorithms.min_plus_apsp(A)


def test_sampler_registration_and_floyd_distribution():
    ours, _ = samplers.build_sampler('min_plus_apsp', 4, seed=9, length=8)
    floyd, _ = samplers.build_sampler('floyd_warshall', 4, seed=9, length=8)
    a, b = ours.next(), floyd.next()
    for x, y in zip(a.features.inputs, b.features.inputs):
        np.testing.assert_array_equal(x.data, y.data)
    assert 'min_plus_apsp' not in specs.CLRS_30_ALGS
