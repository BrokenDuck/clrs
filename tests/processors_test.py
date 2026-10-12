# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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

"""Every upstream processor variant supports PyTorch training."""
import pytest
import torch
from clrs._src import processors

from .processor_cases import KINDS


@pytest.mark.parametrize('kind', KINDS)
def test_forward_backward(kind):
    torch.manual_seed(1)
    processor = processors.get_processor_factory(kind, True, 4, 2)(8)
    nodes = torch.randn(2, 5, 8, requires_grad=True)
    edges = torch.randn(2, 5, 5, 8, requires_grad=True)
    graph = torch.randn(2, 8, requires_grad=True)
    result, edge = processor(nodes, edges, graph, torch.eye(5).expand(2, 5, 5), torch.zeros_like(nodes))
    assert result.shape == nodes.shape
    loss = result.square().sum()
    if edge is not None:
        assert edge.shape == edges.shape
        loss = loss + edge.square().sum()
    loss.backward()
    grads = [p.grad for p in processor.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_d2_processor():
    p = processors.GATv2FullD2(8, 2)
    result = p.d2_forward([torch.randn(2, 3, 8) for _ in range(4)],
                         [torch.randn(2, 3, 3, 8) for _ in range(4)],
                         [torch.randn(2, 8) for _ in range(4)],
                         torch.ones(2, 3, 3), torch.zeros(2, 3, 8))
    assert len(result) == 4
    sum(x.sum() for x in result).backward()


@pytest.mark.parametrize('embedding', [True, False])
def test_memory_hops(embedding):
    p = processors.MemNetMasked(8, 8, 8, embedding_size=8, num_hops=3, apply_embeddings=embedding)
    out = p._apply_memory(torch.ones(2, 8), torch.ones(2, 5, 8))
    assert out.shape == (2, 8)
    out.sum().backward()
