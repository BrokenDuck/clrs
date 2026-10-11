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

"""Native PyTorch implementations of the CLRS processor family."""
import abc
from typing import Callable
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BIG_NUMBER = 1e6
PROCESSOR_TAG = 'clrs_processor'


class Processor(nn.Module, abc.ABC):
    def __init__(self, name='processor'):
        super().__init__()
        self.name = name

    @property
    def inf_bias(self):
        return False

    @property
    def inf_bias_edge(self):
        return False


class GAT(Processor):
    def __init__(self, out_size, nb_heads=1, activation=F.relu,
                 residual=True, use_ln=False, name='gat_aggr'):
        super().__init__(name)
        self.out_size, self.nb_heads = out_size, nb_heads or 1
        if out_size % self.nb_heads:
            raise ValueError('The number of heads must divide the width')
        self.head_size = out_size // self.nb_heads
        self.activation, self.residual = activation, residual
        self.m, self.skip = nn.LazyLinear(out_size), nn.LazyLinear(out_size)
        self.att = nn.ModuleList([nn.LazyLinear(self.nb_heads) for _ in range(4)])
        self.norm = nn.LayerNorm(out_size) if use_ln else nn.Identity()

    def logits(self, z, edge, graph):
        a, b, e, g = self.att
        x = a(z)[:, :, None] + b(z)[:, None, :] + e(edge) + g(graph)[:, None, None]
        return F.leaky_relu(x, negative_slope=0.01).permute(0, 3, 1, 2)

    def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **kwargs):
        b, n, _ = node_fts.shape
        z = torch.cat((node_fts, hidden), -1)
        values = self.m(z).reshape(b, n, self.nb_heads, self.head_size).transpose(1, 2)
        scores = self.logits(z, edge_fts, graph_fts)
        scores = scores.masked_fill(~adj_mat[:, None].bool(), -1e9)
        result = (scores.softmax(-1) @ values).transpose(1, 2).reshape(b, n, self.out_size)
        if self.residual:
            result = result + self.skip(z)
        if self.activation is not None:
            result = self.activation(result)
        return self.norm(result), None


class GATFull(GAT):
    def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **kwargs):
        return super().forward(node_fts, edge_fts, graph_fts, torch.ones_like(adj_mat), hidden)


class GATv2(GAT):
    def __init__(self, out_size, nb_heads=1, mid_size=None, activation=F.relu,
                 residual=True, use_ln=False, name='gatv2_aggr'):
        super().__init__(out_size, nb_heads, activation, residual, use_ln, name)
        self.mid_size = mid_size or out_size
        if self.mid_size % self.nb_heads:
            raise ValueError('The number of heads must divide the message width')
        self.att = nn.ModuleList([nn.LazyLinear(self.mid_size) for _ in range(4)])
        self.heads = nn.ModuleList([nn.Linear(self.mid_size // self.nb_heads, 1) for _ in range(self.nb_heads)])

    def logits(self, z, edge, graph):
        a, b, e, g = self.att
        x = a(z)[:, None, :] + b(z)[:, :, None] + e(edge) + g(graph)[:, None, None]
        x = x.reshape(*x.shape[:-1], self.nb_heads, self.mid_size // self.nb_heads)
        return torch.stack([head(F.leaky_relu(x[..., i, :], 0.01)).squeeze(-1)
                            for i, head in enumerate(self.heads)], 1)


class GATv2Full(GATv2):
    def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **kwargs):
        return super().forward(node_fts, edge_fts, graph_fts, torch.ones_like(adj_mat), hidden)


class GATv2FullD2(GATv2):
    def d2_forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **kwargs):
        if not all(len(x) == 4 for x in (node_fts, edge_fts, graph_fts)):
            raise ValueError('D2 inputs must contain four group actions')
        return [torch.stack([super(GATv2FullD2, self).forward(
            torch.cat((node_fts[g], node_fts[g ^ h]), -1),
            torch.cat((edge_fts[g], edge_fts[g ^ h]), -1),
            torch.cat((graph_fts[g], graph_fts[g ^ h]), -1),
            torch.ones_like(adj_mat), hidden)[0] for h in range(4)]).mean(0) for g in range(4)]


class TripletMessages(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.ModuleList([nn.LazyLinear(width) for _ in range(7)])

    def forward(self, z, edge, graph):
        a, b, c, e, f, g, h = self.layers
        return (a(z)[:, :, None, None] + b(z)[:, None, :, None] + c(z)[:, None, None, :]
                + e(edge)[:, :, :, None] + f(edge)[:, :, None, :] + g(edge)[:, None, :, :]
                + h(graph)[:, None, None, None])


class PGN(Processor):
    def __init__(self, out_size, mid_size=None, mid_act=None, activation=F.relu,
                 reduction=torch.amax, msgs_mlp_sizes=None, use_ln=False,
                 use_triplets=False, nb_triplet_fts=8, gated=False, name='mpnn_aggr'):
        super().__init__(name)
        self.out_size, self.mid_size = out_size, mid_size or out_size
        self.activation, self.mid_act, self.reduction = activation, mid_act, reduction
        self.use_triplets, self.gated = use_triplets, gated
        self.messages = nn.ModuleList([nn.LazyLinear(self.mid_size) for _ in range(4)])
        self.outputs = nn.ModuleList([nn.LazyLinear(out_size) for _ in range(2)])
        layers = []
        for i, width in enumerate(msgs_mlp_sizes or []):
            layers.extend([nn.ReLU(), nn.LazyLinear(width)])
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_size) if use_ln else nn.Identity()
        if use_triplets:
            if nb_triplet_fts <= 0:
                raise ValueError('Triplet feature width must be positive')
            self.triplets = TripletMessages(nb_triplet_fts)
            self.triplet_out = nn.Linear(nb_triplet_fts, out_size)
        if gated:
            self.gate1, self.gate2 = nn.LazyLinear(out_size), nn.LazyLinear(out_size)
            self.gate3 = nn.Linear(out_size, out_size)
            nn.init.constant_(self.gate3.bias, -3.)

    def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **kwargs):
        z = torch.cat((node_fts, hidden), -1)
        a, b, e, g = self.messages
        msgs = self.mlp(a(z)[:, None, :] + b(z)[:, :, None] + e(edge_fts) + g(graph_fts)[:, None, None])
        if self.mid_act is not None:
            msgs = self.mid_act(msgs)
        if self.reduction in (torch.amax, torch.max):
            msgs = msgs.masked_fill(~adj_mat.bool().unsqueeze(-1), -BIG_NUMBER).amax(1)
        elif self.reduction == torch.mean:
            msgs = (msgs * adj_mat.unsqueeze(-1)).sum(1) / adj_mat.sum(1).clamp_min(1).unsqueeze(-1)
        else:
            msgs = self.reduction(msgs * adj_mat.unsqueeze(-1), dim=1)
        ret = self.outputs[0](z) + self.outputs[1](msgs)
        if self.activation is not None:
            ret = self.activation(ret)
        ret = self.norm(ret)
        if self.gated:
            gate = self.gate3(F.relu(self.gate1(z) + self.gate2(msgs))).sigmoid()
            ret = ret * gate + hidden * (1 - gate)
        tri = None
        if self.use_triplets:
            tri = self.triplet_out(self.triplets(z, edge_fts, graph_fts).amax(1))
            if self.activation is not None:
                tri = self.activation(tri)
        return ret, tri


class DeepSets(PGN):
    def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **kwargs):
        adj = torch.eye(adj_mat.shape[-1], device=adj_mat.device).expand_as(adj_mat)
        return super().forward(node_fts, edge_fts, graph_fts, adj, hidden)


class MPNN(PGN):
    def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **kwargs):
        return super().forward(node_fts, edge_fts, graph_fts, torch.ones_like(adj_mat), hidden)


class PGNMask(PGN):
    @property
    def inf_bias(self):
        return True

    @property
    def inf_bias_edge(self):
        return True


class MemNetMasked(Processor):
    """End-to-end memory network, retaining upstream positional embeddings."""
    def __init__(self, vocab_size, sentence_size, linear_output_size,
                 embedding_size=16, memory_size=128, num_hops=1, nonlin=F.relu,
                 apply_embeddings=True, init_func=None, use_ln=False, name='memnet'):
        super().__init__(name)
        self.vocab_size, self.embedding_size = vocab_size, embedding_size
        self.memory_size, self.num_hops = memory_size, num_hops
        self.apply_embeddings, self.nonlin = apply_embeddings, nonlin
        self.register_buffer('encodings', torch.tensor(_position_encoding(sentence_size, embedding_size)))
        if apply_embeddings:
            self.query = nn.Parameter(torch.zeros(vocab_size-1, embedding_size))
            self.story = nn.Parameter(torch.zeros(vocab_size-1, embedding_size))
            self.output = nn.Parameter(torch.zeros(vocab_size-1, embedding_size))
            self.memory = nn.Parameter(torch.zeros(memory_size, embedding_size))
            if init_func is not None:
                for p in (self.query, self.story, self.output, self.memory):
                    with torch.no_grad():
                        init_func(p)
        self.intermediate = nn.Linear(embedding_size if apply_embeddings else sentence_size, embedding_size, bias=False) if num_hops > 1 else nn.Identity()
        self.final_hop = nn.Linear(embedding_size if apply_embeddings or num_hops > 1 else sentence_size, linear_output_size, bias=False)
        self.readout = nn.Linear(linear_output_size, vocab_size, bias=False)
        self.norm = nn.LayerNorm(vocab_size) if use_ln else nn.Identity()

    def embed(self, data, weight):
        weight = torch.cat((weight, weight.new_zeros(1, self.embedding_size)))
        # Match upstream take semantics for integer-valued embeddings; encoded
        # graph features may lie outside the vocabulary, so clamp their indices.
        indices = data.long().clamp(0, self.vocab_size-1)
        return (F.embedding(indices, weight) * self.encodings).sum(-2)

    def _apply_memory(self, queries, stories):
        if self.apply_embeddings:
            if stories.shape[-2] > self.memory_size:
                raise ValueError('Graph exceeds MemNet memory_size')
            memory = self.embed(stories, self.story)
            output = self.embed(stories, self.output)
            pad = self.memory_size - stories.shape[-2]
            memory = F.pad(memory, (0, 0, 0, pad)) + self.memory
            output = F.pad(output, (0, 0, 0, pad))
            query = self.embed(queries, self.query)
        else:
            memory, output, query = stories, stories, queries
        for hop in range(self.num_hops):
            weights = (memory * query.unsqueeze(-2)).sum(-1).softmax(-1)
            result = query + (weights.unsqueeze(-1) * output).sum(-2)
            result = self.final_hop(result) if hop == self.num_hops-1 else self.intermediate(result)
            query = result
            if self.nonlin is not None:
                result = self.nonlin(result)
        return self.norm(self.readout(result))

    def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **kwargs):
        queries = torch.cat((node_fts, graph_fts[:, None]), 1)
        stories = F.pad(edge_fts * adj_mat[..., None], (0, 0, 0, 1, 0, 1))
        result = self._apply_memory(queries, stories)
        return result[:, :-1] + result[:, -1:], None


class MemNetFull(MemNetMasked):
    def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **kwargs):
        return super().forward(node_fts, edge_fts, graph_fts, torch.ones_like(adj_mat), hidden)


ProcessorFactory = Callable[[int], Processor]


def get_processor_factory(kind, use_ln=False, nb_triplet_fts=8, nb_heads=1):
    """Construct all upstream processor variants as registered torch modules."""
    attention = {'gat': GAT, 'gat_full': GATFull, 'gatv2': GATv2, 'gatv2_full': GATv2Full}
    if kind in attention:
        return lambda width: attention[kind](width, nb_heads, use_ln=use_ln)
    if kind in ('memnet_full', 'memnet_masked'):
        cls = MemNetFull if kind == 'memnet_full' else MemNetMasked
        return lambda width: cls(width, width, width, use_ln=use_ln)
    triplet = kind.startswith('triplet_')
    base = kind.removeprefix('triplet_')
    gated = base.startswith('g')
    base = base[1:] if gated else base
    classes = {'deepsets': DeepSets, 'mpnn': MPNN, 'pgn': PGN, 'pgn_mask': PGNMask}
    if base not in classes or (base == 'deepsets' and (triplet or gated)):
        raise ValueError(f'Unexpected processor kind {kind}')
    return lambda width: classes[base](width, msgs_mlp_sizes=[width, width], use_ln=use_ln,
                                      use_triplets=triplet, nb_triplet_fts=nb_triplet_fts, gated=gated)


def _position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls, le = sentence_size+1, embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i-(le-1)/2) * (j-(ls-1)/2)
    return (1 + 4 * encoding / embedding_size / sentence_size).T
