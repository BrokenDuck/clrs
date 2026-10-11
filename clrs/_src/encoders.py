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
"""Encoder utilities."""

import functools
import torch
from torch import nn
from torch.nn import functional as F
from clrs._src import probing
from clrs._src import specs

_Array = torch.Tensor
_DataPoint = probing.DataPoint
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Type = specs.Type


def construct_encoders(stage: str, loc: str, t: str,
                       hidden_dim: int, init: str, name: str):
  """Constructs encoders."""
  if init not in ('default', 'xavier_on_scalars'):
    raise ValueError(f'Encoder initialiser {init} not supported.')
  def linear():
    return ScalarLazyLinear(hidden_dim, scalar_init=(
        init == 'xavier_on_scalars' and stage == _Stage.HINT and t == _Type.SCALAR))
  return nn.ModuleList([linear() for _ in range(2 if loc == _Location.EDGE and t == _Type.POINTER else 1)])


class ScalarLazyLinear(nn.LazyLinear):
  def __init__(self, hidden_dim, scalar_init=False):
    self.scalar_init = scalar_init
    super().__init__(hidden_dim)

  def initialize_parameters(self, inputs):
    pending = self.has_uninitialized_params()
    super().initialize_parameters(inputs)
    if pending and self.scalar_init:
      nn.init.trunc_normal_(self.weight, std=self.out_features ** -0.5)
      nn.init.zeros_(self.bias)


def preprocess(dp: _DataPoint, nb_nodes: int) -> _DataPoint:
  """Pre-process data point.

  Make sure that the data is ready to be encoded into features.
  If the data is of POINTER type, we expand the compressed index representation
  to a full one-hot. But if the data is a SOFT_POINTER, the representation
  is already expanded and we just overwrite the type as POINTER so that
  it is treated as such for encoding.

  Args:
    dp: A DataPoint to prepare for encoding.
    nb_nodes: Number of nodes in the graph, necessary to expand pointers to
      the right dimension.
  Returns:
    The datapoint, with data and possibly type modified.
  """
  new_type = dp.type_
  if dp.type_ in (_Type.POINTER, _Type.SHOULD_BE_PERMUTATION):
    data = F.one_hot(dp.data.long(), nb_nodes).float()
    new_type = _Type.POINTER
  else:
    data = dp.data.float()
    if dp.type_ == _Type.SOFT_POINTER:
      new_type = _Type.POINTER
  dp = probing.DataPoint(
      name=dp.name, location=dp.location, type_=new_type, data=data)

  return dp


def accum_adj_mat(dp: _DataPoint, adj_mat: _Array) -> _Array:
  """Accumulates adjacency matrix."""
  if dp.location == _Location.NODE and dp.type_ in [_Type.POINTER,
                                                    _Type.PERMUTATION_POINTER]:
    adj_mat = adj_mat + ((dp.data + dp.data.transpose(1, 2)) > 0.5)
  elif dp.location == _Location.EDGE and dp.type_ == _Type.MASK:
    adj_mat = adj_mat + ((dp.data + dp.data.transpose(1, 2)) > 0.0)

  return (adj_mat > 0.).float()


def accum_edge_fts(encoders, dp: _DataPoint, edge_fts: _Array) -> _Array:
  """Encodes and accumulates edge features."""
  if dp.location == _Location.NODE and dp.type_ in [_Type.POINTER,
                                                    _Type.PERMUTATION_POINTER]:
    encoding = _encode_inputs(encoders, dp)
    edge_fts = edge_fts + encoding

  elif dp.location == _Location.EDGE:
    encoding = _encode_inputs(encoders, dp)
    if dp.type_ == _Type.POINTER:
      # Aggregate pointer contributions across sender and receiver nodes.
      encoding_2 = encoders[1](dp.data.unsqueeze(-1))
      edge_fts = edge_fts + encoding.mean(dim=1) + encoding_2.mean(dim=2)
    else:
      edge_fts = edge_fts + encoding

  return edge_fts


def accum_node_fts(encoders, dp: _DataPoint, node_fts: _Array) -> _Array:
  """Encodes and accumulates node features."""
  is_pointer = (dp.type_ in [_Type.POINTER, _Type.PERMUTATION_POINTER])
  if ((dp.location == _Location.NODE and not is_pointer) or
      (dp.location == _Location.GRAPH and dp.type_ == _Type.POINTER)):
    encoding = _encode_inputs(encoders, dp)
    node_fts = node_fts + encoding

  return node_fts


def accum_graph_fts(encoders, dp: _DataPoint,
                    graph_fts: _Array) -> _Array:
  """Encodes and accumulates graph features."""
  if dp.location == _Location.GRAPH and dp.type_ != _Type.POINTER:
    encoding = _encode_inputs(encoders, dp)
    graph_fts = graph_fts + encoding

  return graph_fts


def _encode_inputs(encoders, dp: _DataPoint) -> _Array:
  if dp.type_ == _Type.CATEGORICAL:
    encoding = encoders[0](dp.data)
  else:
    encoding = encoders[0](dp.data.unsqueeze(-1))
  return encoding
