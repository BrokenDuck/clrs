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
"""Differentiable, masked CLRS losses for torch tensors."""
import torch
from torch.nn import functional as F
from clrs._src import specs

EPS = 1e-12


def _expand_to(x, y):
    while x.ndim < y.ndim:
        x = x.unsqueeze(-1)
    return x


def _expand_and_broadcast_to(x, y):
    return _expand_to(x, y).expand_as(y)


def _hint_loss(truth_data, truth_type, pred, nb_nodes):
    truth_data = torch.as_tensor(truth_data, device=pred.device)
    t = specs.Type
    mask = None
    if truth_type == t.SCALAR:
        loss = (pred - truth_data.float()).square()
    elif truth_type == t.MASK:
        loss = F.binary_cross_entropy_with_logits(pred, truth_data.float(), reduction='none')
        mask = truth_data != specs.OutputClass.MASKED
    elif truth_type in (t.MASK_ONE, t.CATEGORICAL):
        valid = truth_data == specs.OutputClass.POSITIVE
        target = truth_data.masked_fill(truth_data == specs.OutputClass.MASKED, 0).float()
        loss = -(target * pred.log_softmax(-1)).sum(-1)
        mask = valid.any(-1)
    elif truth_type in (t.POINTER, t.SHOULD_BE_PERMUTATION):
        loss = -pred.log_softmax(-1).gather(-1, truth_data.long().unsqueeze(-1)).squeeze(-1)
    elif truth_type == t.PERMUTATION_POINTER:
        loss = -(truth_data.float() * pred).sum(-1)
    else:
        raise ValueError(f'Unsupported loss type {truth_type}')
    return loss, torch.ones_like(loss, dtype=torch.bool) if mask is None else mask


def _reduce(loss, mask):
    mask = mask.expand_as(loss)
    return loss.masked_fill(~mask.bool(), 0).sum() / mask.sum().clamp_min(1)


def output_loss(truth, pred, nb_nodes):
    return _reduce(*_hint_loss(truth.data, truth.type_, pred, nb_nodes))


def output_loss_chunked(truth, pred, is_last, nb_nodes):
    loss, mask = _hint_loss(truth.data, truth.type_, pred, nb_nodes)
    return _reduce(loss, mask & _expand_to(is_last.bool(), loss))


def hint_loss(truth, preds, lengths, nb_nodes, verbose=False):
    if not len(preds):
        device = torch.as_tensor(truth.data).device
        loss = torch.zeros((), device=device)
    else:
        pred = torch.stack(preds) if isinstance(preds, list) else preds
        # A one-state trajectory has no hint target; the model still takes one
        # processor step to predict its output.
        count = min(pred.shape[0], truth.data.shape[0]-1)
        loss, mask = _hint_loss(truth.data[1:count+1], truth.type_, pred[:count], nb_nodes)
        times = torch.arange(count, device=pred.device)[:, None] + 1
        valid = times < torch.as_tensor(lengths, device=pred.device)
        loss = _reduce(loss, mask & _expand_to(valid, loss))
    return {'loss_' + truth.name: loss} if verbose else loss


def hint_loss_chunked(truth, pred, is_first, nb_nodes):
    loss, mask = _hint_loss(truth.data, truth.type_, pred, nb_nodes)
    return _reduce(loss, mask & _expand_to(~is_first.bool(), loss))


def _is_not_done_broadcast(lengths, i, tensor):
    return _expand_to(lengths > i+1, tensor)
