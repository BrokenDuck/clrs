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

"""PyTorch encode-process-decode networks and recurrent chunk state."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from clrs._src import decoders, encoders, probing, specs, tree


def _expand_to(x, y):
    return x.reshape(x.shape + (1,) * (y.ndim-x.ndim))


def _data_dimensions(features):
    for dp in features.inputs:
        if dp.location in (specs.Location.NODE, specs.Location.EDGE):
            return dp.data.shape[:2]
    raise ValueError('Inputs must include nodes or edges')


def _data_dimensions_chunked(features):
    return _data_dimensions(features._replace(inputs=tree.map_arrays(lambda x: x[0], features.inputs)))


class Net(nn.Module):
    def __init__(self, spec, hidden_dim, encode_hints, decode_hints,
                 processor_factory, use_lstm=False, encoder_init='default',
                 dropout_prob=0., hint_teacher_forcing=0., hint_repred_mode='soft',
                 nb_dims=None, nb_msg_passing_steps=1, debug=False, name='net'):
        super().__init__()
        if encode_hints and not decode_hints:
            raise ValueError('Encoding hints requires decoding hints for autoregressive evaluation')
        if hint_repred_mode not in ('soft', 'hard', 'hard_on_eval'):
            raise ValueError('Invalid hint_repred_mode')
        if not 0 <= hint_teacher_forcing <= 1 or not 0 <= dropout_prob < 1:
            raise ValueError('Invalid teacher forcing or dropout probability')
        if nb_msg_passing_steps < 1:
            raise ValueError('At least one message passing step is required')
        self.spec = spec if isinstance(spec, list) else [spec]
        self.hidden_dim, self.encode_hints, self.decode_hints = hidden_dim, encode_hints, decode_hints
        self.use_lstm, self.encoder_init = use_lstm, encoder_init
        self.dropout_prob, self.hint_teacher_forcing = dropout_prob, hint_teacher_forcing
        self.hint_repred_mode, self.nb_msg_passing_steps = hint_repred_mode, nb_msg_passing_steps
        self.debug = debug
        self.processor = processor_factory(hidden_dim)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim) if use_lstm else None
        self.encoders, self.decoders = nn.ModuleList(), nn.ModuleList()
        shared_algo_encoder = None
        for i, task in enumerate(self.spec):
            enc, dec = nn.ModuleDict(), nn.ModuleDict()
            for name, (stage, loc, typ) in task.items():
                if stage == specs.Stage.INPUT or (stage == specs.Stage.HINT and encode_hints):
                    if name == specs.ALGO_IDX_INPUT_NAME and shared_algo_encoder is not None:
                        enc[name] = shared_algo_encoder
                    else:
                        enc[name] = encoders.construct_encoders(stage, loc, typ, hidden_dim, encoder_init, name)
                        if name == specs.ALGO_IDX_INPUT_NAME:
                            shared_algo_encoder = enc[name]
                if stage == specs.Stage.OUTPUT or (stage == specs.Stage.HINT and decode_hints):
                    dims = nb_dims[i].get(name, 1) if nb_dims is not None else 1
                    dec[name] = decoders.construct_decoders(loc, typ, hidden_dim, dims, name)
            self.encoders.append(enc)
            self.decoders.append(dec)

    def _one_step_pred(self, inputs, hints, hidden, lstm_state, algorithm_index, repred, generator=None):
        b, n, _ = hidden.shape
        nodes = hidden.new_zeros(b, n, self.hidden_dim)
        edges = hidden.new_zeros(b, n, n, self.hidden_dim)
        graph = hidden.new_zeros(b, self.hidden_dim)
        adj = torch.eye(n, device=hidden.device).expand(b, n, n)
        for original in list(inputs) + (list(hints) if self.encode_hints else []):
            dp = encoders.preprocess(original, n)
            enc = self.encoders[algorithm_index][dp.name]
            adj = encoders.accum_adj_mat(dp, adj)
            edges = encoders.accum_edge_fts(enc, dp, edges)
            nodes = encoders.accum_node_fts(enc, dp, nodes)
            graph = encoders.accum_graph_fts(enc, dp, graph)
        nxt = hidden
        for _ in range(self.nb_msg_passing_steps):
            nxt, nxt_edge = self.processor(nodes, edges, graph, adj, nxt)
        if not repred and self.dropout_prob:
            keep = torch.rand(nxt.shape, device=nxt.device, generator=generator) >= self.dropout_prob
            nxt = nxt * keep / (1-self.dropout_prob)
        if self.lstm is not None:
            if lstm_state is None:
                lstm_state = (torch.zeros_like(nxt), torch.zeros_like(nxt))
            h, c = self.lstm(nxt.reshape(b*n, -1), tuple(x.reshape(b*n, -1) for x in lstm_state))
            nxt, lstm_state = h.reshape(b, n, -1), (h.reshape(b, n, -1), c.reshape(b, n, -1))
        h_t = torch.cat((nodes, hidden, nxt), -1)
        e_t = edges if nxt_edge is None else torch.cat((edges, nxt_edge), -1)
        hints, outputs = decoders.decode_fts(self.decoders[algorithm_index], self.spec[algorithm_index],
                                             h_t, adj, e_t, graph, self.processor.inf_bias,
                                             self.processor.inf_bias_edge, repred, generator)
        return nxt, outputs, hints, lstm_state

    def _choose_hints(self, truth, previous, first, n, algorithm_index, repred, generator):
        if not self.encode_hints or not previous:
            return truth
        hard = self.hint_repred_mode == 'hard' or (repred and self.hint_repred_mode == 'hard_on_eval')
        predicted = decoders.postprocess(self.spec[algorithm_index], previous, 0.1, 25, hard)
        forced = first.bool()
        if not repred:
            forced = forced | (torch.rand(first.shape, device=first.device, generator=generator) < self.hint_teacher_forcing)
        result = []
        for dp in truth:
            pred = predicted[dp.name]
            value = dp.data
            typ = dp.type_
            if pred.type_ == specs.Type.SOFT_POINTER:
                value = F.one_hot(value.long(), n).float()
                typ = pred.type_
            data = torch.where(_expand_to(forced, value), value, pred.data)
            result.append(probing.DataPoint(dp.name, dp.location, typ, data))
        return result

    def forward(self, features_list, repred=False, algorithm_index=0,
                return_hints=True, return_all_outputs=False, generator=None):
        if not isinstance(features_list, list):
            features_list = [features_list]
        if algorithm_index == -1:
            result = None
            for i, features in enumerate(features_list):
                result = self.forward(features, repred, i, return_hints, return_all_outputs, generator)
            return result
        if len(features_list) != 1:
            raise ValueError('Pass one task per forward call')
        features = features_list[0]
        b, n = _data_dimensions(features)
        device = features.inputs[0].data.device
        hidden = torch.zeros(b, n, self.hidden_dim, device=device)
        lstm_state, previous, outputs = None, {}, {}
        hint_steps, output_steps, hiddens = [], [], []
        steps = max(1, int(features.lengths.max().item()) - 1)
        for i in range(steps):
            truth = [probing.DataPoint(h.name, h.location, h.type_, h.data[min(i, h.data.shape[0]-1)]) for h in features.hints]
            first = torch.full((b,), i == 0, device=device, dtype=torch.bool)
            hints = self._choose_hints(truth, previous, first, n, algorithm_index, repred, generator)
            hidden, candidate, previous, lstm_state = self._one_step_pred(features.inputs, hints, hidden,
                                                                       lstm_state, algorithm_index, repred, generator)
            outputs = candidate if i == 0 else {k: torch.where(_expand_to(features.lengths > i+1, v), v, outputs[k]) for k, v in candidate.items()}
            hint_steps.append(previous)
            output_steps.append(outputs)
            hiddens.append(hidden)
        if return_all_outputs:
            outputs = {k: torch.stack([x[k] for x in output_steps]) for k in outputs}
        result = (outputs, hint_steps if return_hints else None)
        return (*result, torch.stack(hiddens)) if self.debug else result


@dataclass
class MessagePassingStateChunked:
    hiddens: torch.Tensor
    hint_preds: dict
    inputs: list
    hints: list
    is_first: torch.Tensor
    lstm_state: tuple | None = None

    def detach(self):
        return MessagePassingStateChunked(*(tree.map_arrays(lambda x: x.detach(), x)
            for x in (self.hiddens, self.hint_preds, self.inputs, self.hints,
                      self.is_first, self.lstm_state)))


class NetChunked(Net):
    """Delay the pending step so cross-chunk targets retain their gradients.

    At time t, consume the stored inputs/hints from t-1 and supervise the
    prediction against hints[t]. Output supervision occurs on is_last[t].
    Initial records seed the pending step; singleton records take one step.
    """
    def forward(self, features, state=None, repred=False, algorithm_index=0, generator=None):
        b, n = _data_dimensions_chunked(features)
        device = features.inputs[0].data.device
        if state is None:
            if not features.is_first[0].bool().all():
                raise ValueError('A new chunk stream must begin with is_first for every lane')
            state = MessagePassingStateChunked(
                torch.zeros(b, n, self.hidden_dim, device=device), {},
                tree.map_arrays(lambda x: x[0], features.inputs),
                tree.map_arrays(lambda x: x[0], features.hints), features.is_first[0].bool())
        if state.hiddens.shape[:2] != (b, n):
            raise ValueError('Reset chunk state before changing graph or batch size')
        outputs, hints = [], []
        for t in range(features.is_first.shape[0]):
            first, last = features.is_first[t].bool(), features.is_last[t].bool()
            inp = tree.map_arrays(lambda x: x[t], features.inputs)
            truth = tree.map_arrays(lambda x: x[t], features.hints)
            select = lambda old, new: torch.where(_expand_to(first, new), new, old)
            pending_inputs = tree.map_arrays(select, state.inputs, inp)
            pending_hints = tree.map_arrays(select, state.hints, truth)
            pending_first = first | state.is_first
            current = self._choose_hints(pending_hints, state.hint_preds, pending_first,
                                         n, algorithm_index, repred, generator)
            hidden = state.hiddens.masked_fill(first[:, None, None], 0)
            lstm = tree.map_arrays(lambda x: x.masked_fill(first[:, None, None], 0), state.lstm_state)
            hidden, out, nxt_hints, lstm = self._one_step_pred(
                pending_inputs, current, hidden, lstm, algorithm_index, repred, generator)
            # The first hint is given, not predicted. Its transition is deferred
            # until the next record supplies a target, including across chunks.
            hidden = hidden.masked_fill((first & ~last)[:, None, None], 0)
            lstm = tree.map_arrays(lambda x: x.masked_fill((first & ~last)[:, None, None], 0), lstm)
            outputs.append(out)
            hints.append(nxt_hints)
            state = MessagePassingStateChunked(hidden, nxt_hints, inp, truth, first, lstm)
        stack = lambda sequence: {k: torch.stack([x[k] for x in sequence]) for k in sequence[0]}
        return (stack(outputs), stack(hints)), state
