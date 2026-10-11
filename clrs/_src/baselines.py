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

"""Trainable PyTorch baselines with optimizer and checkpoint management."""
from pathlib import Path
import torch
from torch import nn
from clrs._src import decoders, losses, model, nets, specs, tree
from clrs._src.dataset import to_torch


class BaselineModel(nn.Module, model.Model):
    def __init__(self, spec, dummy_trajectory, processor_factory, hidden_dim=32,
                 encode_hints=False, decode_hints=True, encoder_init='default',
                 use_lstm=False, learning_rate=0.005, grad_clip_max_norm=0.,
                 checkpoint_path='/tmp/clrs3', freeze_processor=False,
                 dropout_prob=0., hint_teacher_forcing=0., hint_repred_mode='soft',
                 name='base_model', nb_msg_passing_steps=1, debug=False,
                 device='cpu', seed=0):
        nn.Module.__init__(self)
        model.Model.__init__(self, spec)
        self.decode_hints = decode_hints
        self.checkpoint_path = Path(checkpoint_path)
        self.grad_clip_max_norm = grad_clip_max_norm
        self.learning_rate, self.freeze_processor = learning_rate, freeze_processor
        self._dummy = dummy_trajectory if isinstance(dummy_trajectory, list) else [dummy_trajectory]
        if len(self._dummy) != len(self._spec):
            raise ValueError('Provide one dummy trajectory per algorithm')
        dims = [{p.name: p.data.shape[-1] if p.type_ == specs.Type.CATEGORICAL else 1
                 for p in list(f.features.inputs)+list(f.features.hints)+list(f.outputs)} for f in self._dummy]
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.net = self._net_class()(self._spec, hidden_dim, encode_hints, decode_hints,
                processor_factory, use_lstm, encoder_init, dropout_prob, hint_teacher_forcing,
                hint_repred_mode, dims, nb_msg_passing_steps, debug)
        self.to(device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.optimizer = None
        self.init([x.features for x in self._dummy], seed)

    @staticmethod
    def _net_class():
        return nets.Net

    @property
    def device(self):
        return next(self.net.parameters()).device

    def init(self, features=None, seed=0):
        """Materialize all task modules before optimizer/checkpoint creation."""
        if self.optimizer is not None:
            self.generator.manual_seed(int(seed))
            return
        features = features or [x.features for x in self._dummy]
        if not isinstance(features, list):
            features = [features]
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(int(seed))
            for i, f in enumerate(features):
                self._materialize(to_torch(f, self.device), i)
        if self.freeze_processor:
            for parameter in self.net.processor.parameters():
                parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam((p for p in self.parameters() if p.requires_grad), lr=self.learning_rate)

    def _materialize(self, features, index):
        self.net(features, repred=True, algorithm_index=index)

    def forward(self, features, algorithm_index=0, repred=False, **kwargs):
        return self.net(to_torch(features, self.device), repred=repred,
                        algorithm_index=algorithm_index, generator=self.generator, **kwargs)

    def loss(self, feedback, algorithm_index=0):
        feedback = to_torch(feedback, self.device)
        result = self.forward(feedback.features, algorithm_index)
        outputs, hints = result[:2]
        n = nets._data_dimensions(feedback.features)[1]
        terms = {p.name: losses.output_loss(p, outputs[p.name], n) for p in feedback.outputs}
        if self.decode_hints:
            terms.update({'hint_' + p.name: losses.hint_loss(p, [h[p.name] for h in hints], feedback.features.lengths, n)
                          for p in feedback.features.hints})
        return sum(terms.values()), terms

    def compute_grad(self, feedback, algorithm_index=0):
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, _ = self.loss(feedback, algorithm_index)
        loss.backward()
        return loss.detach(), {name: None if p.grad is None else p.grad.detach().clone()
                               for name, p in self.named_parameters()}

    def feedback(self, feedback, algorithm_index=0):
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, _ = self.loss(feedback, algorithm_index)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        if self.grad_clip_max_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_max_norm)
        self.optimizer.step()
        return loss.detach()

    @torch.no_grad()
    def predict(self, features, algorithm_index=0, return_hints=True, return_all_outputs=False):
        self.eval()
        result = self.forward(features, algorithm_index, repred=True,
                              return_hints=return_hints, return_all_outputs=return_all_outputs)
        outputs, hints = result[:2]
        post = lambda x: decoders.postprocess(self._spec[algorithm_index], x, 0.1, 25, True)
        return post(outputs), [post(h) for h in hints] if hints is not None else []

    def verbose_loss(self, feedback, algorithm_index=0):
        return self.loss(feedback, algorithm_index)[1]

    def save_model(self, file_name='model.pt'):
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_path / file_name
        torch.save({'model': self.state_dict(), 'optimizer': self.optimizer.state_dict(),
                    'generator': self.generator.get_state(), 'specs': [dict(x) for x in self._spec]}, path)
        return path

    def restore_model(self, file_name='model.pt', only_load_processor=False):
        payload = torch.load(self.checkpoint_path / file_name, map_location=self.device, weights_only=True)
        if only_load_processor:
            prefix = 'net.processor.'
            self.net.processor.load_state_dict({k[len(prefix):]: v for k, v in payload['model'].items() if k.startswith(prefix)})
        else:
            if payload['specs'] != [dict(x) for x in self._spec]:
                raise ValueError('Checkpoint task specs do not match model')
            self.load_state_dict(payload['model'])
            self.optimizer.load_state_dict(payload['optimizer'])
            self.generator.set_state(payload['generator'].cpu())


class BaselineModelChunked(BaselineModel):
    @staticmethod
    def _net_class():
        return nets.NetChunked

    def __init__(self, *args, **kwargs):
        self.mp_states = {}
        super().__init__(*args, **kwargs)

    def _materialize(self, features, index):
        self.net(features, repred=True, algorithm_index=index)

    def reset_state(self):
        self.mp_states.clear()

    def forward(self, features, algorithm_index=0, repred=False, **kwargs):
        key = (repred, algorithm_index)
        result, state = self.net(to_torch(features, self.device), self.mp_states.get(key),
                                repred, algorithm_index, self.generator)
        self.mp_states[key] = state.detach()
        return result

    def loss(self, feedback, algorithm_index=0):
        feedback = to_torch(feedback, self.device)
        outputs, hints = self.forward(feedback.features, algorithm_index)
        n = nets._data_dimensions_chunked(feedback.features)[1]
        terms = {p.name: losses.output_loss_chunked(p, outputs[p.name], feedback.features.is_last, n)
                 for p in feedback.outputs}
        if self.decode_hints:
            terms.update({'hint_' + p.name: losses.hint_loss_chunked(p, hints[p.name], feedback.features.is_first, n)
                          for p in feedback.features.hints})
        return sum(terms.values()), terms

    @torch.no_grad()
    def predict(self, features, algorithm_index=0, return_hints=True, **kwargs):
        self.eval()
        outputs, hints = self.forward(features, algorithm_index, repred=True)
        post = lambda x: decoders.postprocess(self._spec[algorithm_index], x, 0.1, 25, True)
        return post(outputs), post(hints) if return_hints else {}

    def feedback(self, feedback, algorithm_index=0):
        features = feedback.features
        has_target = bool(torch.as_tensor(features.is_last).any()) or (
            self.decode_hints and bool((~torch.as_tensor(features.is_first).bool()).any()))
        if not has_target:
            self.train()
            self.optimizer.zero_grad(set_to_none=True)
            value, _ = self.loss(feedback, algorithm_index)
            return value.detach()
        return super().feedback(feedback, algorithm_index)

    def restore_model(self, *args, **kwargs):
        super().restore_model(*args, **kwargs)
        self.reset_state()
