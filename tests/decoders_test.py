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

import pytest
import torch
from clrs._src import decoders


@pytest.mark.parametrize('zero_diagonal', [True, False])
def test_sinkhorn(zero_diagonal):
    x = torch.randn(10, 10, generator=torch.Generator().manual_seed(42), requires_grad=True)
    y = decoders.log_sinkhorn(x, 25, 1., zero_diagonal, None).exp()
    torch.testing.assert_close(y.sum(-1), torch.ones(10), atol=1e-4, rtol=0)
    torch.testing.assert_close(y.sum(-2), torch.ones(10), atol=1e-4, rtol=0)
    if zero_diagonal:
        assert y.diagonal().sum() == 0
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all()
