# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from triton.runtime.autotuner import Autotuner

from fla.layers import attn
from fla.layers.kronlstm import OUT_GATE_BIAS_INIT
from fla.models import KronLSTMConfig, KronLSTMForCausalLM
from fla.models.kronlstm.modeling_kronlstm import NoRotary
from fla.ops.utils import cumsum
from fla.utils import IS_NVIDIA, assert_close, device

from .test_modeling_base import (
    run_test_generate_matches_forward,
    run_test_generation,
    run_test_model_forward_backward,
)

requires_kernels = pytest.mark.skipif(not IS_NVIDIA, reason='the KronLSTM kernels are validated on NVIDIA GPUs only')
requires_flash_attn = pytest.mark.skipif(attn.flash_attn_func is None, reason='the attention layers need flash-attn')

# the paper's hybrid layout at toy size: attention without positional encoding (NoPE) in a sliding window
NOPE_SWA = {'layers': [1], 'num_heads': 4, 'window_size': 64, 'use_rope': False}


@pytest.fixture
def one_cumsum_config(monkeypatch):
    """One config for the decay's chunk-local cumsum.

    Its autotuner keys on the batch size and on IS_VARLEN, so the fixed-length and the packed forward of test_modeling
    can select different block sizes. Their fp32 sums then differ in the last bits, which the bf16 model amplifies
    past the test's tolerance (seen as a rare failure). The other forward kernels' keys are the same for both passes.
    """
    tuner = cumsum.chunk_local_cumsum_vector_kernel
    while not isinstance(tuner, Autotuner):
        tuner = tuner.fn
    monkeypatch.setattr(tuner, 'configs', tuner.configs[:1])
    monkeypatch.setattr(tuner, 'cache', {})


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@requires_kernels
@pytest.mark.usefixtures('one_cumsum_config')
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, True,  torch.bfloat16),
            (4, 4, 1024, 4, 64, False, torch.bfloat16),
        ]
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    use_l2warp: bool,
    dtype: torch.dtype,
):
    run_test_model_forward_backward(L, B, T, H, D, KronLSTMConfig, use_l2warp=use_l2warp, dtype=dtype)


# ===================================================================================
# Test for Generation
# ===================================================================================
@requires_kernels
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 4, 2000, 8, 64, torch.float16),
        ]
    ],
)
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    run_test_generation(L, B, T, H, D, KronLSTMConfig, dtype)


@requires_kernels
def test_generate_prefill():
    run_test_generate_matches_forward(2, 2, 64, 8, 64, KronLSTMConfig, torch.float32)


# ===================================================================================
# Test for the hybrid with attention without positional encoding
# ===================================================================================
@requires_kernels
@requires_flash_attn
def test_nope_sliding_window_attention():
    """The attention layer against softmax attention without rotary embedding over the last `window_size` positions,
    the token itself included, on a sequence longer than the window."""
    torch.manual_seed(0)
    config = KronLSTMConfig(hidden_size=256, num_hidden_layers=2, num_heads=4, vocab_size=1000, attn=NOPE_SWA)
    layer = KronLSTMForCausalLM(config).model.layers[1].attn.to(device=device, dtype=torch.bfloat16)
    assert isinstance(layer.rotary, NoRotary)
    B, T, H, W = 2, 200, NOPE_SWA['num_heads'], NOPE_SWA['window_size']
    x = torch.randn(B, T, config.hidden_size, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        o = layer(x)[0]
        q, k, v = (proj(x).float().view(B, T, H, -1).transpose(1, 2) for proj in (layer.q_proj, layer.k_proj, layer.v_proj))
        i, j = torch.arange(T, device=device)[:, None], torch.arange(T, device=device)[None, :]
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=(j <= i) & (j > i - W))
        ref = layer.o_proj(ref.transpose(1, 2).reshape(B, T, -1).to(torch.bfloat16))
    assert_close('o', ref, o, 5e-3)


@requires_kernels
@requires_flash_attn
def test_generation_nope_sliding_window_hybrid():
    """Cached generation through both mixers, decoding past the attention window."""
    torch.manual_seed(0)
    config = KronLSTMConfig(hidden_size=256, num_hidden_layers=2, num_heads=4, vocab_size=1000, attn=NOPE_SWA)
    model = AutoModelForCausalLM.from_config(config)
    run_test_generation(2, 4, 512, 4, 64, KronLSTMConfig, torch.bfloat16, model=model, config=config, tol=1e-2)


def test_config_use_rope(tmp_path):
    config = KronLSTMConfig(num_hidden_layers=3, attn={'layers': [1], 'num_heads': 2})
    assert config.attn['use_rope'] is True
    config = KronLSTMConfig(num_hidden_layers=3, attn=[{'layers': [1], 'num_heads': 2, 'use_rope': False}])
    config.save_pretrained(tmp_path)
    assert KronLSTMConfig.from_pretrained(tmp_path).attn[0]['use_rope'] is False
    with pytest.raises(ValueError, match=r"use_rope.*'no'"):
        KronLSTMConfig(num_hidden_layers=3, attn={'layers': [1], 'num_heads': 2, 'use_rope': 'no'})


# ===================================================================================
# Test for the initialisation
# ===================================================================================
def test_init_weights(monkeypatch):
    """The paper's (lm-engine's) initialisation, with KronLSTM's gates initialised by the layer."""
    monkeypatch.setattr(attn, 'flash_attn_func', object())  # builds the attention layer without flash-attn
    torch.manual_seed(0)
    L, std = 4, 0.02
    config = KronLSTMConfig(hidden_size=512, num_hidden_layers=L, num_heads=4, vocab_size=1000, initializer_range=std,
                            attn={'layers': [3], 'num_heads': 4, 'use_rope': False})
    model = KronLSTMForCausalLM(config)
    assert model.lm_head.weight is model.model.embeddings.weight
    for layer in model.model.layers[:3]:
        mixer = layer.attn
        assert torch.equal(mixer.g_proj.weight, torch.zeros_like(mixer.g_proj.weight))
        assert torch.all(mixer.g_proj.bias == OUT_GATE_BIAS_INIT)
        assert mixer.A_log.exp().min() > 0 and mixer.A_log.exp().max() < 16
        dt = F.softplus(mixer.dt_bias)
        assert dt.min() >= 1e-4 * (1 - 1e-6) and dt.max() <= 0.1 * (1 + 1e-6)
        assert torch.all(mixer.o_norm.weight == 1)
    weights = {
        'kronlstm projection': model.model.layers[0].attn.qkv_proj.weight,
        'attention projection': model.model.layers[3].attn.q_proj.weight,
        'mlp projection': model.model.layers[0].mlp.up_proj.weight,
        'embedding': model.model.embeddings.weight,
        'kronlstm output': model.model.layers[0].attn.o_proj.weight,
        'attention output': model.model.layers[3].attn.o_proj.weight,
        'mlp output': model.model.layers[0].mlp.down_proj.weight,
    }
    for name, weight in weights.items():
        target = std / math.sqrt(2 * L) if name.endswith('output') else std
        assert abs(weight.std().item() / target - 1) < 0.05, f'{name}: std {weight.std().item():.5f}, expected {target:.5f}'


def test_from_pretrained_keeps_every_weight(tmp_path):
    """Loading must not re-initialise anything: the layer's gate init writes A_log and dt_bias in place."""
    torch.manual_seed(0)
    config = KronLSTMConfig(hidden_size=256, num_hidden_layers=2, num_heads=2, vocab_size=1000)
    model = KronLSTMForCausalLM(config)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) + 1)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert isinstance(loaded, KronLSTMForCausalLM)
    assert loaded.lm_head.weight is loaded.model.embeddings.weight
    loaded_state = loaded.state_dict()
    assert loaded_state.keys() == state.keys()
    for k, v in state.items():
        assert torch.equal(loaded_state[k], v), k
