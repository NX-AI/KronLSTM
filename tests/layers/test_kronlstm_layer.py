# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from fla.layers import kronlstm
from fla.models.utils import Cache
from fla.ops.kronlstm import naive_recurrent_kronlstm
from fla.utils import IS_NVIDIA, assert_close, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='the KronLSTM kernels are validated on NVIDIA GPUs only')

# the heads of an edge shape of tests/ops/test_kronlstm.py, whose kernel variants these tests reuse
LAYER_SMALL = dict(hidden_size=256, num_k_heads=4, num_v_heads=4, k_head_dim=32, v_head_dim=32)
LAYER_RULES = [
    pytest.param(dict(use_target_gate=False, decouple_erase=False), id='fixed_target'),
    pytest.param(dict(use_target_gate=True, decouple_erase=True), id='target_gate-decouple_erase'),
]


@pytest.fixture(scope='module', autouse=True)
def ieee_fp32_dots():
    # fp32 `tl.dot` defaults to tf32; the fp32 comparisons against the token-by-token reference need ieee
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('TRITON_F32_DEFAULT', 'ieee')
        yield


@pytest.mark.parametrize('rule_kw', LAYER_RULES)
def test_kronlstm_decode_with_cache_matches_full_forward(rule_kw):
    """One call over the prompt, then one-token calls through the cache, against one chunkwise call over the sequence."""
    torch.manual_seed(2)
    layer = kronlstm.KronLSTM(**LAYER_SMALL, mode='chunk', use_short_conv=False, layer_idx=0,
                              **rule_kw).to(device).eval()
    x = torch.randn(2, 100, 256, device=device)
    with torch.no_grad():
        full = layer(x)[0]
        cache = Cache()
        parts = [layer(x[:, :80], past_key_values=cache, use_cache=True)[0]]
        for t in range(80, 100):
            parts.append(layer(x[:, t:t + 1], past_key_values=cache, use_cache=True)[0])
    assert_close('o', full, torch.cat(parts, 1), 1e-4)


@pytest.mark.parametrize('rule_kw', LAYER_RULES)
@pytest.mark.parametrize('decay_gate', ['vector', 'scalar'])
def test_kronlstm_modes_agree_and_train(decay_gate, rule_kw, monkeypatch):
    """Both modes against the same layer with its op replaced by the token-by-token reference; the chunk mode trains."""
    torch.manual_seed(1)
    x = torch.randn(2, 96, 256, device=device)
    layers = {}
    for mode in ('chunk', 'fused_recurrent'):
        layer = kronlstm.KronLSTM(**LAYER_SMALL, mode=mode, decay_gate=decay_gate, layer_idx=0,
                                  **rule_kw).to(device)
        if layers:
            layer.load_state_dict(layers['chunk'].state_dict())
        layers[mode] = layer
    outs = {'chunk': layers['chunk'](x)[0]}
    # with gradients enabled the layer always runs the chunk kernels; the token-by-token one only without
    with torch.no_grad():
        outs['fused_recurrent'] = layers['fused_recurrent'](x)[0]
    with monkeypatch.context() as m, torch.no_grad():
        m.setattr(kronlstm, 'chunk_kronlstm',
                  lambda cu_seqlens, **kwargs: naive_recurrent_kronlstm(**kwargs))
        ref = layers['chunk'](x)[0]
    assert_close('o chunk', ref, outs['chunk'], 1e-4)
    assert_close('o fused_recurrent', ref, outs['fused_recurrent'], 1e-5)
    outs['chunk'].square().mean().backward()
    for name, p in layers['chunk'].named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0, name


@pytest.mark.parametrize('mode', ['chunk', 'fused_recurrent'])
def test_kronlstm_eval_with_grad_uses_chunk(mode):
    """In eval mode with autograd on, short inputs must not take the forward-only token-by-token kernel."""
    torch.manual_seed(42)
    layer = kronlstm.KronLSTM(**LAYER_SMALL, mode=mode, layer_idx=0).to(device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(1, 32, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    with mock.patch(
        'fla.layers.kronlstm.fused_recurrent_kronlstm',
        side_effect=AssertionError('eval with autograd must not use the forward-only fused recurrent kernel'),
    ) as fused_recurrent:
        y = layer(x)[0]
        y.sum().backward()
    fused_recurrent.assert_not_called()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize('rule_kw', LAYER_RULES)
def test_kronlstm_padding_mask_runs_each_sequence_on_its_own(rule_kw):
    """A padding mask packs the valid positions into one variable-length batch.
    Neither the recurrence nor the short convolution crosses a boundary there,
    so the outputs equal dense runs of each sequence alone; the padded positions come back as zeros,
    and the layer trains through it."""
    torch.manual_seed(3)
    layer = kronlstm.KronLSTM(**LAYER_SMALL, mode='chunk', layer_idx=0, **rule_kw).to(device)
    x = torch.randn(2, 48, 256, device=device, requires_grad=True)
    valid = [(0, 48), (10, 41)]                     # the second sequence is padded on both sides
    attention_mask = torch.zeros(2, 48, device=device, dtype=torch.long)
    for i, (s, e) in enumerate(valid):
        attention_mask[i, s:e] = 1
    out = layer(x, attention_mask=attention_mask)[0]
    for i, (s, e) in enumerate(valid):
        ref = layer(x[i:i + 1, s:e])[0]
        assert_close(f'o[{i}]', ref[0], out[i, s:e], 1e-4)
    assert (out[1, :10] == 0).all() and (out[1, 41:] == 0).all()
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_kronlstm_cu_seqlens_is_the_packed_padding_mask():
    """Explicit `cu_seqlens` on a packed `[1, total]` input runs the padding-mask path on the same sequences,
    in both the chunk kernels and the token-by-token one."""
    torch.manual_seed(4)
    layer = kronlstm.KronLSTM(**LAYER_SMALL, mode='chunk', layer_idx=0).to(device)
    x = torch.randn(2, 40, 256, device=device)
    attention_mask = torch.ones(2, 40, device=device, dtype=torch.long)
    attention_mask[0, 25:] = 0
    packed = torch.cat([x[0, :25], x[1]]).unsqueeze(0)
    # int32 offsets, as the padding-mask path derives them: the two paths then run the very same kernel variants
    cu_seqlens = torch.tensor([0, 25, 65], device=device, dtype=torch.int32)
    with torch.no_grad():
        masked = layer(x, attention_mask=attention_mask)[0]
        o = layer(packed, cu_seqlens=cu_seqlens)[0]
        assert torch.equal(o[0, :25], masked[0, :25]) and torch.equal(o[0, 25:], masked[1])
        # in eval mode without gradients, short packed inputs take the token-by-token kernel
        layer.eval()
        o = layer(packed, cu_seqlens=cu_seqlens)[0]
        ref = torch.cat([layer(x[:1, :25])[0][0], layer(x[1:])[0][0]])
    assert_close('o', ref, o[0], 1e-5)


def test_kronlstm_constructor_and_gate_init():
    """Unequal key and value head counts and head dims outside the chunk kernels' range are refused;
    `reset_gate_parameters` restores the layer's own init (decay, output-gate bias, zero output-gate weights)
    after a meta-device build re-initialised with another scheme."""
    with pytest.raises(ValueError, match='head count'):
        kronlstm.KronLSTM(**{**LAYER_SMALL, 'num_v_heads': 8})
    for dims in [dict(k_head_dim=8), dict(v_head_dim=512)]:
        with pytest.raises(ValueError, match='head dims'):
            kronlstm.KronLSTM(**{**LAYER_SMALL, **dims})

    layer = kronlstm.KronLSTM(**LAYER_SMALL)
    no_wd = {n for n, p in layer.named_parameters() if getattr(p, '_no_weight_decay', False)}
    assert no_wd == {'A_log', 'dt_bias', 'g_proj.bias', 'o_norm.weight'}

    with torch.device('meta'):
        layer = kronlstm.KronLSTM(**LAYER_SMALL)
    layer = layer.to_empty(device=device)
    for p in layer.parameters():
        torch.nn.init.normal_(p, std=0.02)
    layer.reset_gate_parameters(generator=torch.Generator(device=device).manual_seed(0))
    A, dt = layer.A_log.exp(), F.softplus(layer.dt_bias)
    assert (A > 0).all() and (A < 16).all()
    assert (dt > 1e-3 * 0.999).all() and (dt < 0.1 * 1.001).all()
    assert (layer.g_proj.weight == 0).all() and (layer.g_proj.bias == 3.0).all()


@pytest.mark.parametrize('regime', ['bf16-params', 'autocast'])
def test_kronlstm_bf16_regimes(regime):
    """bf16 parameters without autocast, as FSDP's mixed precision runs the layer,
    and fp32 parameters under bf16 autocast: either way the op gets bf16 q, k and v, the output is bf16
    and every parameter gets a finite gradient."""
    torch.manual_seed(1)
    layer = kronlstm.KronLSTM(**LAYER_SMALL, layer_idx=0).to(device)
    if regime == 'bf16-params':
        layer = layer.to(torch.bfloat16)
    x = torch.randn(2, 96, 256, device=device, dtype=torch.bfloat16 if regime == 'bf16-params' else torch.float32)
    with mock.patch('fla.layers.kronlstm.chunk_kronlstm', wraps=kronlstm.chunk_kronlstm) as op, \
            torch.autocast('cuda', dtype=torch.bfloat16, enabled=regime == 'autocast'):
        o = layer(x)[0]
    assert {op.call_args.kwargs[n].dtype for n in ('q', 'k', 'v')} == {torch.bfloat16}
    assert o.dtype == torch.bfloat16
    o.float().square().mean().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
