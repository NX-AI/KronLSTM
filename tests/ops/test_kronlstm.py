# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import functools

import pytest
import torch
import torch.nn.functional as F
from triton.runtime.autotuner import Autotuner
from triton.runtime.errors import OutOfResources

from fla.ops.kda.naive import naive_recurrent_kda
from fla.ops.kronlstm import (
    chunk_bwd,
    chunk_fwd,
    chunk_kronlstm,
    fused_recurrent_kronlstm,
    naive_chunk_kronlstm,
    naive_recurrent_kronlstm,
)
from fla.ops.kronlstm.naive import V_NORM_FLOOR
from fla.utils import IS_NVIDIA, assert_close, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='the KronLSTM kernels are validated on NVIDIA GPUs only')

TOL = {torch.float32: 5e-6, torch.bfloat16: 2e-2}
RULES = {'A': dict(decouple_erase=False), 'D': dict(target_gate=True, decouple_erase=False),
         'E': dict(target_gate=True, decouple_erase=True)}
GATES = {'none': None, 'scalar': 'gs', 'vector': 'gv'}
DTYPES = {'fp32': torch.float32, 'bf16': torch.bfloat16}


@pytest.fixture(scope='module', autouse=True)
def ieee_fp32_dots():
    # fp32 `tl.dot` defaults to tf32, which reads ~1e-3 against the float64 reference; the fp32 tolerances need ieee
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('TRITON_F32_DEFAULT', 'ieee')
        yield


def gen(B, T, H, K, V, seed=0, gmag=0.3):
    torch.manual_seed(seed)
    d = dict(dtype=torch.float64, device=device)
    return dict(
        q=F.normalize(torch.randn(B, T, H, K, **d), dim=-1),
        k=F.normalize(torch.randn(B, T, H, K, **d), dim=-1),
        v=F.normalize(torch.randn(B, T, H, V, **d), dim=-1),
        beta=torch.rand(B, T, H, **d) * 0.9 + 0.05,
        gs=-(torch.rand(B, T, H, **d) * 0.9 + 0.1) * gmag,
        gv=-(torch.rand(B, T, H, K, **d) * 0.9 + 0.1) * gmag,
        ct=torch.sigmoid(torch.randn(B, T, H, **d) * 5.0),
        S0=torch.randn(B, H, K, V, **d) * 0.1,
        do=torch.randn(B, T, H, V, **d),
        dS=torch.randn(B, H, K, V, **d) * 0.1,
    )


def rule_kwargs(x, rule, dt):
    # s = 2, the full range of the write strength; the layer tests run its default, s = 1
    kw = dict(RULES[rule], allow_neg_eigval=True)
    if kw.pop('target_gate', False):
        kw['target_gate'] = x['ct'].to(torch.float32 if dt == torch.bfloat16 else dt)
    return kw


def run_grads(op, x, dt, gate, rule, chunk_size=None, initial_state=True):
    """Outputs, final state and the gradients of every input of `op` under a fixed random cotangent."""
    gk = GATES[gate]
    names = ['q', 'k', 'v', 'beta'] + ([gk] if gk else []) + (['S0'] if initial_state else [])
    leaves = {n: x[n].to(dt).detach().clone().requires_grad_(True) for n in names}
    kw = rule_kwargs(x, rule, dt)
    if 'target_gate' in kw:
        kw['target_gate'] = kw['target_gate'].detach().clone().requires_grad_(True)
        leaves['ct'] = kw['target_gate']
    if chunk_size is not None:
        kw['chunk_size'] = chunk_size
    o, S = op(leaves['q'], leaves['k'], leaves['v'], leaves.get(gk) if gk else None, leaves['beta'],
              initial_state=leaves.get('S0'), output_final_state=True, **kw)
    loss = (o.double() * x['do']).sum() + (S.double() * x['dS']).sum()
    grads = torch.autograd.grad(loss, list(leaves.values()))
    return dict(zip(leaves.keys(), grads)), o, S


def assert_all_close(ref, tri, ratio, err_atol=1e-6):
    (g_ref, o_ref, S_ref), (g_tri, o_tri, S_tri) = ref, tri
    assert_close('o', o_ref, o_tri, ratio, err_atol=err_atol)
    assert_close('ht', S_ref, S_tri, ratio, err_atol=err_atol)
    for name in g_ref:
        assert_close(f'd{name}', g_ref[name], g_tri[name], ratio, err_atol=err_atol)


@pytest.mark.parametrize('rule', list(RULES))
def test_naive_recurrent_is_kda_in_kv_space(rule):
    """The state recursion is KDA's with key vec(k v^T), the target as value, beta* as beta and the gate repeated over V."""
    B, T, H, K, V = 2, 96, 3, 16, 8
    x = gen(B, T, H, K, V)
    kw = rule_kwargs(x, rule, torch.float64)
    _, S = naive_recurrent_kronlstm(x['q'], x['k'], x['v'], x['gv'], x['beta'], scale=1.0,
                                    initial_state=x['S0'], output_final_state=True, **kw)
    a = (x['k'].unsqueeze(-1) * x['v'].unsqueeze(-2)).reshape(B, T, H, K * V)
    vn = x['v'].norm(dim=-1) ** 2
    target = vn if rule == 'A' else vn * x['ct']
    if rule == 'E':
        target = target / x['beta']
    beta_star = 2.0 / vn.clamp_min(V_NORM_FLOOR) * x['beta']
    _, S_kda = naive_recurrent_kda(a, a, target.unsqueeze(-1), x['gv'].repeat_interleave(V, dim=-1), beta_star,
                                   scale=1.0, initial_state=x['S0'].reshape(B, H, K * V, 1), output_final_state=True)
    # KDA's reference accumulates in fp32
    assert_close('ht', S_kda.reshape(B, H, K, V), S, 3e-6)


@pytest.mark.parametrize('rule', list(RULES))
@pytest.mark.parametrize('gate', ['none', 'scalar'])
def test_naive_recurrent_all_equal_vector_gate_is_the_scalar_gate(gate, rule):
    x = gen(2, 64, 3, 16, 8)
    kw = rule_kwargs(x, rule, torch.float64)
    g = None if gate == 'none' else x['gs']
    g_eq = torch.zeros_like(x['gv']) if gate == 'none' else x['gs'].unsqueeze(-1).expand_as(x['gv']).contiguous()
    a = naive_recurrent_kronlstm(x['q'], x['k'], x['v'], g, x['beta'], initial_state=x['S0'], output_final_state=True, **kw)
    b = naive_recurrent_kronlstm(x['q'], x['k'], x['v'], g_eq, x['beta'], initial_state=x['S0'], output_final_state=True,
                                 **kw)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize('rule', list(RULES))
@pytest.mark.parametrize('gate', list(GATES))
def test_naive_chunk(gate, rule):
    """The chunkwise reference against the token-by-token one: output, state and every gradient, in float64."""
    x = gen(2, 128, 3, 16, 8)
    ref = run_grads(naive_recurrent_kronlstm, x, torch.float64, gate, rule)
    tri = run_grads(naive_chunk_kronlstm, x, torch.float64, gate, rule, chunk_size=32)
    assert_all_close(ref, tri, 1e-12, err_atol=0.0)


def test_naive_chunk_gradcheck():
    x = gen(1, 24, 2, 4, 4, seed=3)
    names = ['q', 'k', 'v', 'gv', 'beta', 'S0', 'ct']
    leaves = [x[n].clone().requires_grad_(True) for n in names]

    def fn(q, k, v, g, beta, S0, ct):
        return naive_chunk_kronlstm(q, k, v, g, beta, chunk_size=8, initial_state=S0, output_final_state=True,
                                    allow_neg_eigval=True, target_gate=ct, decouple_erase=True)
    assert torch.autograd.gradcheck(fn, leaves, eps=1e-6, atol=1e-6, rtol=1e-5)


# Every (H, K, V, dtype) compiles all configs of the six autotuned kernels, so the shapes are few
# and the other tests reuse them. Every variant, gate rank and dtype runs on the main shape
# (two key blocks, a partial last chunk), and so do larger gates: every exponent the kernels form is referenced
# so that it is <= 0.
# Two variants run on each edge shape: odd T below one chunk with K < 64 (the layer tests' heads),
# one key block with V = 128, V = 256, and K = 256 (four key blocks).
CHUNK_MAIN = (2, 200, 2, 128, 128)
CHUNK_EDGES = [
    ((1, 63, 4, 32, 32), 'fp32'),
    ((1, 63, 4, 32, 32), 'bf16'),
    ((2, 160, 2, 64, 128), 'bf16'),
    ((1, 128, 2, 128, 256), 'fp32'),
    ((1, 100, 2, 256, 64), 'bf16'),
]


def chunk_case(B, T, H, K, V, rule, gate, dtype, gmag=0.3):
    gid = '' if gmag == 0.3 else f'-g{gmag}'
    return pytest.param(B, T, H, K, V, rule, gate, dtype, gmag, id=f'B{B}-T{T}-H{H}-K{K}-V{V}-{rule}-{gate}-{dtype}{gid}')


CHUNK_CASES = (
    [chunk_case(*CHUNK_MAIN, rule, gate, dtype) for rule in RULES for gate in GATES for dtype in DTYPES]
    + [chunk_case(*CHUNK_MAIN, 'A', 'vector', 'fp32', gmag) for gmag in (1.0, 4.0, 8.0)]
    + [chunk_case(*shape, rule, 'vector', dtype) for shape, dtype in CHUNK_EDGES for rule in ('A', 'E')]
)


@pytest.mark.parametrize(('B', 'T', 'H', 'K', 'V', 'rule', 'gate', 'dtype', 'gmag'), CHUNK_CASES)
def test_chunk(B, T, H, K, V, rule, gate, dtype, gmag):
    x = gen(B, T, H, K, V, gmag=gmag)
    ref = run_grads(naive_chunk_kronlstm, x, torch.float64, gate, rule)
    tri = run_grads(chunk_kronlstm, x, DTYPES[dtype], gate, rule)
    # dg is a difference of nearly cancelling terms, which costs a digit in fp32 at |g| >= 4 per step
    assert_all_close(ref, tri, TOL[DTYPES[dtype]] * (4 if gmag >= 4 else 1))


def test_chunk_resume_is_bitwise():
    """Two calls chained through the final state equal one call over the whole sequence, at chunk boundaries."""
    x = gen(1, 256, 2, 128, 128)
    args = [x[n].float() for n in ('q', 'k', 'v', 'gv', 'beta')]
    with torch.no_grad():
        o_full, S_full = chunk_kronlstm(*args, output_final_state=True, decouple_erase=False)
        o1, S1 = chunk_kronlstm(*(t[:, :128] for t in args), output_final_state=True, decouple_erase=False)
        o2, S2 = chunk_kronlstm(*(t[:, 128:] for t in args), initial_state=S1, output_final_state=True,
                                decouple_erase=False)
    assert torch.equal(torch.cat([o1, o2], 1), o_full) and torch.equal(S2, S_full)


def test_chunk_backward_is_bitwise_repeatable():
    x = gen(2, 512, 2, 128, 128, seed=3)
    g_base, _, _ = run_grads(chunk_kronlstm, x, torch.bfloat16, 'vector', 'E')
    for _ in range(20):
        g_new, _, _ = run_grads(chunk_kronlstm, x, torch.bfloat16, 'vector', 'E')
        for name in g_base:
            assert torch.equal(g_new[name], g_base[name]), name


# variable-length inputs: N sequences packed along T with B = 1 and `cu_seqlens`;
# the references are dense runs of each sequence on its own.
# Sequences of any length, an empty one, K < 64 and four key blocks (K = 256), on the shapes of `test_chunk`
VARLEN_CASES = [
    pytest.param(cu, H, K, V, rule, gate, dtype, id=f'N{len(cu) - 1}-T{cu[-1]}-H{H}-K{K}-V{V}-{rule}-{gate}-{dtype}')
    for cu, H, K, V, rule, gate, dtype in [
        ([0, 15, 64, 64, 500, 513, 1000], 2, 128, 128, 'E', 'vector', 'fp32'),
        ([0, 15, 64, 64, 500, 513, 1000], 2, 128, 128, 'D', 'scalar', 'bf16'),
        ([0, 15, 64, 64, 500, 513, 1000], 2, 128, 128, 'A', 'none', 'fp32'),
        ([0, 1, 33, 160], 4, 32, 32, 'E', 'vector', 'fp32'),
        ([0, 200, 456], 2, 256, 64, 'A', 'vector', 'bf16'),
    ]
]


def gen_packed(cu, H, K, V, seed=0):
    """`gen` for `N = len(cu) - 1` sequences packed along T, with one initial state and one state cotangent per sequence."""
    x = gen(1, cu[-1], H, K, V, seed=seed)
    N = len(cu) - 1
    x['S0'], x['dS'] = x['S0'].expand(N, -1, -1, -1).clone(), x['dS'].expand(N, -1, -1, -1).clone()
    x['S0'] += torch.randn_like(x['S0']) * 0.1
    x['dS'] += torch.randn_like(x['dS']) * 0.1
    return x


def run_segments(op, x, dt, gate, rule, cu, packed, grads=True):
    """`run_grads` on packed sequences: one call with `cu_seqlens`, or one dense call per sequence, concatenated.

    An empty sequence contributes no outputs and passes its initial state through."""
    gk = GATES[gate]
    names = ['q', 'k', 'v', 'beta'] + ([gk] if gk else []) + ['S0']
    leaves = {n: x[n].to(dt).detach().clone().requires_grad_(grads) for n in names}
    kw = rule_kwargs(x, rule, dt)
    if 'target_gate' in kw:
        kw['target_gate'] = kw['target_gate'].detach().clone().requires_grad_(grads)
        leaves['ct'] = kw['target_gate']
    g = leaves.get(gk) if gk else None
    if packed:
        o, S = op(leaves['q'], leaves['k'], leaves['v'], g, leaves['beta'], initial_state=leaves['S0'],
                  output_final_state=True, cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=device), **kw)
    else:
        outs, states = [], []
        for i, (s, e) in enumerate(zip(cu[:-1], cu[1:])):
            if s == e:
                states.append(leaves['S0'][i:i + 1])
                continue
            kw_i = {**kw, **({'target_gate': kw['target_gate'][:, s:e]} if 'target_gate' in kw else {})}
            o_i, S_i = op(leaves['q'][:, s:e], leaves['k'][:, s:e], leaves['v'][:, s:e],
                          g[:, s:e] if g is not None else None, leaves['beta'][:, s:e], initial_state=leaves['S0'][i:i + 1],
                          output_final_state=True, **kw_i)
            outs.append(o_i)
            states.append(S_i)
        o, S = torch.cat(outs, 1), torch.cat(states, 0)
    if not grads:
        return {}, o, S
    loss = (o.double() * x['do']).sum() + (S.double() * x['dS']).sum()
    return dict(zip(leaves.keys(), torch.autograd.grad(loss, list(leaves.values())))), o, S


@pytest.mark.parametrize(('cu', 'H', 'K', 'V', 'rule', 'gate', 'dtype'), VARLEN_CASES)
def test_chunk_varlen(cu, H, K, V, rule, gate, dtype):
    """A packed run equals dense runs of each sequence on its own: outputs, final states and every gradient."""
    x = gen_packed(cu, H, K, V)
    ref = run_segments(chunk_kronlstm, x, DTYPES[dtype], gate, rule, cu, packed=False)
    tri = run_segments(chunk_kronlstm, x, DTYPES[dtype], gate, rule, cu, packed=True)
    assert_all_close(ref, tri, TOL[DTYPES[dtype]])


def test_rejects_unsupported_inputs():
    """Head dims outside the kernels' range are refused, and so are fp32 inputs whose state the scan cannot stage;
    `cu_seqlens` needs the sequences packed along T with B = 1 and one initial state per sequence."""
    for K, V in [(8, 32), (32, 8), (512, 32), (32, 512)]:
        x = gen(1, 16, 2, K, V)
        inputs = [x[n].float() for n in ('q', 'k', 'v', 'gv', 'beta')]
        with pytest.raises(ValueError, match='head dims'):
            chunk_kronlstm(*inputs, decouple_erase=False)
        if max(K, V) > 256:
            with pytest.raises(ValueError, match='K, V <= 256'):
                fused_recurrent_kronlstm(*inputs, decouple_erase=False)
    x = gen(1, 16, 2, 256, 256)
    with pytest.raises(ValueError, match='fp32 inputs'):
        chunk_kronlstm(*(x[n].float() for n in ('q', 'k', 'v', 'gv', 'beta')), decouple_erase=False)

    x = gen(2, 64, 2, 32, 32)
    args = [x[n].float() for n in ('q', 'k', 'v', 'gv', 'beta')]
    cu = torch.tensor([0, 32, 64], device=device)
    with pytest.raises(ValueError, match='batch size'):
        chunk_kronlstm(*args, cu_seqlens=cu, decouple_erase=False)
    with pytest.raises(ValueError, match='initial states'):
        chunk_kronlstm(*(t[:1] for t in args), initial_state=x['S0'][:1].float(), cu_seqlens=cu,
                       decouple_erase=False)
    with pytest.raises(ValueError, match='batch size'):
        fused_recurrent_kronlstm(*args, cu_seqlens=cu, decouple_erase=False)


def _autotuner(kernel):
    while not isinstance(kernel, Autotuner):
        kernel = kernel.fn
    return kernel


AUTOTUNERS = {
    'fwd_h': _autotuner(chunk_fwd.chunk_kronlstm_fwd_kernel_h),
    'fwd_intra_diag': _autotuner(chunk_fwd.chunk_kronlstm_fwd_kernel_intra_diag),
    'fwd_inter': _autotuner(chunk_fwd.chunk_kronlstm_fwd_kernel_inter),
    'bwd_dhu': _autotuner(chunk_bwd.chunk_kronlstm_bwd_kernel_dhu),
    'bwd_intra': _autotuner(chunk_bwd.chunk_kronlstm_bwd_kernel_intra),
    'bwd_local': _autotuner(chunk_bwd.chunk_kronlstm_bwd_kernel_local),
}
AUTOTUNE_CASES = [pytest.param(name, i, id=f'{name}-{i}')
                  for name, tuner in AUTOTUNERS.items() for i in range(len(tuner.configs))]


@functools.cache
def _autotune_reference():
    return run_grads(naive_chunk_kronlstm, gen(*CHUNK_MAIN), torch.float64, 'vector', 'D')


@pytest.mark.parametrize(('kernel', 'index'), AUTOTUNE_CASES)
def test_chunk_every_autotune_config(kernel, index, monkeypatch):
    """Whichever config an autotuner may select, forward and backward stay within the bf16 tolerance of the reference."""
    tuner = AUTOTUNERS[kernel]
    monkeypatch.setattr(tuner, 'configs', [tuner.configs[index]])
    try:
        tri = run_grads(chunk_kronlstm, gen(*CHUNK_MAIN), torch.bfloat16, 'vector', 'D')
    except OutOfResources:
        # the autotuner discards a config that does not fit the shape the same way
        pytest.skip('the config does not fit in shared memory at this shape')
    assert_all_close(_autotune_reference(), tri, TOL[torch.bfloat16])


REC_TOL = {torch.float32: 1e-5, torch.bfloat16: 1e-2}
# every variant, gate rank and dtype on the main shape, with and without an initial state;
# two variants on each edge shape: one token with the layer tests' heads, V = 256, and V < K
REC_MAIN = (2, 37, 2, 128, 128)
REC_EDGES = [((1, 1, 4, 32, 32), 'fp32'), ((2, 96, 2, 128, 256), 'bf16'), ((3, 5, 2, 32, 16), 'fp32')]


def rec_case(B, T, H, K, V, rule, gate, dtype, use_h0=True):
    hid = '' if use_h0 else '-no_h0'
    return pytest.param(B, T, H, K, V, rule, gate, dtype, use_h0, id=f'B{B}-T{T}-H{H}-K{K}-V{V}-{rule}-{gate}-{dtype}{hid}')


REC_CASES = (
    [rec_case(*REC_MAIN, rule, gate, dtype) for rule in RULES for gate in GATES for dtype in DTYPES]
    + [rec_case(*REC_MAIN, rule, 'vector', 'fp32', use_h0=False) for rule in ('A', 'E')]
    + [rec_case(*shape, rule, 'vector', dtype) for shape, dtype in REC_EDGES for rule in ('A', 'E')]
)


def rec_call(op, x, dt, gate, rule, initial_state=True, **kwargs):
    gk = GATES[gate]
    g = x[gk].float() if gk else None
    return op(*(x[n].to(dt) for n in ('q', 'k', 'v')), g, x['beta'].to(dt),
              initial_state=x['S0'].float() if initial_state else None, output_final_state=True, **rule_kwargs(x, rule, dt),
              **kwargs)


@pytest.mark.parametrize(('B', 'T', 'H', 'K', 'V', 'rule', 'gate', 'dtype', 'use_h0'), REC_CASES)
def test_fused_recurrent(B, T, H, K, V, rule, gate, dtype, use_h0):
    """The forward-only decode kernel against the token-by-token reference; there is no backward to test."""
    x = gen(B, T, H, K, V)
    o_ref, S_ref = rec_call(naive_recurrent_kronlstm, x, DTYPES[dtype], gate, rule, initial_state=use_h0)
    o_tri, S_tri = rec_call(fused_recurrent_kronlstm, x, DTYPES[dtype], gate, rule, initial_state=use_h0)
    assert_close('o', o_ref, o_tri, REC_TOL[DTYPES[dtype]])
    assert_close('ht', S_ref, S_tri, REC_TOL[DTYPES[dtype]])


@pytest.mark.parametrize(('cu', 'H', 'K', 'V', 'rule', 'gate', 'dtype'), [
    pytest.param(*p, id='N{}-T{}-H{}-K{}-V{}-{}-{}-{}'.format(len(p[0]) - 1, p[0][-1], *p[1:]))
    for p in [
        ([0, 5, 42, 43, 100], 2, 128, 128, 'E', 'vector', 'fp32'),
        ([0, 5, 42, 43, 100], 2, 128, 128, 'E', 'vector', 'bf16'),
        ([0, 1, 1, 2, 3], 2, 128, 128, 'D', 'scalar', 'fp32'),   # one-token sequences, one of them empty
    ]
])
def test_fused_recurrent_varlen(cu, H, K, V, rule, gate, dtype):
    """A packed decode run equals runs of each sequence on its own, in outputs and final states."""
    x = gen_packed(cu, H, K, V)
    dt = DTYPES[dtype]
    _, o_ref, S_ref = run_segments(fused_recurrent_kronlstm, x, dt, gate, rule, cu, packed=False, grads=False)
    _, o_tri, S_tri = run_segments(fused_recurrent_kronlstm, x, dt, gate, rule, cu, packed=True, grads=False)
    assert_close('o', o_ref, o_tri, REC_TOL[dt])
    assert_close('ht', S_ref, S_tri, REC_TOL[dt])


@pytest.mark.parametrize('rule', ['D', 'E'])
def test_fused_recurrent_decoding_matches_chunk(rule):
    """64 one-token steps chained through the final state land where the chunkwise kernel lands over the sequence."""
    x = gen(2, 64, 2, 128, 128, seed=5)
    o_chunk, S_chunk = rec_call(chunk_kronlstm, x, torch.float32, 'vector', rule)
    S = x['S0'].float()
    outs = []
    for t in range(64):
        xt = {n: (x[n][:, t:t + 1] if n not in ('S0', 'dS') else x[n]) for n in x}
        xt['S0'] = S
        o_t, S = rec_call(fused_recurrent_kronlstm, xt, torch.float32, 'vector', rule)
        outs.append(o_t)
    assert_close('o', o_chunk, torch.cat(outs, 1), 1e-5)
    assert_close('ht', S_chunk, S, 1e-5)
