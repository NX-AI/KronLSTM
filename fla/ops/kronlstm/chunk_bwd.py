# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Chunkwise KronLSTM backward: `chunk_kronlstm_bwd`, called by `chunk.ChunkKronLSTMFunction`.

The stages follow GDN-2's backward: KDA's `chunk_kda_bwd_dAv` for the read-out Gram,
the reverse scan `chunk_kronlstm_bwd_kernel_dhu`, the key Gram's cotangent `chunk_kronlstm_bwd_kernel_dAkk`,
both Grams' gradients into q and k in `chunk_kronlstm_bwd_kernel_intra`,
the fused chunk-local gradients in `chunk_kronlstm_bwd_kernel_local`, and a reverse cumsum for the gate;
`README.md` has the derivation.

The chunk state gradients `dh` are stored in the input dtype, as GDN-2 stores them; `dh0` is fp32.
"""

import torch
import triton
import triton.language as tl

from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs, check_shared_mem

NUM_WARPS = [4, 8]
NUM_STAGES = [1, 2, 3] if check_shared_mem('ampere') else [1, 2]


def _prune_dhu_configs(configs, nargs, **kwargs):
    # ptxas 12.8, which Triton 3.7 bundles, miscompiles 8 warps with 1 stage on Hopper when V <= 32:
    # at K <= 64 a wgmma descriptor is built from uniform registers that are never written,
    # so the kernel reads shared memory out of range, and at K >= 128 the kernel faults with an illegal address too.
    # ptxas 13.1 compiles both correctly. At V <= 32 that config is at most 1.5% faster than the next one,
    # so it is dropped there for every K.
    args = {**(nargs or {}), **kwargs}
    if args['V'] <= 32:
        return [c for c in configs if (c.num_warps, c.num_stages) != (8, 1)]
    return configs


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[triton.Config({}, num_warps=nw, num_stages=ns) for nw in NUM_WARPS for ns in NUM_STAGES],
    key=['H', 'K', 'V', 'BT'],
    prune_configs_by={'early_config_prune': _prune_dhu_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kronlstm_bwd_kernel_dhu(
    q_up,       # [B, T, H, K]   q * 2^gc
    k_up,       # [B, T, H, K]   k * 2^gc
    k_tail,     # [B, T, H, K]   k * 2^(gc_C - gc)
    v,          # [B, T, H, V]
    gc,         # [B, T, H, K]
    Tinv,       # [B, T, H, BT]
    bs,         # [B, T, H]      beta*
    do,         # [B, T, H, V]
    dv_intra,   # [B, T, H, V]   A_qk^T do
    dv_new,     # out [B, T, H, V]     total cotangent of u * v
    dr,         # out [B, T, H]        Tinv^T du
    dh,         # out [B, NT, H, K, V] dh[n] = dS_{n+1}, the gradient of the state after chunk n
    dh0,        # out [N, H, K, V] or None
    dht,        # [N, H, K, V] or None
    scale,
    cu_seqlens,     # [N + 1] or None
    chunk_offsets,  # [N + 1] the first chunk of each sequence (varlen only)
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos = i_n * T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    dh += (boh * H + i_h).to(tl.int64) * K * V
    v += (bos * H + i_h).to(tl.int64) * V
    do += (bos * H + i_h).to(tl.int64) * V
    dv_intra += (bos * H + i_h).to(tl.int64) * V
    dv_new += (bos * H + i_h).to(tl.int64) * V
    q_up += (bos * H + i_h).to(tl.int64) * K
    k_up += (bos * H + i_h).to(tl.int64) * K
    k_tail += (bos * H + i_h).to(tl.int64) * K
    gc += (bos * H + i_h).to(tl.int64) * K
    bs += (bos * H + i_h).to(tl.int64)
    dr += (bos * H + i_h).to(tl.int64)
    Tinv += (bos * H + i_h).to(tl.int64) * BT
    if USE_INITIAL_STATE:
        dh0 = dh0 + i_nh * K * V
    if USE_FINAL_STATE_GRADIENT:
        dht = dht + i_nh * K * V

    o_v = tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    o_j = tl.arange(0, BT)

    if USE_FINAL_STATE_GRADIENT:
        b_dh1 += tl.load(dht + o_k1[:, None] * V + o_v[None, :], mask=m_k1[:, None] & m_v[None, :], other=0.).to(tl.float32)
        if K > 64:
            b_dh2 += tl.load(
                dht + o_k2[:, None] * V + o_v[None, :],
                mask=m_k2[:, None] & m_v[None, :],
                other=0.,
            ).to(tl.float32)
        if K > 128:
            b_dh3 += tl.load(
                dht + o_k3[:, None] * V + o_v[None, :],
                mask=m_k3[:, None] & m_v[None, :],
                other=0.,
            ).to(tl.float32)
        if K > 192:
            b_dh4 += tl.load(
                dht + o_k4[:, None] * V + o_v[None, :],
                mask=m_k4[:, None] & m_v[None, :],
                other=0.,
            ).to(tl.float32)

    for i_t in range(NT - 1, -1, -1):
        i_t64 = i_t.to(tl.int64)
        o_t = i_t64 * BT + o_j
        m_t = o_t < T
        m_tv = m_t[:, None] & m_v[None, :]

        # dh[n] = dS_{n+1}, the gradient of the state this chunk produced
        p_dh = dh + i_t64 * H * K * V
        tl.store(p_dh + o_k1[:, None] * V + o_v[None, :], b_dh1.to(p_dh.dtype.element_ty), mask=m_k1[:, None] & m_v[None, :])
        if K > 64:
            tl.store(p_dh + o_k2[:, None] * V + o_v[None, :],
                     b_dh2.to(p_dh.dtype.element_ty), mask=m_k2[:, None] & m_v[None, :])
        if K > 128:
            tl.store(p_dh + o_k3[:, None] * V + o_v[None, :],
                     b_dh3.to(p_dh.dtype.element_ty), mask=m_k3[:, None] & m_v[None, :])
        if K > 192:
            tl.store(p_dh + o_k4[:, None] * V + o_v[None, :],
                     b_dh4.to(p_dh.dtype.element_ty), mask=m_k4[:, None] & m_v[None, :])

        # dv_new = dv_intra + K_tail dS_{n+1}
        b_kt = tl.load(k_tail + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None] & m_k1[None, :], other=0.)
        b_dv = tl.dot(b_kt, b_dh1.to(b_kt.dtype))
        if K > 64:
            b_kt = tl.load(k_tail + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None] & m_k2[None, :], other=0.)
            b_dv += tl.dot(b_kt, b_dh2.to(b_kt.dtype))
        if K > 128:
            b_kt = tl.load(k_tail + o_t[:, None] * (H * K) + o_k3[None, :], mask=m_t[:, None] & m_k3[None, :], other=0.)
            b_dv += tl.dot(b_kt, b_dh3.to(b_kt.dtype))
        if K > 192:
            b_kt = tl.load(k_tail + o_t[:, None] * (H * K) + o_k4[None, :], mask=m_t[:, None] & m_k4[None, :], other=0.)
            b_dv += tl.dot(b_kt, b_dh4.to(b_kt.dtype))
        b_dv += tl.load(dv_intra + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.).to(tl.float32)
        tl.store(dv_new + o_t[:, None] * (H * V) + o_v[None, :], b_dv.to(dv_new.dtype.element_ty), mask=m_tv)

        # back through u = Tinv (pay - beta* p), du -> dr -> dp; the forward's coupling over V, mirrored:
        # one scalar per row from a sum over all of V
        b_v = tl.load(v + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.).to(tl.float32)
        b_du = tl.sum(b_dv * b_v, 1)                                                  # [BT]
        b_Ti = tl.load(Tinv + o_t[:, None] * (H * BT) + o_j[None, :], mask=m_t[:, None], other=0.).to(tl.float32)
        b_dr = tl.sum(b_Ti * b_du[:, None], 0)                                        # dr_j = sum_t Tinv[t,j] du_t
        tl.store(dr + o_t * H, b_dr, mask=m_t)
        b_bs = tl.load(bs + o_t * H, mask=m_t, other=0.).to(tl.float32)
        b_dpv = ((-b_bs * b_dr)[:, None] * b_v).to(k_up.dtype.element_ty)             # (dp . v)  [BT, BV]
        b_do = tl.load(do + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.)

        # dS_n = D dS_{n+1} + scale Q_up^T do + K_up^T (dp . v)
        last_idx = min((i_t + 1) * BT, T) - 1
        p_gl = gc + (last_idx * H).to(tl.int64) * K
        b_dh1 *= exp2(tl.load(p_gl + o_k1, mask=m_k1, other=0.).to(tl.float32))[:, None]
        b_qt = tl.load(q_up + o_k1[:, None] + o_t[None, :] * (H * K), mask=m_k1[:, None] & m_t[None, :], other=0.)
        b_kut = tl.load(k_up + o_k1[:, None] + o_t[None, :] * (H * K), mask=m_k1[:, None] & m_t[None, :], other=0.)
        b_dh1 += tl.dot(b_qt, b_do) * scale + tl.dot(b_kut, b_dpv)
        if K > 64:
            b_dh2 *= exp2(tl.load(p_gl + o_k2, mask=m_k2, other=0.).to(tl.float32))[:, None]
            b_qt = tl.load(q_up + o_k2[:, None] + o_t[None, :] * (H * K), mask=m_k2[:, None] & m_t[None, :], other=0.)
            b_kut = tl.load(k_up + o_k2[:, None] + o_t[None, :] * (H * K), mask=m_k2[:, None] & m_t[None, :], other=0.)
            b_dh2 += tl.dot(b_qt, b_do) * scale + tl.dot(b_kut, b_dpv)
        if K > 128:
            b_dh3 *= exp2(tl.load(p_gl + o_k3, mask=m_k3, other=0.).to(tl.float32))[:, None]
            b_qt = tl.load(q_up + o_k3[:, None] + o_t[None, :] * (H * K), mask=m_k3[:, None] & m_t[None, :], other=0.)
            b_kut = tl.load(k_up + o_k3[:, None] + o_t[None, :] * (H * K), mask=m_k3[:, None] & m_t[None, :], other=0.)
            b_dh3 += tl.dot(b_qt, b_do) * scale + tl.dot(b_kut, b_dpv)
        if K > 192:
            b_dh4 *= exp2(tl.load(p_gl + o_k4, mask=m_k4, other=0.).to(tl.float32))[:, None]
            b_qt = tl.load(q_up + o_k4[:, None] + o_t[None, :] * (H * K), mask=m_k4[:, None] & m_t[None, :], other=0.)
            b_kut = tl.load(k_up + o_k4[:, None] + o_t[None, :] * (H * K), mask=m_k4[:, None] & m_t[None, :], other=0.)
            b_dh4 += tl.dot(b_qt, b_do) * scale + tl.dot(b_kut, b_dpv)

    if USE_INITIAL_STATE:
        tl.store(dh0 + o_k1[:, None] * V + o_v[None, :], b_dh1.to(dh0.dtype.element_ty), mask=m_k1[:, None] & m_v[None, :])
        if K > 64:
            tl.store(dh0 + o_k2[:, None] * V + o_v[None, :], b_dh2.to(dh0.dtype.element_ty), mask=m_k2[:, None] & m_v[None, :])
        if K > 128:
            tl.store(dh0 + o_k3[:, None] * V + o_v[None, :], b_dh3.to(dh0.dtype.element_ty), mask=m_k3[:, None] & m_v[None, :])
        if K > 192:
            tl.store(dh0 + o_k4[:, None] * V + o_v[None, :], b_dh4.to(dh0.dtype.element_ty), mask=m_k4[:, None] & m_v[None, :])


def chunk_kronlstm_bwd_dhu(q_up, k_up, k_tail, v, gc, Tinv, bs, do, dv_intra, dht, has_h0, scale, chunk_size,
                           cu_seqlens=None, chunk_indices=None, chunk_offsets=None):
    B, T, H, K = k_up.shape
    V = v.shape[-1]
    if cu_seqlens is None:
        N, NT = B, triton.cdiv(T, chunk_size)
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
    # the chunk state gradients in the input dtype, as GDN-2 stores dh
    dh = k_up.new_empty(B, NT, H, K, V, dtype=k_up.dtype)
    dh0 = k_up.new_zeros(N, H, K, V, dtype=torch.float32) if has_h0 else None
    dv_new = torch.empty_like(do)
    dr = k_up.new_empty(B, T, H, dtype=torch.float32)
    chunk_kronlstm_bwd_kernel_dhu[(N * H,)](
        q_up=q_up, k_up=k_up, k_tail=k_tail, v=v, gc=gc, Tinv=Tinv, bs=bs, do=do,
        dv_intra=dv_intra, dv_new=dv_new, dr=dr, dh=dh, dh0=dh0, dht=dht, scale=scale, cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets, T=T, H=H, K=K, V=V, BT=chunk_size, BV=max(16, triton.next_power_of_2(V)),
    )
    return dh, dh0, dv_new, dr

# the chunk-local half


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_kronlstm_bwd_kernel_dAkk(
    v,          # [B, T, H, V]
    dr,         # [B, T, H]   Tinv^T du
    u,          # [B, T, H]
    bs,         # [B, T, H]   beta*
    dA_kk,      # out [B, T, H, BT] fp32   dT . beta* . A_vv on the strict lower triangle
    A_vv,       # out [B, T, H, BT] fp32   V V^T, the plain value Gram, full
    cu_seqlens,     # [N + 1] or None
    chunk_indices,  # [NT, 2] (sequence, chunk within it) of every chunk (varlen only)
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """The key Gram's cotangent, needed before the Gram backward and the chunk-local kernel run:
    `dT = -(dr (x) u)` on the strict lower triangle, and `dA_kk = dT . beta* . A_vv`.

    It rebuilds the value Gram, which the forward does not keep, and hands it on to the chunk-local kernel.
    `b_v` is a dot operand only and the launch is `num_stages=1`:
    with software pipelining, a tile that is both a `tl.dot` operand through `tl.trans` and a register value
    can read different pipeline stages on Hopper.
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_j = tl.arange(0, BT)
    row = (bos + o_t) * H + i_h                                   # [BT] row index in [B*T, H]

    b_vv = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v < V)[None, :]
        b_v = tl.load(v + row[:, None] * V + o_v[None, :], mask=m_tv, other=0.)
        b_vv += tl.dot(b_v, tl.trans(b_v))
    off_A = row[:, None] * BT + o_j[None, :]
    tl.store(A_vv + off_A, b_vv, mask=m_t[:, None])

    b_dr = tl.load(dr + row, mask=m_t, other=0.).to(tl.float32)
    b_u = tl.load(u + row, mask=m_t, other=0.).to(tl.float32)
    b_bs = tl.load(bs + row, mask=m_t, other=0.).to(tl.float32)
    m_s = (o_j[:, None] > o_j[None, :]) & m_t[:, None]
    b_dA = tl.where(m_s, -(b_dr[:, None] * b_u[None, :]) * b_bs[:, None] * b_vv, 0.)
    tl.store(dA_kk + off_A, b_dA, mask=m_t[:, None])


def chunk_kronlstm_bwd_dAkk(v, dr, u, bs, chunk_size, cu_seqlens=None, chunk_indices=None):
    B, T, H = bs.shape
    V = v.shape[-1]
    NT = triton.cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)
    dA_kk = bs.new_empty(B, T, H, chunk_size, dtype=torch.float32)
    A_vv = bs.new_empty(B, T, H, chunk_size, dtype=torch.float32)
    chunk_kronlstm_bwd_kernel_dAkk[(NT, B * H)](
        v=v, dr=dr, u=u, bs=bs, dA_kk=dA_kk, A_vv=A_vv, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, V=V, BT=chunk_size, BV=min(64, triton.next_power_of_2(V)), num_warps=4, num_stages=1,
    )
    return dA_kk, A_vv


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[triton.Config({}, num_warps=nw, num_stages=ns) for nw in [1, 2, 4, 8] for ns in [2, 3, 4]],
    key=['BK', 'NC', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kronlstm_bwd_kernel_intra(
    q,          # [B, T, H, K]
    k,          # [B, T, H, K]
    gc,         # [B, T, H, K]   fp32
    dA_qk,      # [B, T, H, BT]  fp32, causal incl. diagonal
    dA_kk,      # [B, T, H, BT]  fp32, strict lower
    dq_g,       # out [B, T, H, K] fp32   dA_qk's gradient into q
    dk_g,       # out [B, T, H, K] fp32   dA_qk's into k (column side) + dA_kk's into k (both sides)
    dgc_g,      # out [B, T, H, K] fp32   q . dq_g + k . (dk_row - dk_col), natural-log units
    cu_seqlens,     # [N + 1] or None
    chunk_indices,  # [NT, 2] (sequence, chunk within it) of every chunk (varlen only)
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """One program per (K block, sub-chunk i, chunk, head): GDN-2's `chunk_gdn2_bwd_kernel_intra` without its erase gate.

    Row side: `2^(gc_i - gn) sum_j dA[i, j] (k_j 2^(gn - gc_j))` over columns j <= i,
    with gn = gc on the sub-chunk's first row.
    Column side: `2^(gn' - gc_i) sum_j dA[j, i]^T (x_j 2^(gc_j - gn'))`, with gn' = gc on its last row,
    x = q for dA_qk and x = k for dA_kk.
    Within the sub-chunk both sides loop over its rows, and every exponent is <= 0.
    """
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_b * T
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    gc += (bos * H + i_h) * K
    dq_g += (bos * H + i_h) * K
    dk_g += (bos * H + i_h) * K
    dgc_g += (bos * H + i_h) * K
    dA_qk += (bos * H + i_h) * BT
    dA_kk += (bos * H + i_h) * BT
    o_i = tl.arange(0, BC)
    o_c = i_ti + o_i
    m_c = o_c < T
    m_kc = m_c[:, None] & m_k[None, :]
    b_g = tl.load(gc + o_c[:, None] * (H * K) + o_k[None, :], mask=m_kc, other=0.).to(tl.float32)

    # row side, earlier sub-chunks
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    b_dkr = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        b_gn = tl.load(gc + i_ti * (H * K) + o_k, mask=m_k, other=0.).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            m_jk = m_j[:, None] & m_k[None, :]
            m_dj = m_c[:, None] & ((i_j * BC + o_i)[None, :] < BT)
            b_kj = tl.load(k + o_j[:, None] * (H * K) + o_k[None, :], mask=m_jk, other=0.)
            b_gk = tl.load(gc + o_j[:, None] * (H * K) + o_k[None, :], mask=m_jk, other=0.).to(tl.float32)
            b_kg = b_kj * exp2(b_gn - b_gk)
            b_dAqk = tl.load(dA_qk + o_c[:, None] * (H * BT) + (i_j * BC + o_i)[None, :], mask=m_dj, other=0.)
            b_dAkk = tl.load(dA_kk + o_c[:, None] * (H * BT) + (i_j * BC + o_i)[None, :], mask=m_dj, other=0.)
            b_dq += tl.dot(b_dAqk, b_kg)
            b_dkr += tl.dot(b_dAkk, b_kg)
        b_gqn = exp2(b_g - b_gn)
        b_dq *= b_gqn
        b_dkr *= b_gqn
    # row side, within the sub-chunk
    o_dA = o_c * (H * BT) + i_i * BC
    p_kj = k + i_ti * (H * K) + o_k
    p_gkj = gc + i_ti * (H * K) + o_k
    for j in range(0, min(BC, T - i_ti)):
        b_dAqk = tl.load(dA_qk + o_dA + j, mask=m_c, other=0.)
        b_dAkk = tl.load(dA_kk + o_dA + j, mask=m_c, other=0.)
        b_kj = tl.load(p_kj, mask=m_k, other=0.).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0.).to(tl.float32)
        m_i = o_i[:, None] >= j
        b_gqk = exp2(b_g - b_gkj[None, :])
        b_dq += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)
        b_dkr += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.)
        p_kj += H * K
        p_gkj += H * K
    b_q = tl.load(q + o_c[:, None] * (H * K) + o_k[None, :], mask=m_kc, other=0.).to(tl.float32)
    b_k = tl.load(k + o_c[:, None] * (H * K) + o_k[None, :], mask=m_kc, other=0.).to(tl.float32)
    b_dgc = b_q * b_dq + b_k * b_dkr                        # row sides carry +

    # column side, later sub-chunks
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
    NCe = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NCe - 1:
        b_gn = tl.load(gc + (min(i_ti + BC, T) - 1) * (H * K) + o_k, mask=m_k, other=0.).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NCe):
            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            m_jk = m_j[:, None] & m_k[None, :]
            m_dj = ((i_i * BC + o_i)[:, None] < BT) & m_j[None, :]
            b_qj = tl.load(q + o_j[:, None] * (H * K) + o_k[None, :], mask=m_jk, other=0.)
            b_kj = tl.load(k + o_j[:, None] * (H * K) + o_k[None, :], mask=m_jk, other=0.)
            b_gk = tl.load(gc + o_j[:, None] * (H * K) + o_k[None, :], mask=m_jk, other=0.).to(tl.float32)
            b_dAqk = tl.load(dA_qk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (H * BT), mask=m_dj, other=0.)
            b_dAkk = tl.load(dA_kk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (H * BT), mask=m_dj, other=0.)
            b_gkn = tl.where(m_j[:, None], exp2(b_gk - b_gn), 0.)
            b_dkt += tl.dot(b_dAqk, b_qj * b_gkn)
            b_dkt += tl.dot(b_dAkk, b_kj * b_gkn)
        b_dkt *= exp2(b_gn - b_g)
    # column side, within the sub-chunk
    o_dA = i_ti * (H * BT) + i_i * BC + o_i
    p_qj = q + i_ti * (H * K) + o_k
    p_kj = k + i_ti * (H * K) + o_k
    p_gkj = gc + i_ti * (H * K) + o_k
    for j in range(0, min(BC, T - i_ti)):
        b_dAqk = tl.load(dA_qk + o_dA + j * (H * BT), mask=m_c, other=0.)
        b_dAkk = tl.load(dA_kk + o_dA + j * (H * BT), mask=m_c, other=0.)
        b_qj = tl.load(p_qj, mask=m_k, other=0.).to(tl.float32)
        b_kj = tl.load(p_kj, mask=m_k, other=0.).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0.).to(tl.float32)
        m_i = o_i[:, None] <= j
        b_gkq = exp2(b_gkj[None, :] - b_g)
        b_dkt += tl.where(m_i, (b_dAqk[:, None] * b_qj[None, :] + b_dAkk[:, None] * b_kj[None, :]) * b_gkq, 0.)
        p_qj += H * K
        p_kj += H * K
        p_gkj += H * K
    b_dgc -= b_k * b_dkt                                    # column sides carry -

    tl.store(dq_g + o_c[:, None] * (H * K) + o_k[None, :], b_dq, mask=m_kc)
    tl.store(dk_g + o_c[:, None] * (H * K) + o_k[None, :], b_dkr + b_dkt, mask=m_kc)
    tl.store(dgc_g + o_c[:, None] * (H * K) + o_k[None, :], b_dgc, mask=m_kc)


def chunk_kronlstm_bwd_intra(q, k, gc, dA_qk, dA_kk, chunk_size, cu_seqlens=None, chunk_indices=None):
    """Both Grams' cotangents into q, k and the gate in one launch; returns `(dq_g, dk_g, dgc_g)` in fp32."""
    B, T, H, K = k.shape
    BT = chunk_size
    BC = min(16, BT)
    NC = BT // BC
    BK = min(32, triton.next_power_of_2(K))
    NK = triton.cdiv(K, BK)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    dq_g = torch.empty_like(q, dtype=torch.float32)
    dk_g = torch.empty_like(k, dtype=torch.float32)
    dgc_g = torch.empty_like(gc)
    kw = dict(q=q, k=k, gc=gc, dA_qk=dA_qk, dA_kk=dA_kk, dq_g=dq_g, dk_g=dk_g, dgc_g=dgc_g,
              cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC)
    chunk_kronlstm_bwd_kernel_intra[(NK * NC, NT, B * H)](**kw)
    return dq_g, dk_g, dgc_g


@triton.jit(do_not_specialize=['T'])
def _chunk_kronlstm_bwd_kernel_local(
    q,          # [B, T, H, K]
    k,          # [B, T, H, K]
    v,          # [B, T, H, V]
    v_new,      # [B, T, H, V]   u . v
    do,         # [B, T, H, V]
    dv_new,     # [B, T, H, V]   the total cotangent of u . v (from the reverse scan)
    gc,         # [B, T, H, K]   fp32, cumulative log2 decay within the chunk
    beta,       # [B, T, H]
    ct,         # [B, T, H] fp32 or None
    A_kk,       # [B, T, H, BT] fp32, zero on and above the diagonal
    A_vv,       # [B, T, H, BT]  fp32, full (from `chunk_kronlstm_bwd_kernel_dAkk`)
    bs,         # [B, T, H]      beta*
    u,          # [B, T, H]      the write strengths
    dr,         # [B, T, H]      Tinv^T du
    h,          # [B, NT, H, K, V]  S0, the state at the chunk's start (input dtype)
    dh,         # [B, NT, H, K, V]  dS_{n+1}, the gradient of the state after the chunk
    dq_g,       # [B, T, H, K]   fp32, the Grams' gradient into q (from the intra kernel)
    dk_g,       # [B, T, H, K]   fp32, the Grams' gradient into k
    dgc_g,      # [B, T, H, K]   fp32, the Grams' part of the gate gradient, natural-log units
    dq,         # out [B, T, H, K]   final
    dk,         # out [B, T, H, K]   final
    dv,         # out [B, T, H, V]   final
    dbeta,      # out [B, T, H]      final
    dct,        # out [B, T, H] fp32 or None
    dgc,        # out [B, T, H, K]   fp32, per key dim, chunk-end terms on the last row
    scale,
    s,          # 2 under allow_neg_eigval, else 1
    cu_seqlens,     # [N + 1] or None
    chunk_indices,  # [NT, 2] (sequence, chunk within it) of every chunk (varlen only)
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DECOUPLE_ERASE: tl.constexpr,
    HAS_CT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """One program per (chunk, head), in the shape of GDN-2's fused chunk-local backward.

    Pass 1 walks the state in K blocks with V blocks inside, finishing dq, dk and dgc for each K block,
    and accumulates the score `p` on the way.
    The row scalars follow (dbeta*, dbeta, dct, d||v||^2), and pass 2 walks the state again, V blocks outside, for dv.
    State blocks are loaded transposed from memory, as GDN-2 loads `h`,
    so no loaded tile is both a `tl.trans` operand and a register value
    (the hazard `chunk_kronlstm_bwd_kernel_dAkk` notes).
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        # the chunk's row in h / dh is its global index; its position in the sequence comes from chunk_indices
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_b * T
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    last_idx = min(T, i_t * BT + BT) - 1
    f_last = (o_t == last_idx).to(tl.float32)
    o_j = tl.arange(0, BT)
    row = (bos + o_t) * H + i_h                                   # [BT] row index in [B*T, H]
    h += (i_tg * H + i_h) * K * V
    dh += (i_tg * H + i_h) * K * V
    p_gl = gc + ((bos + last_idx) * H + i_h) * K                  # gc on the chunk's last row

    # per-row scalars
    b_bs = tl.load(bs + row, mask=m_t, other=0.).to(tl.float32)
    b_u = tl.load(u + row, mask=m_t, other=0.).to(tl.float32)
    b_dr = tl.load(dr + row, mask=m_t, other=0.).to(tl.float32)
    b_dp = -b_bs * b_dr                                          # the score's cotangent

    # through the solve: dT = -(dr (x) u) on the strict lower triangle
    off_A = row[:, None] * BT + o_j[None, :]
    m_s = (o_j[:, None] > o_j[None, :]) & m_t[:, None]
    b_Akk = tl.where(m_s, tl.load(A_kk + off_A, mask=m_t[:, None], other=0.), 0.)
    b_Avv = tl.load(A_vv + off_A, mask=m_t[:, None], other=0.)
    b_vn = tl.sum(tl.where(o_j[:, None] == o_j[None, :], b_Avv, 0.), 1)          # v_t . v_t
    b_dT = tl.where(m_s, -(b_dr[:, None] * b_u[None, :]), 0.)
    b_dbs = tl.sum(b_dT * b_Akk * b_Avv, 1)                                        # dbeta*, part 1
    b_dAvv = b_dT * b_bs[:, None] * b_Akk
    b_sym = b_dAvv + tl.trans(b_dAvv)                                              # dv takes (dA_vv + dA_vv^T) V

    # pass 1 over the state: K blocks outside, V blocks inside
    b_p = tl.zeros([BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_kk = m_t[:, None] & m_k[None, :]
        off_k = row[:, None] * K + o_k[None, :]
        b_q = tl.load(q + off_k, mask=m_kk, other=0.).to(tl.float32)
        b_k = tl.load(k + off_k, mask=m_kk, other=0.).to(tl.float32)
        b_gc = tl.load(gc + off_k, mask=m_kk, other=0.).to(tl.float32)
        b_gl = tl.load(p_gl + o_k, mask=m_k, other=0.).to(tl.float32)
        b_gam = exp2(b_gc)                                                         # 2^gc
        b_tail = exp2(b_gl[None, :] - b_gc)                                        # 2^(gc_C - gc)
        b_qup = b_q * b_gam
        b_kup = b_k * b_gam
        b_ktail = b_k * b_tail
        b_dqup = tl.zeros([BT, BK], dtype=tl.float32)
        b_dkup = tl.zeros([BT, BK], dtype=tl.float32)
        b_dktail = tl.zeros([BT, BK], dtype=tl.float32)
        b_hd = tl.zeros([BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_v = o_v < V
            m_tv = m_t[:, None] & m_v[None, :]
            m_vk = m_v[:, None] & m_k[None, :]
            off_v = row[:, None] * V + o_v[None, :]
            b_S0T = tl.load(h + o_v[:, None] + o_k[None, :] * V, mask=m_vk, other=0.)    # S0^T     [BV, BK]
            b_dS1T = tl.load(dh + o_v[:, None] + o_k[None, :] * V, mask=m_vk, other=0.)  # dS_{n+1}^T
            b_do = tl.load(do + off_v, mask=m_tv, other=0.)
            b_vnew = tl.load(v_new + off_v, mask=m_tv, other=0.)
            b_v = tl.load(v + off_v, mask=m_tv, other=0.)
            b_dqup += tl.dot(b_do, b_S0T.to(b_do.dtype))                               # do S0^T
            b_dktail += tl.dot(b_vnew, b_dS1T.to(b_vnew.dtype))                        # v_new dS^T
            b_dpv = (b_dp[:, None] * b_v.to(tl.float32)).to(b_v.dtype)
            b_dkup += tl.dot(b_dpv, b_S0T.to(b_v.dtype))                               # (dp . v) S0^T
            b_p += tl.sum(tl.dot(b_v, b_S0T.to(b_v.dtype)) * b_kup, 1)                 # rowsum((V S0^T) . K_up)
            b_hd += tl.sum(b_S0T.to(tl.float32) * b_dS1T.to(tl.float32), 0)            # (dS . S0) summed over V
            tl.debug_barrier()
        b_dqup *= scale
        b_dqg = tl.load(dq_g + off_k, mask=m_kk, other=0.)
        b_dkg = tl.load(dk_g + off_k, mask=m_kk, other=0.)
        b_dgcg = tl.load(dgc_g + off_k, mask=m_kk, other=0.)
        # the gate in natural-log units: each decayed operand times its cotangent,
        # the chunk-end terms (the tail scaling and the state decay) on the chunk's last row, and the Grams' part
        b_end = tl.sum(b_ktail * b_dktail, 0) + exp2(b_gl) * b_hd                      # [BK]
        b_dgc = (b_qup * b_dqup + b_kup * b_dkup - b_ktail * b_dktail
                 + f_last[:, None] * b_end[None, :] + b_dgcg)
        b_dq = b_dqup * b_gam + b_dqg
        b_dk = b_dkup * b_gam + b_dktail * b_tail + b_dkg
        tl.store(dq + off_k, b_dq.to(dq.dtype.element_ty), mask=m_kk)
        tl.store(dk + off_k, b_dk.to(dk.dtype.element_ty), mask=m_kk)
        tl.store(dgc + off_k, b_dgc, mask=m_kk)

    # the row scalars: r = pay - beta* p took the solve's cotangent
    b_dbs -= b_dr * b_p
    b_beta = tl.load(beta + row, mask=m_t, other=0.).to(tl.float32)
    # ||v||^2 is floored at V_NORM_FLOOR in beta_max only
    b_unfl = (b_vn > 1e-18).to(tl.float32)
    b_vnc = tl.maximum(b_vn, 1e-18)
    b_bmax = s / b_vnc
    b_tgt = b_vn
    if HAS_CT:
        b_ct = tl.load(ct + row, mask=m_t, other=0.).to(tl.float32)
        b_tgt = b_tgt * b_ct
    if DECOUPLE_ERASE:
        # under decouple_erase, beta comes off the write
        b_dtgt = b_dr * b_bmax
    else:
        b_dtgt = b_dr * b_bs
    b_dgate = b_dr * b_tgt
    b_dvn = b_dtgt
    if HAS_CT:
        b_dvn = b_dvn * b_ct
        tl.store(dct + row, (b_dtgt * b_vn).to(dct.dtype.element_ty), mask=m_t)
    if DECOUPLE_ERASE:
        b_dbmax = b_dgate
    else:
        b_dbs += b_dgate
        b_dbmax = tl.zeros([BT], dtype=tl.float32)
    tl.store(dbeta + row, (b_dbs * b_bmax).to(dbeta.dtype.element_ty), mask=m_t)
    b_dbmax += b_dbs * b_beta
    b_dvn -= b_dbmax * b_bmax / b_vnc * b_unfl

    # pass 2 over the state, V blocks outside and K inside:
    # dv = dp . (K_up S0) + u . dv_new + (dA_vv + dA_vv^T) V + 2 V d||v||^2
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        off_v = row[:, None] * V + o_v[None, :]
        b_dvp = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            m_kk = m_t[:, None] & m_k[None, :]
            off_k = row[:, None] * K + o_k[None, :]
            b_k = tl.load(k + off_k, mask=m_kk, other=0.)
            b_gc = tl.load(gc + off_k, mask=m_kk, other=0.).to(tl.float32)
            b_dpk = (b_dp[:, None] * b_k.to(tl.float32) * exp2(b_gc)).to(b_k.dtype)   # dp . K_up
            b_S0 = tl.load(h + o_k[:, None] * V + o_v[None, :], mask=m_k[:, None] & m_v[None, :], other=0.)
            b_dvp += tl.dot(b_dpk, b_S0.to(b_k.dtype))
        b_v = tl.load(v + off_v, mask=m_tv, other=0.)
        b_dvnew = tl.load(dv_new + off_v, mask=m_tv, other=0.).to(tl.float32)
        b_dv = (b_dvp + b_u[:, None] * b_dvnew + tl.dot(b_sym.to(b_v.dtype), b_v)
                + 2. * b_v.to(tl.float32) * b_dvn[:, None])
        tl.store(dv + off_v, b_dv.to(dv.dtype.element_ty), mask=m_tv)


chunk_kronlstm_bwd_kernel_local = triton.heuristics({
    'HAS_CT': lambda args: args['ct'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})(fla_cache_autotune(
    configs=[triton.Config({}, num_warps=nw, num_stages=ns) for nw in NUM_WARPS for ns in NUM_STAGES],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'DECOUPLE_ERASE'],
    **autotune_cache_kwargs,
)(_chunk_kronlstm_bwd_kernel_local))


def chunk_kronlstm_bwd_local(q, k, v, v_new, do, dv_new, gc, beta, target_gate, A_kk, A_vv, bs, u, dr, h, dh,
                             dq_g, dk_g, dgc_g, scale, beta_max_scaler, decouple_erase, chunk_size,
                             cu_seqlens=None, chunk_indices=None):
    """The chunk-local gradients in one launch."""
    B, T, H, K = k.shape
    V = v.shape[-1]
    NT = triton.cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)
    dq, dk, dv, dbeta = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(beta)
    dct = torch.empty_like(target_gate) if target_gate is not None else None
    dgc = torch.empty_like(gc)
    kw = dict(q=q, k=k, v=v, v_new=v_new, do=do, dv_new=dv_new, gc=gc, beta=beta, ct=target_gate, A_kk=A_kk,
              A_vv=A_vv, bs=bs, u=u, dr=dr, h=h, dh=dh, dq_g=dq_g, dk_g=dk_g, dgc_g=dgc_g,
              dq=dq, dk=dk, dv=dv, dbeta=dbeta, dct=dct, dgc=dgc, scale=scale, s=beta_max_scaler,
              cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T,
              H=H, K=K, V=V, BT=chunk_size, BK=min(64, triton.next_power_of_2(K)),
              BV=min(64, triton.next_power_of_2(V)), DECOUPLE_ERASE=decouple_erase)
    chunk_kronlstm_bwd_kernel_local[(NT, B * H)](**kw)
    return dq, dk, dv, dbeta, dct, dgc


def chunk_kronlstm_bwd(q, k, v, beta, target_gate, gc, A_kk, A_qk, Tinv, bs, u, h, v_new,
                       q_up, k_up, k_tail, has_h0, do, dht, scale, beta_max_scaler, decouple_erase,
                       chunk_size, need_dg=True, cu_seqlens=None, chunk_indices=None, chunk_offsets=None):
    """The backward on tensors padded to a multiple of `chunk_size`, or packed by `cu_seqlens`, with `g` at rank 4.

    Returns `(dq, dk, dv, dbeta, dg, dct, dh0)`, each in its input's dtype.
    `dg` is `[B, T, H, K]` fp32, or `None` when `need_dg` is `False`;
    a per-head gate's reduction over K happens in the autograd ops outside the Function.
    """
    C = chunk_size
    if cu_seqlens is not None:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, C)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, C)
    dA_qk, dv_intra = chunk_kda_bwd_dAv(q=q, k=k, v=v_new, do=do, A=A_qk, scale=scale, chunk_size=C,
                                        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    dh, dh0, dv_new, dr = chunk_kronlstm_bwd_dhu(q_up, k_up, k_tail, v, gc, Tinv, bs, do, dv_intra,
                                                 dht, has_h0, scale, C, cu_seqlens=cu_seqlens,
                                                 chunk_indices=chunk_indices, chunk_offsets=chunk_offsets)
    dA_kk, A_vv = chunk_kronlstm_bwd_dAkk(v, dr, u, bs, C, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    dq_g, dk_g, dgc_g = chunk_kronlstm_bwd_intra(q, k, gc, dA_qk, dA_kk, C, cu_seqlens=cu_seqlens,
                                                 chunk_indices=chunk_indices)
    dq, dk, dv, dbeta, dct, dgc = chunk_kronlstm_bwd_local(
        q, k, v, v_new, do, dv_new, gc, beta, target_gate, A_kk, A_vv, bs, u, dr, h, dh,
        dq_g, dk_g, dgc_g, scale, beta_max_scaler, decouple_erase, C, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices)
    dg = chunk_local_cumsum(dgc.contiguous(), chunk_size=C, reverse=True, cu_seqlens=cu_seqlens,
                            chunk_indices=chunk_indices) if need_dg else None
    return dq, dk, dv, dbeta, dg, dct, dh0
