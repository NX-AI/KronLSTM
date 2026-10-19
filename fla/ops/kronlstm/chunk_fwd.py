# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Chunkwise KronLSTM forward: `chunk_kronlstm_fwd`, called by `chunk.chunk_kronlstm`.

Five launches in GDN-2's shape: the log2 gate cumsum,
`chunk_kronlstm_fwd_kernel_intra_diag` and `chunk_kronlstm_fwd_kernel_inter` for the intra-chunk half
(both key Grams, the erase strengths `beta*` and `(I + T)^-1`), the scan `chunk_kronlstm_fwd_kernel_h`,
and GLA's read-out `chunk_gla_fwd_o_gk`; `README.md` has the derivation.

fp32 inputs run tf32 dots, which read about 1e-3 against the float64 reference, unless `TRITON_F32_DEFAULT=ieee` is set.
"""

import torch
import triton
import triton.language as tl

from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs, check_shared_mem

NUM_WARPS = [4, 8]
NUM_STAGES = [1, 2, 3] if check_shared_mem('ampere') else [1, 2]


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[triton.Config({}, num_warps=nw, num_stages=ns) for nw in NUM_WARPS for ns in NUM_STAGES],
    key=['H', 'K', 'V', 'BT', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kronlstm_fwd_kernel_h(
    k_up,       # [B, T, H, K] k * 2^gc, for the score
    k_tail,     # [B, T, H, K] k * 2^(gc_C - gc), for the write
    v,          # [B, T, H, V]
    gc,         # [B, T, H, K]   cumulative log2 decay within the chunk
    Tinv,       # [B, T, H, BT] (I + T)^-1
    bs,         # [B, T, H]      beta*
    pay,        # [B, T, H]      write_gate * target
    v_new,      # out [B, T, H, V]     u * v
    u,          # out [B, T, H] write strengths
    h,          # out [B, NT, H, K, V] state at each chunk start, in the input dtype
    h0,         # [N, H, K, V] or None
    ht,         # [N, H, K, V] or None
    cu_seqlens,     # [N + 1] or None
    chunk_offsets,  # [N + 1] the first chunk of each sequence (varlen only)
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """One program per (sequence, head) holding the whole [K, V] state in registers as 64-row key blocks,
    like GDN-2's scan; unlike it, V is not tiled across programs, since the score `p` sums over all of V.
    """
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

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    h += (boh * H + i_h).to(tl.int64) * K * V
    v += (bos * H + i_h).to(tl.int64) * V
    v_new += (bos * H + i_h).to(tl.int64) * V
    u += (bos * H + i_h).to(tl.int64)
    k_up += (bos * H + i_h).to(tl.int64) * K
    k_tail += (bos * H + i_h).to(tl.int64) * K
    gc += (bos * H + i_h).to(tl.int64) * K
    bs += (bos * H + i_h).to(tl.int64)
    pay += (bos * H + i_h).to(tl.int64)
    Tinv += (bos * H + i_h).to(tl.int64) * BT
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

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

    if USE_INITIAL_STATE:
        b_h1 += tl.load(h0 + o_k1[:, None] * V + o_v[None, :], mask=m_k1[:, None] & m_v[None, :], other=0.).to(tl.float32)
        if K > 64:
            b_h2 += tl.load(h0 + o_k2[:, None] * V + o_v[None, :], mask=m_k2[:, None] & m_v[None, :], other=0.).to(tl.float32)
        if K > 128:
            b_h3 += tl.load(h0 + o_k3[:, None] * V + o_v[None, :], mask=m_k3[:, None] & m_v[None, :], other=0.).to(tl.float32)
        if K > 192:
            b_h4 += tl.load(h0 + o_k4[:, None] * V + o_v[None, :], mask=m_k4[:, None] & m_v[None, :], other=0.).to(tl.float32)

    for i_t in range(NT):
        i_t64 = i_t.to(tl.int64)
        o_t = i_t64 * BT + o_j
        m_t = o_t < T

        # the state at this chunk's start, for the read-out
        p_h = h + i_t64 * H * K * V
        tl.store(p_h + o_k1[:, None] * V + o_v[None, :], b_h1.to(p_h.dtype.element_ty), mask=m_k1[:, None] & m_v[None, :])
        if K > 64:
            tl.store(p_h + o_k2[:, None] * V + o_v[None, :], b_h2.to(p_h.dtype.element_ty), mask=m_k2[:, None] & m_v[None, :])
        if K > 128:
            tl.store(p_h + o_k3[:, None] * V + o_v[None, :], b_h3.to(p_h.dtype.element_ty), mask=m_k3[:, None] & m_v[None, :])
        if K > 192:
            tl.store(p_h + o_k4[:, None] * V + o_v[None, :], b_h4.to(p_h.dtype.element_ty), mask=m_k4[:, None] & m_v[None, :])

        # the score on the decayed state, rowsum(((Gamma * K) S0) . V)
        b_ku = tl.load(k_up + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None] & m_k1[None, :], other=0.)
        b_pv = tl.dot(b_ku, b_h1.to(b_ku.dtype))
        if K > 64:
            b_ku = tl.load(k_up + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None] & m_k2[None, :], other=0.)
            b_pv += tl.dot(b_ku, b_h2.to(b_ku.dtype))
        if K > 128:
            b_ku = tl.load(k_up + o_t[:, None] * (H * K) + o_k3[None, :], mask=m_t[:, None] & m_k3[None, :], other=0.)
            b_pv += tl.dot(b_ku, b_h3.to(b_ku.dtype))
        if K > 192:
            b_ku = tl.load(k_up + o_t[:, None] * (H * K) + o_k4[None, :], mask=m_t[:, None] & m_k4[None, :], other=0.)
            b_pv += tl.dot(b_ku, b_h4.to(b_ku.dtype))
        b_v = tl.load(v + o_t[:, None] * (H * V) + o_v[None, :], mask=m_t[:, None] & m_v[None, :], other=0.)
        b_p = tl.sum(b_pv * b_v.to(tl.float32), 1)                                 # [BT]

        # apply the solve, u = Tinv (pay - beta* p), and scale the values by the write strengths
        b_bs = tl.load(bs + o_t * H, mask=m_t, other=0.).to(tl.float32)
        b_pay = tl.load(pay + o_t * H, mask=m_t, other=0.).to(tl.float32)
        b_Ti = tl.load(Tinv + o_t[:, None] * (H * BT) + o_j[None, :], mask=m_t[:, None], other=0.).to(tl.float32)
        b_u = tl.sum(b_Ti * (b_pay - b_bs * b_p)[None, :], 1)                    # [BT]
        tl.store(u + o_t * H, b_u, mask=m_t)
        b_uv = b_u[:, None] * b_v.to(tl.float32)                                   # [BT, BV]
        tl.store(v_new + o_t[:, None] * (H * V) + o_v[None, :],
                 b_uv.to(v_new.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

        # decay the state by 2^gc at the chunk's last step, per key dimension
        last_idx = min((i_t + 1) * BT, T) - 1
        p_gl = gc + (last_idx * H).to(tl.int64) * K
        b_h1 *= exp2(tl.load(p_gl + o_k1, mask=m_k1, other=0.).to(tl.float32))[:, None]
        if K > 64:
            b_h2 *= exp2(tl.load(p_gl + o_k2, mask=m_k2, other=0.).to(tl.float32))[:, None]
        if K > 128:
            b_h3 *= exp2(tl.load(p_gl + o_k3, mask=m_k3, other=0.).to(tl.float32))[:, None]
        if K > 192:
            b_h4 *= exp2(tl.load(p_gl + o_k4, mask=m_k4, other=0.).to(tl.float32))[:, None]

        # write: S += (k * 2^(gc_C - gc))^T (u * v)
        b_uv = b_uv.to(k_tail.dtype.element_ty)
        b_kt = tl.load(k_tail + o_k1[:, None] + o_t[None, :] * (H * K), mask=m_k1[:, None] & m_t[None, :], other=0.)
        b_h1 += tl.dot(b_kt, b_uv)
        if K > 64:
            b_kt = tl.load(k_tail + o_k2[:, None] + o_t[None, :] * (H * K), mask=m_k2[:, None] & m_t[None, :], other=0.)
            b_h2 += tl.dot(b_kt, b_uv)
        if K > 128:
            b_kt = tl.load(k_tail + o_k3[:, None] + o_t[None, :] * (H * K), mask=m_k3[:, None] & m_t[None, :], other=0.)
            b_h3 += tl.dot(b_kt, b_uv)
        if K > 192:
            b_kt = tl.load(k_tail + o_k4[:, None] + o_t[None, :] * (H * K), mask=m_k4[:, None] & m_t[None, :], other=0.)
            b_h4 += tl.dot(b_kt, b_uv)

    if STORE_FINAL_STATE:
        tl.store(ht + o_k1[:, None] * V + o_v[None, :], b_h1.to(ht.dtype.element_ty), mask=m_k1[:, None] & m_v[None, :])
        if K > 64:
            tl.store(ht + o_k2[:, None] * V + o_v[None, :], b_h2.to(ht.dtype.element_ty), mask=m_k2[:, None] & m_v[None, :])
        if K > 128:
            tl.store(ht + o_k3[:, None] * V + o_v[None, :], b_h3.to(ht.dtype.element_ty), mask=m_k3[:, None] & m_v[None, :])
        if K > 192:
            tl.store(ht + o_k4[:, None] * V + o_v[None, :], b_h4.to(ht.dtype.element_ty), mask=m_k4[:, None] & m_v[None, :])


def chunk_kronlstm_fwd_h(k_up, k_tail, v, gc, Tinv, bs, pay, initial_state, output_final_state, chunk_size,
                         cu_seqlens=None, chunk_indices=None, chunk_offsets=None):
    B, T, H, K = k_up.shape
    V = v.shape[-1]
    if cu_seqlens is None:
        N, NT = B, triton.cdiv(T, chunk_size)
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
    # the chunk states in the input dtype, as GDN-2 stores h
    h = k_up.new_empty(B, NT, H, K, V, dtype=k_up.dtype)
    v_new = torch.empty_like(v)
    u = k_up.new_empty(B, T, H, dtype=torch.float32)
    ht = k_up.new_zeros(N, H, K, V, dtype=torch.float32) if output_final_state else None
    chunk_kronlstm_fwd_kernel_h[(N * H,)](
        k_up=k_up, k_tail=k_tail, v=v, gc=gc, Tinv=Tinv, bs=bs, pay=pay, v_new=v_new, u=u, h=h,
        h0=initial_state, ht=ht, cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets, T=T, H=H, K=K, V=V,
        BT=chunk_size, BV=max(16, triton.next_power_of_2(V)),
    )
    return h, v_new, u, ht


# the intra-chunk half in GDN-2's two-kernel shape:
# a token-parallel kernel for the diagonal 16x16 blocks of both key Grams,
# and one kernel per (chunk, head) for the other blocks, beta*, (I + T)^-1 and the decayed keys;
# exponents are referenced to sub-chunk boundaries, 2^(gc_t - gc_s) = 2^(gc_t - gn) 2^(gn - gc_s),
# with both factors <= 1, so nothing overflows at any gate magnitude

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[triton.Config({'BH': BH}, num_warps=nw) for BH in [1, 2, 4, 8] for nw in [1, 2, 4, 8]],
    key=['K', 'H'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'N'])
def chunk_kronlstm_fwd_kernel_intra_diag(
    q,          # [B, T, H, K]
    k,          # [B, T, H, K]
    gc,         # [B, T, H, K]   fp32, cumulative log2 decay within the chunk
    A_qk,       # out [B, T, H, BT] fp32   the diagonal 16x16 blocks, scale folded in, causal incl. diagonal
    A_kk,       # out [B, T, H, BT] fp32   the diagonal 16x16 blocks, strict lower
    scale,
    cu_seqlens,     # [N + 1] or None
    N,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """One program per (token, group of BH heads) writes the token's row of the diagonal 16x16 block of both Grams.

    Every entry of the row is written, zeros above the diagonal included; `inter` writes the blocks off the diagonal.
    """
    i_tg, i_hg = tl.program_id(0).to(tl.int64), tl.program_id(1)
    if IS_VARLEN:
        # the token's sequence by binary search over the offsets, as GDN-2's token-parallel kernel finds it
        left, right = 0, N
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T
    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_ts = i_c * BT + i_s * BC
    q += bos * H * K
    k += bos * H * K
    gc += bos * H * K
    A_qk += bos * H * BT
    A_kk += bos * H * BT
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, BK)
    m_h = (i_hg * BH + o_h) < H
    m_k = o_k < K
    off = i_t * H * K + (i_hg * BH + o_h)[:, None] * K + o_k[None, :]
    b_q = tl.load(q + off, mask=m_h[:, None] & m_k[None, :], other=0.).to(tl.float32)
    b_k = tl.load(k + off, mask=m_h[:, None] & m_k[None, :], other=0.).to(tl.float32)
    b_g = tl.load(gc + off, mask=m_h[:, None] & m_k[None, :], other=0.).to(tl.float32)
    for j in range(i_ts, min(T, i_ts + BC)):
        offj = j * H * K + (i_hg * BH + o_h)[:, None] * K + o_k[None, :]
        b_kj = tl.load(k + offj, mask=m_h[:, None] & m_k[None, :], other=0.).to(tl.float32)
        b_gj = tl.load(gc + offj, mask=m_h[:, None] & m_k[None, :], other=0.).to(tl.float32)
        # 2^(gc_t - gc_j) <= 1 for j <= t; for j > t the exponent is clamped, so no Inf meets the zero (Inf * 0 is NaN)
        b_kgj = tl.where(m_k[None, :], b_kj * exp2(tl.minimum(b_g - b_gj, 0.)), 0.)
        b_Aqk = tl.where(j <= i_t, tl.sum(b_q * b_kgj, 1) * scale, 0.)
        b_Akk = tl.where(j < i_t, tl.sum(b_k * b_kgj, 1), 0.)
        tl.store(A_qk + i_t * H * BT + (i_hg * BH + o_h) * BT + j % BT, b_Aqk, mask=m_h)
        tl.store(A_kk + i_t * H * BT + (i_hg * BH + o_h) * BT + j % BT, b_Akk, mask=m_h)


@triton.heuristics({
    'HAS_CT': lambda args: args['ct'] is not None,
    'STORE_QUP': lambda args: args['q_up'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[triton.Config({'BK': BK}, num_warps=nw) for BK in [32, 64] for nw in [1, 2, 4]],
    key=['H', 'K', 'V', 'BT', 'BC', 'BV', 'DECOUPLE_ERASE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kronlstm_fwd_kernel_inter(
    q,          # [B, T, H, K]   only read when STORE_QUP
    k,          # [B, T, H, K]
    v,          # [B, T, H, V]
    gc,         # [B, T, H, K]   fp32
    beta,       # [B, T, H]
    ct,         # [B, T, H] or None
    A_qk,       # in/out [B, T, H, BT] fp32: diagonal blocks in, the other blocks out
    A_kk,       # in/out [B, T, H, BT] fp32   same
    q_up,       # out [B, T, H, K]   q * 2^gc (or None)
    k_up,       # out [B, T, H, K]   k * 2^gc
    k_tail,     # out [B, T, H, K]   k * 2^(gc_C - gc)
    bs,         # out [B, T, H]      beta* = s beta / ||v||^2
    pay,        # out [B, T, H]      write_gate * target
    Tinv,       # out [B, T, H, BT] fp32   (I + T)^-1, T = strict(beta* . A_kk . A_vv)
    scale,
    s,          # 2 under allow_neg_eigval, else 1
    cu_seqlens,     # [N + 1] or None
    chunk_indices,  # [NT, 2] (sequence, chunk within it) of every chunk (varlen only)
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DECOUPLE_ERASE: tl.constexpr,
    HAS_CT: tl.constexpr,
    STORE_QUP: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_b * T
    if i_t * BT >= T:
        return
    tc0 = i_t * BT
    tc1 = tc0 + BC
    tc2 = tc0 + 2 * BC
    tc3 = tc0 + 3 * BC
    # the rows each sub-chunk's decay is referenced to; a packed sequence's last chunk may end before a sub-chunk starts,
    # and that sub-chunk serves no stored row, so its anchor stays on the sequence's last row
    # instead of reading into the next sequence or past the packed tensor
    if IS_VARLEN:
        tn1, tn2, tn3 = min(tc1, T - 1), min(tc2, T - 1), min(tc3, T - 1)
    else:
        tn1, tn2, tn3 = tc1, tc2, tc3
    last_idx = min(T, tc0 + BT) - 1
    o_i = tl.arange(0, BC)
    m0 = (tc0 + o_i) < T
    m1 = (tc1 + o_i) < T
    m2 = (tc2 + o_i) < T
    m3 = (tc3 + o_i) < T
    # row indices into [B * T, H], one vector per sub-chunk
    r0 = (bos + tc0 + o_i) * H + i_h
    r1 = (bos + tc1 + o_i) * H + i_h
    r2 = (bos + tc2 + o_i) * H + i_h
    r3 = (bos + tc3 + o_i) * H + i_h
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    # the value Gram, every block on and below the diagonal, A_vv_ij = V_i V_j^T;
    # V_j^T is loaded transposed, so no loaded tile is also a `tl.trans` operand
    b_vv00 = tl.zeros([BC, BC], dtype=tl.float32)
    if NC > 1:
        b_vv10 = tl.zeros([BC, BC], dtype=tl.float32)
        b_vv11 = tl.zeros([BC, BC], dtype=tl.float32)
    if NC > 2:
        b_vv20 = tl.zeros([BC, BC], dtype=tl.float32)
        b_vv21 = tl.zeros([BC, BC], dtype=tl.float32)
        b_vv22 = tl.zeros([BC, BC], dtype=tl.float32)
        b_vv30 = tl.zeros([BC, BC], dtype=tl.float32)
        b_vv31 = tl.zeros([BC, BC], dtype=tl.float32)
        b_vv32 = tl.zeros([BC, BC], dtype=tl.float32)
        b_vv33 = tl.zeros([BC, BC], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        b_v0 = tl.load(v + r0[:, None] * V + o_v[None, :], mask=m0[:, None] & m_v[None, :], other=0.)
        b_v0t = tl.load(v + r0[None, :] * V + o_v[:, None], mask=m0[None, :] & m_v[:, None], other=0.)
        b_vv00 += tl.dot(b_v0, b_v0t)
        if NC > 1:
            b_v1 = tl.load(v + r1[:, None] * V + o_v[None, :], mask=m1[:, None] & m_v[None, :], other=0.)
            b_v1t = tl.load(v + r1[None, :] * V + o_v[:, None], mask=m1[None, :] & m_v[:, None], other=0.)
            b_vv10 += tl.dot(b_v1, b_v0t)
            b_vv11 += tl.dot(b_v1, b_v1t)
        if NC > 2:
            b_v2 = tl.load(v + r2[:, None] * V + o_v[None, :], mask=m2[:, None] & m_v[None, :], other=0.)
            b_v2t = tl.load(v + r2[None, :] * V + o_v[:, None], mask=m2[None, :] & m_v[:, None], other=0.)
            b_v3 = tl.load(v + r3[:, None] * V + o_v[None, :], mask=m3[:, None] & m_v[None, :], other=0.)
            b_v3t = tl.load(v + r3[None, :] * V + o_v[:, None], mask=m3[None, :] & m_v[:, None], other=0.)
            b_vv20 += tl.dot(b_v2, b_v0t)
            b_vv21 += tl.dot(b_v2, b_v1t)
            b_vv22 += tl.dot(b_v2, b_v2t)
            b_vv30 += tl.dot(b_v3, b_v0t)
            b_vv31 += tl.dot(b_v3, b_v1t)
            b_vv32 += tl.dot(b_v3, b_v2t)
            b_vv33 += tl.dot(b_v3, b_v3t)

    # beta_max and beta* per sub-chunk, as the references compute them, with ||v||^2 floored at V_NORM_FLOOR
    b_vn0 = tl.sum(tl.where(m_I, b_vv00, 0.), 1)
    b_bmax0 = s / tl.maximum(b_vn0, 1e-18)
    b_bs0 = b_bmax0 * tl.load(beta + r0, mask=m0, other=0.).to(tl.float32)
    b_tgt0 = b_vn0
    if HAS_CT:
        b_tgt0 = b_tgt0 * tl.load(ct + r0, mask=m0, other=0.).to(tl.float32)
    if DECOUPLE_ERASE:
        b_pay0 = b_bmax0 * b_tgt0
    else:
        b_pay0 = b_bs0 * b_tgt0
    tl.store(bs + r0, b_bs0, mask=m0)
    tl.store(pay + r0, b_pay0, mask=m0)
    if NC > 1:
        b_vn1 = tl.sum(tl.where(m_I, b_vv11, 0.), 1)
        b_bmax1 = s / tl.maximum(b_vn1, 1e-18)
        b_bs1 = b_bmax1 * tl.load(beta + r1, mask=m1, other=0.).to(tl.float32)
        b_tgt1 = b_vn1
        if HAS_CT:
            b_tgt1 = b_tgt1 * tl.load(ct + r1, mask=m1, other=0.).to(tl.float32)
        if DECOUPLE_ERASE:
            b_pay1 = b_bmax1 * b_tgt1
        else:
            b_pay1 = b_bs1 * b_tgt1
        tl.store(bs + r1, b_bs1, mask=m1)
        tl.store(pay + r1, b_pay1, mask=m1)
    if NC > 2:
        b_vn2 = tl.sum(tl.where(m_I, b_vv22, 0.), 1)
        b_bmax2 = s / tl.maximum(b_vn2, 1e-18)
        b_bs2 = b_bmax2 * tl.load(beta + r2, mask=m2, other=0.).to(tl.float32)
        b_tgt2 = b_vn2
        if HAS_CT:
            b_tgt2 = b_tgt2 * tl.load(ct + r2, mask=m2, other=0.).to(tl.float32)
        if DECOUPLE_ERASE:
            b_pay2 = b_bmax2 * b_tgt2
        else:
            b_pay2 = b_bs2 * b_tgt2
        tl.store(bs + r2, b_bs2, mask=m2)
        tl.store(pay + r2, b_pay2, mask=m2)
        b_vn3 = tl.sum(tl.where(m_I, b_vv33, 0.), 1)
        b_bmax3 = s / tl.maximum(b_vn3, 1e-18)
        b_bs3 = b_bmax3 * tl.load(beta + r3, mask=m3, other=0.).to(tl.float32)
        b_tgt3 = b_vn3
        if HAS_CT:
            b_tgt3 = b_tgt3 * tl.load(ct + r3, mask=m3, other=0.).to(tl.float32)
        if DECOUPLE_ERASE:
            b_pay3 = b_bmax3 * b_tgt3
        else:
            b_pay3 = b_bs3 * b_tgt3
        tl.store(bs + r3, b_bs3, mask=m3)
        tl.store(pay + r3, b_pay3, mask=m3)

    # the key Grams' off-diagonal blocks and the decayed keys, in one pass over K:
    # A_ij = (X_i 2^(gc_i - gn_i)) (K_j 2^(gn_i - gc_j))^T with gn_i = gc on block i's first row,
    # X = q for A_qk (scale folded in) and X = k for A_kk; block j's factor is shared by both
    if NC > 1:
        b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)
    if NC > 2:
        b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_gl = tl.load(gc + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.).to(tl.float32)
        # block 0
        mk0 = m0[:, None] & m_k[None, :]
        b_k0 = tl.load(k + r0[:, None] * K + o_k[None, :], mask=mk0, other=0.).to(tl.float32)
        b_g0 = tl.load(gc + r0[:, None] * K + o_k[None, :], mask=mk0, other=0.).to(tl.float32)
        b_e0 = exp2(b_g0)
        tl.store(k_up + r0[:, None] * K + o_k[None, :], (b_k0 * b_e0).to(k_up.dtype.element_ty), mask=mk0)
        tl.store(k_tail + r0[:, None] * K + o_k[None, :],
                 (b_k0 * exp2(b_gl[None, :] - b_g0)).to(k_tail.dtype.element_ty), mask=mk0)
        if STORE_QUP:
            b_q0 = tl.load(q + r0[:, None] * K + o_k[None, :], mask=mk0, other=0.).to(tl.float32)
            tl.store(q_up + r0[:, None] * K + o_k[None, :], (b_q0 * b_e0).to(q_up.dtype.element_ty), mask=mk0)
        if NC > 1:
            mk1 = m1[:, None] & m_k[None, :]
            b_q1 = tl.load(q + r1[:, None] * K + o_k[None, :], mask=mk1, other=0.).to(tl.float32)
            b_k1 = tl.load(k + r1[:, None] * K + o_k[None, :], mask=mk1, other=0.).to(tl.float32)
            b_g1 = tl.load(gc + r1[:, None] * K + o_k[None, :], mask=mk1, other=0.).to(tl.float32)
            b_e1 = exp2(b_g1)
            tl.store(k_up + r1[:, None] * K + o_k[None, :], (b_k1 * b_e1).to(k_up.dtype.element_ty), mask=mk1)
            tl.store(k_tail + r1[:, None] * K + o_k[None, :],
                     (b_k1 * exp2(b_gl[None, :] - b_g1)).to(k_tail.dtype.element_ty), mask=mk1)
            if STORE_QUP:
                tl.store(q_up + r1[:, None] * K + o_k[None, :], (b_q1 * b_e1).to(q_up.dtype.element_ty), mask=mk1)
            b_gn1 = tl.load(gc + ((bos + tn1) * H + i_h) * K + o_k, mask=m_k, other=0.).to(tl.float32)
            b_gqn1 = tl.where(m1[:, None], exp2(b_g1 - b_gn1[None, :]), 0.)
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            b_Aqk10 += tl.dot(b_q1 * b_gqn1, b_kgt)
            b_Akk10 += tl.dot(b_k1 * b_gqn1, b_kgt)
        if NC > 2:
            mk2 = m2[:, None] & m_k[None, :]
            b_q2 = tl.load(q + r2[:, None] * K + o_k[None, :], mask=mk2, other=0.).to(tl.float32)
            b_k2 = tl.load(k + r2[:, None] * K + o_k[None, :], mask=mk2, other=0.).to(tl.float32)
            b_g2 = tl.load(gc + r2[:, None] * K + o_k[None, :], mask=mk2, other=0.).to(tl.float32)
            b_e2 = exp2(b_g2)
            tl.store(k_up + r2[:, None] * K + o_k[None, :], (b_k2 * b_e2).to(k_up.dtype.element_ty), mask=mk2)
            tl.store(k_tail + r2[:, None] * K + o_k[None, :],
                     (b_k2 * exp2(b_gl[None, :] - b_g2)).to(k_tail.dtype.element_ty), mask=mk2)
            if STORE_QUP:
                tl.store(q_up + r2[:, None] * K + o_k[None, :], (b_q2 * b_e2).to(q_up.dtype.element_ty), mask=mk2)
            b_gn2 = tl.load(gc + ((bos + tn2) * H + i_h) * K + o_k, mask=m_k, other=0.).to(tl.float32)
            b_gqn2 = tl.where(m2[:, None], exp2(b_g2 - b_gn2[None, :]), 0.)
            b_qg2 = b_q2 * b_gqn2
            b_kg2 = b_k2 * b_gqn2
            b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
            b_Aqk20 += tl.dot(b_qg2, b_kgt)
            b_Akk20 += tl.dot(b_kg2, b_kgt)
            b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
            b_Aqk21 += tl.dot(b_qg2, b_kgt)
            b_Akk21 += tl.dot(b_kg2, b_kgt)
            mk3 = m3[:, None] & m_k[None, :]
            b_q3 = tl.load(q + r3[:, None] * K + o_k[None, :], mask=mk3, other=0.).to(tl.float32)
            b_k3 = tl.load(k + r3[:, None] * K + o_k[None, :], mask=mk3, other=0.).to(tl.float32)
            b_g3 = tl.load(gc + r3[:, None] * K + o_k[None, :], mask=mk3, other=0.).to(tl.float32)
            b_e3 = exp2(b_g3)
            tl.store(k_up + r3[:, None] * K + o_k[None, :], (b_k3 * b_e3).to(k_up.dtype.element_ty), mask=mk3)
            tl.store(k_tail + r3[:, None] * K + o_k[None, :],
                     (b_k3 * exp2(b_gl[None, :] - b_g3)).to(k_tail.dtype.element_ty), mask=mk3)
            if STORE_QUP:
                tl.store(q_up + r3[:, None] * K + o_k[None, :], (b_q3 * b_e3).to(q_up.dtype.element_ty), mask=mk3)
            b_gn3 = tl.load(gc + ((bos + tn3) * H + i_h) * K + o_k, mask=m_k, other=0.).to(tl.float32)
            b_gqn3 = tl.where(m3[:, None], exp2(b_g3 - b_gn3[None, :]), 0.)
            b_qg3 = b_q3 * b_gqn3
            b_kg3 = b_k3 * b_gqn3
            b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
            b_Aqk30 += tl.dot(b_qg3, b_kgt)
            b_Akk30 += tl.dot(b_kg3, b_kgt)
            b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
            b_Aqk31 += tl.dot(b_qg3, b_kgt)
            b_Akk31 += tl.dot(b_kg3, b_kgt)
            b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
            b_Aqk32 += tl.dot(b_qg3, b_kgt)
            b_Akk32 += tl.dot(b_kg3, b_kgt)

    # store the Grams: the off-diagonal blocks, and zeros in the blocks above the diagonal
    b_zero = tl.zeros([BC, BC], dtype=tl.float32)
    c0 = o_i[None, :]
    c1 = (BC + o_i)[None, :]
    c2 = (2 * BC + o_i)[None, :]
    c3 = (3 * BC + o_i)[None, :]
    if NC > 1:
        tl.store(A_qk + r1[:, None] * BT + c0, b_Aqk10 * scale, mask=m1[:, None])
        tl.store(A_kk + r1[:, None] * BT + c0, b_Akk10, mask=m1[:, None])
        tl.store(A_qk + r0[:, None] * BT + c1, b_zero, mask=m0[:, None])
        tl.store(A_kk + r0[:, None] * BT + c1, b_zero, mask=m0[:, None])
    if NC > 2:
        tl.store(A_qk + r2[:, None] * BT + c0, b_Aqk20 * scale, mask=m2[:, None])
        tl.store(A_qk + r2[:, None] * BT + c1, b_Aqk21 * scale, mask=m2[:, None])
        tl.store(A_qk + r3[:, None] * BT + c0, b_Aqk30 * scale, mask=m3[:, None])
        tl.store(A_qk + r3[:, None] * BT + c1, b_Aqk31 * scale, mask=m3[:, None])
        tl.store(A_qk + r3[:, None] * BT + c2, b_Aqk32 * scale, mask=m3[:, None])
        tl.store(A_kk + r2[:, None] * BT + c0, b_Akk20, mask=m2[:, None])
        tl.store(A_kk + r2[:, None] * BT + c1, b_Akk21, mask=m2[:, None])
        tl.store(A_kk + r3[:, None] * BT + c0, b_Akk30, mask=m3[:, None])
        tl.store(A_kk + r3[:, None] * BT + c1, b_Akk31, mask=m3[:, None])
        tl.store(A_kk + r3[:, None] * BT + c2, b_Akk32, mask=m3[:, None])
        tl.store(A_qk + r0[:, None] * BT + c2, b_zero, mask=m0[:, None])
        tl.store(A_qk + r0[:, None] * BT + c3, b_zero, mask=m0[:, None])
        tl.store(A_qk + r1[:, None] * BT + c2, b_zero, mask=m1[:, None])
        tl.store(A_qk + r1[:, None] * BT + c3, b_zero, mask=m1[:, None])
        tl.store(A_qk + r2[:, None] * BT + c3, b_zero, mask=m2[:, None])
        tl.store(A_kk + r0[:, None] * BT + c2, b_zero, mask=m0[:, None])
        tl.store(A_kk + r0[:, None] * BT + c3, b_zero, mask=m0[:, None])
        tl.store(A_kk + r1[:, None] * BT + c2, b_zero, mask=m1[:, None])
        tl.store(A_kk + r1[:, None] * BT + c3, b_zero, mask=m1[:, None])
        tl.store(A_kk + r2[:, None] * BT + c3, b_zero, mask=m2[:, None])

    # T = strict(beta* . A_kk . A_vv); its diagonal blocks read A_kk's from the diagonal kernel
    b_T00 = tl.where(m_A, b_bs0[:, None] * tl.load(A_kk + r0[:, None] * BT + c0, mask=m0[:, None], other=0.) * b_vv00, 0.)
    if NC > 1:
        b_T11 = tl.where(m_A, b_bs1[:, None] * tl.load(A_kk + r1[:, None] * BT + c1, mask=m1[:, None], other=0.) * b_vv11, 0.)
        b_T10 = b_bs1[:, None] * b_Akk10 * b_vv10
    if NC > 2:
        b_T22 = tl.where(m_A, b_bs2[:, None] * tl.load(A_kk + r2[:, None] * BT + c2, mask=m2[:, None], other=0.) * b_vv22, 0.)
        b_T33 = tl.where(m_A, b_bs3[:, None] * tl.load(A_kk + r3[:, None] * BT + c3, mask=m3[:, None], other=0.) * b_vv33, 0.)
        b_T20 = b_bs2[:, None] * b_Akk20 * b_vv20
        b_T21 = b_bs2[:, None] * b_Akk21 * b_vv21
        b_T30 = b_bs3[:, None] * b_Akk30 * b_vv30
        b_T31 = b_bs3[:, None] * b_Akk31 * b_vv31
        b_T32 = b_bs3[:, None] * b_Akk32 * b_vv32

    # (I + T)^-1 on the diagonal blocks by forward substitution, solve_tril's 16x16 step:
    # row i is e_i - sum_{j<i} T[i, j] row_j, and the identity is added at the end
    b_Ai00 = -b_T00
    for i in range(2, min(BC, T - tc0)):
        b_a = -tl.sum(tl.where((o_i == i)[:, None], b_T00, 0.), 0)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a, b_Ai00)
    b_Ai00 += m_I
    if NC > 1:
        b_Ai11 = -b_T11
        for i in range(2, min(BC, T - tc1)):
            b_a = -tl.sum(tl.where((o_i == i)[:, None], b_T11, 0.), 0)
            b_a = tl.where(o_i < i, b_a, 0.)
            b_a += tl.sum(b_a[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i)[:, None], b_a, b_Ai11)
        b_Ai11 += m_I
    if NC > 2:
        b_Ai22 = -b_T22
        for i in range(2, min(BC, T - tc2)):
            b_a = -tl.sum(tl.where((o_i == i)[:, None], b_T22, 0.), 0)
            b_a = tl.where(o_i < i, b_a, 0.)
            b_a += tl.sum(b_a[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i)[:, None], b_a, b_Ai22)
        b_Ai22 += m_I
        b_Ai33 = -b_T33
        for i in range(2, min(BC, T - tc3)):
            b_a = -tl.sum(tl.where((o_i == i)[:, None], b_T33, 0.), 0)
            b_a = tl.where(o_i < i, b_a, 0.)
            b_a += tl.sum(b_a[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i)[:, None], b_a, b_Ai33)
        b_Ai33 += m_I

    # the inverse's off-diagonal blocks, solve_tril's merge step, with fp32 `ieee` dots
    tl.store(Tinv + r0[:, None] * BT + c0, b_Ai00, mask=m0[:, None])
    if NC > 1:
        b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_T10, input_precision='ieee'), b_Ai00, input_precision='ieee')
        tl.store(Tinv + r1[:, None] * BT + c0, b_Ai10, mask=m1[:, None])
        tl.store(Tinv + r1[:, None] * BT + c1, b_Ai11, mask=m1[:, None])
        tl.store(Tinv + r0[:, None] * BT + c1, b_zero, mask=m0[:, None])
    if NC > 2:
        b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_T21, input_precision='ieee'), b_Ai11, input_precision='ieee')
        b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_T32, input_precision='ieee'), b_Ai22, input_precision='ieee')
        b_Ai20 = -tl.dot(b_Ai22, tl.dot(b_T20, b_Ai00, input_precision='ieee')
                         + tl.dot(b_T21, b_Ai10, input_precision='ieee'), input_precision='ieee')
        b_Ai31 = -tl.dot(b_Ai33, tl.dot(b_T31, b_Ai11, input_precision='ieee')
                         + tl.dot(b_T32, b_Ai21, input_precision='ieee'), input_precision='ieee')
        b_Ai30 = -tl.dot(b_Ai33, tl.dot(b_T30, b_Ai00, input_precision='ieee')
                         + tl.dot(b_T31, b_Ai10, input_precision='ieee')
                         + tl.dot(b_T32, b_Ai20, input_precision='ieee'), input_precision='ieee')
        tl.store(Tinv + r2[:, None] * BT + c0, b_Ai20, mask=m2[:, None])
        tl.store(Tinv + r2[:, None] * BT + c1, b_Ai21, mask=m2[:, None])
        tl.store(Tinv + r2[:, None] * BT + c2, b_Ai22, mask=m2[:, None])
        tl.store(Tinv + r3[:, None] * BT + c0, b_Ai30, mask=m3[:, None])
        tl.store(Tinv + r3[:, None] * BT + c1, b_Ai31, mask=m3[:, None])
        tl.store(Tinv + r3[:, None] * BT + c2, b_Ai32, mask=m3[:, None])
        tl.store(Tinv + r3[:, None] * BT + c3, b_Ai33, mask=m3[:, None])
        tl.store(Tinv + r0[:, None] * BT + c2, b_zero, mask=m0[:, None])
        tl.store(Tinv + r0[:, None] * BT + c3, b_zero, mask=m0[:, None])
        tl.store(Tinv + r1[:, None] * BT + c2, b_zero, mask=m1[:, None])
        tl.store(Tinv + r1[:, None] * BT + c3, b_zero, mask=m1[:, None])
        tl.store(Tinv + r2[:, None] * BT + c3, b_zero, mask=m2[:, None])


def chunk_kronlstm_fwd_intra(q, k, v, gc, beta, target_gate, scale, beta_max_scaler, decouple_erase, chunk_size,
                             need_q_up=False, cu_seqlens=None, chunk_indices=None):
    """The intra-chunk half in two launches: both key Grams, `beta*`, `(I + T)^-1` and the decayed keys.

    Returns `(A_qk, A_kk, Tinv, q_up, k_up, k_tail, bs, pay)`; the three `[B, T, H, BT]` tensors are fp32
    and fully defined, `A_qk` carries `scale` and `A_kk` does not.
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = chunk_size
    BC = min(16, BT)
    NC = BT // BC
    assert NC in (1, 2, 4), "the inter kernel is written for 1, 2 or 4 sub-chunks of 16"
    if cu_seqlens is None:
        N, NT = B, triton.cdiv(T, BT)
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
    A_qk = k.new_empty(B, T, H, BT, dtype=torch.float32)
    A_kk = k.new_empty(B, T, H, BT, dtype=torch.float32)
    Tinv = k.new_empty(B, T, H, BT, dtype=torch.float32)
    k_up, k_tail = torch.empty_like(k), torch.empty_like(k)
    q_up = torch.empty_like(q) if need_q_up else None
    bs = k.new_empty(B, T, H, dtype=torch.float32)
    pay = k.new_empty(B, T, H, dtype=torch.float32)

    def grid(meta):
        return (B * T, triton.cdiv(H, meta['BH']))
    chunk_kronlstm_fwd_kernel_intra_diag[grid](
        q=q, k=k, gc=gc, A_qk=A_qk, A_kk=A_kk, scale=scale, cu_seqlens=cu_seqlens, N=N, T=T, H=H, K=K,
        BK=max(16, triton.next_power_of_2(K)), BT=BT, BC=BC,
    )
    kw = dict(q=q, k=k, v=v, gc=gc, beta=beta, ct=target_gate, A_qk=A_qk, A_kk=A_kk, q_up=q_up, k_up=k_up,
              k_tail=k_tail, bs=bs, pay=pay, Tinv=Tinv, scale=scale, s=beta_max_scaler, cu_seqlens=cu_seqlens,
              chunk_indices=chunk_indices, T=T, H=H, K=K, V=V, BT=BT, BC=BC, NC=NC,
              BV=min(64, triton.next_power_of_2(V)), DECOUPLE_ERASE=decouple_erase)
    chunk_kronlstm_fwd_kernel_inter[(NT, B * H)](**kw)
    return A_qk, A_kk, Tinv, q_up, k_up, k_tail, bs, pay


def chunk_kronlstm_fwd(q, k, v, beta, g, scale, initial_state, output_final_state, chunk_size,
                       beta_max_scaler, target_gate, decouple_erase, for_backward=False,
                       cu_seqlens=None, chunk_indices=None, chunk_offsets=None):
    """The forward on tensors padded to a multiple of `chunk_size`, or packed by `cu_seqlens`.

    With `for_backward`, it also returns the intermediates `chunk_kronlstm_bwd` consumes.
    """
    C = chunk_size
    if cu_seqlens is not None:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, C)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, C)
    # the log2 cumulative gate: the natural-log gate is scaled by 1 / ln 2 once, as GDN-2 does
    gc = chunk_local_cumsum(g, chunk_size=C, scale=RCP_LN2, cu_seqlens=cu_seqlens,
                            chunk_indices=chunk_indices)  # [B, T, H, K] fp32
    A_qk, A_kk, Tinv, q_up, k_up, k_tail, bs, pay = chunk_kronlstm_fwd_intra(
        q, k, v, gc, beta, target_gate, scale, beta_max_scaler, decouple_erase, C, need_q_up=for_backward,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    h, v_new, u, ht = chunk_kronlstm_fwd_h(k_up, k_tail, v, gc, Tinv, bs, pay, initial_state, output_final_state, C,
                                           cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                                           chunk_offsets=chunk_offsets)
    # the read-out Gram carries `scale`; `fwd_o_gk` masks it at load and scales the inter term
    o = chunk_gla_fwd_o_gk(q=q, v=v_new, g=gc, A=A_qk, h=h, scale=scale, chunk_size=C, cu_seqlens=cu_seqlens,
                           chunk_indices=chunk_indices)
    if not for_backward:
        return o, ht
    return o, ht, (gc, A_kk, A_qk, Tinv, bs, u, h, v_new, q_up, k_up, k_tail)
