# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Token-by-token KronLSTM for inference: `fused_recurrent_kronlstm`, forward only.

One program per (sequence, head) holds the whole `[K, V]` state in registers and walks the sequence a token at a time,
with no chunk padding and no Gram matrices; it repeats `naive_recurrent_kronlstm` step for step in fp32.
Unlike GDN-2's recurrent kernel it does not tile V across programs:
the score `y_t = k_t^T S v_t` sums over all of V, so a V-tile would need a reduction across programs at every token.
"""

import torch
import triton
import triton.language as tl

from fla.ops.kronlstm.naive import _gate_rank
from fla.ops.utils.op import exp
from fla.utils import input_guard


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_G': lambda args: args['g'] is not None,
    'HAS_CT': lambda args: args['ct'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_kronlstm_fwd_kernel(
    q,          # [B, T, H, K]
    k,          # [B, T, H, K]
    v,          # [B, T, H, V]
    beta,       # [B, T, H]
    g,          # [B, T, H, K] natural-log decay, or None
    ct,         # [B, T, H] target gate, or None
    o,          # out [B, T, H, V]
    h0,         # [N, H, K, V] fp32 or None
    ht,         # out [N, H, K, V] fp32 or None
    cu_seqlens,     # [N + 1] or None
    scale,
    s,          # 2 under allow_neg_eigval, else 1
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DECOUPLE_ERASE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_G: tl.constexpr,
    HAS_CT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T

    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * H + i_h) * V + o_v
    p_o = o + (bos * H + i_h) * V + o_v
    if USE_G:
        p_g = g + (bos * H + i_h) * K + o_k
    p_beta = beta + bos * H + i_h
    if HAS_CT:
        p_ct = ct + bos * H + i_h

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :], mask=m_h, other=0.).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=m_k, other=0.).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=m_k, other=0.).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0.).to(tl.float32)
        if USE_G:
            b_h *= exp(tl.load(p_g, mask=m_k, other=0.).to(tl.float32))[:, None]
        # the score on the decayed state: a full contraction over K and V
        b_y = tl.sum(tl.sum(b_h * b_k[:, None], 0) * b_v)
        b_vn = tl.sum(b_v * b_v)
        # beta_max = s / ||v||^2 with ||v||^2 floored at V_NORM_FLOOR
        b_bmax = s / tl.maximum(b_vn, 1e-18)
        b_bs = b_bmax * tl.load(p_beta).to(tl.float32)
        b_tgt = b_vn
        if HAS_CT:
            b_tgt = b_tgt * tl.load(p_ct).to(tl.float32)
        if DECOUPLE_ERASE:
            b_u = b_bmax * b_tgt - b_bs * b_y
        else:
            b_u = b_bs * (b_tgt - b_y)
        b_vnew = b_u * b_v
        b_h += b_k[:, None] * b_vnew[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)

        p_q += H * K
        p_k += H * K
        p_v += H * V
        p_o += H * V
        if USE_G:
            p_g += H * K
        p_beta += H
        if HAS_CT:
            p_ct += H

    if STORE_FINAL_STATE:
        tl.store(ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :], b_h.to(ht.dtype.element_ty), mask=m_h)


def _num_warps(BK: int, BV: int) -> int:
    """Four warps up to a `[128, 128]` state and eight beyond:
    the two full-state reductions per token grow more expensive across warps faster than occupancy improves.
    """
    return 4 if BK * BV <= 16384 else 8


@input_guard
def fused_recurrent_kronlstm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    allow_neg_eigval: bool = False,
    target_gate: torch.Tensor | None = None,
    decouple_erase: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
):
    """Token-by-token KronLSTM forward for inference and decoding.

    Backward is not supported; train with `chunk_kronlstm`.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`, assumed to be unit-norm.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            natural-log decay, `[B, T, H]` (one rate per head) or `[B, T, H, K]` (one rate per key dimension),
            or `None` for no decay.
        beta (torch.Tensor):
            erase gate (the step size of the Kronecker delta rule) of shape `[B, T, H]`, in `[0, 1]`.
        scale (float, Optional):
            read-out scale. Default: `K ** -0.5`.
        initial_state (torch.Tensor, Optional):
            initial state of shape `[N, H, K, V]` for `N` input sequences, which is `B` for equal-length sequences;
            it is read in fp32. Default: `None`.
        output_final_state (bool, Optional):
            whether to return the final state. Default: `False`.
        allow_neg_eigval (bool, Optional):
            whether `beta* = s * beta / ||v||^2` may reach the negative-eigenvalue range, `s = 2`; else `s = 1`.
            Default: `False`.
        target_gate (torch.Tensor, Optional):
            target gate `c` of shape `[B, T, H]`, after a sigmoid; the write target becomes `c * ||v||^2`.
            Default: `None`.
        decouple_erase (bool, Optional):
            whether `beta` gates the erase only: the write keeps the cap `s / ||v||^2`,
            so `target_gate` alone gates it. Requires `target_gate`. Default: `True`.
        cu_seqlens (torch.LongTensor, Optional):
            cumulative sequence lengths of shape `[N + 1]` for variable-length inputs packed along `T`,
            consistent with the FlashAttention API; `B` must be 1. Default: `None`.

    Returns:
        o (torch.Tensor):
            outputs of shape `[B, T, H, V]`, in `v`'s dtype.
        final_state (torch.Tensor):
            final state of shape `[N, H, K, V]` in fp32 if `output_final_state`, else `None`.
    """
    if decouple_erase and target_gate is None:
        raise ValueError("`decouple_erase=True` requires `target_gate`.")
    g_rank = _gate_rank(g, k.shape[-1])
    B, T, H, K = k.shape
    V = v.shape[-1]
    if K > 256 or V > 256:
        raise ValueError(f"The recurrent kernel holds the whole [K, V] state per program; K, V <= 256 (got {K}, {V}).")
    N = B
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {B} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        N = len(cu_seqlens) - 1
        if initial_state is not None and initial_state.shape[0] != N:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {N} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = K ** -0.5

    q, k, v = (x.contiguous() for x in (q, k, v))
    beta = beta.contiguous()
    target_gate = target_gate.contiguous() if target_gate is not None else None
    if g_rank == 0:
        g = None
    elif g_rank == 3:
        g = g.float().unsqueeze(-1).expand(B, T, H, K).contiguous()
    else:
        g = g.float().contiguous()
    h0 = initial_state.to(torch.float32).contiguous() if initial_state is not None else None
    ht = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    o = torch.empty_like(v)
    BK, BV = triton.next_power_of_2(K), triton.next_power_of_2(V)
    fused_recurrent_kronlstm_fwd_kernel[(N * H,)](
        q=q, k=k, v=v, beta=beta, g=g, ct=target_gate, o=o, h0=h0, ht=ht, cu_seqlens=cu_seqlens, scale=scale,
        s=2.0 if allow_neg_eigval else 1.0, T=T, H=H, K=K, V=V, BK=BK, BV=BV, DECOUPLE_ERASE=decouple_erase,
        num_warps=_num_warps(BK, BV), num_stages=2,
    )
    return o, ht
