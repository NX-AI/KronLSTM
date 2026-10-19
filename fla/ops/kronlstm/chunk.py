# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Chunkwise KronLSTM for training: `chunk_kronlstm`, the public entry point, and its autograd Function.

The forward lives in `chunk_fwd.py` and the backward in `chunk_bwd.py`; `README.md` describes both.
"""

import torch

from fla.ops.kronlstm.chunk_bwd import chunk_kronlstm_bwd
from fla.ops.kronlstm.chunk_fwd import chunk_kronlstm_fwd
from fla.ops.kronlstm.naive import _gate_rank
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class ChunkKronLSTMFunction(torch.autograd.Function):
    """Autograd wrapper around the KronLSTM chunkwise forward and backward."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta, g, scale, initial_state, output_final_state, chunk_size,
                beta_max_scaler, target_gate, decouple_erase, need_dg, cu_seqlens, chunk_indices, chunk_offsets):
        # grad mode is off inside Function.forward, so ask the ctx, not the tensors
        need_grad = any(ctx.needs_input_grad)
        out = chunk_kronlstm_fwd(q, k, v, beta, g, scale, initial_state, output_final_state,
                                 chunk_size, beta_max_scaler, target_gate, decouple_erase, for_backward=need_grad,
                                 cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets)
        if not need_grad:
            return out
        o, ht, (gc, A_kk, A_qk, Tinv, bs, u, h, v_new, q_up, k_up, k_tail) = out
        ctx.save_for_backward(q, k, v, beta, target_gate, gc, A_kk, A_qk, Tinv, bs, u, h, v_new,
                              q_up, k_up, k_tail, cu_seqlens, chunk_indices, chunk_offsets)
        ctx.has_h0 = initial_state is not None
        ctx.scale, ctx.chunk_size = scale, chunk_size
        ctx.beta_max_scaler, ctx.decouple_erase, ctx.need_dg = beta_max_scaler, decouple_erase, need_dg
        return o, ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        (q, k, v, beta, target_gate, gc, A_kk, A_qk, Tinv, bs, u, h, v_new, q_up, k_up, k_tail,
         cu_seqlens, chunk_indices, chunk_offsets) = ctx.saved_tensors
        dq, dk, dv, dbeta, dg, dct, dh0 = chunk_kronlstm_bwd(
            q, k, v, beta, target_gate, gc, A_kk, A_qk, Tinv, bs, u, h, v_new, q_up, k_up, k_tail,
            ctx.has_h0, do.contiguous(), None if dht is None else dht.contiguous(), ctx.scale,
            ctx.beta_max_scaler, ctx.decouple_erase, ctx.chunk_size, need_dg=ctx.need_dg,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets)
        return dq, dk, dv, dbeta, dg, None, dh0, None, None, None, dct, None, None, None, None, None


@torch.compiler.disable
def chunk_kronlstm(
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
    """Chunkwise KronLSTM, forward and backward in Triton.

    One step on the state `S in R^{K x V}`, with a unit-norm `k`, is

        y_t = k_t^T Diag(exp(g_t)) S_{t-1} v_t
        u_t = (s * beta_t / ||v_t||^2) (||v_t||^2 - y_t)
        S_t = Diag(exp(g_t)) S_{t-1} + u_t k_t v_t^T,    o_t = scale * q_t^T S_t

    with `s = 2` under `allow_neg_eigval`, else `s = 1`;
    `target_gate` and `decouple_erase` change the write as `README.md` describes. The chunk length is 64.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`, assumed to be unit-norm.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            natural-log decay, `[B, T, H]` (one rate per head) or `[B, T, H, K]` (one rate per key dimension),
            or `None` for no decay. A per-head gate is broadcast over K and runs the per-key-dimension kernels.
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
    if not (16 <= K <= 256 and 16 <= V <= 256):
        raise ValueError(
            f"The chunk kernels support head dims K, V in [16, 256] (`tl.dot` needs 16, and the scan holds the whole "
            f"[K, V] state per program), got K={K}, V={V}.",
        )
    if k.dtype == torch.float32 and K > 128 and V > 128:
        raise ValueError(
            f"With fp32 inputs, K <= 128 or V <= 128 is required, got K={K}, V={V}: "
            f"the scan stages every 64-row block of its [K, V] state in shared memory, in k's dtype "
            f"(320 KB at K = V = 256 in fp32, against 227 KB on an H100). Pass bf16 inputs.",
        )
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {B} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = K ** -0.5
    beta_max_scaler = 2.0 if allow_neg_eigval else 1.0
    chunk_size = 64

    q, k, v, beta = (x.contiguous() for x in (q, k, v, beta))
    if g_rank == 0:
        g = torch.zeros(B, T, H, K, dtype=torch.float32, device=k.device)
    elif g_rank == 3:
        g = g.float().unsqueeze(-1).expand(B, T, H, K).contiguous()
    else:
        g = g.float().contiguous()
    target_gate = target_gate.contiguous() if target_gate is not None else None

    chunk_indices, chunk_offsets = None, None
    pad = 0
    if cu_seqlens is not None:
        # packed sequences cannot be padded one by one; the kernels mask each sequence's last chunk instead
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
    else:
        pad = (chunk_size - T % chunk_size) % chunk_size
    if pad:
        # padded steps neither decay (g = 0) nor write (v = 0, hence beta* = 0);
        # T is dim -3 of the [B, T, H, .] tensors and dim -2 of beta
        q, k, v, g = (torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v, g))
        beta = torch.nn.functional.pad(beta, (0, 0, 0, pad))
        target_gate = torch.nn.functional.pad(target_gate, (0, 0, 0, pad)) if target_gate is not None else None
    if initial_state is not None:
        initial_state = initial_state.to(torch.float32).contiguous()

    o, ht = ChunkKronLSTMFunction.apply(q, k, v, beta, g, scale, initial_state,
                                        output_final_state, chunk_size, beta_max_scaler,
                                        target_gate, decouple_erase, g_rank != 0, cu_seqlens, chunk_indices, chunk_offsets)
    return o[:, :T], ht
