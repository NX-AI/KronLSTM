# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.kronlstm import chunk_kronlstm, fused_recurrent_kronlstm

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


# `chunk` trains (the chunkwise Triton kernels, forward and backward); `fused_recurrent` is the token-by-token forward
MODES = ('chunk', 'fused_recurrent')

# 'vector' decays each key dimension at its own rate, as GDN-2 does (`g` of shape [B, T, H, K]);
# 'scalar' is the per-head reduction of the same parametrisation (`g` of shape [B, T, H])
DECAY_GATES = ('vector', 'scalar')

# the output gate's bias at initialisation: sigmoid(3) ~ 0.95, so the layer starts close to an ungated one;
# it is exempt from weight decay, which would pull the gate back towards half open
OUT_GATE_BIAS_INIT = 3.0


@torch.compiler.disable
def _eager(fn, *args, **kwargs):
    """Run the op or the gated norm outside torch.compile, so that dynamo does not trace into their autograd Functions."""
    return fn(*args, **kwargs)


class KronLSTM(nn.Module):
    """KronLSTM layer.

    The matrix state `S in R^{K x V}` is written along one rank-one Kronecker key `k_t v_t^T` per step,
    with a scalar write strength that closes the residual to a target:

        y_t = k_t^T Diag(exp(g_t)) S_{t-1} v_t
        u_t = beta*_t (c_t ||v_t||^2 - y_t),    beta*_t = s beta_t / ||v_t||^2
        S_t = Diag(exp(g_t)) S_{t-1} + u_t k_t v_t^T

    with unit-norm q and k, a per-head erase gate `beta_t`, a per-head write (target) gate `c_t`,
    `s = 2` under `allow_neg_eigval` and 1 otherwise,
    and GDN-2's channel-wise decay `g_t = -exp(A_log) * softplus(f(x_t) + dt_bias)`.
    Under `decouple_erase` the learnt `beta` comes off the write and gates the erase only.
    See `fla/ops/kronlstm/README.md` and the KronLSTM paper (https://arxiv.org/abs/2610.26834).

    The projections keep `nn.Linear`'s init: a model or training framework initialises them with its own scheme
    and then calls `reset_gate_parameters` for the parameters whose init is the layer's own.
    `A_log`, `dt_bias`, every bias and the norm weights are marked `_no_weight_decay`.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        k_head_dim (int, Optional):
            The dimension of each query and key head. Default: 128.
        v_head_dim (int, Optional):
            The dimension of each value head. Default: 128.
        num_k_heads (int, Optional):
            The number of query and key heads. Default: 16.
        num_v_heads (int, Optional):
            The number of value heads, which must equal `num_k_heads`: there is no grouped value attention. Default: 16.
        mode (str, Optional):
            Which kernel to use, `chunk` or `fused_recurrent`. With gradients enabled the layer always uses `chunk`,
            and it falls back to `fused_recurrent` for short inference sequences (`q_len <= 64`). Default: `chunk`.
        use_short_conv (bool, Optional):
            Whether to apply one depthwise short convolution to the concatenated q, k and v. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolution. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution. Default: `False`.
        normalize_v (bool, Optional):
            Whether to L2-normalize v per head; q and k always are, since the recurrence assumes a unit-norm k.
            Default: `True`.
        use_beta (bool, Optional):
            Whether to learn the per-head erase gate; with `False`, `beta = 1` and `allow_neg_eigval` is off.
            Default: `True`.
        allow_neg_eigval (bool, Optional):
            Whether `s = 2` in `beta* = s beta / ||v||^2`, which admits negative eigenvalues;
            otherwise `s = 1`, which caps the write at the no-overshoot point. Default: `False`.
        use_target_gate (bool, Optional):
            Whether to learn the target gate `c`, the fraction of `||v||^2` the write aims for;
            with `False`, the target is `||v||^2`. Default: `True`.
        decouple_erase (bool, Optional):
            Whether to take `beta` off the write, so that it gates the erase only
            and the target gate alone gates the write. Requires `use_target_gate`. Default: `True`.
        decay_gate (str, Optional):
            `vector` for one decay rate per key dimension, `scalar` for one per head. Default: `vector`.
        use_output_gate (bool, Optional):
            Whether to use the sigmoid-gated output norm.
            Its gate starts at `sigmoid(3)` for every token: bias 3 and zero weights. Default: `True`.
        layer_idx (int, Optional):
            The index of the layer, required when a cache is used. Default: `None`.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        k_head_dim: int = 128,
        v_head_dim: int = 128,
        num_k_heads: int = 16,
        num_v_heads: int = 16,
        mode: str = 'chunk',
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        normalize_v: bool = True,
        use_beta: bool = True,
        allow_neg_eigval: bool = False,
        use_target_gate: bool = True,
        decouple_erase: bool = True,
        decay_gate: str = 'vector',
        use_output_gate: bool = True,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()

        if mode not in MODES:
            raise ValueError(f"mode=`{mode}` is not one of {MODES}.")
        if decay_gate not in DECAY_GATES:
            raise ValueError(f"decay_gate=`{decay_gate}` is not one of {DECAY_GATES}.")
        if num_k_heads != num_v_heads:
            raise ValueError(
                f"KronLSTM shares one head count across q/k/v, got num_k_heads={num_k_heads} and "
                f"num_v_heads={num_v_heads}.",
            )
        # training always runs the chunk kernels
        if not (16 <= k_head_dim <= 256 and 16 <= v_head_dim <= 256):
            raise ValueError(
                f"KronLSTM's chunk kernels support head dims in [16, 256], got k_head_dim={k_head_dim} and "
                f"v_head_dim={v_head_dim}.",
            )
        if decouple_erase and not use_target_gate:
            raise ValueError("`decouple_erase=True` requires `use_target_gate=True`.")

        self.mode = mode
        self.hidden_size = hidden_size
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.num_heads = num_v_heads
        self.key_dim = self.num_heads * k_head_dim
        self.value_dim = self.num_heads * v_head_dim
        self.layer_idx = layer_idx

        self.normalize_v = normalize_v
        self.use_beta = use_beta
        # with beta = 1 everywhere, s = 2 would put every step on the eigenvalue -1 edge
        self.allow_neg_eigval = allow_neg_eigval and use_beta
        self.use_target_gate = use_target_gate
        self.decouple_erase = decouple_erase
        self.decay_gate = decay_gate
        self.use_output_gate = use_output_gate

        # q, k and v in one projection and one depthwise convolution over the concatenated channels,
        # which is three independent convolutions on the slices
        self.qkv_proj = nn.Linear(hidden_size, 2 * self.key_dim + self.value_dim, bias=False)

        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.conv_size = conv_size
            self.qkv_conv1d = ShortConvolution(2 * self.key_dim + self.value_dim, conv_size, bias=conv_bias,
                                               activation='silu')
        else:
            warnings.warn("Short convolution is not used.", stacklevel=2)

        # the per-head erase gate, a plain sigmoid in [0, 1];
        # `allow_neg_eigval` opens the negative-eigenvalue range in the op
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # the target gate, per head; after its sigmoid the target is a bounded fraction of ||v||^2
        if use_target_gate:
            self.c_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # one decay channel per (head, key dimension) for 'vector', one per head for 'scalar'
        self.gate_dim = self.key_dim if decay_gate == 'vector' else self.num_heads
        # low rank through v_head_dim, as GDN-2 and KDA factor it
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, v_head_dim, bias=False),
            nn.Linear(v_head_dim, self.gate_dim, bias=False),
        )
        # a per-head decay rate (A_log) and a per-channel softplus bias (dt_bias), as GDN-2 and KDA place them
        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.gate_dim, dtype=torch.float32))
        # drawn here, in creation order: moving the draw would change the layer a seed builds
        self._reset_decay_parameters()

        # the output path: a sigmoid-gated RMSNorm with a dense hidden -> value_dim gate projection, then o_proj
        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=True)
            self.o_norm = FusedRMSNormGated(v_head_dim, activation='sigmoid', eps=norm_eps)
        else:
            self.o_norm = RMSNorm(v_head_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self._reset_gates()

        for name, p in self.named_parameters():
            if name in ('A_log', 'dt_bias') or name.endswith('.bias') or name.startswith('o_norm.'):
                p._no_weight_decay = True

    @torch.no_grad()
    def reset_gate_parameters(self, generator: torch.Generator | None = None) -> None:
        """(Re)initialise the parameters whose init is the layer's own: the decay's `A_log` and `dt_bias`,
        and the output gate's bias and weights.

        `__init__` initialises them the same way. A framework that allocates the layer on the meta device,
        materialises it and re-initialises every projection with its own scheme (FSDP) calls this afterwards,
        with its own `generator`; the updates are in place, so they also work on sharded parameters.
        """
        self._reset_decay_parameters(generator)
        self._reset_gates()

    @torch.no_grad()
    def _reset_decay_parameters(self, generator: torch.Generator | None = None) -> None:
        # A_log ~ log U(0, 16) per head; dt_bias = inverse softplus of logU(1e-3, 1e-1), floored at 1e-4.
        # Nothing chains on what `nn.init` returns: transformers 4 swaps these functions for ones returning None
        # while `from_pretrained` builds the model
        nn.init.uniform_(self.A_log, 0, 16, generator=generator)
        self.A_log.log_()
        nn.init.uniform_(self.dt_bias, generator=generator)
        dt = torch.exp(self.dt_bias * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    @torch.no_grad()
    def _reset_gates(self) -> None:
        if self.use_output_gate:
            nn.init.constant_(self.g_proj.bias, OUT_GATE_BIAS_INIT)
            # zero weights: the gate is exactly sigmoid(bias) for every token at initialisation
            nn.init.zeros_(self.g_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        """Args:
            hidden_states (torch.Tensor):
                Inputs of shape `[B, T, hidden_size]`, or `[1, total_tokens, hidden_size]` with `cu_seqlens`.
            attention_mask (torch.Tensor, Optional):
                A `[B, T]` 0/1 padding mask.
                The valid positions are packed into one variable-length batch (`cu_seqlens`), run per sequence,
                and padded back. Default: `None`.
            past_key_values (Cache, Optional):
                The cache holding the convolution and recurrent states. Default: `None`.
            use_cache (bool, Optional):
                Whether to return the updated cache. Default: `False`.
            output_attentions (bool, Optional):
                Unused: no attention weights are returned. Default: `False`.
            cu_seqlens (torch.LongTensor, Optional):
                Cumulative sequence lengths of shape `[N + 1]` for variable-length inputs packed along `T`;
                the batch size must then be 1. Passed through `kwargs`. Default: `None`.
        """
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "Expected attention_mask as a [batch_size, seq_len] 0/1 padding mask.",
            )

        batch_size, q_len, _ = hidden_states.shape
        last_state = get_layer_cache(self, past_key_values)

        # a padding mask packs the valid positions into one variable-length batch, as the other FLA layers do;
        # explicit `cu_seqlens` take precedence and leave the input as it is
        cu_seqlens = kwargs.get('cu_seqlens')
        hidden_states, indices, cu_seqlens = unpad_hidden_states(hidden_states, cu_seqlens, attention_mask, q_len)

        qkv = self.qkv_proj(hidden_states)
        conv_state = None
        if self.use_short_conv:
            if last_state is not None:
                conv_state = last_state['conv_state']
            qkv, conv_state = self.qkv_conv1d(x=qkv, cache=conv_state, output_final_state=use_cache,
                                              cu_seqlens=cu_seqlens)
        else:
            qkv = F.silu(qkv)
        q, k, v = qkv.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)

        q, k = (rearrange(x, '... (h d) -> ... h d', d=self.k_head_dim) for x in (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.v_head_dim)
        # back to the projection's dtype after the norm, which autocast runs in fp32:
        # under bf16 autocast the op then runs its bf16 kernels, whose scan holds the [K, V] state at every head dim
        # (the fp32 one stops at 128 x 128)
        q, k = (F.normalize(x, dim=-1).to(x.dtype) for x in (q, k))
        if self.normalize_v:
            v = F.normalize(v, dim=-1).to(v.dtype)

        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones(*hidden_states.shape[:-1], self.num_heads,
                              dtype=hidden_states.dtype, device=hidden_states.device)

        # the decay in explicit fp32 (GDN-2, arXiv:2605.22791, App. D.1): `.float()` before the bias,
        # so the softplus, the exponential and the product run in fp32 whatever dtype the projection ran in
        g = F.softplus(self.f_proj(hidden_states).float() + self.dt_bias)
        if self.decay_gate == 'vector':
            g = rearrange(g, '... (h d) -> ... h d', d=self.k_head_dim)
            g = -self.A_log.float().exp().unsqueeze(-1) * g
        else:
            g = -self.A_log.float().exp() * g

        # the target gate's pre-activation in fp32 as well:
        # bf16's 8-bit mantissa is ~0.4% on the pre-activation itself
        target_gate = self.c_proj(hidden_states).float().sigmoid() if self.use_target_gate else None

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        # as in fla/layers/gated_deltanet.py: with gradients enabled always the chunk kernels,
        # since the token-by-token kernel has no backward;
        # otherwise short inference inputs (decoding steps, short prompts) take the latter
        if torch.is_grad_enabled():
            mode = 'chunk'
        elif q_len <= 64 and not self.training:
            mode = 'fused_recurrent'
        else:
            mode = self.mode
        op = {'chunk': chunk_kronlstm, 'fused_recurrent': fused_recurrent_kronlstm}[mode]
        o, recurrent_state = _eager(
            op,
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            allow_neg_eigval=self.allow_neg_eigval,
            target_gate=target_gate,
            decouple_erase=self.decouple_erase,
            cu_seqlens=cu_seqlens,
        )

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=conv_state,
            offset=q_len,
        )

        if self.use_output_gate:
            o = _eager(self.o_norm, o, rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.v_head_dim))
        else:
            o = _eager(self.o_norm, o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        # back to the model dtype at the kernel boundary (GDN-2 App. D.3): `o` comes out in v's dtype
        o = self.o_proj(o.to(hidden_states.dtype))
        o = repad_hidden_states(o, indices, batch_size, q_len)
        return o, None, past_key_values
