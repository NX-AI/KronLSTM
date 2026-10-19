# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

from transformers.configuration_utils import PretrainedConfig

from fla.models.hybrid import HybridAttentionConfig, _HybridAttentionConfigMixin


class KronLSTMConfig(_HybridAttentionConfigMixin, PretrainedConfig):
    """The configuration of a KronLSTM language model.

    The defaults are the paper's 1.5B model: 24 layers of KronLSTM and a SwiGLU MLP
    (intermediate size 5632 at hidden size 2048), tied embeddings and the Llama-2 vocabulary.

    `attn` places softmax attention at selected layers, as in FLA's other hybrid models (`fla.models.hybrid`).
    Each specification additionally takes `use_rope` (default `True`): `False` gives attention without
    positional encoding (NoPE), as in the paper's hybrid models, and the specification's `rope_theta` is then
    unused. `window_size` counts the attending token itself, so a window that sees the token and the
    2048 tokens before it is `window_size=2049`.
    """

    model_type = 'kronlstm'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        attn_mode: str = 'chunk',
        hidden_size: int = 2048,
        num_heads: int = 16,
        k_head_dim: int = 128,
        v_head_dim: int = 128,
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
        max_position_embeddings: int = 8192,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = 'swish',
        num_hidden_layers: int = 24,
        norm_eps: float = 1e-5,
        attn: HybridAttentionConfig = None,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        **kwargs,
    ):
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.normalize_v = normalize_v
        self.use_beta = use_beta
        self.allow_neg_eigval = allow_neg_eigval
        self.use_target_gate = use_target_gate
        self.decouple_erase = decouple_erase
        self.decay_gate = decay_gate
        self.use_output_gate = use_output_gate
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError("`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.")
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _normalize_hybrid_attention_config(self, attn: HybridAttentionConfig) -> HybridAttentionConfig:
        attn = super()._normalize_hybrid_attention_config(attn)
        # `use_rope` is KronLSTM's addition to FLA's specification, which keeps unknown keys as they are
        for spec in [attn] if isinstance(attn, dict) else attn or []:
            use_rope = spec.setdefault('use_rope', True)
            if not isinstance(use_rope, bool):
                raise ValueError(f"attn specification field 'use_rope' must be a Boolean; got {use_rope!r}")
        return attn
