# KronLSTM

KronLSTM[^1] is a linear-attention architecture built on the **Kronecker Delta Rule**, an update rule for associative matrix memories. In place of the delta rule's vector residual $\mathbf{S}\mathbf{k}_t-\mathbf{v}_t$, it corrects the single scalar association between a key and a value, and it does so along the outer product $\mathbf{k}_t\mathbf{v}_t^\top$ of that pair alone. The memory is revised along one direction in matrix space, so a write interferes only with stored pairs that overlap the new one in both key and value. Three gates control the memory: a vector forget gate $\mathbf{f}_t$, an erase gate $\beta_t$ and a write gate $c_t$. There is no normalizer or stabilizer state. This directory provides:

| Function                   | File                 | Use                                                         |
| -------------------------- | -------------------- | ----------------------------------------------------------- |
| `chunk_kronlstm`           | `chunk.py`           | training: chunkwise Triton forward and backward             |
| `fused_recurrent_kronlstm` | `fused_recurrent.py` | inference: token-by-token Triton forward, no backward       |
| `naive_recurrent_kronlstm` | `naive.py`           | reference: token by token, differentiable                   |
| `naive_chunk_kronlstm`     | `naive.py`           | reference: chunkwise, differentiable, the kernels' oracle   |

## The Kronecker Delta Rule

In the paper's orientation, $\mathbf{S}\in\mathbb{R}^{d_v\times d_k}$ and the scalar $\mathbf{v}_t^\top\mathbf{S}\mathbf{k}_t$ measures how strongly the memory associates $\mathbf{k}_t$ with $\mathbf{v}_t$. The rule moves this scalar towards a target $c_t$:

```math
\mathbf{S}_{t+1} = \mathbf{S}_t - \beta_t\left(\mathbf{v}_t^\top\mathbf{S}_t\mathbf{k}_t - c_t\right)\mathbf{v}_t\mathbf{k}_t^\top .
```

With the Kronecker feature $\mathbf{a}_t=\mathbf{k}_t\otimes\mathbf{v}_t=\mathrm{vec}(\mathbf{v}_t\mathbf{k}_t^\top)$ and $\mathbf{s}=\mathrm{vec}(\mathbf{S})$, the association is $\mathbf{a}_t^\top\mathbf{s}$ and $\lVert\mathbf{a}_t\rVert^2=\lVert\mathbf{k}_t\rVert^2\lVert\mathbf{v}_t\rVert^2$. The rule then becomes the Widrow–Hoff delta rule with input $\mathbf{a}_t$ and a scalar target $c_t$, which gives it its name:

```math
\mathbf{s}_{t+1} = \left(\mathbf{I} - \beta_t\,\mathbf{a}_t\mathbf{a}_t^\top\right)\mathbf{s}_t + \beta_t\,c_t\,\mathbf{a}_t .
```

The paper proves the following properties of the rule; they shape the kernels below:

- **Error correction.** Presenting the same pair again scales the residual by $1-\beta_t\lVert\mathbf{a}_t\rVert^2$. The step therefore contracts the residual exactly when $0<\beta_t\lVert\mathbf{a}_t\rVert^2<2$, and $\beta_t\lVert\mathbf{a}_t\rVert^2=1$ meets the target in one step. Over $(1,2)$ the transition has a negative eigenvalue, which is what gives delta-rule models their state tracking.
- **Interference.** For another stored pair $(\mathbf{k}_i,\mathbf{v}_i)$, the update changes that pair's association by $-\beta_t(\mathbf{v}_t^\top\mathbf{S}_t\mathbf{k}_t-c_t)\,(\mathbf{v}_i^\top\mathbf{v}_t)(\mathbf{k}_t^\top\mathbf{k}_i)$. This is the product of the value overlap and the key overlap, so orthogonality in either one protects the stored pair. In the delta rule the key overlap alone decides.
- **Chunkwise training.** The transition is a generalized Householder matrix that does not depend on $c_t$. The chunkwise WY/UT scheme of DeltaNet therefore applies with $\mathbf{a}_t$ in place of $\mathbf{k}_t$, and the target enters only the driving term.
- **Bounded memory.** In the non-expansive regime $\beta_t\lVert\mathbf{a}_t\rVert^2\le 1$, the expected memory norm is bounded at every step by a constant set by the data distribution alone. The memory converges to a unique steady state whose mean is the least-squares memory.

## Recurrence in the kernels

The kernels store the transpose, in FLA's layout: $\mathbf{S}\in\mathbb{R}^{K\times V}$, written along $\mathbf{k}_t\mathbf{v}_t^\top$ and read out as $\mathbf{o}_t=\mathbf{q}_t^\top\mathbf{S}_t$. Keys are unit-norm. Values may have any norm, and the erase gate is rescaled by $\lVert\mathbf{v}_t\rVert^2$ so that $\beta^*_t\lVert\mathbf{a}_t\rVert^2=s\beta_t$ whatever the value norm. Here $s=2$ under `allow_neg_eigval` and $s=1$ otherwise. One step of the default rule is

```math
\begin{aligned}
\tilde{\mathbf{S}}_{t-1} &= \mathbf{D}_t\,\mathbf{S}_{t-1}, \qquad \mathbf{D}_t=\mathrm{Diag}(\exp\mathbf{g}_t)=\mathrm{Diag}(\mathbf{f}_t) && \text{forget} \\
y_t &= \mathbf{k}_t^\top \tilde{\mathbf{S}}_{t-1}\,\mathbf{v}_t && \text{score on the decayed memory} \\
\beta^*_t &= s\,\beta_t / \lVert\mathbf{v}_t\rVert^2, \qquad \beta^{\max}_t = s / \lVert\mathbf{v}_t\rVert^2 \\
u_t &= \beta^{\max}_t\,c_t\lVert\mathbf{v}_t\rVert^2 - \beta^*_t\,y_t && \text{write minus erase} \\
\mathbf{S}_t &= \tilde{\mathbf{S}}_{t-1} + u_t\,\mathbf{k}_t\mathbf{v}_t^\top \\
\mathbf{o}_t &= \text{scale}\cdot\mathbf{q}_t^\top\mathbf{S}_t
\end{aligned}
```

The erase gate is `beta`, the write gate $c_t$ is `target_gate` (after a sigmoid), and the forget gate is $\mathbf{g}_t=\log\mathbf{f}_t$. With $\ell_2$-normalized values and $s=1$, the write coefficient reduces to $u_t=c_t-\beta_t y_t$, the paper's KronLSTM update in transposed form:

```math
\mathbf{S}_t = \mathbf{D}_t\mathbf{S}_{t-1} \;-\; \beta_t\,y_t\,\mathbf{k}_t\mathbf{v}_t^\top \;+\; c_t\,\mathbf{k}_t\mathbf{v}_t^\top ,
```

so forgetting, erasing and writing are each set by their own gate. This form uses the decoupled target $\tilde c_t=c_t/\beta_t$: the learnt $\beta_t$ is taken off the write, so $\beta_t$ alone sets how much of the stored association is erased and $c_t$ alone how much is written.

## Variants

All variants share the score $y_t$, the forget gate and the erase rate. They differ only in the write coefficient. The paper's ablation (Appendix A6, Table A3) compares them; the first row is KronLSTM.

| Paper                         | Op arguments                                  | $u_t$                                                                   | $u_t$ at $\lVert\mathbf{v}\rVert=1$, $s=1$ |
| ----------------------------- | --------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------ |
| KronLSTM (decoupled target)   | `target_gate=c` (default `decouple_erase=True`) | $\beta^{\max}_t c_t\lVert\mathbf{v}_t\rVert^2-\beta^*_t y_t$          | $c_t-\beta_t y_t$                          |
| "$\beta$ on both sides"       | `target_gate=c`, `decouple_erase=False`       | $\beta^*_t\,(c_t\lVert\mathbf{v}_t\rVert^2-y_t)$                        | $\beta_t\,(c_t-y_t)$                       |
| "non-learnable target"        | `decouple_erase=False`                        | $\beta^*_t\,(\lVert\mathbf{v}_t\rVert^2-y_t)$                           | $\beta_t\,(1-y_t)$                         |

`decouple_erase=True` needs a `target_gate`, so a call without one must pass `decouple_erase=False`. The fourth ablation, "no $\ell_2$-norm on $\mathbf{v}_t$", is the layer's `normalize_v=False`. The ops handle it through the $1/\lVert\mathbf{v}_t\rVert^2$ in $\beta^*_t$ and $\beta^{\max}_t$, with no change of rule.

Substituting the write back into the score gives the fixed point each variant drives the association to:

```math
\begin{aligned}
\text{coupled:}\quad & y_t^{+} - c_t\lVert\mathbf{v}_t\rVert^2 = (1 - s\beta_t)\,\left(y_t - c_t\lVert\mathbf{v}_t\rVert^2\right), \\
\text{decoupled:}\quad & y_t^{+} = (1 - s\beta_t)\,y_t + s\,c_t\lVert\mathbf{v}_t\rVert^2, \qquad \text{fixed point } c_t\lVert\mathbf{v}_t\rVert^2/\beta_t .
\end{aligned}
```

In every variant the residual contracts at the rate $1-s\beta_t$, independently of $\lVert\mathbf{v}_t\rVert$. A pair that is already stored is not written again, a missing pair is written, and an over-stored pair is partly erased. With $s=1$, every step is non-expansive, the regime of the paper's memory-norm bound. $s=2$ admits the negative-eigenvalue half of the range, which the paper uses for state tracking.

Every variant is the gated delta rule on $\mathbf{s}=\mathrm{vec}(\mathbf{S})$, with key $\mathbf{a}_t$, erase strength $\beta^*_t$ and a driving term $w_t\tau_t\mathbf{a}_t$:

```math
\mathbf{s}_t = \left(\mathbf{I} - \beta^*_t\,\mathbf{a}_t\mathbf{a}_t^\top\right)\tilde{\mathbf{D}}_t\,\mathbf{s}_{t-1} + w_t\,\tau_t\,\mathbf{a}_t ,
```

with target $\tau_t=c_t\lVert\mathbf{v}_t\rVert^2$, or $\lVert\mathbf{v}_t\rVert^2$ without a target gate, and $w_t=\beta^{\max}_t$ under `decouple_erase`, else $\beta^*_t$. The transition is the same for all variants, so they share one chunkwise algorithm, and the only inner product it needs is $\mathbf{a}_i^\top\mathbf{a}_j=(\mathbf{k}_i^\top\mathbf{k}_j)(\mathbf{v}_i^\top\mathbf{v}_j)$. The $KV$-dimensional features are never formed.

## Forget gate

The forget gate acts on the key axis, the rows of $\mathbf{S}$ in FLA's layout. It is the only part of the recurrence with a free rank, and the rank of `g` selects it:

| `g`            | Forget gate                                                           |
| -------------- | --------------------------------------------------------------------- |
| `None`         | none                                                                  |
| `[B, T, H]`    | one rate per head, $\mathbf{S}\leftarrow e^{g_t}\mathbf{S}$, as in Gated DeltaNet |
| `[B, T, H, K]` | one rate per key dimension, $\mathbf{S}\leftarrow\mathrm{Diag}(e^{\mathbf{g}_t})\mathbf{S}$, as in KDA and GDN-2 |

`g` is the natural log of the forget gate, $\le 0$. Because the memory is addressed by keys, a per-key-dimension rate lets one stored pair fade while another persists. The score $y_t$ is a full contraction over both $K$ and $V$, so the write coefficient stays a scalar and the write stays rank one under either rank. In `naive_recurrent_kronlstm`, an all-equal rank-4 gate reproduces the rank-3 result exactly. The Triton ops broadcast a rank-3 gate to rank 4 and run one path.

`fla.layers.kronlstm.KronLSTM` builds the gate as in the paper: a sigmoid raised to a learned positive per-head power, following Mamba and GDN-2,

```math
\mathbf{f}_t = \sigma\!\left(-(\mathbf{s}_f)_t\right)^{a}, \qquad \mathbf{g}_t = \log\mathbf{f}_t = -a\,\mathrm{softplus}\!\left((\mathbf{s}_f)_t\right), \qquad a=\exp(a_{\log}),
```

with $a_{\log}$ the per-head `A_log`, $(\mathbf{s}_f)_t$ the output of the low-rank projection `f_proj` plus the per-channel bias `dt_bias`, and the exponent rescaling each head's retention horizon by $1/a$.

## Chunkwise form

This is the paper's chunkwise derivation (Appendix A8), transposed to FLA's layout. Take a chunk of length $C$ that starts from $\mathbf{S}_0$, and let $`\boldsymbol{\gamma}_t=\exp(\sum_{i\le t}\mathbf{g}_i)\in\mathbb{R}^K`$ be the cumulative decay. Dividing the decay out of the memory, the score of step $t$ is

```math
y_t = p_t + \sum_{j\lt t} u_j\,A^{kk}_{tj}A^{vv}_{tj}, \qquad p_t=(\boldsymbol{\gamma}_t\odot\mathbf{k}_t)^\top\mathbf{S}_0\,\mathbf{v}_t ,
```

where $p_t$ is the score of the pair on the incoming memory. The two Grams are a decayed key Gram and a plain value Gram:

```math
A^{kk}_{tj}=\sum_d k_{td}\,k_{jd}\,\frac{\gamma_{td}}{\gamma_{jd}}, \qquad A^{vv}_{tj}=\mathbf{v}_t^\top\mathbf{v}_j .
```

By the mixed-product property, $\mathbf{A}^{kk}\odot\mathbf{A}^{vv}$ is the Gram matrix of the decay-weighted Kronecker features. The value side carries no decay because the gate acts on the key axis. Substituting the write coefficient gives a unit lower-triangular system for the vector $\mathbf{u}$ of write coefficients of the chunk,

```math
\left(\mathbf{I}+\mathrm{Diag}(\boldsymbol{\beta}^*)\,\mathrm{strict}\!\left(\mathbf{A}^{kk}\odot\mathbf{A}^{vv}\right)\right)\mathbf{u}=\mathbf{w}\odot\boldsymbol{\tau}-\boldsymbol{\beta}^*\odot\mathbf{p},
```

and it is invertible for every gate value and every input, including under `allow_neg_eigval`. The matrix is the same for every variant; the variants only move the right-hand side. Write $\mathbf{T}$ for its strictly lower part. Then $\mathbf{u}=(\mathbf{I}+\mathbf{T})^{-1}(\mathbf{w}\odot\boldsymbol{\tau}-\boldsymbol{\beta}^*\odot\mathbf{p})$, and only $\mathbf{p}$ depends on $\mathbf{S}_0$. The output and the next memory are

```math
\begin{aligned}
\mathbf{o}_t &= \text{scale}\left[(\boldsymbol{\gamma}_t\odot\mathbf{q}_t)^\top\mathbf{S}_0+\sum_{j\le t}u_j\,A^{qk}_{tj}\,\mathbf{v}_j\right], \qquad A^{qk}_{tj}=\sum_d q_{td}\,k_{jd}\,\frac{\gamma_{td}}{\gamma_{jd}} \\
\mathbf{S}_C &= \mathrm{Diag}(\boldsymbol{\gamma}_C)\,\mathbf{S}_0+\sum_t u_t\left(\frac{\boldsymbol{\gamma}_C}{\boldsymbol{\gamma}_t}\odot\mathbf{k}_t\right)\mathbf{v}_t^\top .
\end{aligned}
```

The pseudo-value of token $t$ is the value $\mathbf{v}_t$ scaled by the scalar $u_t$. The code names follow the paper: `k_up` $=\boldsymbol{\gamma}\odot\mathbf{K}$, `k_tail` $=(\boldsymbol{\gamma}_C/\boldsymbol{\gamma})\odot\mathbf{K}$, `q_up` $=\boldsymbol{\gamma}\odot\mathbf{Q}$, `A_kk`, `A_vv`, `A_qk` for the Grams and `Tinv` for $(\mathbf{I}+\mathbf{T})^{-1}$.

The one structural difference from KDA and GDN-2 is $p_t$. The incoming memory enters the solve through $C$ scalars instead of $C$ read-outs: their scans read $\mathbf{W}\mathbf{S}_0\in\mathbb{R}^{C\times V}$, whose columns are independent, while $p_t$ sums over all of $V$. A scan that tiles $V$ across programs therefore has to exchange $\mathbf{p}$ once per chunk. The extra cost over KDA is the value Gram $\mathbf{A}^{vv}$, and $\mathbf{p}$ replaces $\mathbf{W}\mathbf{S}_0$ at equal cost.

## Kernels

All kernels work in log2: the natural-log gate is cumulated once with `scale=1/ln 2` and every kernel uses `exp2`, as GDN-2 and KDA do. Exponents are referenced to sub-chunk boundaries the way GLA does it, $2^{g_t-g_s}=2^{g_t-g_n}\,2^{g_n-g_s}$ with both factors $\le 1$, so nothing overflows at any gate magnitude.

Forward (`chunk_fwd.py`):

| Stage                       | Kernel                                  | Produces                                                                  |
| --------------------------- | --------------------------------------- | ------------------------------------------------------------------------- |
| gate cumsum                 | `chunk_local_cumsum` (`fla.ops.utils`)  | log2 cumulative gate                                                      |
| diagonal Gram blocks        | `chunk_kronlstm_fwd_kernel_intra_diag`  | the $16\times16$ diagonal blocks of $\mathbf{A}^{qk}$, $\mathbf{A}^{kk}$  |
| off-diagonal blocks, solve  | `chunk_kronlstm_fwd_kernel_inter`       | the other blocks, $\boldsymbol{\beta}^*$, $\mathbf{w}\odot\boldsymbol{\tau}$, $(\mathbf{I}+\mathbf{T})^{-1}$, the decay-scaled keys |
| scan over chunks            | `chunk_kronlstm_fwd_kernel_h`           | chunk states $\mathbf{h}$, $\mathbf{u}\odot\mathbf{V}$, $\mathbf{u}$      |
| read-out                    | `chunk_gla_fwd_o_gk` (`fla.ops.gla`)    | $\mathbf{o}$                                                              |

The inverse is built in registers: forward substitution on the four diagonal $16\times16$ blocks, then the block formulas with fp32 `ieee` dots, the two steps of `solve_tril` in one kernel. The solve is the precision-sensitive step; the other products run on tensor cores. The scan holds the whole $[K, V]$ state of one (sequence, head) in registers, as up to four 64-row key blocks, so $K, V\le 256$. Unlike GDN-2's scan it does not tile $V$ across programs: the score $\mathbf{p}$ sums over all of $V$, so a $V$-tile would need an exchange between programs at every chunk.

Variable-length inputs follow FLA's convention: $N$ sequences packed along $T$ with batch size 1 and the offsets `cu_seqlens`. The two scans take one program per (sequence, head) and read their range from `cu_seqlens`, with the chunk states of all sequences in one `[1, NT, H, K, V]` tensor at the offsets `chunk_offsets`; every per-chunk kernel reads its (sequence, chunk) from `chunk_indices`, as the shared GLA and KDA kernels do; the token-parallel kernel finds a token's sequence by binary search. A sequence's last chunk is masked rather than padded, so the dense path, which pads $T$ to a multiple of the chunk size, and the packed path run the same kernels.

Backward (`chunk_bwd.py`):

| Stage                     | Kernel                                    | Produces                                                         |
| ------------------------- | ----------------------------------------- | ---------------------------------------------------------------- |
| read-out Gram cotangent   | `chunk_kda_bwd_dAv` (`fla.ops.kda`)       | $d\mathbf{A}^{qk}$, the intra-chunk part of $d(\mathbf{u}\odot\mathbf{V})$ |
| reverse scan              | `chunk_kronlstm_bwd_kernel_dhu`           | $d\mathbf{h}$, $d\mathbf{S}_0$, $d(\mathbf{u}\odot\mathbf{V})$, $(\mathbf{I}+\mathbf{T})^{-\top}d\mathbf{u}$ |
| key Gram cotangent        | `chunk_kronlstm_bwd_kernel_dAkk`          | $d\mathbf{A}^{kk}$, $\mathbf{A}^{vv}$                            |
| Gram cotangents into q, k | `chunk_kronlstm_bwd_kernel_intra`         | both Grams' gradients into $\mathbf{q}$, $\mathbf{k}$ and the gate |
| chunk-local gradients     | `chunk_kronlstm_bwd_kernel_local`         | final $d\mathbf{q}$, $d\mathbf{k}$, $d\mathbf{v}$, $d\beta$, $dc$ and the per-row gate gradient |
| gate                      | `chunk_local_cumsum(reverse=True)`        | $d\mathbf{g}$                                                    |

The reverse scan mirrors the forward's coupling: the value cotangent becomes one scalar per row, $`du_t=d(u_t\mathbf{v}_t)^\top\mathbf{v}_t`$, which goes back through the solve and corrects the state gradient by $`\mathbf{K}_{\uparrow}^\top\left((-\boldsymbol{\beta}^*\odot d\mathbf{r})\odot\mathbf{V}\right)`$, where KDA's scan subtracts $\mathbf{W}^\top d\mathbf{V}$. Two points differ from the delta-rule backward, as the paper notes (Appendix A8.8). The values enter the system matrix through $\mathbf{A}^{vv}$, so $d\mathbf{v}$ also receives a contribution through the inverse. And $`\mathrm{Diag}(\boldsymbol{\beta}^*)`$ sits inside the Hadamard product, so it stays inside the accumulation of the system matrix's cotangent. Gate gradients are assembled in natural-log units, since $2^{g/\ln 2}=e^{g}$: each contribution is a decayed operand times its cotangent, the chunk-end terms sit on the chunk's last row, and one reverse cumsum turns them into $d\mathbf{g}$. The forward saves its intermediates (both Grams, the inverse, the chunk states, $\mathbf{u}\odot\mathbf{V}$, the decay-scaled keys) instead of recomputing them in the backward.

## Numerics

- The recurrent state is fp32 in every kernel and in the returned final state. The chunk states $\mathbf{h}$ and $d\mathbf{h}$ are stored in the input dtype, as GDN-2 stores them.
- With fp32 inputs, `tl.dot` runs in tf32 unless `TRITON_F32_DEFAULT=ieee` is set; the outputs then differ from the float64 reference by about $10^{-3}$. bf16 inputs are unaffected.

## Limitations

- Variable-length inputs (`cu_seqlens`) need the sequences packed along $T$ with batch size 1, as in the other FLA ops.
- No context parallelism and no grouped value attention.
- The chunk length of `chunk_kronlstm` is fixed at 64.
- $K, V\le 256$, since every scan holds the whole $[K, V]$ state of a (sequence, head) in one program; `chunk_kronlstm` also needs $K, V\ge 16$ for its `tl.dot`s.
- With fp32 inputs, `chunk_kronlstm` needs $K\le 128$ or $V\le 128$: the scan keeps its state in registers but stages each 64-row block in shared memory as a `tl.dot` operand, in the inputs' dtype, and in fp32 the state of $K=V=256$ takes 320 KB. The layer hands the op q, k and v in the projections' dtype, so under bf16 autocast it runs the bf16 kernels.
- `fused_recurrent_kronlstm` is forward-only; training uses `chunk_kronlstm`.

[^1]: A.-R. Hartl, L. Zólyomi, M. List, G. Klambauer, S. Böck, S. Hochreiter. *Kronecker Delta Rule for Long Context Linear Attention*, 2026. https://arxiv.org/abs/2610.26834.
