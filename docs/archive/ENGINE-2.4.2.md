# foldjet-engine 2.4.2: what moved here

> Historical migration record for that engine release. The integration that superseded it is described in [BACKENDS.md](../guides/BACKENDS.md), [../ENGINE-UNSUPPORTED.md](../ENGINE-UNSUPPORTED.md) and [CHECKPOINT-FORMAT.md](../guides/CHECKPOINT-FORMAT.md).

foldjet-engine 2.4.2 removed `foldjet_engine.architecture` and `foldjet_engine.ops`
(engine `docs/CHANGELOG.md`, 2.4.2 "Removed"). FoldJet pins `foldjet-engine==2.4.2`.

| was | now |
|---|---|
| `architecture.diffusion.*` (base, edm, decoupled_xpred, augmentation, schedules, samplers) | `foldjet.diffusion` (same layout) |
| `architecture.utils.align` / `se3` / `transform` | `foldjet.diffusion.geometry.alignment` / `se3` / `kabsch` |
| `architecture.tensor_layout` | `foldjet.diffusion.tensor_layout` |
| `architecture.typing.typecheck` | `foldjet_engine.utils.typecheck.typecheck` (inside `foldjet.diffusion`: its own `_typecheck`) |
| `architecture.exceptions` / `_engine_impl` (`ImplementationType`, `to_engine_impl`) | `foldjet.layers.implementation` |
| `architecture.blocks.pairformer` | `foldjet.blocks.pairformer` |
| `architecture.checkpoints.backend_attention` | `foldjet.checkpoints.backend_attention` |
| `architecture.primitives.LayerNorm` | `foldjet_engine.modules.primitives.LayerNorm` |
| `ops.layer_norm` / `ops.swiglu_ffn` / `ops.triangle_multiplicative_update` | `foldjet_engine.kernels.layer_norm` / `swiglu_ffn` / `triangle_multiplicative_update` |
| `ops.transition` | `foldjet_engine.kernels.transition_layer` |
| `ops.augmented_attention_pair_bias` | `foldjet_engine.kernels.augmented_attention.whole_op.augmented_attention_pair_bias` |
| `ops.layer_norm_linear` | `foldjet_engine.kernels.layernorm_linear.interface.layer_norm_linear` |
| `ops.gated_linear` | removed; `linear(sigmoid(gate) * value)` (no CUDA gate-row kernel any more) |
| `foldjet_engine.integrations.anthropic*` | `foldjet_engine.baselines.anthropic*` |
| `foldjet_engine.integrations.token_dit` | `foldjet_engine.kernels.integrations.token_dit` |

`foldjet.diffusion` imports torch, numpy, scipy, pydantic and jaxtyping only, so the folder can be moved into its own package.

## Update: engine modules only (branch `migrate/engine-modules`)

FoldJet no longer calls engine kernel functions for model computation. A dense module the engine has a module for gets that
module at install time (`modules/engine/modules.py`, now including `Transition` -> `foldjet_engine.modules.Transition`), and
`NativeLayerNorm` hands its BF16 calls to `foldjet_engine.modules.primitives.LayerNorm`. Everything else runs its PyTorch
statements: the rows above that map to `kernels.layer_norm`, `swiglu_ffn`, `triangle_multiplicative_update`, `transition_layer`,
`augmented_attention_pair_bias` and `layer_norm_linear` are gone, as is `foldjet.checkpoints` (`backend_attention`,
`gate_projection`). Remaining infrastructure imports from `foldjet_engine.kernels`: `_capture` (graph capture scopes),
`sampler_ops.interface` (sampler glue), `transition.cuda._weights` (weight re-layout cache clear),
`integrations.token_dit.serves_extra_bias`, `dit_variants.cuda.parallel_residual`.
