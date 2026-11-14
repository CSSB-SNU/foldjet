# Requests to the miniworld-engine team (2.5.0 - 2.5.6)

Requests only: what FoldForge would like the engine to provide. What the engine does not support and what FoldForge does meanwhile is in
[ENGINE-UNSUPPORTED.md](ENGINE-UNSUPPORTED.md); each request names the rows there it would retire (`U-<section>`).

Priority: **P1** blocks a clean integration or costs most of a fold's speed on the main path (H100, bf16); **P2** removes a FoldForge workaround
or a per-family slow path; **P3** convenience, documentation, or a wish for exactness in a corner.

Numbering is stable (R1 ... R27); a request that is done is struck out here, not renumbered. Earlier lists were merged: the former items 1 / 5
are R1, 4 / 8 are R2, 3 / F4 are R6, 6 is R5, 9 / D2 are R4, 7 / 11 are R10, D1 is R7, F1 is R8, F2 is R9, F3 is R11.

| # | Priority | Request | Retires |
| --- | --- | --- | --- |
| ~~R1~~ | – | ~~Public CUDA-graph capture and weight-cache API~~ Done: `miniworld_engine.capture` (2.5.6), used by FoldForge `modules/engine/capture.py` | U-removed (capture), U-layout (caches) |
| ~~R2~~ | – | ~~Kernel-served `OuterProductMean` with `count_offset` (AF3's `n + 1e-3` divisor)~~ Done (engine kernels serve it; FoldForge defaults to it since 2026-11-14) | U-backend (OPM, closed) |
| ~~R3~~ | – | ~~`AttentionPairBias` `cuda` / `triton` with an additive `extra_pair_bias`~~ Fixed (served on `cuda` in 2.5.4, eval only) | U-backend (single attention, closed) |
| R4 | P2 | `protenix_v2` key (or documented alias) on `TriangleAttention` / `TriangleMultiplication` | U-class (protenix2 keys) |
| R5 | P2 | Separate model keys for the triangle attention and the triangle multiplication of a `PairformerBlock` | U-layout (RoseTTAFold3 fast) |
| ~~R6~~ | – | ~~ESMFold2 atom block with the atom pair (and rotary keyed by `uid`), or a documented non-equivalence~~ Done: `AtomDiT("esmfold2_pair")` (2.5.3), used by FoldForge `b36209c` / `fcddf56` | U-class (ESMFold2 atom stack, closed) |
| ~~R7~~ | – | ~~Chai-1 `TokenDiTChai1` / `AtomDiTChai1` on `cuda` with FoldForge's mixed dtypes (bf16 `single`, fp32 `cond` / `pair`)~~ Fixed (served on `cuda` in 2.5.4) | U-backend (Chai-1 DiT, closed) |
| R8 | P2 | fp32 `TriangleMultiplication` / `TriangleAttention` / `Transition` kernels (`precision: fp32` runs them on `pytorch`) | U-backend (fp32) |
| R9 | P3 | Chai-1 `circular` atom key window, or a documented equivalence of `pad` + `uid` | U-backend (Chai-1 key window) |
| R10 | P3 | Stack classes for the DiT blocks (`TokenDiTStack` / `AtomDiTStack`) and a `from_shared_pair(...)` helper for the released shared pair norm | U-layout (DiT stacks, shared pair norm) |
| R11 | P3 | A Chai-1 MSA block (`outer_product_mean(groups=8)` into parallel MSA / pair stages) and a general MSA block with the AF3 / Boltz orders | U-class (MSA block) |
| R12 | P3 | `parallel_residual`: narrowed, still open. The engine has `ParallelResidual` (`cuda`, `triton`; at most 4 branches); FoldForge's MSA-stack sum (2-3 branches) can use it, the pairformer's 5 cannot (R22) | U-removed (parallel_residual) |
| R13 | P3 | An option on `AtomDiTOpenFold3` for AF3's mask form under 128 real atoms | U-equivalence (atom window < 128 atoms) |
| R14 | P3 | Class-docstring invariants for the shared-KV atom AdaLN and RoseTTAFold3's zero query bias (or a `query_bias` constructor option) | U-layout (shared KV, zero query bias) |
| ~~R15~~ | – | ~~Export the dead-module loader~~ Closed: the engine declined; solved in FoldForge (`modules/dense/dead.py`) | U-removed (dead-module loader) |
| ~~R16~~ | – | ~~`AtomDiT` blocks send a gradient to the atom-pair input~~ Withdrawn 2026-11-12: not an engine bug (FoldForge training policy, see the paragraph) | U-backend (training: atom-pair conditioning) |
| ~~R17~~ | – | ~~An autograd-capable `PoolAtoms` (or a backward for the `sampler_ops` module)~~ Done: engine `PoolAtoms` backward (2.5.3); FoldForge training uses it since 2026-11-14 (`foldforge_sampler_glue` off, pooling on; 64 of 68 calls per bench served by the engine) | U-backend (training: pooling, closed) |
| R18 | P2 | A public multi-fold (trunk_seed x diffusion_seed) contract for the DiT blocks | U-class (grouped 5x5 sampler) |
| R19 | P2 | Smaller temporaries for the 2.5.0 `PairformerBlock` batched over samples (the confidence head's CUDA-graph pool) | U-class (confidence-head graph memory) |
| ~~R20~~ | – | ~~2.5.1 sm90 atom-DiT backward `CUDA_ERROR_ILLEGAL_ADDRESS`~~ Fixed in 2.5.3 (a Protenix v1 training smoke on `cuda` without the `pytorch` workaround runs) | U-backend (training: atom DiT cuda backward) |
| ~~R21~~ | – | ~~`MSAPairWeightedAveraging` `cuda` for `d_pair` 256 (Chai-1) past N = 1024, and the memory it then needs~~ Fixed (served at N = 1536 / 2048 in 2.5.4; the memory was not re-measured) | U-class (PWA wide pair, closed) |
| R22 | P2 | The engine `PairformerBlock._pair_updates` sums its parallel branches in ATen (`sub` / `add`): a fused path for 5 branches (ParallelResidual caps at 4) | U-removed (parallel_residual) |
| R23 | P3 | Engine-internal ATen glue at n = 1 (`inverse_length`, the fp32 -> bf16 `copy_` of `trimul_h100`) | – |
| R24 | P2 | An Euler update op without Kabsch alignment (the `AlignEuler` module always aligns) | U-backend (glue modules) |
| R25 | P3 | RoseTTAFold3 chirality-gradient op (`modules/dense/chirality.py`, FoldForge-own) | U-class (chirality features) |
| ~~R26~~ | – | ~~A contract to refresh the hoisted per-fold tables (token-DiT pair bias, atom-block conditioning) IN PLACE into the same buffers, so a sampler CUDA graph can be reused across folds~~ Done: `refresh_static_inputs` / `StaticTables` (2.5.5), used by `models/sampler_graph.py` | – |
| ~~R27~~ | – | ~~The H100 token-DiT cuda path at small L: the stack-level conditioning hoist of 2.4.4 is gone (ESMFold2 and Boltz-2 at d_cond 768: 11 kernels per block instead of about 7; RoseTTAFold3 and Chai-1 smaller)~~ Fixed in 2.5.6 (checked 2026-11-14): ESMFold2 5I28 warm n = 1 / 2 / 4 = 0.245 / 0.433 / 1.021 s on 2.5.6 (2.5.4: 0.299 / 0.476 / 1.098, 2.4.4: 0.254 / 0.445 / 1.055; the 2.5.4 and 2.4.4 numbers are the 2026-11-12 record, not a same-day run) | – |

## P1

**R1. Public capture API.** FoldForge's CUDA-graph wrapper (`models/execution.py::_static_inputs`) enters
`miniworld_engine.kernels._capture.static_weights()` / `static_inputs()` around every capture (via `modules/engine/capture.py`), and `modules/engine/capture.py::
clear_weight_caches` empties `kernels._capture._FORMS`, `kernels._tf32._CACHE` and the `_PACKS*` dicts of `kernels.integrations.*`
before weights are replaced. All of that is private API, confined to `modules/engine/capture.py`. Please export `static_weights`, `static_inputs` and `clear_weight_caches` from the
package root with the same semantics (a model that builds a second set of weights, or a test that builds many models, needs the clear).

**R2. DONE: OPM `count_offset` on kernels.** AF3's divisor is `n + 1e-3`; the `cuda` / `triton` backends serve only `clamp(n, 1)` and refuse a call that
sets `count_offset`, so AF3-family rows choose between the kernels (clamp divisor, a relative difference of 1e-3 / n) and exactness (`pytorch`).
FoldForge builds `count_offset=None` by default and runs the exact form on `pytorch` only where the mode asks for it
(`modules.OPM_COUNT_OFFSET`). A kernel with the offset removes the choice.

**R16. WITHDRAWN (2026-11-12): not an engine bug.** The 8 atom-pair parameters without gradient (`embed_pair_distances`, `embed_pair_offsets`, `single_to_pair_cond_row` / `_col` without `_1`) are unused by design in every model: AF3 discards that output; in the other models the converter copies one original weight to both slots and only the `_1` slot is used. The same per-parameter gradient state appears with the atom blocks on `cuda`, on `pytorch`, and with every op on `pytorch` (Protenix v1 4,139 parameters, Chai-1 3,817). The other groups of the 17: `embed_ref_charge` and `bond_embedding` are data dependent (gradient flows on 7fvu, 26 charged atoms and 41 ligand bonds; 8ayg has neither); `msa_stack.3` does not exist in the original; `experimentally_resolved_*` is missing from FoldForge's training loss; two pairformer groups hold checkpoint values of 1e-37..1e-12 whose gradients underflow (the same on pytorch). The fixes are FoldForge's (frozen-parameter policy, grad check, loss), see docs/guides/TRAINING.md. My first reading of this as an engine defect was wrong. (Original text:) In a training step of Protenix v1 the parameters that build the atom pair (the input embedder and diffusion head `embed_pair_distances`, `embed_pair_offsets`, `single_to_pair_cond_row` / `_col`: 8 tensors) receive no gradient at all: the engine `AtomDiT` blocks appear not to send one back through their pair-bias input. Observed, root cause not confirmed. Without it those parameters never train. Please check, and if it is missing, provide backward for the pair input of the `cuda` / `triton` / `pytorch` backends of `AtomDiT*`.

## P2

**R3. FIXED in 2.5.4 (checked 2026-11-12). `AttentionPairBias` with `extra_pair_bias`.** OpenDDE's structural-token refiner adds an `[L, L]` bias to every head of the single
attention. Engine 2.5.0 / 2.5.1 refused it on `cuda`, so those calls (4 per fold) ran `pytorch`. In 2.5.4 `AttentionPairBias._supports_cuda` accepts it (the planes are
added to the hoisted bias; eval only, with a gradient the call is refused). Audit on H100 (OpenDDE, eager `engine_dispatch`, n = 1 / 8 / 12): all 216 `AttentionPairBias`
calls (56 modules) on `cuda`, no refusal.

**R4. `protenix_v2` key.** `TokenDiT` / `AtomDiT` have `protenix_v2`; `TriangleAttention` / `TriangleMultiplication` do not, so protenix2 builds
them with `protenix` (same math: transposed ending bias). A documented alias lets the family key map to one name.

**R5. Separate triangle keys.** A `PairformerBlock` has one `model` key for both the triangle attention and the multiplication. RoseTTAFold3's
fast mode wants RF3 attention with the plain multiplication, so FoldForge replaces the block's two multiplications after building it
(`modules/engine/pairformer.py`). A second key (for example `trimul_model=`) removes the replaced sub-modules.

**R6. DONE (engine 2.5.3, used by FoldForge 2026-11-12). ESMFold2 atom stack with the pair.** The engine now has `AtomDiT("esmfold2_pair")` (`AtomDiTESMFold2Pair`: the AF3 block over the
`[windows, 32, 192, d_pair]` atom pair with the family's rotary and RMS / SiLU AdaLN; `backends = pytorch / triton / cuda`). FoldForge runs the ESMFold2 atom
stacks on it since commit `b36209c` (`fcddf56` adds `key_dim_per_head`), keeping the own stack as the fallback on engines without the class. Audit on H100 (engine 2.5.4,
n = 1 / 8 / 12 / 16): 9 blocks served on `cuda`, no refusal. Measured gain (5I28, ESMFold2, warm forward, H100, 2026-11-12): n = 1 0.405 -> 0.299 s, n = 8 4.096 -> 3.404 s
(kit 0.311 / 3.919 s). Before: ESMFold2's sampler took 0.514 s against Protenix v1's 0.429 s at n = 1 although its token DiT has half the blocks (12 against 24);
the engine's fused atom kernels (`pre90_split`, `post90_fwd`, `h_attn_fwd`) were absent from its profile (`docs/records/5i28_foldforge_20261111.md`).

**R7. FIXED in 2.5.4 (checked 2026-11-12). Chai-1 DiT on `cuda`.** FoldForge hands the token block bf16 `single` and fp32 `cond` / `pair` (the conditioning is built in fp32).
Engine 2.5.1 refused that in the Chai-1 classes ("all bfloat16 or all float32"; the atom block also under the fp32 sampler glue), so the 64 token-DiT and 27 atom-DiT
calls per fold ran `pytorch`. In 2.5.4 both classes (`MasterEntryCast`) are served on `cuda`: audit on H100 (eager `engine_dispatch`, n = 8 / 12 / 16, fp32 conditioning
inputs): `TokenDiTChai1` 16 modules / 48 calls, `AtomDiTChai1` 9 modules / 21 calls, no refusal. The engine accepts the fp32 inputs; a FoldForge-side dtype change for them
is separate work and not a refusal.

**R8. fp32 pair-stack kernels.** An fp32-activation (`precision: fp32`) fold of the Protenix lineage runs `TriangleMultiplication`,
`TriangleAttention` and `Transition` on `pytorch` (the kernels are bf16-only). Documented behaviour; listed because every fp32 reference run is
mostly `pytorch`.

**R15. Dead-module loader (closed: the engine declined it; FoldForge owns it).** Why it matters: some checkpoints carry residual modules whose every parameter is ~1e-37 (Protenix v1: 76 modules, the confidence head's 8 triangle attentions, 2 triangle multiplications and pair transition, 34 attention-pair-bias and 31 transition modules of the trunk). Their update is exactly zero, so skipping them leaves the output bit-identical and saves their time (the Protenix kits skip them). 2.4.x had a loader that flagged them; 2.5.x keeps only the plain `dead` flag on `Transition`, `AttentionPairBias`, `TriangleAttention`, `TriangleMultiplication`, `OuterProductMean` and `MSAPairWeightedAveraging` (flagged: return the residual input without grad outside training). Where it is solved: FoldForge, `modules/dense/dead.py::mark_dead` (all parameters below 1e-30, called once the weights are final; `engine.skip_dead_modules: false` for training). The remaining risk is the engine renaming or changing the `dead` flag, which would stop the skipping silently (output unchanged, only slower): `tests/models/test_dead_modules.py` now pins that the six engine classes declare `dead` (default False) and that a flagged `Transition` returns its input.

**R17. `PoolAtoms` backward (training).** The `sampler_ops` modules (`AugmentNoise`, `AlignEuler`, `PoolAtoms`) are inference-only by design (no autograd), so a training step cannot use `PoolAtoms` and FoldForge switches every module to the torch pooling (`foldforge_sampler_ops=False`, `train/prepare.py`). The same op (gather + relu + masked mean over the atoms of a token) with a backward would remove the switch and the slower torch gather in the training graph (`indexing_backward_kernel` is the largest PyTorch backward kernel of a step).

**R18. Multi-fold sampling contract.** FoldForge's trunk_seed x diffusion_seed sampler (`models/grouped.py`) runs the DiT blocks once over `[A, NT, ...]` inputs where each of the NT folds has its own pair, conditioning and atom pair; the engine's grouped-bias attention reads sample s's pair from fold `s % nbg`. That behaviour is not documented, and FoldForge reaches it through `stack.blocks` of the DiT stacks. Please state the contract (for example a public `pair_groups` argument) so it can be relied on. Measured end to end on FoldBench (1,522 assemblies): 1.5x for assemblies up to 256 tokens, 1.08x over the whole set, 1.0x above 1,024 tokens.

**R19. Confidence-head graph memory.** With one CUDA-graph pool per captured graph, the 2.5.0 `PairformerBlock` batched over samples (`engine/graph.py::make_pairformer_block(..., batched="sample")`) needs 2.4x the pool of the 2.4.0 fused `AF3Pairformer` in the confidence head: summed over the captures of 70 ascending-size FoldBench assemblies, 26.9 GiB against 11.3 GiB (largest bucket 5.67 against 3.04 GiB). Together with the evoformer pools (about 37-41 GiB) the private pools reached 80 GiB at the ~400-token bucket and a capture failed (five assemblies; `release-graphs target` passed). FoldForge now evicts the callable's oldest graphs before a capture when memory is short (`models/execution.py`), but the size of the temporaries is the engine's: please check whether the batched block can use a fused path or state a workspace-size contract. The temporaries were estimated from pool sizes, not profiled.

**New candidates from the 2.5.4 per-model audits (2026-11-12, H100).** R22 - R25 come from torch profiles of the audited models; graph replays carry no python stacks, so some
attributions are inferred from eager runs or by kernel name (marked "inferred"). Treat the percentages as estimates.

**R22. Fused parallel update in `PairformerBlock`.** `PairformerBlock._pair_updates` (`modules/pairformer.py` ~line 293) sums the parallel branches' deltas with ATen `sub` / `add`
(`module(x) - x` per branch, then `+`). Chai-1 (bench, n = 8, 200 steps, wall 14.33 s): `aten::sub` 570 ms (3.98%) + `aten::add` 457 ms (3.19%) = 7.2% of the wall (trunk 896 ms,
confidence head 93 ms, template 38 ms; attributed from eager, inferred for the graph). `ParallelResidual` caps at `MAX_BRANCHES = 4`, the Chai-1 pair update has 5 branches (two
multiplications, two attentions, the transition). Please lift the cap to 5 (or fuse the sum inside the pairformer). Retires part of R12's scope.

**R24. Euler update without Kabsch alignment.** `AlignEuler` always aligns (Kabsch); the families with `realign_sampler=False` (protenix1 / protenix2, opendde, af3, openfold3,
boltz2, chai1, rosettafold3) therefore get no engine sampler glue for the Euler step and run FoldForge's torch update. From the profiles: 3-5% of the n = 1 time (inferred, by kernel name).
An op that takes the denoised coordinates and sigma step without the alignment would serve them.

**R20. Atom-DiT `cuda` backward (training), engine 2.5.1. FIXED in 2.5.3** (the training smoke runs without the `pytorch` workaround; the kernel comment names this request). Engine 2.5.0 refused a training call of `AtomDiTProtenix` on `cuda` ("the backward kernels know the chained K / V AdaLN and the clipped 'pad' window only") and the guard ran `pytorch`; 2.5.1 serves it, and on Protenix v1's blocks (128 tokens, 8ayg, batch of 8 noisy samples) the backward fails: `miniworld_engine/kernels/integrations/local_dit_h100.py:436 _bwd` -> `k90.finalize(...)` -> `cuLaunchKernelEx: CUDA_ERROR_ILLEGAL_ADDRESS` (grid (16, 27, 1), seen with `CUDA_LAUNCH_BLOCKING=1`; without it the same fault surfaces later as `CUBLAS_STATUS_EXECUTION_FAILED` in a bf16 GEMM). Please fix the kernel or refuse the call as 2.5.0 did. Also: the block's output is a view created inside a custom autograd Function, so an in-place write to it fails when a gradient is recorded (`Output 0 of View is a view and is being modified inplace`); FoldForge now writes out of place when grad is enabled (`modules/dense/atom_cross_attention.py::_mask_atoms`), but returning a fresh tensor would avoid the trap for other callers. FoldForge's training script runs `atom_blocks` on `pytorch` meanwhile (`scripts/train_protenix1.py`).

**R21. FIXED in 2.5.4 (checked 2026-11-12): served at N = 1536 / 2048.** The H100 Chai-1 audit (eager `engine_dispatch`, n = 12 and 16) shows `MSAPairWeightedAveraging` (4 modules, 16 calls) on `cuda`
with no refusal; the engine docstring now lists `pwa_wide` at N <= 2048 for inference. The memory this path needs (77 GiB at n = 16) was NOT re-measured. Original text: On the engine's `redesign` branch (a01c952e1) the Chai-1 5I28 run still logs
"MSAPairWeightedAveraging backend cuda refuses a call: d_pair 256: the pair side is pwa_wide's sm_90a kernels (N <= 1024), got N=2048 on sm_90;
running its pytorch backend": the 1536 / 2048 serving covers `d_pair` 128 only. Chai-1 n = 16 (2048 tokens) is the one cell that then needs 77 GiB
of the 80 (it runs alone in a fresh process, 70.6 s against the kit's 89.7 s, and runs out of memory behind the smaller sizes of one process), so
a `cuda` path for `d_pair` 256 is the difference between a margin and none. Measured 2026-11-11 with `benchmarks/5i28_ff/`.

## P3

**R9. Chai-1 `circular` window.** `AtomDiTChai1(key_window="pad")` + `uid` equals the old exact mode unless a conformer spans both ends of the atom
list (or the system has fewer than 128 atoms). If the release's `get_qkv_indices_for_blocks` really wraps, an engine `circular` window makes
FoldForge's `exact` mode exact again.

**R10. DiT stacks and the shared pair norm.** FoldForge assembles `ModuleList`s of `TokenDiT*` / `AtomDiT*` blocks itself
(`modules/engine/transformers.py`, with the `extra_pair_bias` / `uid` plumbing), and converts the released shared pair LayerNorm / projection
once, offline, into per-block copies (`models/checkpoints/legacy/released.py`). Reference stack classes and a `from_shared_pair(...)` helper would let
every caller of a released AF3-family checkpoint use one engine implementation.

**R11. MSA block.** The engine has no MSA block; FoldForge composes `modules/engine/graph.py::EvoformerBlock` from engine parts, including
Chai-1's parallel schedule (each part owns its residual, so a stage's update is the sum of the parts' deltas `part(x) - x`).

**R12. `parallel_residual` (narrowed, still open).** `EvoformerBlock._forward_parallel` (Chai-1's MSA block) sums parallel branches in torch (`x + sum_i (y_i - x)`) because the
parts own their residuals. The 2.5.0 modules had no fused equivalent; engine 2.5.4 has `ParallelResidual` (`pytorch`, `cuda`, `triton`; `MAX_BRANCHES = 4`). FoldForge's
MSA-stack sum has 2 branches (MSA attention + MSA transition) or 3 (two triangle multiplications + pair transition; then 2 triangle attentions), so it could use the module:
FoldForge has not switched yet. The pairformer's pair update has 5 branches (two multiplications, two attentions, the transition) and cannot (R22). The pairformer blocks use the engine's own `parallel=True`.

**R13. AF3 mask under 128 atoms.** `AtomDiT*` switch every masked key off; AF3's own form (`+1e9` only where both query and key are padding) lets a
real query read the padded keys of its slid window below 128 real atoms. Identical from 128 atoms (tested at 130 and 200).

**R23. ATen glue inside engine modules at n = 1.** Examples: `modules/functional.py` ~365 `inverse_length` (`copy_`, `sum`, `reciprocal` per triangle multiplication) and the fp32 -> bf16
`copy_` in `kernels/integrations/trimul_h100.py` ~709 (`_update_inference_body`): about 3% of the RoseTTAFold3 n = 1 time (`~/foldforge-scratch/cuda254/rosettafold3/classified.txt`, eager
attribution; Chai-1 n = 1: `trimul_h100.py:709` 3.47 ms = 0.52% of a 662 ms wall). Small per call, but at n = 1 they are not hidden by the kernels.

**R25. RoseTTAFold3 chirality gradient.** `modules/dense/chirality.py` (`dihedral_error_gradient`, FoldForge-own, run per sampler step in the atom encoder) is 3.4% of the eager n = 8
time and 14.6% of the eager n = 1 time (eager profile; the compiled / graph runs show it as small `bmm` / `gemv` kernels, `chirality.py:53`). A fused op for it would be an engine
family of its own: a wish, not a blocker.

**R14. Documented invariants.** `AtomDiTBoltz2` / `AtomDiTRoseTTAFold3` / `AtomDiTChai1` share one AdaLN for q and k / v; FoldForge checks the
released `k_adaptive_layernorm` equals the query one before discarding it. RoseTTAFold3's `FlatQKNormAugmentedAttention` has no query bias; FoldForge
drops the released zero bias after checking it. If the releases are documented to do this, say so in the class docstrings so the checks are
confirmations, not inferences.

**R26. In-place refresh of the hoisted per-fold tables (sampler graph reuse).** FoldForge captures the sampler step as a CUDA graph (`models/sampler_graph.py`, 2 eager warm-up steps + `torch.cuda.graph`) once per fold, i.e. in every forward: Protenix v1 on an H100 measures `sampler_graph_captures` = 4 for 4 forwards, 73-79 ms of capture + warm-up inside the warm forward at n = 1 (about 15% of 0.485 s) and 94-97 ms at n = 8 (about 2.4% of 3.9 s); the coordinates of every forward are identical. The graph bakes in the addresses of the per-fold tensors (`batch`, `embeddings`, `single_base`, `pair_cond`, `key_mask`, `atom_state`), and the engine's own tables hoisted during the warm-up (the token DiT pair bias, the atom blocks' conditioning) are cached by tensor OBJECT identity; `miniworld_engine.capture` exposes `static_inputs`, `static_weights` and `clear_weight_caches` only, nothing that writes a new fold's tables into the existing buffers. A graph reused across folds without that would read the first fold's conditioning and pair bias (a wrong result), which is why FoldForge re-captures. Please provide a contract (for example `refresh_static_inputs(module, ...)`, or hoisted tables that live in caller-owned buffers) so the graph can be captured once per shape. FoldForge would then keep the per-fold conditioning in static buffers and cache the graph per shape. Measured by a probe of the FoldForge sampler (not an engine profile); the bound on the gain is the capture + warm-up cost above, the per-fold table computation stays.

**R27. Token-DiT cuda path at small L: kernels per block and the lost conditioning hoist (ESMFold2, Boltz-2, RoseTTAFold3, Chai-1).** ESMFold2's 5I28 time at n = 1..4 is 0.034-0.045 s slower on engine 2.5.4 than on 2.4.4 with the same FoldForge-side inputs. A one-to-one run on the same H100 node (engine 2.4.4 + FoldForge 491f4f0 against engine 2.5.4 + FoldForge main, old / new alternated, 5 repeats) reproduces it (n = 1 0.254 vs 0.299 s, n = 2 0.445 vs 0.476, n = 4 1.055 vs 1.098; the 10-29 record's 0.254 / 0.443 / 1.052 is reproduced to 0.002 s). The profile attributes it to the sampler's token DiT (12 blocks x 202 calls per fold; d_single / d_cond 768, d_pair 256, 16 heads, S = 5): 2.4.4 ran `DiTVariantStack` (`nvjet 128x128` + two `res_kernel` + `glu_kernel`, about 42 us per block, block period at n = 1 / 4 / 8 of 44.9 / 125.6 / 242.5 us), 2.5.4 runs `TokenDiT("esmfold2")` = `TokenDiTProtenix` fused cuda (`gemm_swiglu_sm90`, `resgate_adaln_rows`, `adaln_in_rows`, `resgate_out_rows`, `cond_rows<768>` and two bias GEMMs `nvjet_72x64` / `40x64`: 11 kernels, about 55 us, period 60.2 / 135.1 / 249.8 us). Engine-only micro-benchmark of 12 blocks at L = 128: 596 us / 102 kernels (2.4.4) against 794 us / 132 kernels (2.5.4). The sampler step period at n = 1 grows 947 -> 1114 us (+33 ms over 200 steps; the whole n = 1 gap is 0.043 s) and the captured step holds 454 kernels instead of 332 (more capture time). 2.4.4's `DiTBlock(MINIWORLD)` refused the fused path at d_cond 768 (it required `cond.shape[-1] == 384`), which is why the old path was the variant stack. The gap shrinks with n (step period +88 us at n = 4, +9 us at n = 8: the GEMMs dominate). Measurements: `~/foldforge-scratch/old_vs_new/` (profile-based, one node; the 2.4.0 candidate of the 10-29 record was not run, 2.4.4 was). **Engine side: being fixed in the existing H100 token-DiT cuda family (fewer launches at d_cond 768 and small L; target = no slower than 2.4.4 per block); it also covers Boltz-2 (d_cond 768). The engine will say which version lands it** (message from the engine session, 2026-11-12). FoldForge side, separate and still ours: the trunk's fp32 -> bf16 `bfloat16_copy` (2,915 launches, about 4-5 ms at n = 1) and the confidence head's fp32 LayerNorm / add.

R27 addendum (engine-side module audit of the diffusion path, engine 2.4.4 against main on an H100, 2026-11-12, from the engine session; report `~/miniworld-engine-scratch/audit24/diffusion/report.md`): the cause is the token-DiT stack-level conditioning hoist of 2.4.4 that is gone. It shows at small L for boltz2 / esmfold2 and also for rosettafold3 (L128 1.16x, L256 1.07x) and chai1 (L128 1.11x); the protenix family ties 2.4.4 (so Protenix v2 / OpenDDE n = 1 gaps are not the token DiT). Three more engine-side findings of the same audit are being fixed there and matter to FoldForge: the MiniWorld bias-only token DiT copies the bias per call (up to 2.6x at L768; no FoldForge family uses it), `AttentionPairBias` with B = samples > 1 at L128 is slower (the confidence head's single attention: S = 8 1.5x, S = 5 1.1x), and `PoolAtoms`' launch configuration costs 3-4%. No cuda -> pytorch fallback was found anywhere. The engine fix is released as one version; its number goes into the pin and here.
