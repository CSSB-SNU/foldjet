# What foldjet-engine 2.5.6 does not support

> **Currency (marked 2026-11-25):** written against engine 2.5.0 - 2.5.6 (then named miniworld-engine; dated rows go to 2026-11-14). FoldJet now pins
> `foldjet-engine==1.0.0`, the renamed engine; this page has not been re-audited against it, so a row may already be closed.

For every gap: **what**, **where FoldJet needs it**, **what FoldJet does meanwhile**. Requests to the engine team that follow from a row are
in [ENGINE-REQUESTS.md](ENGINE-REQUESTS.md) (`R<n>`). The sections are the categories:

| section | meaning |
| --- | --- |
| U-removed | an engine API FoldJet used that 2.5.0 removed |
| U-class | no engine class for what a family computes: FoldJet-own module |
| U-backend | the class exists but a backend does not, or refuses the call: the engine `pytorch` backend runs it |
| U-layout | the engine computes the same thing in another layout: FoldJet adapts it (mostly in the offline converter) |
| U-equivalence | the engine's form differs in a stated corner from the release / the old FoldJet graph |
| U-precision | dtype policy |

The engine never falls back (a call outside a backend's range raises `BackendUnsupported`). FoldJet's policy (`modules/engine/policy.py`) asks
`module.unsupported(*args)` before every call of a non-`pytorch` module and, under `engine.unsupported: fallback` (default), runs that module's
engine `pytorch` backend for the call and logs the reason once; `error` raises. See [guides/BACKENDS.md](guides/BACKENDS.md).

## U-removed: engine API that is gone

| What | Where FoldJet needed it | Meanwhile |
| --- | --- | --- |
| `foldjet_engine.settings` (process-global policy) | `api.py` snapshot / restore around a fold | Removed. Only torch's matmul precision is restored (`modules/engine/configure.py::inference_settings`). |
| Autotune cache (`autotune`, cache root, `run_all`, shape keys) | `engine_cache_dir`, `tools/build_autotune_cache.py`, `bucketing.ATOM_SHAPES` | `engine_cache_dir` is accepted and ignored with one `DeprecationWarning`; the builder tool and its guide are deleted; `ATOM_SHAPES` is a FoldJet constant. Launch configs come from the engine's pretuned tables. |
| Public `static_weights` / `static_inputs` / `clear_weight_caches` | `models/execution.py` graph capture, `models/loading.py` | Private `kernels._capture` and the known weight-cache dicts (R1). |
| `mark_dead_modules` / `clear_dead_modules` (dead-module loader, 2.4.x) | `load_model` skip of dead modules | Not restored by the engine (R15 closed). FoldJet's `modules/dense/dead.py` flags the engine's plain `dead` attribute (76 modules on Protenix v1, bit-identical output); `tests/models/test_dead_modules.py` pins the engine side of that contract. |
| `kernels.sampler_ops.interface` | `models/sampler_graph.py`, `atom_cross_attention._pool_engine` | The modules `AugmentNoise`, `AlignEuler`, `PoolAtoms` (cuda on sm90, triton elsewhere). |
| `kernels.augmented_attention.whole_op` | `tools/audit_af3_batch.py` | Wraps `AugmentedAttentionPairBias.forward`. |
| `modules.dispatch`, `baselines` payload variables, `ImplementationType` | `modules/engine/backends.py` | Engine backend names (`pytorch | cuda | triton | anthropic | cueq`); `miniworld` and `cuequivariance` are FoldJet policy aliases (`backends.ALIASES`). |
| `from_released` / `modules/released.py` (twin builders) | the old dense twins | `models/checkpoints/legacy/released.py`, used only by the offline converter. |
| `runtime.parallel_residual`, `runtime.token_dit_serves_extra_bias`, `DiTVariantStack` | pairformer parallel blocks, DiT | `graph.EvoformerBlock._forward_parallel` sums the parallel MSA-stack parts in torch (the trunk, confidence and template blocks use the engine's own `parallel=True` pairformer, whose `_pair_updates` also sums in ATen: R22) (R12; the engine has `ParallelResidual` since 2.5.x, up to 4 branches, not used yet); `TokenDiT*.unsupported(...)`; per-block stacks (`transformers.TokenDiTStack`) (R10). |
| Dense twins (`engine_of`, `served_by`, `serve_token` / `serve_atom`) | the old attach-an-engine-module mechanism | Deleted with their shims (`routing.py`, `attention_ops.py`). FoldJet's graph holds the engine classes directly. |

## U-class: no engine class

| What | Where | Meanwhile |
| --- | --- | --- |
| ~~ESMFold2 atom stack WITH the atom pair~~ (`AtomDiTESMFold2` is the pair-free SWA block; the engine has `AtomDiT("esmfold2_pair")` = `AtomDiTESMFold2Pair`, `backends = pytorch / triton / cuda`, since 2.5.3) | `esmfold2` / `esmfold2-fast` atom encoder, decoder and diffusion atom blocks (AF3 block over a 32 x 192 atom pair + RMS / SiLU AdaLN, raw gates, rank window +-64, rotary) | **Closed 2026-11-12 (engine 2.5.4):** the atom stacks run on the engine block (`ModelKeys.atom_dit = "esmfold2_pair"`, FoldJet `b36209c`, `fcddf56`); the released layout is converted per block (`released.esmfold2_atom_blocks_from_released`), the own `DiffusionCrossAttTransformer` stays only as the fallback on engines without the class (2.5.1). Audit on H100 (engine 2.5.4, n = 1 / 8 / 12 / 16): `AtomDiTESMFold2Pair` 9 modules served on `cuda`, no refusal (R6). |
| MSA / Evoformer block | every family with an MSA stack | `engine/graph.py::EvoformerBlock` composes engine parts in the AF3 order, the Boltz / OpenDDE order (`spec.msa_update_before_opm`) or Chai-1's parallel schedule (R11). |
| Template embedders, confidence / distogram heads, AF3 atom encoder / decoder, diffusion-head pre-norm and Fourier, sampler body, language models, `SummedInputEmbedder`, Boltz-2's confidence re-embedding, RoseTTAFold3's chirality features, `distogram_bias`, OpenDDE's structural-token expander, ESM-C towers | the graph | FoldJet-own modules; only their inner pairformer stacks, transitions, OPM and PWA are engine classes. |
| fp32 torch `nn.LayerNorm` on pair-sized tensors although the engine `LayerNorm` (`cuda` / `triton`) accepts every pair width the models use (64 / 128 / 256 / 384) | about 12 sites: template embedder, confidence head, recycle, diffusion conditioning | **FoldJet-own, numerics decision pending (no engine request: the engine side is ready).** Swapping changes the output dtype and rounding: torch returns fp32 under the bf16 autocast, the engine returns the input dtype. Potential: about 1% of the n = 8 time; engine micro-benchmark (H100, `[1, 1024, 1024, 128]` fp32): 0.37 ms against 1.56 ms for torch. |
| `TokenPairInit` generalised (pair width 128 / 256 / 384 / 512, relpos 139 / 143, bond forms) and Chai-1's token embedder | trunk pair initialisation | FoldJet's own `dense/pair_init.py`. |
| `TriangleAttention(model=)` / `TriangleMultiplication(model=)` key `protenix_v2` | protenix2 | `ModelKeys("protenix", "protenix_v2", "protenix_v2")`: the triangle classes use `protenix` (same math), the DiT classes `protenix_v2` (R4). |

## U-backend: refused calls, missing backends

| What | Where | Meanwhile |
| --- | --- | --- |
| ~~`OuterProductMean`: `count_offset` only on `pytorch`~~ (no `cueq` backend; an `anthropic` backend exists since 2.5.1) | every AF3-family MSA stack | **Closed 2026-11-14:** the engine `cuda` (sm_90a, sm_80) and `triton` kernels serve `count_offset`; `modules.OPM_COUNT_OFFSET` now defaults to `1e-3` (the exact AF3 divisor; `None` selects the clamp). Warm 5I28 n = 1, H100: protenix1 0.352 s (exact) vs 0.353 s (clamp), af3 0.380 s vs 0.380 s: no cost. Grouped OPM (Chai-1) has no division: no offset. |
| `MSAPairWeightedAveraging`: no `cueq` (an `anthropic` backend exists since 2.5.1; 2.5.0 had none) | MSA stack | `engine.ops: {msa_pwa: cueq}` is refused by name; fallback `pytorch`. |
| `ConditionedTransition` and `AdaptiveLayerNorm` (`backends = pytorch / triton`: no `cuda`, the diffusion head's conditioning runs `triton` / `pytorch`); `cuda` of the other classes is H100 (sm_90) / A100 only. ~~`TriangleMultiplicationRF3`~~ is not a gap any more: engine 2.5.4 lists `pytorch / anthropic / cuda / triton`, and the RoseTTAFold3 audit on H100 served `TriangleMultiplicationRF3` (110 modules) and `TriangleAttentionRF3` (110) on `cuda`, no refusal (n = 8 / 12 / 16). `BidirectionalTriangleMultiplication` lists `pytorch / cuda / triton` (no FoldJet model in the audits uses it) | RoseTTAFold3 exact mode, diffusion head | `choose_backend` gives `pytorch` with the reason; `miniworld` on another card means `triton` where the class has it. |
| `TriangleAttention*` / `TriangleMultiplication` `anthropic` / `cueq` ranges (shapes, masks, grad) | trunk | The class's own `_supports_<backend>`; the guard sends the refused call to `pytorch`. |
| ~~`AttentionPairBias` `cuda` refuses `extra_pair_bias`~~ | OpenDDE's refiner | **Closed 2026-11-12 (engine 2.5.4):** `AttentionPairBias._supports_cuda` takes `extra_pair_bias` (`[L, L]` / `[H, L, L]` / `[B, H, L, L]`, eval only: with a gradient it is refused, so a training call still runs `pytorch`). Audit on H100 (OpenDDE, n = 1 / 8 / 12): 56 modules, 216 calls on `cuda`, no refusal (R3). |
| ~~`TokenDiTChai1` / `AtomDiTChai1` `cuda` with FoldJet's bf16 flow~~ | Chai-1 diffusion | **Closed 2026-11-12 (engine 2.5.4):** both classes (`MasterEntryCast`) are served on `cuda` with FoldJet's flow (fp32 conditioning inputs accepted): `TokenDiTChai1` 16 modules / 48 calls, `AtomDiTChai1` 9 modules / 21 calls, no refusal at n = 8 / 12 / 16. A separate FoldJet-side dtype fix for the fp32 inputs is not a refusal (R7). |
| Atom DiT key windows: Chai-1 exact's `circular` removed (`KEY_WINDOWS = pad / slide / slide_qblock`) | `chai1` exact mode | `engine_key_window("circular")` returns `pad` plus the conformer-id mask `uid` (R9, see U-equivalence). |
| fp32 `TriangleMultiplication` / `TriangleAttention` / `Transition` | `precision: fp32` | bf16-only kernels: `pytorch` (R8). |
| Glue modules need fp32 (`AugmentNoise`, `AlignEuler`) / bf16 (`PoolAtoms`) CUDA tensors, scalar sigma, non-batched sigma | EDM step, encoder pooling | `module.unsupported(...)` is not None: the FoldJet torch path runs (same results, more launches). |
| One `model` key for the triangle attention AND the multiplication of a `PairformerBlock` | RoseTTAFold3 fast mode (RF3 attention + plain multiplication) | `engine/pairformer.py::pairformer_block` builds the block and replaces its two multiplications (R5). |

## U-layout: the same computation, another layout

| What | Meaning for FoldJet |
| --- | --- |
| Engine tensors are batched (`[B, L, ...]`, bool key mask `[B, L]`; DiT `single` / `cond` `[A, B, L, d]`) | The graph is unbatched (`[L, ...]`, `[L, L]` pair masks); `graph.run_block` and the DiT stacks add and remove the batch axis; masks come from `modules.residue_mask(pair_mask)`. |
| The DiT blocks have no shared pair norm / projection (AF3's super-block projection) | Broadcast / sliced per block by `released.token_dit_blocks_from_released`, **once, offline** (miniworld format). |
| Shared-KV atom blocks (Boltz-2 / RoseTTAFold3 / Chai-1) have no `ada_ln_kv` | The converter discards the released `k_adaptive_layernorm` only if it equals the query AdaLN exactly (`released._check_shared_kv_adaln`, else `ValueError`). Verified on the real blobs (every atom block of encoder, decoder and diffusion stack equal) (R14). |
| Pair LayerNorm offsets (MSA PWA, single attention, DiT) | Discarded by the converter (`released.DISCARDED`): softmax-invariant constants. |
| RoseTTAFold3's bias-free query (`FlatQKNormAugmentedAttention`, `query_bias=False`) | The released zero `q_projection.bias` is dropped only if all zero (`miniworld.DROPPABLE_ZERO`, `ConversionError` otherwise); all zero in the real blob. |
| The engine DiT blocks own their residuals and hoist their own pair bias | FoldJet's stacks (`TokenDiTStack` / `AtomDiTStack`) run blocks in order; `extra_pair_bias` (OpenDDE) is added to the hoisted bias; the atom block derives its 128-key window from the key mask (real atoms first, which FoldJet's atom layout guarantees) (R10). |
| Protenix atom blocks chain the K / V AdaLN (`kv_norm == "chained"`) and release the atom pair norm per block | Nothing discarded or sliced for protenix (the AF3 family's shared one is sliced). |
| protenix1 template stack: triangle multiplication hidden 128 on a 64-channel pair, attention 4 x 32 | `PairformerConfig(d_hidden_tri_multi=128)`; `graph.make_pairformer_block(tri_hidden=...)`. |
| Chai-1: identity AdaLN, no gating query, no atom output projection, one shared atom AdaLN; grouped OPM (8 groups x 8, engine `d_hidden` is the TOTAL width); confidence head `TriangleAttentionChai1Confidence` (released `output_projection_transposed` -> `to_out_t`) | Engine classes' own forms; `ModelKeys.confidence = "chai1_confidence"`. AdaLN epsilon: the engine default 0.1 is the release's; fast mode passes `spec.adaptive_norm_eps = 1e-5` (`graph.chai1_adaln_eps`). |
| Boltz-2 up-gated transitions | `UpGatedConditionedTransition` inside the `TokenDiTBoltz2` / `AtomDiTBoltz2` blocks (the block owns the transition). |
| OPM forms | Fast: divide after the projection (`normalize_before_proj=False`, clamp divisor); exact: `normalize_before_proj=True` + clamp; RoseTTAFold3: `projection_bias=True`; Chai-1: grouped. |
| Atom windows | Fast: `key_window="slide"`; exact: `"pad"` (the clipped `32 w - 48` window); Boltz-2 / RoseTTAFold3 as AF3. |
| Converter discards (real blobs) | Only softmax-invariant offsets (`single_pair_logits_norm.bias`, `msa_attention1.pair_norm.bias`, the pair-norm offsets above) and Chai-1's tied K / V AdaLN. Anything else unread makes `convert` refuse. |

## U-equivalence: where the engine form differs in a corner

| What | Meaning |
| --- | --- |
| Atom attention under 128 real atoms | The old AF3 mask form (`+1e9` only where BOTH query and key are padding) let a real query read the padded keys of its slid window; `AtomDiT*` switches every masked key off. Same result from 128 real atoms (tested at 130 and 200); below that the engine's form is the intended one (R13). |
| Chai-1 exact mode: `circular` key window removed | Exact mode runs `key_window="pad"` plus the same-conformer mask `uid`. The output equals the old exact output when no conformer covers both ends of the atom list and the system has at least 128 atoms (tested; real blob, 26 tokens, exact: sampled coordinates 3.4e-6 relative); where a conformer spans both ends (a one-conformer system) or there are fewer than 128 atoms the old exact output cannot be reproduced. So "exact" means the release's equations up to that window convention. Fast mode (`slide`) is unchanged (R9). |
| OPM divisor | exact (`n + 1e-3`) by default since 2026-11-14 (`OPM_COUNT_OFFSET = 1e-3`, served by the kernels); Boltz lineage keeps the clamp (`opm_clamped_norm`). |
| `AtomDiT*` window from the key mask | The engine derives each window from the key mask; the old graph gathered it from `queries_to_keys`. Same window with FoldJet's real-atoms-first layout, so `AtomDiTStack` ignores `queries_to_keys` / `keys_mask` / `keys_single_cond`. |

## U-precision

Engine dtypes are `master` (fp32 parameters, bf16 autocast) or `fp32`; the trunk is always `master`. FoldJet parameters are always fp32. The
`af3_default` policy loads the released values exactly into fp32 parameters (a bf16 value is exact in fp32), runs the trunk as `master` and keeps
its diffusion-head TF32 reference mode (`reference_precision`). The load report carries `precision` (`master` / `fp32`), `activation_dtype` and
`parameter_dtype` (always float32). Bit-for-bit bf16 *storage* of released weights is not reproducible (the matmul inputs are the same bf16
roundings under autocast; the fp32 residual / norm parameters are exact). The pair stream dtype is the activation dtype: the residual adds of the
FoldJet-owned glue run in whatever dtype autocast gives them.

Two consequences measured on FoldBench (115 assemblies, 2026-11-10/11):

- **Engine modules take the autocast dtype, never fp32** (`modules/engine/configure.py::engine_activation`, used by `graph.run_block` and the
  atom DiT stack): the engine's `MasterWeights` contract is fp32 masters with bf16 activations, and its `cuda` backends refuse an fp32 activation
  ("bf16 only") so the call ran the unchunked pytorch backend, whose fp32 triangle-attention logits are N^3 x 4 heads x 4 B (16 / 54 / 128 GiB at
  1024 / 1536 / 2048 tokens). Before 2026-11-11 the confidence head (it took its activation dtype from its fp32 weights), the template embedder and
  the atom transformer's conditioning handed fp32 over, in every family; the confidence head, the first refiner block and the template pairformer now
  run in bf16 under `master`, as the trunk always did. `fp32` precision (no autocast) is unchanged. Checked 2026-11-12 on an H100
  against the released structures (`tools/references/validate.sh fast`, 5I28 and 3PTB, three seeds, backend pytorch): 17 of 19 families x targets PASS.
  The two FAILs (5I28 IntelliFold2 pLDDT 69.0 against the release's 60.2; 5I28 OpenDDE structure 2.33 A against a bound of 2.21) are the same on the
  commit before these changes (`6bbca72`: 69.02 and 2.34 A), so they are not from this change, which moved those pLDDTs by 0.01; they are
  regressions since the 2026-09-27 reports (IntelliFold2 60.0 then), not yet traced. 1A1K and the MSA condition (5I28-msa) were not rerun.
- **The diffusion sampler loop runs OUTSIDE the activation autocast** (`models/architectures/af3.py::_sample_diffusion`; only the network calls `denoise` and
  the conditioning table open it). Inside it, the random rigid motion and the Kabsch alignment matmuls rounded the fp32 coordinates to bf16 at every one of
  the 200 steps: consecutive CA-CA distance std 0.10-0.21 A instead of 0.02-0.05 A and FoldBench lDDT 0.88 -> 0.17 with TM-score and DockQ unchanged.
- **`af3_default` runs the diffusion head in fp32** (since 2026-11-12; before it was bit-identical to `bf16`). AF3 runs the diffusion module in fp32 while the trunk is bf16:
  the trunk and the heads stay `master` (fp32 masters, bf16 activations), while the denoiser and the conditioning table (`models/architectures/af3.py::_sample_diffusion`)
  run under `modules/engine/configure.py::diffusion_autocast`, which DISABLES the autocast (the entry points open the bf16 autocast around the whole forward and
  `engine_activation` would turn fp32 head activations back into bf16 inside it). The head gets fp32 activations (`reference_precision` casts the pair / single
  trunk outputs) on the fp32 masters. The engine `cuda` backends of the head refuse them (matmul precision `highest`: "float32 activations run on TF32 tensor cores only")
  and the engine `pytorch` backend serves them in true fp32 (logged, `engine.unsupported: fallback`); `execution.tf32: true` makes them TF32 cores.
  `bf16` is unchanged (125 of 125 CIFs md5-identical on the 5 FoldBench targets). Measured 2026-11-12 on H100 (protenix1, 5 targets x 5 seeds x 5 samples): not
  closer to fp32 than `bf16` (all-atom RMSD to fp32 0.26 / 2.44 / 11.86 / 1.02 / 0.16 A against 0.29 / 2.54 / 11.81 / 1.01 / 0.21 A: noise; FoldBench scores equal),
  and 1.5x slower (5I28 warm forward n=1 0.44 -> 0.68 s, n=4 1.29 -> 2.00 s; `fp32`: 1.05 s, 10.3 s). It is the AF3 policy as defined, not an accuracy gain.
  Training (`train/step.py`) keeps the head under the bf16 autocast.

## The legacy converter graph (not a gap, a boundary)

The released Haiku layout is dealt with in one place, `src/foldjet/models/checkpoints/legacy/` (the dense released-layout modules and the Haiku
record codec), and only by the offline converter (`foldjet weights convert`, `docs/guides/CHECKPOINT-FORMAT.md`). Nothing at inference or
training imports it (guarded by `tests/models/test_miniworld_resolution.py`).
