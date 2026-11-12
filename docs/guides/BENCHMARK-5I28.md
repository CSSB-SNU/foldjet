# The 5I28 benchmark

The model-level speed benchmark of FoldForge: every supported model on the engine against its Anthropic kit (and the vendor's stock code),
on **5I28 (azurin, 128 residues)**, one H100 80GB.

**Ownership.** The protocol was first written in miniworld-engine (`benchmarks/5i28.md`, commit `71c800225`, 2026-10-30; removed from the
engine tree by the 2.5.0 redesign). FoldForge owns it from 2026-11-11: this page is the protocol, FoldForge runs it, and the results
are recorded here. The engine commit keeps the earlier records (all nine models, Protenix v1/v2 at 10 trunk passes, the kit
forward-definition audit): `docs/records/5i28_*.md` and `anthropic_kits_fwd_definition_audit_20261029.md` in that commit.

## Why 5I28

5I28 is 128 residues long, exactly one token-bucket unit. FoldForge rounds tokens up to multiples of 128
(`models/bucketing.py::TOKEN_BUCKET_SIZE = 128`), so n identical chains give 128 n tokens, a whole number of buckets for every n:
the benchmark measures the model at 128, 256, ... 2048 tokens with **no padding rows** in the timed forward, and the size axis is
plain "n chains". A target that is not a multiple of 128 would time padded tokens and make the sizes of different targets
incomparable. One fixed target, one known sequence and one real MSA / template set also make every model and kit see the same input.

## Inputs (both sides, every model, every n)

- **Size:** n identical chains, n = 1, 2, 4, 8, 12, 16 (128 n tokens).
- **MSA:** the real MSA, always. FoldForge: `msa_db: ~/data/BioMolDB_20260224/a3m_16k.lmdb`, key `P0113359` (6790 rows); kits: the same
  rows as an a3m. There is no run without the MSA.
- **Templates:** 4 distinct real templates, always: `5SYD_A`, `6KOL_A`, `5YT7_A`, `6TFD_A`.
  - FoldForge: `template_db: .../template.lmdb`, key `P0113359`, `template_n: 4`. The LMDB stores every chain of an entry (20 records,
    6 distinct sequences). **Keep `template_dedup: true` (the default)**: `load_templates` then keeps one record per hit sequence,
    which gives these four. (FoldBench specs set `template_dedup: false`; a 5I28 spec must not.)
  - Kits: those 4, in each kit's own input form. For the Protenix-family kits (Protenix v1/v2, OpenDDE) use the whole 20-hit a3m;
    their search keeps one hit per sequence. Never pass a list that a kit's dedupe would shrink.
  - A kit run counts only when its log shows 4 real templates (`real=4/4`, `Found 4 templates`).
  - RoseTTAFold3 takes one coordinate set per chain: the same single template on both sides. ESMFold2 has no template input.
- **Sampling:** 4 recycles, 200 steps, 5 samples, bf16. The same seeds where the kit allows.

## Timing

One complete model forward (trunk passes, diffusion sampler, confidence heads), CUDA-synchronized before and after, excluding
featurization, loading and output writing. Report the median of the warm forwards; the first forward is cold.
- FoldForge: `model_seconds_warm`.
- Kits: their `PHASE ... fwd_s` (or the kit's equivalent span, per the kit forward-definition audit). Not the kits' wider
  "Model forward time" lines.
- FoldForge runs on prebuilt kernels and warm caches: no JIT compile inside a timed run.
- **The FoldForge number includes one sampler CUDA-graph capture per forward** (2 eager warm-up steps + capture, `models/sampler_graph.py`: the graph is
  built again for every fold). Measured on Protenix v1 (H100, 2026-11-12, engine 2.5.4): 73-79 ms at n = 1 (about 15% of the 0.485 s warm forward) and
  94-97 ms at n = 8 (about 2.4% of 3.9 s); a second call of the model pays it again. The kits' own spans are not changed by this. Records written before a
  fix keep their numbers; engine request R26 in [ENGINE-REQUESTS.md](../ENGINE-REQUESTS.md) is what would let the graph be captured once per shape.

## Determinism

The same input, the same seeds and the same code give bit-identical output across processes, also with a fresh inductor / Triton cache
(checked 2026-11-13: all nine benchmark models at n = 1 in two processes each, RoseTTAFold3 also at n = 4 and 8; coordinates and every
confidence array with `np.array_equal`). Two sources of run-to-run differences were removed: Chai-1's post-atom-conditioning LayerNorm, which
inductor autotuned by timing inside the compiled step (a different reduction order per process; now made once per fold outside the step),
and RoseTTAFold3's chirality-gradient `index_add_` (float atomics on overlapping centres; now a sorted segment sum). A new engine kernel that
uses atomics or a timing-chosen configuration breaks this guarantee: check it with two fresh processes. Chai-1 and RoseTTAFold3 outputs
changed once with the fix (a new deterministic variant); references made before 2026-11-13 differ from them by the old spread.

## Reporting

- Show the ratio as **kit / FoldForge** (> 1 means FoldForge is faster), never FoldForge / kit.
- A number taken any other way (no MSA, no templates, fewer than 4 real templates, a wider timing span) is not a 5I28 number.
- Records are dated files (`docs/records/`, created with the first FoldForge record) and are not edited afterwards. The first:
  [2026-11-11](../records/5i28_foldforge_20261111.md).
  The kit side (every kit, n = 1..16, method, versions, caveats): [2026-11-11](../records/5i28_anthropic_kits_20261111.md); its harness is `benchmarks/5i28_kits/`.

## Preflight before any sweep

Run n = 1 of every model on BOTH sides first and check: the checkpoint loads; the trunk passes equal the kit's; the templates and
MSA rows are the real ones; zero compiles in the timed region; every setting matched to the kit (never a FoldForge default).
Never edit a script a running queue executes.
