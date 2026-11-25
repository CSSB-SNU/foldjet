# Archive: dated records (historical)

Everything here is a snapshot of its date. It is kept as evidence and is **not** the current behaviour, interface or benchmark
table; the current documents are listed in [../README.md](../README.md). Paths, engine versions, backend names (`miniworld`,
`team-gm`, `libs/...`) and commands in these files are those of their date. The JSON / CSV files are the raw evidence the
Markdown records cite; do not edit them.

## 2026-09-13 .. 2026-09-15: first integration (team-gm era)

| record | what it records | evidence |
|---|---|---|
| [ENVIRONMENT-20260913.md](ENVIRONMENT-20260913.md) | the consumer environment of the first integration (uv, torch 2.10) | |
| [ENGINE-INTEGRATION-20260913.md](ENGINE-INTEGRATION-20260913.md) | engine pin and consumer API repairs | |
| [CODE-UNIFICATION-20260913.md](CODE-UNIFICATION-20260913.md), [continued](CODE-UNIFICATION-CONTINUED-20260913.md) | first code-unification audit (CCD, Pairformer, MSA, diffusion, templates, precision) | `model-boundaries-20260913.csv` |
| [SHARED-CORE-20260914.md](SHARED-CORE-20260914.md) | diffusion / neural core moved from model copies to team-gm | `shared-core-*.json` |
| [SHARED-IO-20260914.md](SHARED-IO-20260914.md) | one typed request / runtime for every CLI | `shared-layout-migration.json` |
| [MODEL-CORE-UNIFICATION-20260914.md](MODEL-CORE-UNIFICATION-20260914.md) | common checkpoint core, FoldCP removal | `model-core-*.json`, `model-boundaries-20260914.csv` |
| [QUALIFICATION-20260914.md](QUALIFICATION-20260914.md) | inference qualification: complex inputs, compile / CUDA-graph boundaries | `qualification-*.json`, `qualification-cache-misses.csv` |
| [CLOSEOUT-20260914.md](CLOSEOUT-20260914.md) | follow-up: operational DB reads, A5000, the large-input AF3 compiler regression | `operational-input-results.json` |
| [CSSB3-WORKSPACE-20260914.md](CSSB3-WORKSPACE-20260914.md) | the cssb3 (A100) workspace migration | |
| [INFERENCE-BACKENDS-BUCKETS-20260914.md](INFERENCE-BACKENDS-BUCKETS-20260914.md) | backends and token buckets (padding policy superseded on 09-15) | |
| [TOKEN-BUCKETS-20260915.md](TOKEN-BUCKETS-20260915.md) | token axes rounded to multiples of 128 | `layout-validation-20260915.json` |
| [INFERENCE-BUCKETS-E2E-20260915.md](INFERENCE-BUCKETS-E2E-20260915.md) | buckets and end-to-end measurements | `inference-e2e-results-20260915.json` |
| [BACKEND-PURITY-20260915.md](BACKEND-PURITY-20260915.md) | PyTorch backend execution audit (before the attention wiring fix) | `BACKEND-PURITY-20260915.json` |
| [BACKEND-WIRING-20260915.md](BACKEND-WIRING-20260915.md) | checkpoint backend dispatch and the CUDA-graph experiment | `BACKEND-WIRING-20260915.json` |
| [checkpoint-kernel-wiring-20260915.md](checkpoint-kernel-wiring-20260915.md) | pointer to the engine's checkpoint shape census | |
| [PORTING-esmfold2.md](PORTING-esmfold2.md) | how ESMFold2 was ported from team-gm into FoldJet | |

## Benchmark reports replaced by [../benchmark_results.md](../benchmark_results.md)

- [benchmark-20260916/](benchmark-20260916/benchmark_results.md): 4YX2, four models x five configurations (2026-09-16), with its `assets/`.
- [benchmark-20260918-4yx2/](benchmark-20260918-4yx2/benchmark_results.md): the 4YX2 report of 2026-09-18 and its notes.

## Engine migration records (superseded by [../ENGINE-UNSUPPORTED.md](../ENGINE-UNSUPPORTED.md) and [../guides/BACKENDS.md](../guides/BACKENDS.md))

- [BACKENDS-engine-2.3.md](BACKENDS-engine-2.3.md): backend policy and measurements on the pre-2.5.0 engine.
- [engine-kernel-gaps-20261021.md](engine-kernel-gaps-20261021.md): engine work that was still needed to connect every family (engine 2.3.3, 2026-10-21;
  moved here from `docs/engine-kernel-gaps.md`).
- [ENGINE-2.4.2.md](ENGINE-2.4.2.md), [ENGINE-2.4.3.md](ENGINE-2.4.3.md): what moved into the engine in 2.4.2, released-checkpoint parity in 2.4.3
  (moved here from `docs/guides/`).
