# 5I28 benchmark: RoseTTAFold3 Anthropic kit

Spec: `docs/guides/BENCHMARK-5I28.md`. The kit is `$KITS_DIR/rosettafold3_cu130`, used read-only.
Fast and exact mode run `source configs/h100.env; ROSETTAFOLD3_OPT=<mode>; opt/venv/bin/python fwdtimer.py ... rf3.cli:app fold`;
off mode uses the stock `venv/bin/python`. It runs in the kits' container (`cu130kit.sh`, a copy with its singularity cache/tmp moved
here).

## Run

```
CUDA_VISIBLE_DEVICES=<gpu> bash run.sh N          # N = 1 2 4 8 12 16; MODE=fast|exact|off (default fast); COPIES (default 4 at n=1, else 3)
```

One process folds COPIES identical items, so it does COPIES forwards and the first is cold. The full log goes to `logs/x<N>.log`.
Stdout is a single line:
`KIT rosettafold3 n=<N> exit=<rc> fwd=[...] warm_median=<s> templates_real=<k>/1 msa_rows=<r>`. For a mode other than fast, the kit
name is `rosettafold3:<mode>`.

- `fwd` is the `FWDTIME` line of `fwdtimer.py`: `RF3WithConfidence.forward` between two `torch.cuda.synchronize()` calls. It is the
  same timer as the earlier `fwdtimer_real.py`, and the audit marks this span as matching FoldJet's. `warm_median` is the median of
  every forward except the first.
- `templates_real` is 1/1 only when every forward's model input has the template on every chain: its `REALINPUT` line must show
  `templated_tokens = 88*N` and `template_pairs = N*88*88` (intra-chain only). Otherwise it is 0/1. The kit's own
  `TEMPLATE selection=N` line is copied into the log.
- `msa_rows` is the full MSA depth handed to atomworks' FeaturizeMSALikeAF3 (`MSAFULL` line, printed outside the timed region). RF3
  then feeds 1024 rows per recycle (`msa_stack` in `REALINPUT`).

## Inputs (`inputs/`)

- **MSA:** `msa/5i28.a3m` is the real 5I28 MSA (6789 a3m rows, same file as the earlier runs). It goes into `msa_paths` for every
  chain.
- **Template:** RF3 has one template track, a CA distogram of the input structure (one coordinate set per chain). The input structure
  `x<N>/tpl_x<N>.pdb` is threaded: the query sequence, carrying 5syd_A's N/CA/C/O coordinates on its 88 aligned query positions
  (41-128), with occupancy 0 elsewhere. `template_selection` is `<chain>/*/41-128` for every chain.
  - It is built by `mkthread.py` from `~/data/cif_raw/sy/5syd.cif.gz` and the LMDB 20-hit a3m.
  - It is byte-identical to the earlier `rosettafold3_realmsa` input (`check/cpu_check.out`).
  - 5syd_A is FoldJet's first template of its 4.
- **Sampling:** `n_recycles=4 num_steps=200 diffusion_batch_size=5 early_stopping_plddt_threshold=0 seed=0`.
- **Caches:** `cache/` is a copy of the kit's warm `cache/rosettafold3_gpu130`, with the paths in Triton's group files rewritten.
  TMPDIR and the outputs are under this directory.

## FoldJet side

FoldJet's `rosettafold3` family has `use_input_templates=False` (`~/FoldForge/src/foldjet/modules/dense/spec.py`), so its template
embedder runs on zero features. To give it the kit's input it needs two changes:

- Set `use_input_templates=True` for that family.
- Feed exactly one template, 5SYD_A (`template_n: 1`; it is first in the LMDB order). FoldJet's RF3 `_single_pass` averages the
  features of every present slot, so with `template_n: 4` it would condition on a mean of 4 distograms, which the kit cannot express.

The template pair stack runs once in either case. The timing difference is only the per-template feature build.
