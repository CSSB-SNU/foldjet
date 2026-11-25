# RTX 5090 (cssb-rtx), engine 3.0.6 campaign: notes on the CSVs

Checkout `~/ff/paper306` (paper-306 6a0593e + the rc fix of 6bafced / 0b01766 as working-tree changes; the rows carry `6a0593e0a`), env engine 3.0.6,
sm120 kernels 3.0.6, torch 2.13.0+cu129. Every run under `~/mwe/gpu.sh` (GPU lock, 16 GB host-RAM cap).

- **Checks:** AF3 5I28 n=1 finite on cuda (all 9 engine classes served by cuda, 0 refused; msa_stack.2/3.msa_transition max |x| 65,280, no non-finite;
  the 3.0.2 failing input replays finite on cuda / triton / pytorch). AF3 n=19-22 complete with the pair-band fix (1e6d215): reserved 22.0 / 23.2 / 26.3 / 27.9 GiB.
  Raw: H100 cluster `~/foldforge-scratch/paper/F9/rtx5090/check306/`.
- **ESMFold2 / ESMFold2-Fast: not feasible** (F3, F4, F5, F9): loading ESM-C 6B (24 GB fp32 safetensors) exceeds the 5090 host's 16 GB RAM cap (killed by systemd-oomd; user rule, cap not raised);
  FoldJet has no low-RAM ESM-C load.
- **Kits:** not re-run (5090 kits of 2026-11-18 in `F3_5i28_times/data/existing/`); F9 has no kit arm (user decision). F10 removed from the paper.
- **F3:** one process per (model, repetition) over n = 1..16. Protenix v2 is out of memory at n = 16 there although a fresh process completes up to n = 19 (F9):
  memory left by the smaller sizes of the same process.
- **F4:** every error row is out of memory (74, mostly the PyTorch arm at the large targets): results, not failures.
- **F6:** engine arm: 48 augmentations in one pass and chunks 24 / 12 run out of memory, chunk 8 trains (recorded `aug_chunk`). PyTorch eager runs out of memory
  at every chunk down to 2; PyTorch+compile and cuEq+compile train at chunk 8. Engine-arm runs: 0 calls served by Triton / PyTorch
  (`~/foldforge-scratch/paper/engine_issues/rtx5090_f6_backend_calls.md`).
- **F5:** sample-to-sample pairs only (`F5_rtx5090_pairs_summary.csv`); the FoldBench-evaluator ground-truth scoring of the 5090 predictions is not run
  (predictions on the H100 cluster: `~/foldforge-scratch/paper/F4/rtx5090/pred/`).
