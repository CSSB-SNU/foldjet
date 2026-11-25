# scripts/

Training drivers. Inference has no script: use `foldjet fold <model>` or the Python API (see the [root README](../README.md)).

| path | what |
|---|---|
| `train_protenix1.py` | FoldJet's own training driver (Protenix v1 / AF3 on the one graph, the MiniWorld recipe of `foldjet.train.config.TrainConfig`): `--smoke`, `--batch-source <module>:<fn>`, `--bench` (the F6 / F10 paper harnesses run it unchanged). Guide: [docs/guides/TRAINING.md](../docs/guides/TRAINING.md) |
| `finetune/` | the Protenix v1 fine-tune on MiniWorld data: the MiniWorld data adapter, Slurm run scripts and conformer-cache tools ([finetune/README.md](finetune/README.md)) |

Utilities that are not training (checks, audits, conversion, profiling) are in [`../tools/`](../tools/README.md).
