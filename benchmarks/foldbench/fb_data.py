"""(target, trunk_seed) -> the featurised model input, built in torch DataLoader workers (CPU), consumed in order by the GPU process.

One item per (target, trunk_seed), in processing order (every target, its trunk_seeds one after the other); fb_multi_worker.py runs the same
logic in forked feature processes, so its queue (as a DataLoader's `num_workers` x `prefetch_factor`) keeps the next trunk_seeds and the next targets ready while the GPU works. A worker keeps the last target's
loaded input (and, through the forked MSA-feature cache, its MSA features), so the trunk_seeds that land on the same worker do not redo them."""

from pathlib import Path

from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, targets, seeds, specs, model, msa_depth, bucketing):
        self.targets, self.seeds, self.specs = list(targets), list(seeds), str(specs)
        self.model, self.msa_depth, self.bucketing = (
            model,
            int(msa_depth),
            bool(bucketing),
        )
        self._last = (None, None)

    def __len__(self):
        return len(self.targets) * len(self.seeds)

    def __getitem__(self, idx):
        t, s = self.targets[idx // len(self.seeds)], self.seeds[idx % len(self.seeds)]
        try:
            import foldforge as ff
            from foldforge.data.inputs.build import limit_msa
            from foldforge.models.io import dense_atoms

            if self._last[0] != t:
                self._last = (
                    t,
                    limit_msa(
                        ff.load_inputs(Path(self.specs) / f"{t}.yaml").target,
                        self.msa_depth,
                    ),
                )  # the cap `prepare_prediction` applies
            target = self._last[1]
            key, result, _ = dense_atoms._prefetch_job(
                target, self.model, s, str(target.spec.ccd_db), self.bucketing
            )
            return {
                "target": t,
                "seed": s,
                "slot": (target.spec.name, int(s), self.bucketing),
                "key": key,
                "sig": dense_atoms._spec_signature(target, s, self.bucketing),
                "result": result,
                "error": None,
            }
        except Exception as exc:  # one bad input must not end the iterator
            return {
                "target": t,
                "seed": s,
                "slot": None,
                "key": None,
                "sig": None,
                "result": None,
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            }
