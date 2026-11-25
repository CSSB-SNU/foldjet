"""Real-data batch source for ``scripts/train_protenix1.py``: MiniWorld's PDB sampler + loader -> FoldJet ``TrainBatch``.

    python scripts/train_protenix1.py --batch-source ft_adapter:batches ...        (PYTHONPATH must contain this directory)

``batches(cfg, device)`` returns an endless iterator of ``TrainBatch`` (one example, fixed bucket shape: 384 tokens / 4096 atoms / MSA pool /
4 templates). Knobs are environment variables (all optional), see README.md; they can also be passed as keyword arguments to ``make_source``.

Data flow (per rank): MiniWorld ``WeightedSampler`` index stream (rank-distinct, seed + epoch + rank, deterministic in the stream position) ->
torch DataLoader with CPU worker processes (spawn) -> each worker: MiniWorld ``Preprocessor.process`` (CIF load, crop, MSA, templates) with a rng
seeded from (seed, rank, position) -> ``convert.Converter`` (AF3 / FoldJet featurisation of the crop) -> CPU tensors -> main process moves them to
the GPU and wraps them in ``TrainBatch``.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
for extra in (
    HERE.parent / "extra_site",
):  # pyarrow, symlinked from MiniWorld's env (no install); appended last so it never shadows the env
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.append(str(extra))

import _mwshim  # noqa: F401


def _env(name: str, default, cast=str):
    value = os.environ.get(name)
    return default if value in (None, "") else cast(value)


@dataclass
class Settings:
    # default: MiniWorld v2.4.0's data (BioMolDB_20261014 on node-local NVMe, sampler antibody_protein 10 / protein_rna 20); was H100_default.yaml (older data)
    data_config: str = field(
        default_factory=lambda: _env(
            "FT_DATA_CONFIG", str(HERE / "data_20261014_local_ft.yaml")
        )
    )
    overlay: str = field(
        default_factory=lambda: _env("FT_OVERLAY", str(HERE / "overlay_pdb_only.yaml"))
    )
    seed: int | None = field(
        default_factory=lambda: _env("FT_SEED", None, int)
    )  # default: cfg.seed
    workers: int = field(default_factory=lambda: _env("FT_WORKERS", 14, int))
    prefetch: int = field(
        default_factory=lambda: _env("FT_PREFETCH", 4, int)
    )  # batches in flight per worker
    start_step: int = field(
        default_factory=lambda: _env("FT_START_STEP", 0, int)
    )  # optimizer step the run (re)starts at
    epoch_size: int = field(
        default_factory=lambda: _env("FT_EPOCH_SIZE", 1 << 17, int)
    )  # indices drawn per rank per sampler "epoch"
    tokens: int = field(default_factory=lambda: _env("FT_TOKENS", 384, int))
    atoms: int = field(default_factory=lambda: _env("FT_ATOMS", 4096, int))
    msa_pool: int = field(default_factory=lambda: _env("FT_MSA_POOL", 8192, int))
    templates: int = field(default_factory=lambda: _env("FT_TEMPLATES", 4, int))
    bond_rows: int = field(default_factory=lambda: _env("FT_BOND_ROWS", 128, int))
    log_every: int = field(default_factory=lambda: _env("FT_LOG_EVERY", 25, int))
    max_failures: int = field(
        default_factory=lambda: _env("FT_MAX_FAILURES", 200, int)
    )  # converted-sample failures (skipped) tolerated per worker
    model: str = "protenix1"
    mode: str = field(
        default_factory=lambda: _env("FT_MODE", "fast")
    )  # FoldJet DenseSpec mode ("fast" = the training script default)


def rank_world() -> tuple[int, int]:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:  # noqa: S110 - torch.distributed probe is optional
        pass
    return _env("RANK", 0, int), _env("WORLD_SIZE", 1, int)


def sample_seed(base_seed: int, rank: int, position: int) -> int:
    """32-bit seed of the sample at stream position ``position`` of ``rank`` (pure function: a restart reproduces it)."""
    return int(
        np.random.SeedSequence(
            [int(base_seed) & 0xFFFFFFFF, int(rank), int(position)]
        ).generate_state(1, dtype=np.uint32)[0]
    )


# -------------------------------------------------------------------------------------------------------------------- worker side
class SampleDataset:
    """Map-style dataset the DataLoader workers index with ``(position, catalog_index)``. Built lazily in each worker process."""

    def __init__(self, settings: Settings, base_seed: int, rank: int) -> None:
        self.settings, self.base_seed, self.rank = settings, base_seed, rank
        self._ds = None
        self._conv = None
        self.failures = 0

    def _setup(self) -> None:
        import torch

        torch.set_num_threads(1)
        import mwconfig
        import mwsample
        from convert import Converter

        data = mwconfig.load_data_dict(self.settings.data_config, self.settings.overlay)
        self._ds = mwsample.build_dataset(data, light=True)
        s = self.settings
        self._conv = Converter(
            family=s.model,
            mode=s.mode,
            tokens=s.tokens,
            atoms=s.atoms,
            msa_pool=s.msa_pool,
            templates=s.templates,
            bond_rows=s.bond_rows,
        )

    def __getitem__(self, key: tuple[int, int]) -> dict[str, Any]:
        import mwsample
        import torch
        from convert import ConversionError

        if self._ds is None:
            self._setup()
        position, index, spares = key
        spares = list(spares)
        seed = sample_seed(self.base_seed, self.rank, position)
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        c0 = time.process_time()
        for attempt in range(
            8
        ):  # a sample that cannot be expressed in the dense layout is replaced by the next draw of the same rng
            try:
                raw = mwsample.process_item(self._ds, index, rng, spares=spares)
                t_mw = time.perf_counter() - t0
                out = self._conv.convert(raw, seed + attempt)
                break
            except ConversionError as exc:
                self.failures += 1
                print(f"FT_SKIP pos={position} idx={index} {exc}", flush=True)
                if not spares:
                    raise RuntimeError(f"position {position}: no spare index left")  # noqa: B904 - keep the traceback as is
                index = int(spares.pop(0))
            except Exception as exc:  # one bad record (unreadable LMDB value, RDKit failure, ...) must not end a multi-day run
                self.failures += 1
                print(
                    f"FT_ERROR pos={position} idx={index} {type(exc).__name__}: {str(exc)[:300]}",
                    flush=True,
                )
                if not spares:
                    raise RuntimeError(f"position {position}: no spare index left")  # noqa: B904 - keep the traceback as is
                index = int(spares.pop(0))
        else:
            raise RuntimeError(
                f"position {position}: 8 consecutive unconvertible samples"
            )
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        return {
            "name": out.name,
            "features": {k: t(v) for k, v in out.features.items()},
            "atom_positions": t(out.atom_positions),
            "atom_mask": t(out.atom_mask),
            "bond_pairs": t(out.bond_pairs),
            "present_index": t(out.present_index),
            "present_valid": t(out.present_valid),
            "info": {
                **out.info,
                "position": position,
                "mw_s": t_mw,
                "total_s": time.perf_counter() - t0,
                "cpu_s": time.process_time() - c0,
                "attempts": attempt + 1,
                "pid": os.getpid(),
            },
        }


class _Positions:
    """Sampler of ``(position, catalog_index, spare_indices)``: MiniWorld WeightedSampler draws, epoch after epoch, from stream position ``start``.
    ``spare_indices`` (SPARES per position, a second weighted draw) replace a sample that fails, so a retry stays inside the sampler distribution.
    """

    SPARES = 16

    def __init__(self, sampler, spare_sampler, epoch_size: int, start: int) -> None:
        self.sampler, self.spare_sampler, self.epoch_size, self.start = (
            sampler,
            spare_sampler,
            epoch_size,
            start,
        )

    def __iter__(self):
        position = self.start
        while True:
            epoch, offset = divmod(position, self.epoch_size)
            self.sampler.set_epoch(epoch)
            indices = list(iter(self.sampler))
            self.spare_sampler.set_epoch(epoch)
            spares = np.fromiter(
                iter(self.spare_sampler),
                dtype=np.int64,
                count=self.epoch_size * self.SPARES,
            ).reshape(self.epoch_size, self.SPARES)
            for i in range(offset, self.epoch_size):
                yield position, int(indices[i]), spares[i].tolist()
                position += 1


# -------------------------------------------------------------------------------------------------------------------- main side
class Source:
    """Iterator of ``TrainBatch`` (+ statistics: ``stats()``)."""

    def __init__(self, cfg, device, settings: Settings | None = None) -> None:
        import mwconfig
        import mwsample
        from torch.utils.data import DataLoader

        self.settings = s = settings or Settings()
        self.device = device
        self.rank, self.world = rank_world()
        self.seed = int(s.seed if s.seed is not None else getattr(cfg, "seed", 0))
        accum = int(getattr(cfg, "grad_accum_steps", 1))
        self.start_position = (
            int(s.start_step) * accum
        )  # the run consumed ``accum`` samples per optimizer step on every rank
        data = mwconfig.load_data_dict(s.data_config, s.overlay)
        t0 = time.time()
        full = mwsample.build_dataset(
            data
        )  # catalog + source/type-balanced weights (this process only draws indices)
        from miniworld.data.dataloader.sampler import WeightedSampler

        sampler = WeightedSampler(
            full,
            full.weights,
            num_samples=s.epoch_size,
            num_replicas=self.world,
            rank=self.rank,
            shuffle=True,
            seed=self.seed,
            replacement=full.config.DB_config.sample_with_replacement,
        )
        self.catalog_size = len(full)
        spare_sampler = WeightedSampler(
            full,
            full.weights,
            num_samples=s.epoch_size * _Positions.SPARES,
            num_replicas=self.world,
            rank=self.rank,
            shuffle=True,
            seed=self.seed + 1_000_003,
            replacement=full.config.DB_config.sample_with_replacement,
        )
        positions = _Positions(
            sampler, spare_sampler, s.epoch_size, self.start_position
        )
        del full
        dataset = SampleDataset(s, self.seed, self.rank)
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ.setdefault(var, "1")
        self.loader = DataLoader(
            dataset,
            batch_size=None,
            sampler=positions,
            num_workers=s.workers,
            prefetch_factor=s.prefetch if s.workers else None,
            multiprocessing_context="spawn" if s.workers else None,
            persistent_workers=bool(s.workers),
            pin_memory=False,
            collate_fn=_identity,
        )
        self._it = iter(self.loader)
        self._q = None
        if (
            os.environ.get("FT_PUMP", "0") == "1"
        ):  # a thread takes the worker results (unpickle + shared-memory rebuild) off the training thread
            import queue
            import threading

            self._q = queue.Queue(maxsize=int(os.environ.get("FT_PUMP_DEPTH", "3")))
            threading.Thread(target=self._pump, name="ft-pump", daemon=True).start()
        self.n = 0
        self.wait_s = 0.0
        self.last_return = None
        self.compute_s = 0.0
        self.sum_info: dict[str, float] = {}
        self.setup_s = time.time() - t0
        print(
            f"FT_SOURCE rank={self.rank}/{self.world} seed={self.seed} start_position={self.start_position} workers={s.workers} "
            f"prefetch={s.prefetch} catalog={self.catalog_size} msa_pool={s.msa_pool} setup_s={self.setup_s:.1f}",
            flush=True,
        )

    def _pump(self) -> None:
        try:
            for item in self._it:
                self._q.put(item)
            self._q.put(StopIteration())
        except BaseException as exc:  # (handed to the training thread)
            self._q.put(exc)

    def _next_item(self):
        if self._q is None:
            return next(self._it)
        item = self._q.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def __iter__(self):
        return self

    def __next__(self):
        from foldjet.train.data import TrainBatch, features_to_device

        t_call = time.perf_counter()
        if self.last_return is not None:
            self.compute_s += (
                t_call - self.last_return
            )  # time the consumer spent between two next() calls (the GPU work of the micro step)
        item = self._next_item()
        t_got = time.perf_counter()
        self.wait_s += t_got - t_call
        dev = self.device
        batch = TrainBatch(
            features=features_to_device(item["features"], dev),
            atom_positions=item["atom_positions"].to(dev, non_blocking=True),
            atom_mask=item["atom_mask"].to(dev, non_blocking=True),
            bond_pairs=item["bond_pairs"].to(dev, non_blocking=True),
            name=item["name"],
            present_index=item["present_index"].to(dev, non_blocking=True),
            present_valid=item["present_valid"].to(dev, non_blocking=True),
        )
        self.n += 1
        for k, v in item["info"].items():
            if isinstance(v, (int, float)):
                self.sum_info[k] = self.sum_info.get(k, 0.0) + v
        if self.n <= 8 or self.n % self.settings.log_every == 0:
            self._log(item, t_got - t_call)
        self.last_return = time.perf_counter()
        return batch

    def _log(self, item, wait: float) -> None:
        i = item["info"]
        print(
            f"FT_DATA rank={self.rank} n={self.n} pos={i['position']} item={item['name']} tokens={i['tokens']} atoms={i['atoms']} "
            f"msa={i['msa_rows']} templates={i['templates']} bonds={i['bonds']} worker_s={i['total_s']:.2f} (mw {i['mw_s']:.2f}, "
            f"convert {i['convert_s']:.2f}) wait_ms={1000 * wait:.1f} mean_wait_ms={1000 * self.wait_s / self.n:.1f} "
            f"mean_between_next_ms={1000 * self.compute_s / max(self.n - 1, 1):.0f}",
            flush=True,
        )

    def stats(self) -> dict[str, float]:
        n = max(self.n, 1)
        return {
            "samples": self.n,
            "wait_s": self.wait_s,
            "mean_wait_ms": 1000 * self.wait_s / n,
            "mean_compute_ms": 1000 * self.compute_s / max(n - 1, 1),
            **{f"mean_{k}": v / n for k, v in self.sum_info.items()},
        }


def make_source(cfg, device, **overrides) -> Source:
    settings = Settings()
    for key, value in overrides.items():
        setattr(settings, key, value)
    return Source(cfg, device, settings)


def batches(cfg, device):
    """``--batch-source ft_adapter:batches``."""
    return make_source(cfg, device)


def _identity(x):
    return x
