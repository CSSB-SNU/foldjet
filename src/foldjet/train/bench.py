"""Speed measurement of real training steps (``scripts/train_protenix1.py --bench``).

The data pool, the deterministic micro-step plan, the engine-dispatch table and the
step-time summary. Everything here is plain Python / torch so it is unit-tested on the
CPU.

The data are STAND-INS: FoldJet's own featurisation of FoldBench targets with the
model's own pre-training prediction as ground truth (the real ground-truth atoms are
not mapped to the dense layout; the real loader is MiniWorld's adapter). Step time
does not depend on the label values, it depends on the shapes (tokens, atoms, MSA
rows, templates), which the stand-ins have.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

import foldjet as ff
from foldjet.train.bucket_batch import BucketSpec, bucket_example, count_signatures
from foldjet.train.data import TrainBatch, features_to_device, fixed_atom_index

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from foldjet.api import Predictor
    from foldjet.train.config import TrainConfig
    from foldjet.train.dist import WorldInfo

SPECS = Path("/home/psk6950/foldforge-scratch/foldbench/specs_all")
#: ``target<TAB>tokens<TAB>atoms<TAB>...`` of every FoldBench target
#: (``foldbench/size_scan.py``: tokens / atoms after FoldJet's featurisation).
SIZE_TSV = Path("/home/psk6950/foldforge-scratch/foldbench/size_sorted.tsv")
#: MiniWorld's recipe crop (configs/miniworld/data/H100_default.yaml: crop.max_tokens
#: 384, max_atoms 4096; phase2a_*_b200_joint: crop L384).
RECIPE_TOKENS = 384
#: length of the present-atom index of every pool example (--bench-bucket sets it to
#: --bucket-atoms: one signature above 4096 atoms; else a per-example multiple of 512)
FIXED_LENGTH = 4096


# ------------------------------------------------------------- the deterministic plan
@dataclass(frozen=True)
class MicroPlan:
    """Pool example, trunk pass count and torch seed of one micro step."""

    example: int  # index into the pool
    passes: int  # trunk passes (recycles) of this micro step
    # seed of the torch RNG (noise, rotations, sigma) before the micro step
    torch_seed: int


def global_micro_index(it: int, micro: int, accum: int, rank: int, world: int) -> int:
    """Micro step number over the whole job.

    Rank ``r`` takes ``r, r + W, ...`` of the ``accum * world`` micro steps of
    optimizer step ``it``.
    """
    return (it * accum + micro) * world + rank


def micro_plan(
    base_seed: int, global_index: int, pool_size: int, cfg: TrainConfig
) -> MicroPlan:
    """Pool example, recycle count and RNG seed of one micro step.

    A pure function of ``(base_seed, global_index)``. The three backends and every
    rank draw the same data stream, and a data-parallel run equals the single-process
    run that takes the same global indices.
    """
    rng = random.Random(base_seed * 1_000_003 + global_index)
    example = rng.randrange(pool_size)
    passes = rng.randint(cfg.recycles_min, cfg.recycles_max)
    # e.g. "1,2,3,4": measurement only, recycle counts cycle by global micro index
    # (capture-order test)
    forced = os.environ.get("FF_BENCH_PASSES_SEQ")
    if forced:
        seq = [int(x) for x in forced.split(",")]
        passes = seq[global_index % len(seq)]
    return MicroPlan(example=example, passes=passes, torch_seed=rng.getrandbits(62))


# ----------------------------------------------------------------------------- the pool
def read_sizes(tsv: Path = SIZE_TSV, column: int = 1) -> dict[str, int]:
    """``{target: int(column)}`` of the size table (rows with a numeric column)."""
    sizes = {}
    for line in tsv.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) > column and parts[column].isdigit():
            sizes[parts[0]] = int(parts[column])
    return sizes


def pick_pool(
    tokens: int = RECIPE_TOKENS,
    count: int = 8,
    sizes: dict[str, int] | None = None,
    spec_dir: Path = SPECS,
    low_fraction: float = 0.7,
    high_fraction: float = 1.0,
    atoms: dict[str, int] | None = None,
    max_atoms: int = 4096,
) -> list[str]:
    """``count`` targets whose size-table tokens lie in a range.

    The range is ``low_fraction .. high_fraction`` x ``tokens``; the targets are
    evenly spread over it (a target above ``tokens`` or with more than ``max_atoms``
    atoms would be cropped by the recipe; below it the batch is padded up to the token
    bucket). Deterministic: sorted by (tokens, name).
    """
    atoms = (read_sizes(column=2) if sizes is None else {}) if atoms is None else atoms
    sizes = read_sizes() if sizes is None else sizes
    low, high = int(low_fraction * tokens), int(high_fraction * tokens)
    rows = sorted(
        (n, name)
        for name, n in sizes.items()
        if low <= n <= high
        and atoms.get(name, 0) <= max_atoms
        and (spec_dir / f"{name}.yaml").exists()
        and name.endswith("-assembly1")
    )
    if len(rows) < count:
        message = (
            f"only {len(rows)} FoldBench targets have {low}..{high} tokens, "
            f"{count} wanted"
        )
        raise ValueError(message)
    step = (len(rows) - 1) / (count - 1) if count > 1 else 0
    return [rows[round(i * step)][1] for i in range(count)]


def parse_pool(
    spec: str, tokens: int = RECIPE_TOKENS, count: int = 8, max_atoms: int = 4096
) -> list[str]:
    """Parse ``--bench-pool``.

    ``default`` (the targets near ``tokens``) or a comma separated list of FoldBench
    targets.
    """
    if spec in ("", "default"):
        return pick_pool(tokens, count, max_atoms=max_atoms)
    return [t.strip() for t in spec.split(",") if t.strip()]


def real_tokens(example: TrainBatch) -> int:
    """Tokens of ``example`` with at least one atom (the padding excluded)."""
    mask = example.features["pred_dense_atom_mask"]
    return int(mask.reshape(mask.shape[0], -1).any(-1).sum())


def standin_example(
    pred: Predictor, target: str, seed: int | None = None
) -> TrainBatch:
    """One example: FoldJet's featurisation of a FoldBench target.

    The ground truth is the model's own pre-training prediction of it (a STAND-IN: the
    real ground-truth atoms are not mapped to the dense layout in phase 0/1). ``seed``
    seeds the torch RNG before the prediction.
    """
    net = pred._network  # noqa: SLF001 - the stand-in wraps the predictor's network
    captured: dict = {}
    original = net.run_trunk

    def wrapped(batch: Any) -> Any:
        captured["batch"] = batch
        out = original(batch)
        captured["trunk"] = out
        return out

    net.run_trunk = wrapped
    try:
        state = pred.run_trunk(ff.load_inputs(SPECS / f"{target}.yaml"))
    finally:
        net.run_trunk = original
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), pred._scope():  # noqa: SLF001 - the predictor's run scope
        samples = net.sample_diffusion(captured["trunk"])["diffusion_samples"][
            "atom_positions"
        ]
    state.close()
    # leave inference mode: autograd needs ordinary tensors
    features = {k: v.clone() for k, v in captured["batch"].items()}
    # the GatherInfo input_shape metadata stays on the host (no sync per micro step)
    features = features_to_device(features, samples.device)
    index, valid = fixed_atom_index(
        features["pred_dense_atom_mask"].cpu(), FIXED_LENGTH
    )
    return TrainBatch(
        features=features,
        atom_positions=samples[0].clone().float(),
        name=target,
        present_index=index.to(samples.device),
        present_valid=valid.to(samples.device),
    )


def build_pool(  # noqa: C901 - two small closures (build / finish) share the arguments
    pred: Predictor,
    targets: Sequence[str],
    info: WorldInfo,
    cache: Path | None = None,
    base_seed: int = 0,
    barrier: Callable[[], object] | None = None,
    log: Callable[..., object] = print,
    max_tokens: int | None = None,
    bucket: BucketSpec | None = None,
) -> list[TrainBatch]:
    """Build (or load) the examples of ``targets`` on the GPU.

    With ``cache`` the pool is built ONCE (rank 0) and loaded by every rank and every
    later run: the stand-in ground truth is then identical for the three backends.
    Without it each process builds its own (the ground truth is the prediction of the
    backend being measured). With ``bucket`` every example is then padded to that ONE
    shape (``train/bucket_batch.py``, in memory after the build / load: the cache file
    keeps the raw examples, so one cache serves the bucketed and the plain pool);
    without it the pool is unchanged.
    """

    def build() -> list[TrainBatch]:
        out = []
        for i, target in enumerate(targets):
            example = standin_example(pred, target, seed=base_seed + i)
            # the recipe would crop it; the stand-in cannot
            if max_tokens is not None and real_tokens(example) > max_tokens:
                log(
                    f"POOL skipped {target} tokens={real_tokens(example)} "
                    f"> {max_tokens}"
                )
                continue
            log(
                f"POOL built {target} tokens={real_tokens(example)} "
                f"padded_tokens={example.atom_positions.shape[0]}"
            )
            out.append(example)
        return out

    def finish(pool: list[TrainBatch]) -> list[TrainBatch]:
        if bucket is None:
            log(f"POOL signatures={len(count_signatures(pool))}")
            return pool
        out = [bucket_example(e, bucket) for e in pool]
        log(
            f"POOL bucketed to {bucket}: signatures {len(count_signatures(pool))} -> "
            f"{len(count_signatures(out))}"
        )
        return out

    device = pred.device
    if cache is None:
        return finish(build())
    if cache.exists():
        requested, loaded = _load_pool(cache, device)
        if requested == list(targets):
            return finish(loaded)
    if info.is_main:
        pool = build()
        _save_pool(pool, cache, list(targets))
    if barrier is not None:
        barrier()
    return finish(_load_pool(cache, device)[1])


def _save_pool(pool: Sequence[TrainBatch], path: Path, requested: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = [
        {
            "name": e.name,
            "features": {k: v.cpu() for k, v in e.features.items()},
            "atom_positions": e.atom_positions.cpu(),
            "atom_mask": None if e.atom_mask is None else e.atom_mask.cpu(),
            "bond_pairs": None if e.bond_pairs is None else e.bond_pairs.cpu(),
        }
        for e in pool
    ]
    tmp = path.with_suffix(".tmp")
    torch.save({"requested": requested, "examples": blob}, tmp)
    tmp.replace(path)


def _load_pool(path: Path, device: torch.device) -> tuple[list[str], list[TrainBatch]]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    blob = saved["examples"]

    def to(t: torch.Tensor | None) -> torch.Tensor | None:
        return None if t is None else t.to(device)

    # sample-fixed present-atom index of the mask the step uses, computed here on the
    # host data
    def fixed(b: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        mask = (
            b["atom_mask"]
            if b["atom_mask"] is not None
            else b["features"]["pred_dense_atom_mask"]
        )
        index, valid = fixed_atom_index(mask, FIXED_LENGTH)
        return index.to(device), valid.to(device)

    pool = []
    for b in blob:
        present_index, present_valid = fixed(b)
        pool.append(
            TrainBatch(
                features=features_to_device(b["features"], device),
                atom_positions=b["atom_positions"].to(device),
                atom_mask=to(b["atom_mask"]),
                bond_pairs=to(b["bond_pairs"]),
                name=b["name"],
                present_index=present_index,
                present_valid=present_valid,
            )
        )
    return saved["requested"], pool


def pool_source(
    pool: Sequence[TrainBatch],
) -> Iterator[TrainBatch]:  # pragma: no cover - trivial
    """Cycle over ``pool`` forever."""
    while True:
        yield from pool


# ---------------------------------------------------------------- engine dispatch table
def engine_table(observations: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``report_observations(model)`` aggregated per engine class.

    Module count, served host calls per backend, refused calls per reason.
    """
    rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "modules": 0,
            "selected": Counter(),
            "served": Counter(),
            "refused": Counter(),
        }
    )
    for module in observations["modules"].values():
        row = rows[module["class"]]
        row["modules"] += 1
        row["selected"][str(module["backend"])] += 1
        row["served"].update(module["served_host_calls"])
        row["refused"].update(module["refused_host_calls"])
    return {
        cls: {k: (dict(v) if isinstance(v, Counter) else v) for k, v in row.items()}
        for cls, row in sorted(rows.items())
    }


def format_table(table: dict[str, dict[str, Any]], title: str = "ENGINE") -> list[str]:
    """One line per class.

    ``selected`` = the backend each module of the class is set to, ``served`` = calls
    that ran on each backend (a refused call is served by pytorch), ``refused`` = calls
    the selected backend refused, by reason.
    """
    lines = [f"{title} class modules selected served refused(reason)"]
    for cls, row in table.items():
        refused = (
            "; ".join(f"{n}x {reason}" for reason, n in sorted(row["refused"].items()))
            or "-"
        )
        lines.append(
            f"{title} {cls:34s} n={row['modules']:<4d} "
            f"selected={_fmt(row['selected'])} served={_fmt(row['served'])} "
            f"refused={refused}"
        )
    return lines


def _fmt(counter: dict[str, int]) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(counter.items())) or "-"


# ------------------------------------------------------------------------------ summary
def summarize_steps(walls: Sequence[float]) -> dict[str, float | None]:
    """Summarise the step wall times.

    Step 1 (first-call overheads) separately; mean / median / min / max over steps
    2..N.
    """
    rest = list(walls[1:])
    return {
        "first_step_s": walls[0] if walls else None,
        "mean_s": statistics.fmean(rest) if rest else None,
        "median_s": statistics.median(rest) if rest else None,
        "min_s": min(rest) if rest else None,
        "max_s": max(rest) if rest else None,
        "total_wall_s": sum(walls),
    }


def git_commit(path: Path | None = None) -> str:
    """HEAD commit of the repository holding ``path`` (``unknown`` on any failure)."""
    here = path or Path(__file__).resolve().parent
    try:
        return subprocess.run(  # noqa: S603 - fixed git command, no user input
            ["git", "-C", str(here), "rev-parse", "HEAD"],  # noqa: S607 - git from PATH
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - informational only
        return "unknown"


def bench_json(payload: dict[str, Any]) -> str:
    """Format ``payload`` as the ``BENCH {json}`` log line."""
    return "BENCH " + json.dumps(payload, default=str, sort_keys=True)
