"""Speed measurement of real training steps (``scripts/train_protenix1.py --bench``): the data pool, the deterministic micro-step plan, the
engine-dispatch table and the step-time summary. Everything here is plain Python / torch so it is unit-tested on the CPU.

The data are STAND-INS: FoldForge's own featurisation of FoldBench targets with the model's own pre-training prediction as ground truth (the
real ground-truth atoms are not mapped to the dense layout; the real loader is MiniWorld's adapter). Step time does not depend on the label
values, it depends on the shapes (tokens, atoms, MSA rows, templates), which the stand-ins have.
"""

from __future__ import annotations

import json
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

import foldforge as ff
from foldforge.train.data import TrainBatch

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from foldforge.train.config import TrainConfig
    from foldforge.train.dist import WorldInfo

SPECS = Path("/home/psk6950/foldforge-scratch/foldbench/specs_all")
#: ``target<TAB>tokens<TAB>atoms<TAB>...`` of every FoldBench target (``foldbench/size_scan.py``: tokens / atoms after FoldForge's featurisation).
SIZE_TSV = Path("/home/psk6950/foldforge-scratch/foldbench/size_sorted.tsv")
#: MiniWorld's recipe crop (configs/miniworld/data/H100_default.yaml: crop.max_tokens 384, max_atoms 4096; phase2a_*_b200_joint: crop L384).
RECIPE_TOKENS = 384


# ---------------------------------------------------------------------------------------------------------------- the deterministic plan
@dataclass(frozen=True)
class MicroPlan:
    example: int  # index into the pool
    passes: int  # trunk passes (recycles) of this micro step
    torch_seed: int  # seed of the torch RNG (noise, rotations, sigma) before the micro step


def global_micro_index(it: int, micro: int, accum: int, rank: int, world: int) -> int:
    """Micro step number over the whole job: rank ``r`` takes ``r, r + W, ...`` of the ``accum * world`` micro steps of optimizer step ``it``."""
    return (it * accum + micro) * world + rank


def micro_plan(base_seed: int, global_index: int, pool_size: int, cfg: TrainConfig) -> MicroPlan:
    """Pool example, recycle count and RNG seed of one micro step: a pure function of ``(base_seed, global_index)``. The three backends and every
    rank draw the same data stream, and a data-parallel run equals the single-process run that takes the same global indices."""
    rng = random.Random(base_seed * 1_000_003 + global_index)
    example = rng.randrange(pool_size)
    passes = rng.randint(cfg.recycles_min, cfg.recycles_max)
    return MicroPlan(example=example, passes=passes, torch_seed=rng.getrandbits(62))


# ---------------------------------------------------------------------------------------------------------------- the pool
def read_sizes(tsv: Path = SIZE_TSV, column: int = 1) -> dict[str, int]:
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
    """``count`` targets whose size-table tokens lie in ``low_fraction .. high_fraction`` x ``tokens``, evenly spread over that range (a target above ``tokens`` or with more than ``max_atoms`` atoms would be
    cropped by the recipe; below it the batch is padded up to the token bucket). Deterministic: sorted by (tokens, name)."""
    atoms = (read_sizes(column=2) if sizes is None else {}) if atoms is None else atoms
    sizes = read_sizes() if sizes is None else sizes
    low, high = int(low_fraction * tokens), int(high_fraction * tokens)
    rows = sorted(
        (n, name)
        for name, n in sizes.items()
        if low <= n <= high and atoms.get(name, 0) <= max_atoms and (spec_dir / f"{name}.yaml").exists() and name.endswith("-assembly1")
    )
    if len(rows) < count:
        message = f"only {len(rows)} FoldBench targets have {low}..{high} tokens, {count} wanted"
        raise ValueError(message)
    step = (len(rows) - 1) / (count - 1) if count > 1 else 0
    return [rows[round(i * step)][1] for i in range(count)]


def parse_pool(spec: str, tokens: int = RECIPE_TOKENS, count: int = 8) -> list[str]:
    """``--bench-pool``: ``default`` (the targets near ``tokens``) or a comma separated list of FoldBench targets."""
    if spec in ("", "default"):
        return pick_pool(tokens, count)
    return [t.strip() for t in spec.split(",") if t.strip()]


def real_tokens(example: TrainBatch) -> int:
    mask = example.features["pred_dense_atom_mask"]
    return int(mask.reshape(mask.shape[0], -1).any(-1).sum())


def standin_example(pred, target: str, seed: int | None = None) -> TrainBatch:
    """One example: FoldForge's featurisation of a FoldBench target; the ground truth is the model's own pre-training prediction of it (a STAND-IN:
    the real ground-truth atoms are not mapped to the dense layout in phase 0/1). ``seed`` seeds the torch RNG before the prediction."""
    net = pred._network
    captured: dict = {}
    original = net.run_trunk

    def wrapped(batch):
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
    with torch.no_grad(), pred._scope():
        samples = net.sample_diffusion(captured["trunk"])["diffusion_samples"]["atom_positions"]
    state.close()
    features = {k: v.clone() for k, v in captured["batch"].items()}  # leave inference mode: autograd needs ordinary tensors
    return TrainBatch(features=features, atom_positions=samples[0].clone().float(), name=target)


def build_pool(
    pred,
    targets: Sequence[str],
    info: WorldInfo,
    cache: Path | None = None,
    base_seed: int = 0,
    barrier=None,
    log=print,
    max_tokens: int | None = None,
) -> list[TrainBatch]:
    """The examples of ``targets`` on the GPU. With ``cache`` the pool is built ONCE (rank 0) and loaded by every rank and every later run: the
    stand-in ground truth is then identical for the three backends. Without it each process builds its own (the ground truth is the
    prediction of the backend being measured)."""

    def build() -> list[TrainBatch]:
        out = []
        for i, target in enumerate(targets):
            example = standin_example(pred, target, seed=base_seed + i)
            if max_tokens is not None and real_tokens(example) > max_tokens:  # the recipe would crop it; the stand-in cannot
                log(f"POOL skipped {target} tokens={real_tokens(example)} > {max_tokens}")
                continue
            log(f"POOL built {target} tokens={real_tokens(example)} padded_tokens={example.atom_positions.shape[0]}")
            out.append(example)
        return out

    device = pred.device
    if cache is None:
        return build()
    if cache.exists():
        requested, loaded = _load_pool(cache, device)
        if requested == list(targets):
            return loaded
    if info.is_main:
        pool = build()
        _save_pool(pool, cache, list(targets))
    if barrier is not None:
        barrier()
    return _load_pool(cache, device)[1]


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

    def to(t):
        return None if t is None else t.to(device)

    return saved["requested"], [
        TrainBatch(
            features={k: v.to(device) for k, v in b["features"].items()},
            atom_positions=b["atom_positions"].to(device),
            atom_mask=to(b["atom_mask"]),
            bond_pairs=to(b["bond_pairs"]),
            name=b["name"],
        )
        for b in blob
    ]


def pool_source(pool: Sequence[TrainBatch]) -> Iterator[TrainBatch]:  # pragma: no cover - trivial
    while True:
        yield from pool


# ---------------------------------------------------------------------------------------------------------------- engine dispatch table
def engine_table(observations: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``report_observations(model)`` aggregated per engine class: module count, served host calls per backend, refused calls per reason."""
    rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"modules": 0, "selected": Counter(), "served": Counter(), "refused": Counter()}
    )
    for module in observations["modules"].values():
        row = rows[module["class"]]
        row["modules"] += 1
        row["selected"][str(module["backend"])] += 1
        row["served"].update(module["served_host_calls"])
        row["refused"].update(module["refused_host_calls"])
    return {
        cls: {k: (dict(v) if isinstance(v, Counter) else v) for k, v in row.items()} for cls, row in sorted(rows.items())
    }


def format_table(table: dict[str, dict[str, Any]], title: str = "ENGINE") -> list[str]:
    """One line per class: ``selected`` = the backend each module of the class is set to, ``served`` = calls that ran on each backend (a refused
    call is served by pytorch), ``refused`` = calls the selected backend refused, by reason."""
    lines = [f"{title} class modules selected served refused(reason)"]
    for cls, row in table.items():
        refused = "; ".join(f"{n}x {reason}" for reason, n in sorted(row["refused"].items())) or "-"
        lines.append(
            f"{title} {cls:34s} n={row['modules']:<4d} selected={_fmt(row['selected'])} served={_fmt(row['served'])} refused={refused}"
        )
    return lines


def _fmt(counter: dict[str, int]) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(counter.items())) or "-"


# ---------------------------------------------------------------------------------------------------------------- summary
def summarize_steps(walls: Sequence[float]) -> dict[str, float | None]:
    """Step 1 (first-call overheads) separately; mean / median / min / max over steps 2..N."""
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
    here = path or Path(__file__).resolve().parent
    try:
        return subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - informational only
        return "unknown"


def bench_json(payload: dict[str, Any]) -> str:
    return "BENCH " + json.dumps(payload, default=str, sort_keys=True)
