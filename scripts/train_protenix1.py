r"""Protenix v1 finetuning on the FoldJet graph (MiniWorld recipe).

Phase 0/1: eager steps, one GPU, gradient accumulation.

  # smoke: one fixed example (FoldBench target) overfit + gradient coverage + kernel
  # table + checkpoint round trip
  python scripts/train_protenix1.py --smoke --target 8ayg-assembly1 --steps 50 \
      --warmup-steps 0 --run-dir RUN

  # training from a data source (MiniWorld provides the adapter; FoldJet does not
  # import MiniWorld)
  python scripts/train_protenix1.py --batch-source my_pkg.adapter:batches \
      --steps 100000 --run-dir RUN [--ckpt X.miniworld.pt]

  # speed measurement of a backend (pytorch / cueq / miniworld): 30 real optimizer
  # steps, effective batch 256 (8 GPUs: torchrun --nproc-per-node 8);
  # docs/guides/TRAINING.md
  python scripts/train_protenix1.py --bench --backend cueq --effective-batch 256 \
      --steps 30 --warmup-steps 10 --bench-pool-cache POOL.pt

The recipe (loss weights, Adam, clip, EMA, recycles, augmentations) is
:class:`foldjet.train.config.TrainConfig`; read the lr warning there.
"""

# ruff: noqa: INP001, T201 - standalone script, print logging

from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

import foldjet as ff
from foldjet.modules.engine.configure import activation_autocast
from foldjet.train import (
    TrainBatch,
    TrainConfig,
    bench,
    bucket_batch,
    checkpoint,
    data,
    dist,
    freeze,
    fused_opt,
    optim,
    prepare,
    recipes,
    step,
)
from foldjet.train import (
    compile as train_compile,
)
from foldjet.train.profile import PhaseTimer, tensor_bytes

if TYPE_CHECKING:
    from collections.abc import Iterator

SPECS = Path("/home/psk6950/foldforge-scratch/foldbench/specs_all")

#: |loss| above this counts as diverged (and as non-finite in --bench)
_DIVERGED_LOSS = 1e6


def smoke_batches(pred: Any, target: str) -> Iterator[TrainBatch]:
    """Yield one fixed example: FoldJet's own featurisation of a FoldBench target.

    The ground truth is the model's own pre-training prediction of it (a STAND-IN:
    the real ground-truth atoms are not mapped to the dense layout in phase 0/1).
    """
    example = bench.standin_example(pred, target)
    while True:
        yield example


def split_sizes(
    model: torch.nn.Module,
    example: TrainBatch,
    cfg: TrainConfig,  # noqa: ARG001 - kept for the call signature
) -> dict:
    """Bytes a 2-GPU split of the step would move (the mini-rollout on a second GPU).

    The diffusion-head parameters (re-synced after every optimizer step), the trunk
    outputs the rollout reads (single / pair / target features, per crop), the
    predicted coordinates back.
    """
    features = {
        k: (v.float() if v.dtype == torch.float64 else v)
        for k, v in example.features.items()
    }
    with torch.no_grad(), activation_autocast(model):
        batch_data = model._prepare_batch(features)  # noqa: SLF001 - the model's own batch preparation
        emb = step.run_trunk_train(model, batch_data, 1)
    head = model.diffusion_head
    res, atoms, _ = example.atom_positions.shape
    return {
        "num_res": int(res),
        "atoms_per_token": int(atoms),
        "diffusion_head_params": sum(p.numel() for p in head.parameters()),
        "diffusion_head_param_bytes_fp32": sum(
            p.numel() * p.element_size() for p in head.parameters()
        ),
        "single_bytes": tensor_bytes(emb["single"]),
        "pair_bytes": tensor_bytes(emb["pair"]),
        "target_feat_bytes": tensor_bytes(emb["target_feat"]),
        "rollout_inputs_bytes": tensor_bytes(emb),
        "x_mini_bytes_fp32": res * atoms * 3 * 4,
        "pair_dtype": str(emb["pair"].dtype),
        "single_dtype": str(emb["single"].dtype),
    }


#: columns of a :func:`grad_coverage` row
COVERAGE_COLUMNS = (
    "total",
    "none",
    "zero_live",
    "nonfinite",
    "underflow",
    "data_dependent",
    "frozen",
)


def grad_coverage(model: torch.nn.Module) -> dict:
    """Count which parameters got a gradient from one step.

    Groups by the first three name components; the confidence head is in the list
    (single-stage training; only ``--no-confidence`` leaves it without a loss). A
    parameter is classified by ``train/freeze.py``: frozen (no path to the loss /
    not in the original / fixed in the original: not counted as a defect),
    ``data_dependent`` (needs charged atoms / bonds in the example: smoke 7fvu),
    ``underflow`` (a decayed module: the gradient is exactly zero because the
    weights are ~1e-37..1e-12 relative to the model's typical weight) and the real
    defects ``none`` (no gradient), ``zero_live`` (zero gradient on a live weight)
    and ``nonfinite``.
    """
    scale = freeze.weight_scale(model)
    groups: dict[str, list[int]] = defaultdict(lambda: [0] * len(COVERAGE_COLUMNS))
    column = {name: i for i, name in enumerate(COVERAGE_COLUMNS)}
    for name, param in model.named_parameters():
        row = groups[".".join(name.split(".")[:3])]
        row[column["total"]] += 1
        if not param.requires_grad:
            row[column["frozen"]] += 1
            continue
        kind = freeze.classify_gradient(name, param, scale)
        if kind != "ok":
            row[column[{"zero": "zero_live"}.get(kind, kind)]] += 1
    return dict(groups)


def data_dependent_flow(model: torch.nn.Module) -> dict[str, bool]:
    """Report, per data-dependent parameter, whether a nonzero gradient reached it.

    Data-dependent: the charge and bond embeddings; checked on this example.
    """
    flow = {}
    for name, param in model.named_parameters():
        if freeze.DATA_DEPENDENT.fullmatch(name):
            flow[name] = param.grad is not None and bool(param.grad.any())
    return flow


def kernel_table(
    model: torch.nn.Module, example: TrainBatch, cfg: TrainConfig, top: int = 30
) -> list[tuple[str, float, int]]:
    """CUDA kernels of one forward + backward by total GPU time.

    Engine kernels versus plain aten GEMMs / elementwise (= the pytorch path).
    """
    from torch.profiler import (  # noqa: PLC0415 - only the --smoke kernel table profiles
        ProfilerActivity,
        profile,
    )

    model.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step.micro_step(model, example, cfg, passes=1)
        torch.cuda.synchronize()
    rows = [
        (e.key, e.device_time_total / 1000.0, e.count)
        for e in prof.key_averages()
        if e.device_time_total > 0
    ]
    rows.sort(key=lambda r: -r[1])
    return rows[:top]


def load_summary(model: torch.nn.Module, backend: str) -> dict:
    """Summarise what the loader chose.

    The plan (backend per op, fallbacks) and every engine module that does NOT run
    the requested backend, with the reason.
    """
    report = getattr(model, "foldjet_load_report", {}) or {}
    plan = report.get("backends", {})
    modules = (report.get("engine_backends") or {}).get("modules", {})
    off = {}
    for _name, row in modules.items() if isinstance(modules, dict) else []:
        if isinstance(row, dict) and row.get("reason"):
            key = f"{row.get('class')}:{row.get('op')}"
            off.setdefault(
                key,
                {
                    "modules": 0,
                    "backend": row.get("backend"),
                    "requested": row.get("requested"),
                    "reason": row.get("reason"),
                },
            )["modules"] += 1
    return {"backend": backend, "plan": plan, "modules_not_on_requested_backend": off}


def main() -> None:  # noqa: C901, PLR0912, PLR0915 - one linear training driver
    """Parse the flags, load the model and run the phase-0 checks and the training."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--target", default="8ayg-assembly1")
    ap.add_argument("--model", default="protenix1")
    ap.add_argument(
        "--batch-source",
        default=None,
        help="module:callable(cfg, device) -> iterator of TrainBatch",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="optimizer steps (default 50; 30 with --bench)",
    )
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--num-augment", type=int, default=None)
    ap.add_argument("--max-lr", type=float, default=None)
    ap.add_argument("--min-lr", type=float, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--trunk-lr-scale", type=float, default=None)
    ap.add_argument(
        "--recipe",
        choices=recipes.RECIPES,
        default="miniworld",
        help="miniworld (default, TrainConfig as it is) or protenix: Protenix v1 "
        "fine-tuning recipe (train/recipes.py: grad clip 10, Adam wd 1e-8, lr 1e-3 "
        "warmup 2000 af3 decay, loss weights, EMA 0.999); explicit flags below still "
        "override it",
    )
    ap.add_argument(
        "--precision",
        choices=("bf16", "af3_default"),
        default=None,
        help="model precision (default: the recipe's; protenix -> af3_default, "
        "miniworld -> bf16). af3_default: bf16 trunk on fp32 masters, the diffusion "
        "head in fp32 (AF3 / Protenix); bf16: everything under the bf16 autocast",
    )
    ap.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="global grad-norm clip (default: the recipe, miniworld 1.0 / protenix 10)",
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Adam weight decay (default: the recipe, miniworld 0 / protenix 1e-8)",
    )
    ap.add_argument(
        "--no-confidence",
        action="store_true",
        help="DEBUG ONLY: skip the confidence head (the finetuning is one stage; "
        "every parameter trains)",
    )
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument(
        "--profile-phases",
        action="store_true",
        help="per-phase GPU timing (CUDA events) of every step: trunk fwd per "
        "recycle, diffusion fwd / bwd, trunk bwd, mini-rollout, confidence fwd / bwd, "
        "optimizer + EMA",
    )
    ap.add_argument(
        "--rollout-order",
        choices=("after", "before", "side", "graph"),
        default=None,
        help="where the mini-rollout runs: after the trunk backward (default, least "
        "memory), before the diffusion forward, or on a second CUDA stream (side), or "
        "graph: the 20-step rollout as ONE CUDA graph per shape, replayed after the "
        "trunk backward (train/rollout_graph.py; the noise is drawn inside the graph "
        "from the default CUDA generator; measured on the 5I28-size pool the losses "
        "and parameters are bitwise those of --rollout-order after)",
    )
    ap.add_argument(
        "--augment-chunk",
        type=int,
        default=None,
        help="diffusion augmentations per forward/backward chunk (0 = all)",
    )
    ap.add_argument(
        "--recycles", type=int, default=None, help="fix the trunk passes (min = max)"
    )
    ap.add_argument(
        "--only",
        choices=("trunk", "diffusion", "rollout", "confidence"),
        default=None,
        help="MEASUREMENT: run one part of the step alone",
    )
    ap.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="recompute the pair-stack blocks in the backward instead of storing "
        "their activations",
    )
    ap.add_argument(
        "--ckpt-policy",
        default="all",
        help="which blocks --activation-checkpointing wraps: all (default) | none | "
        "pairformer (trunk only) | evoformer | except:<trunk|msa|template|confidence> "
        "| every:N / first:K / last:K (those trunk pairformer blocks plus every "
        "non-trunk block); 'none' also switches checkpointing off",
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help="start from this .miniworld.pt (default: the registered model's "
        "checkpoint)",
    )
    ap.add_argument("--run-dir", default="runs/train_protenix1")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument(
        "--keep-last",
        type=int,
        default=3,
        help="--save-every: keep the newest N step checkpoints (and every "
        "--keep-every-th step); 0 keeps all",
    )
    ap.add_argument(
        "--keep-every",
        type=int,
        default=500,
        help="--save-every: steps that are multiples of this are never pruned",
    )
    ap.add_argument(
        "--resume",
        default=None,
        help="continue from this step<N>.miniworld.pt (+ its .train.pt: optimizer, "
        "EMA, step); --steps is then the TOTAL step count",
    )
    ap.add_argument(
        "--epoch-samples",
        type=int,
        default=25600,
        help="samples per epoch (MiniWorld train.train_item, the composed v200 "
        "recipe: 25600): steps per epoch = this / effective batch",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="TOTAL epochs (replaces --steps: steps = epochs x steps per epoch); "
        "with --resume this is also the total",
    )
    ap.add_argument(
        "--save-every-epochs",
        type=int,
        default=0,
        help="save epoch<E>.miniworld.pt (+ .train.pt) at the end of every K-th "
        "epoch; never pruned (MiniWorld save_freq 5)",
    )
    ap.add_argument(
        "--save-last",
        action="store_true",
        help="at the end of EVERY epoch overwrite last.miniworld.pt (+ .train.pt, "
        "last.step), like MiniWorld's last.pt; resume picks the newer of it and the "
        "epoch<E> checkpoints",
    )
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument(
        "--no-save",
        action="store_true",
        help="skip the final checkpoint (measurement runs)",
    )
    ap.add_argument(
        "--backend",
        choices=("pytorch", "cueq", "miniworld"),
        default="miniworld",
        help="engine backend of every module: pytorch (reference), cueq (NVIDIA "
        "cuEquivariance where it has the op, the rest falls back to pytorch), "
        "miniworld (our kernels: cuda per module class on H100)",
    )
    ap.add_argument(
        "--no-activation-checkpointing",
        action="store_true",
        help="--bench turns --activation-checkpointing on (pytorch and cueq OOM an "
        "80 GB H100 at 384 tokens without it); this keeps it off",
    )
    ap.add_argument(
        "--effective-batch",
        type=int,
        default=None,
        help="micro batch 1 x grad_accum x world size; sets grad_accum = N / world "
        "size (error if not divisible)",
    )
    ap.add_argument(
        "--dist-backend",
        default="nccl",
        help="torch.distributed backend under torchrun (gloo: two processes on one "
        "GPU, for tests)",
    )
    ap.add_argument(
        "--seed", type=int, default=None, help="base seed (default: TrainConfig.seed)"
    )
    ap.add_argument(
        "--fused-opt",
        nargs="?",
        const="fused",
        default=None,
        choices=("foreach", "fused", "triton"),
        help="clip + Adam + EMA as multi-tensor launches (train/fused_opt.py; "
        "capturable Adam, device lr); default: the per-tensor path of train/optim.py",
    )
    ap.add_argument(
        "--bench",
        action="store_true",
        help="speed measurement of --steps real optimizer steps (implies --no-save "
        "--no-profile; no phase-0 micro step / coverage / kernel table)",
    )
    ap.add_argument(
        "--bench-pool",
        default="default",
        help="'default' (targets near --bench-tokens) or comma separated FoldBench "
        "targets; STAND-IN examples",
    )
    ap.add_argument(
        "--bench-tokens",
        type=int,
        default=bench.RECIPE_TOKENS,
        help="token count of the default pool (MiniWorld recipe crop: 384)",
    )
    ap.add_argument(
        "--micro-log",
        action="store_true",
        help="--bench: print one MICRO line per micro step on every rank (global "
        "index, example, recycles, loss) to compare runs micro step by micro step",
    )
    ap.add_argument(
        "--bench-dump",
        default=None,
        help="--bench: directory to write the step-0 gradients (grads_step0.pt) and "
        "the final parameters (params.pt) of rank 0, for run-to-run / eager-vs-compile "
        "comparison",
    )
    ap.add_argument(
        "--bench-pool-count", type=int, default=8, help="targets of the default pool"
    )
    ap.add_argument(
        "--bench-pool-cache",
        default=None,
        help="build the pool once (rank 0) into this file and load it everywhere: "
        "identical stand-in labels for every backend",
    )
    ap.add_argument(
        "--step-graph",
        action="store_true",
        help="the WHOLE micro step (trunk recycles + diffusion + confidence + "
        "backward, gradient accumulation into persistent .grad) as ONE CUDA graph per "
        "(recycles, shape signature), replayed (train/step_graph.py); the mini-rollout "
        "is recorded inline; needs --bench-bucket (one shape) to stay at <= 4 graphs; "
        "default off",
    )
    ap.add_argument(
        "--compile",
        nargs="?",
        const="all",
        default=None,
        choices=tuple(train_compile.REGIONS),
        help="torch.compile(dynamic=False) of the pair-stack / diffusion blocks "
        "(train/compile.py); composes with --step-graph and "
        "--activation-checkpointing; default off (eager)",
    )
    ap.add_argument(
        "--msa-rows",
        type=int,
        default=None,
        help="MSA rows drawn per recycle (default: the model's 1024; MiniWorld "
        "recipe bucket_msa_size: 128)",
    )
    ap.add_argument(
        "--bench-bucket",
        action="store_true",
        help="pad every example to ONE shape (train/bucket_batch.py: --bench-tokens "
        "tokens, --bucket-atoms atoms, --bucket-msa-pool raw MSA rows, "
        "--bucket-templates), so a CUDA graph of the micro step has one signature; "
        "also applies to --batch-source streams; default off",
    )
    ap.add_argument(
        "--bucket-atoms",
        type=int,
        default=4096,
        help="--bench-bucket: flat atoms (MiniWorld crop.max_atoms 4096)",
    )
    ap.add_argument(
        "--bucket-msa-pool",
        type=int,
        default=16384,
        help="--bench-bucket: raw MSA pool rows before the per-recycle draw (16384 = "
        "the largest FoldJet MSA bucket: nothing is cut)",
    )
    ap.add_argument(
        "--bucket-templates", type=int, default=4, help="--bench-bucket: templates"
    )
    args = ap.parse_args()

    world = dist.world_info()
    if args.bench:
        args.no_save = args.no_profile = True
        if args.steps is None:
            args.steps = 30
        if args.warmup_steps is None:
            args.warmup_steps = 10
    if args.steps is None:
        args.steps = 50

    def say(*a: object, **k: Any) -> None:  # only rank 0 prints
        if world.is_main:
            print(*a, **k)

    # before the flags: an explicit flag overrides the recipe value
    cfg = recipes.apply(args.recipe, TrainConfig())
    if args.effective_batch is not None:
        if args.grad_accum is not None:
            msg = "--effective-batch and --grad-accum are exclusive"
            raise SystemExit(msg)
        try:
            args.grad_accum = dist.accum_for_effective_batch(
                args.effective_batch, world.world_size
            )
        except ValueError as error:
            raise SystemExit(str(error)) from None
    if args.bench and not args.no_activation_checkpointing:
        args.activation_checkpointing = True
    if args.bench and args.augment_chunk is None:
        # No chunking: the recipe runs all augmentations in one forward / backward.
        # (An earlier --bench default of 8 was never asked for; it made eager ~60%
        # slower at 384 tokens.) With activation checkpointing (the --bench default)
        # 48 augmentations at 384 peak 36 GiB on miniworld.
        args.augment_chunk = 0
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    changes = {"grad_accum_steps": 1 if args.smoke else cfg.grad_accum_steps}
    for key, value in (
        ("grad_accum_steps", args.grad_accum),
        ("num_augment", args.num_augment),
        ("max_lr", args.max_lr),
        ("min_lr", args.min_lr),
        ("warmup_steps", args.warmup_steps),
        ("trunk_lr_scale", args.trunk_lr_scale),
        ("grad_clip_max_norm", args.grad_clip),
        ("weight_decay", args.weight_decay),
        ("rollout_steps", args.rollout_steps),
        ("rollout", args.rollout_order),
        ("step_graph", True if args.step_graph else None),
        ("compile", args.compile),
        ("augment_chunk", args.augment_chunk),
        ("only", args.only),
        ("recycles_min", args.recycles),
        ("recycles_max", args.recycles),
    ):
        if value is not None:
            changes[key] = value  # noqa: PERF403 - kept as the loop over the flag table
    if args.no_confidence:
        changes["confidence"] = False
    cfg = replace(cfg, **changes)
    effective = cfg.grad_accum_steps * world.world_size
    if args.epoch_samples % effective:
        msg = (
            f"--epoch-samples {args.epoch_samples} is not a multiple of the effective "
            f"batch {effective}"
        )
        raise SystemExit(msg)
    steps_per_epoch = args.epoch_samples // effective
    # --epochs replaces --steps (the default of 50 is only a placeholder)
    if args.epochs is not None:
        args.steps = args.epochs * steps_per_epoch
    run_dir = Path(args.run_dir)
    if world.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    dist.init_process_group(world, args.dist_backend)
    # legacy streams (smoke / --batch-source): rank r seeds seed + r (rank 0 = the
    # single-process run); --bench draws every micro step from
    # bench.micro_plan(seed, global micro index) instead, which does not depend on the
    # order or the rank count
    rng = prepare.seed_everything(cfg.seed + world.rank)

    # the diffusion head runs in fp32 in the Protenix recipe (as AF3 / Protenix train)
    precision = args.precision or (
        "af3_default" if args.recipe == "protenix" else "bf16"
    )
    say("PRECISION", precision, flush=True)
    # dead modules must run in training (their parameters train): do not flag them at
    # load
    pred = ff.load_model(
        args.model,
        config={
            "backend": args.backend,
            "precision": precision,
            "engine": {"skip_dead_modules": False, "unsupported": "fallback"},
        },  # atom_blocks: engine default (2.5.3 fixed the sm90 atom-DiT backward, R20)
        checkpoint=args.ckpt,
        device=None
        if world.world_size == 1
        else torch.device("cuda", torch.cuda.current_device()),
    )
    model = pred._network  # noqa: SLF001 - the predictor's network is the model trained
    device = pred.device
    say(
        "LOAD_REPORT",
        json.dumps(load_summary(model, args.backend), default=str),
        flush=True,
    )
    say(
        "CONFIG",
        json.dumps(
            {
                k: (v if not hasattr(v, "__dataclass_fields__") else v.__dict__)
                for k, v in cfg.__dict__.items()
            },
            default=str,
        ),
        flush=True,
    )

    if args.bench_bucket:
        # present-atom index of one length for every example (else 512-multiples per
        # example split the step-graph key)
        bench.FIXED_LENGTH = max(bench.FIXED_LENGTH, args.bucket_atoms)
    bucket_spec = (
        bucket_batch.BucketSpec(
            tokens=args.bench_tokens,
            atoms=args.bucket_atoms,
            msa_pool_rows=args.bucket_msa_pool,
            templates=args.bucket_templates,
        )
        if args.bench_bucket
        else None
    )
    pool: list[TrainBatch] = []
    if args.bench:
        targets = bench.parse_pool(
            args.bench_pool,
            args.bench_tokens,
            args.bench_pool_count,
            max_atoms=args.bucket_atoms,
        )
        t_pool = time.time()
        pool = bench.build_pool(
            pred,
            targets,
            world,
            cache=Path(args.bench_pool_cache) if args.bench_pool_cache else None,
            base_seed=cfg.seed,
            barrier=lambda: dist.barrier(world),
            max_tokens=args.bench_tokens if args.bench_pool == "default" else None,
            bucket=bucket_spec,
            log=say,
        )
        say(
            f"POOL targets={','.join(targets)} "
            f"tokens={[bench.real_tokens(e) for e in pool]} "
            f"build_s={time.time() - t_pool:.1f}",
            flush=True,
        )
        source = iter(())
        first = pool[0]
    elif args.smoke:
        source = smoke_batches(pred, args.target)
        first = next(source)
    elif args.batch_source:
        source = iter(data.load_source(args.batch_source)(cfg, device))
        if bucket_spec is not None:
            source = bucket_batch.bucket_source(source, bucket_spec)
        first = next(source)
    else:
        msg = "give --smoke, --bench or --batch-source module:callable"
        raise SystemExit(msg)
    say(
        "INFO",
        json.dumps(
            {
                "num_res": int(first.atom_positions.shape[0]),
                "atoms_per_token": int(first.atom_positions.shape[1]),
                "target": first.name,
            }
        ),
        flush=True,
    )
    real_atoms = int(first.features["pred_dense_atom_mask"].sum())
    resolved_atoms = int(
        (
            first.atom_mask
            if first.atom_mask is not None
            else first.features["pred_dense_atom_mask"]
        ).sum()
    )
    say(
        f"ATOMS first example: real={real_atoms:d} resolved={resolved_atoms:d} (the "
        "experimentally-resolved loss classifies the real ones)",
        flush=True,
    )
    prep_info = prepare.enable_training(model, cfg)
    if args.msa_rows is not None:
        # MSA rows per recycle (the recipe: bucket_msa_size 128; FoldJet default
        # 1024)
        model.evoformer.num_msa = args.msa_rows
    dist.broadcast_parameters(
        list(model.parameters())
        + [b for b in model.buffers() if b.is_floating_point()],
        world,
    )
    if cfg.compile:
        say(
            f"COMPILE blocks={train_compile.enable_compile(model, cfg.compile):d} "
            f"region={cfg.compile}",
            flush=True,
        )  # before the checkpoint wrapper: it recomputes the compiled forward
    if args.activation_checkpointing:
        blocks = prepare.enable_activation_checkpointing(model, args.ckpt_policy)
        say(f"ACT_CKPT policy={args.ckpt_policy} blocks={blocks:d}", flush=True)
    trunk, heads = optim.split_parameters(model)
    say(
        "PREPARE",
        json.dumps(
            {
                **prep_info,
                "params_trunk": sum(p.numel() for _, p in trunk),
                "params_heads": sum(p.numel() for _, p in heads),
            }
        ),
        flush=True,
    )
    by_rule: dict[str, list[str]] = defaultdict(list)
    for frozen_name, rule in freeze.frozen_parameters(model, cfg.loss).items():
        by_rule[rule.key].append(frozen_name)
    for rule_key, names in sorted(by_rule.items()):
        say("FROZEN", rule_key, len(names), "e.g.", names[0], flush=True)

    if args.profile_phases:
        say("SIZES", json.dumps(split_sizes(model, first, cfg)), flush=True)

    # -- phase 0: one micro step -> gradient coverage (+ kernels)
    if not args.bench:
        model.zero_grad(set_to_none=True)
        parts = step.micro_step(model, first, cfg, passes=cfg.recycles_max)
        say("FIRST_STEP", json.dumps(parts), flush=True)
        cover = grad_coverage(model)
        column = {name: i for i, name in enumerate(COVERAGE_COLUMNS)}
        defects = ("none", "zero_live", "nonfinite")
        in_loss = lambda k: cfg.confidence or not k.startswith("confidence_head")
        bad = {
            k: v
            for k, v in cover.items()
            if any(v[column[c]] for c in defects) and in_loss(k)
        }
        say(
            f"GRAD_COVERAGE groups={len(cover):d}  groups_with_defects={len(bad):d}  "
            "(defect = no gradient / zero gradient on a live weight / nonfinite; "
            "frozen, data-dependent and underflow are classified, not defects)",
            flush=True,
        )
        sums = [0] * len(COVERAGE_COLUMNS)
        for key, row in sorted(cover.items()):
            if in_loss(key):
                sums = [a + b for a, b in zip(sums, row, strict=False)]
            defect = any(row[column[c]] for c in defects)
            flag = (
                ""
                if not defect
                else (
                    "  <-- expected (no loss: --no-confidence)"
                    if not in_loss(key)
                    else "  <-- CHECK"
                )
            )
            say(
                f"GRAD {key:60s} "
                + " ".join(
                    f"{c}={v:4d}" for c, v in zip(COVERAGE_COLUMNS, row, strict=False)
                )
                + flag,
                flush=True,
            )
        say(
            "GRAD_SUM "
            + " ".join(
                f"{c}={v}" for c, v in zip(COVERAGE_COLUMNS, sums, strict=False)
            ),
            flush=True,
        )
        for name, flowing in sorted(data_dependent_flow(model).items()):
            say(f"DATA_DEPENDENT {name} gradient_flows={flowing}", flush=True)
        tmask = first.features.get("template_atom_mask")
        if tmask is not None:
            real_templates = int(tmask.reshape(tmask.shape[0], -1).any(-1).sum())
            real_template_atoms = int(tmask.sum())
            say(
                f"TEMPLATES real_templates={real_templates:d} "
                f"real_atoms={real_template_atoms:d}",
                flush=True,
            )
        if args.smoke and not args.no_profile:
            for name, ms, count in kernel_table(model, first, cfg):
                say(f"KERNEL {ms:10.2f} ms  x{count:<6d} {name[:150]}", flush=True)

    # -- training loop
    fopt = fused_opt.build_fused(model, cfg, args.fused_opt) if args.fused_opt else None
    optimizer = fopt.optimizer if fopt else optim.build_optimizer(model, cfg)
    ema = (
        fopt.ema if fopt else (optim.EMA(model, cfg.ema_decay) if cfg.use_ema else None)
    )
    params = [p for g in optimizer.param_groups for p in g["params"]]
    start_step = 0
    if args.resume:
        start_step = checkpoint.resume(
            model, Path(args.resume), optimizer=optimizer, ema=ema
        )
        say("RESUME", args.resume, "step", start_step, flush=True)
    history: list[float] = []
    bench_walls: list[float] = []
    bench_nonfinite = 0
    bench_micro = 0
    bench_tokens = 0
    bench_peak = [0.0, 0.0]
    diverged_printed = False
    accum = cfg.grad_accum_steps
    # the SUM over ranks of the scaled gradients is the mean over accum x world micro
    # steps
    micro_scale = 1.0 / (accum * world.world_size)
    from foldjet.modules.engine.serving import (  # noqa: PLC0415 - imported after the model load, as before
        report_observations,
        reset_observations,
    )

    # the dispatch table counts the training steps only (not the load / pool build)
    reset_observations(model)
    if args.bench_dump and world.is_main:
        Path(args.bench_dump).mkdir(parents=True, exist_ok=True)
        torch.save(
            {n: p.detach().cpu() for n, p in model.named_parameters()},
            Path(args.bench_dump) / "params_init.pt",
        )
    compile_warmup_s = 0.0
    # compile every variant (no-grad recycles, grad + checkpoint recompute, 48-sample
    # diffusion, rollout) before the timed steps; a step graph warms up in its capture
    if cfg.compile and not cfg.step_graph:
        t_warm = time.time()
        step.micro_step(model, first, cfg, passes=cfg.recycles_max, scale=0.0)
        torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=True)
        compile_warmup_s = time.time() - t_warm
        say(f"COMPILE_WARMUP seconds={compile_warmup_s:.1f}", flush=True)
    t0 = time.time()
    guard = contextlib.ExitStack()
    # nothing compiles after the warm-up (a step graph checks its own captures)
    if cfg.compile and args.bench and not cfg.step_graph:
        guard.enter_context(train_compile.strict())
    for it in range(start_step, args.steps):
        lr = fopt.set_lr(it, cfg) if fopt else optim.set_lr(optimizer, it, cfg)
        # a step graph accumulates into the SAME .grad tensors: zeroed in place
        optimizer.zero_grad(set_to_none=not cfg.step_graph)
        totals = defaultdict(float)
        timer = PhaseTimer(args.profile_phases)
        if args.profile_phases or args.bench:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t_step = time.time()
        for micro in range(accum):
            if args.bench:
                plan = bench.micro_plan(
                    cfg.seed,
                    bench.global_micro_index(
                        it, micro, accum, world.rank, world.world_size
                    ),
                    len(pool),
                    cfg,
                )
                torch.manual_seed(plan.torch_seed)
                example, passes = pool[plan.example], plan.passes
                bench_tokens += bench.real_tokens(example)
            else:
                example = first if micro == 0 and it == 0 else next(source)
                passes = prepare.draw_passes(cfg, rng)
            out = step.micro_step(
                model,
                example,
                cfg,
                passes=passes,
                scale=micro_scale,
                timer=timer,
            )
            bench_micro += 1
            if args.bench and args.micro_log:
                g = bench.global_micro_index(
                    it, micro, accum, world.rank, world.world_size
                )
                print(
                    f"MICRO rank={world.rank:d} g={g:d} example={example.name} "
                    f"passes={passes:d} loss={out['loss']:.7f} "
                    f"diffusion={out['diffusion']:.7f}",
                    flush=True,
                )
            for k, v in out.items():
                totals[k] += v / accum
            if it == 0 and micro < 2:  # noqa: PLR2004 - the first two micro steps
                for line in bench.format_table(
                    bench.engine_table(report_observations(model)),
                    f"ENGINE_AFTER_MICRO{micro + 1}",
                ):
                    say(line, flush=True)
        with timer.phase("allreduce"):
            dist.allreduce_gradients(params, world)
            if args.bench_dump and it == 0 and world.is_main:
                Path(args.bench_dump).mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        n: p.grad.detach().cpu()
                        for n, p in model.named_parameters()
                        if p.grad is not None
                    },
                    Path(args.bench_dump) / "grads_step0.pt",
                )
        with timer.phase("optimizer_ema"):
            if fopt is not None:
                norm_t = fopt.step()
            else:
                norm_t = optim.clip_grad_norm(params, cfg.grad_clip_max_norm)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
        norm = float(norm_t)
        totals = defaultdict(float, dist.mean_over_ranks(totals, world, device))
        if args.bench:
            torch.cuda.synchronize()
            wall = time.time() - t_step
            bench_walls.append(wall)
            peak, reserved = (
                torch.cuda.max_memory_allocated() / 2**30,
                torch.cuda.max_memory_reserved() / 2**30,
            )
            bench_peak = [max(bench_peak[0], peak), max(bench_peak[1], reserved)]
            finite = (
                totals["loss"] == totals["loss"]
                and abs(totals["loss"]) < _DIVERGED_LOSS
                and norm == norm  # noqa: PLR0124 - NaN check
                and abs(norm) != float("inf")
            )
            bench_nonfinite += 0 if finite else 1
            say(
                f"BENCHSTEP it={it:d} wall_s={wall:.3f} loss={totals['loss']:.4f} "
                f"lr={lr:.3e} peak_gib={peak:.2f} "
                f"micro_steps={accum * world.world_size:d} gnorm={norm:.6f}",
                flush=True,
            )
        if args.profile_phases:
            torch.cuda.synchronize()
            phases = timer.result()
            phases["step_wall"] = (time.time() - t_step) * 1000.0
            peaks = [v for k, v in phases.items() if k.endswith(":peak_gib")]
            phases["step_peak_gib"] = max(peaks) if peaks else 0.0
            phases["persist_after_step_gib"] = torch.cuda.memory_allocated() / 2**30
            phases["reserved_gib"] = torch.cuda.memory_reserved() / 2**30
            say(
                f"PHASES {it:4d} "
                f"{json.dumps({k: round(v, 2) for k, v in phases.items()})}",
                flush=True,
            )
            if it == 0:
                gib = 2**30
                say(
                    "PERSISTENT",
                    json.dumps(
                        {
                            "params_fp32_gib": round(
                                sum(
                                    p.numel() * p.element_size()
                                    for p in model.parameters()
                                )
                                / gib,
                                2,
                            ),
                            "grads_gib": round(
                                sum(
                                    p.grad.numel() * p.grad.element_size()
                                    for p in model.parameters()
                                    if p.grad is not None
                                )
                                / gib,
                                2,
                            ),
                            "adam_states_gib": round(
                                tensor_bytes(
                                    [list(s.values()) for s in optimizer.state.values()]
                                )
                                / gib,
                                2,
                            ),
                            "ema_gib": round(
                                tensor_bytes(list(ema.shadow.values())) / gib, 2
                            )
                            if ema is not None
                            else 0.0,
                        }
                    ),
                    flush=True,
                )
        history.append(totals["loss"])
        say(
            f"STEP {it:4d} lr={lr:.2e} loss={totals['loss']:.4f} "
            f"diffusion={totals['diffusion']:.4f} bond={totals['bond']:.4f} "
            f"distogram={totals['distogram']:.4f} plddt={totals['plddt']:.4f} "
            f"pde={totals['pde']:.4f} pae={totals['pae']:.4f} "
            f"resolved={totals['resolved']:.4f} gnorm={norm:.3f} "
            f"t={time.time() - t0:.1f}s",
            flush=True,
        )
        if totals["loss"] != totals["loss"] or abs(totals["loss"]) > _DIVERGED_LOSS:
            if args.bench:  # keep timing: NaN does not change the step-time class
                if not diverged_printed:
                    say("DIVERGED at step", it, "(--bench keeps timing)", flush=True)
                    diverged_printed = True
            else:
                say("DIVERGED at step", it, flush=True)
                break
        if (
            args.save_every
            and not args.bench
            and world.is_main
            and (it + 1) % args.save_every == 0
        ):
            checkpoint.save(
                model,
                run_dir / f"step{it + 1}.miniworld.pt",
                family=args.model,
                source="train",
                ema=ema,
                optimizer=optimizer,
                step=it + 1,
            )
            # prune: the newest --keep-last step checkpoints and every --keep-every-th
            # step stay
            if args.keep_last > 0:
                done = sorted(
                    (int(p.name[4:].split(".")[0]), p)
                    for p in run_dir.glob("step*.miniworld.pt")
                    if p.name[4:].split(".")[0].isdigit()
                )
                for n, p in done[: -args.keep_last]:
                    if args.keep_every and n % args.keep_every == 0:
                        continue
                    p.unlink(missing_ok=True)
                    p.with_name(
                        p.name.removesuffix(".miniworld.pt") + ".train.pt"
                    ).unlink(missing_ok=True)
        if (
            not args.bench
            and world.is_main
            and args.save_every_epochs
            and (it + 1) % steps_per_epoch == 0
            and ((it + 1) // steps_per_epoch) % args.save_every_epochs == 0
        ):
            epoch = (it + 1) // steps_per_epoch
            checkpoint.save(
                model,
                run_dir / f"epoch{epoch:04d}.miniworld.pt",
                family=args.model,
                source="train",
                ema=ema,
                optimizer=optimizer,
                step=it + 1,
            )
            say("SAVED epoch", epoch, "step", it + 1, flush=True)
        if (
            not args.bench
            and world.is_main
            and args.save_last
            and (it + 1) % steps_per_epoch == 0
        ):
            # MiniWorld's last.pt: overwritten at the end of every epoch (written under
            # a temporary name and renamed, so a kill mid-write leaves the previous one
            # intact)
            tmp = run_dir / "last_tmp.miniworld.pt"
            checkpoint.save(
                model,
                tmp,
                family=args.model,
                source="train",
                ema=ema,
                optimizer=optimizer,
                step=it + 1,
            )
            for suffix in (".miniworld.pt", ".train.pt"):
                (run_dir / f"last_tmp{suffix}").replace(run_dir / f"last{suffix}")
            (run_dir / "last.step").write_text(str(it + 1))
            say(
                "SAVED last epoch",
                (it + 1) // steps_per_epoch,
                "step",
                it + 1,
                flush=True,
            )

    if history:
        n = max(1, min(10, len(history) // 2))
        say(
            f"CURVE first{n:d}_mean={sum(history[:n]) / n:.4f} "
            f"last{n:d}_mean={sum(history[-n:]) / n:.4f} min={min(history):.4f}",
            flush=True,
        )

    # numerics comparison: the parameters after the last step (the gradients of step 0
    # are saved in the loop)
    if args.bench_dump and world.is_main:
        torch.save(
            {n: p.detach().cpu() for n, p in model.named_parameters()},
            Path(args.bench_dump) / "params.pt",
        )
    checksum = dist.param_checksum(params)
    agree = dist.ranks_agree(checksum, world, device)
    say(
        f"PARAM_CHECKSUM sum_abs={checksum:.10f} ranks_identical={agree}",
        flush=True,
    )
    if args.bench:
        import importlib.metadata  # noqa: PLC0415 - only --bench reports the engine

        import miniworld_engine  # noqa: PLC0415 - only --bench reports the engine

        engine_version = importlib.metadata.version("miniworld-engine")

        table = bench.engine_table(report_observations(model))
        for line in bench.format_table(table, "ENGINE_FINAL"):
            say(line, flush=True)
        if cfg.compile:
            say(
                "COMPILE",
                json.dumps(
                    {
                        "region": cfg.compile,
                        "warmup_s": round(compile_warmup_s, 1),
                        **train_compile.report(),
                    }
                ),
                flush=True,
            )
        if "_foldjet_step_graph" in model.__dict__:
            say(
                "STEP_GRAPH",
                json.dumps(model.__dict__["_foldjet_step_graph"].report()),
                flush=True,
            )
        # --rollout-order graph: captures, replays, cache entries, bytes held
        if "_foldjet_rollout_graph" in model.__dict__:
            say(
                "ROLLOUT_GRAPH",
                json.dumps(model.__dict__["_foldjet_rollout_graph"].report()),
                flush=True,
            )
        stats = bench.summarize_steps(bench_walls)
        total_micro = args.steps * accum * world.world_size
        tokens = torch.tensor([float(bench_tokens)], dtype=torch.float64, device=device)
        if world.world_size > 1:
            torch.distributed.all_reduce(tokens)
        payload = {
            "backend": args.backend,
            "world_size": world.world_size,
            "grad_accum": accum,
            "effective_batch": accum * world.world_size,
            "steps": args.steps,
            "warmup_steps": cfg.warmup_steps,
            **stats,
            "micro_steps_total": total_micro,
            "micro_steps_per_s": total_micro / stats["total_wall_s"],
            "seconds_per_micro_step_per_gpu": stats["total_wall_s"]
            / (args.steps * accum),
            "tokens_per_s": float(tokens.item()) / stats["total_wall_s"],
            "nonfinite_steps": bench_nonfinite,
            "peak_allocated_gib": bench_peak[0],
            "peak_reserved_gib": bench_peak[1],
            "param_checksum": checksum,
            "ranks_identical": agree,
            "engine_table": table,
            "pool": [e.name for e in pool],
            "commit": bench.git_commit(),
            "engine_version": engine_version,
            "engine_file": next(iter(miniworld_engine.__path__)),
            "gpu": torch.cuda.get_device_name(device),
            "dist_backend": args.dist_backend if world.world_size > 1 else None,
        }
        say(bench.bench_json(payload), flush=True)
    if world.world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    if args.no_save or not world.is_main:
        return
    # -- checkpoint round trip
    path = checkpoint.save(
        model,
        run_dir / "final.miniworld.pt",
        family=args.model,
        source="train",
        ema=ema,
        optimizer=optimizer,
        step=start_step + len(history),
    )
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = model.state_dict()
    worst = max(
        float(
            (blob["state_dict"][k].to(state[k].device).float() - state[k].float())
            .abs()
            .max()
        )
        for k in list(state)[:50]
    )
    say(
        "CHECKPOINT",
        path,
        "keys",
        len(blob["state_dict"]),
        "format",
        blob["format"],
        f"max_abs_diff(first50)={worst:.3g}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except torch.OutOfMemoryError as error:
        print(
            "OOM",
            str(error)[:240].replace("\n", " "),
            f"max_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.1f}",
            flush=True,
        )
