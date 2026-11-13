"""Protenix v1 finetuning on the FoldForge graph (MiniWorld recipe). Phase 0/1: eager steps, one GPU, gradient accumulation.

  # smoke: one fixed example (FoldBench target) overfit + gradient coverage + kernel table + checkpoint round trip
  python scripts/train_protenix1.py --smoke --target 8ayg-assembly1 --steps 50 --warmup-steps 0 --run-dir RUN

  # training from a data source (MiniWorld provides the adapter; FoldForge does not import MiniWorld)
  python scripts/train_protenix1.py --batch-source my_pkg.adapter:batches --steps 100000 --run-dir RUN [--ckpt X.miniworld.pt]

  # speed measurement of a backend (pytorch / cueq / miniworld): 30 real optimizer steps, effective batch 256 (8 GPUs: torchrun --nproc-per-node 8); docs/guides/TRAINING.md
  python scripts/train_protenix1.py --bench --backend cueq --effective-batch 256 --steps 30 --warmup-steps 10 --bench-pool-cache POOL.pt

The recipe (loss weights, Adam, clip, EMA, recycles, augmentations) is :class:`foldforge.train.config.TrainConfig`; read the lr warning there.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

import foldforge as ff
from foldforge.train import (
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
    step,
)
from foldforge.modules.engine.configure import activation_autocast
from foldforge.train.profile import PhaseTimer, tensor_bytes

SPECS = Path("/home/psk6950/foldforge-scratch/foldbench/specs_all")


def smoke_batches(pred, target: str):
    """One fixed example: FoldForge's own featurisation of a FoldBench target; the ground truth is the model's own pre-training prediction of it
    (a STAND-IN: the real ground-truth atoms are not mapped to the dense layout in phase 0/1)."""
    example = bench.standin_example(pred, target)
    while True:
        yield example


def split_sizes(model, example, cfg) -> dict:
    """Bytes a 2-GPU split of the step would move (the mini-rollout on a second GPU): the diffusion-head parameters (re-synced after every
    optimizer step), the trunk outputs the rollout reads (single / pair / target features, per crop), the predicted coordinates back."""
    features = {
        k: (v.float() if v.dtype == torch.float64 else v) for k, v in example.features.items()
    }
    with torch.no_grad(), activation_autocast(model):
        batch_data = model._prepare_batch(features)
        emb = step.run_trunk_train(model, batch_data, 1)
    head = model.diffusion_head
    res, atoms, _ = example.atom_positions.shape
    return {
        "num_res": int(res),
        "atoms_per_token": int(atoms),
        "diffusion_head_params": sum(p.numel() for p in head.parameters()),
        "diffusion_head_param_bytes_fp32": sum(p.numel() * p.element_size() for p in head.parameters()),
        "single_bytes": tensor_bytes(emb["single"]),
        "pair_bytes": tensor_bytes(emb["pair"]),
        "target_feat_bytes": tensor_bytes(emb["target_feat"]),
        "rollout_inputs_bytes": tensor_bytes(emb),
        "x_mini_bytes_fp32": res * atoms * 3 * 4,
        "pair_dtype": str(emb["pair"].dtype),
        "single_dtype": str(emb["single"].dtype),
    }


#: columns of a :func:`grad_coverage` row
COVERAGE_COLUMNS = ("total", "none", "zero_live", "nonfinite", "underflow", "data_dependent", "frozen")


def grad_coverage(model: torch.nn.Module) -> dict:
    """Which parameters got a gradient from one step: groups by the first three name components; the confidence head is in the list (single-stage
    training; only ``--no-confidence`` leaves it without a loss). A parameter is classified by ``train/freeze.py``: frozen (no path to the loss /
    not in the original / fixed in the original: not counted as a defect), ``data_dependent`` (needs charged atoms / bonds in the example: smoke 7fvu),
    ``underflow`` (a decayed module: the gradient is exactly zero because the weights are ~1e-37..1e-12 relative to the model's typical weight) and
    the real defects ``none`` (no gradient), ``zero_live`` (zero gradient on a live weight) and ``nonfinite``."""
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
    """Per data-dependent parameter (charge, bond embeddings): did a nonzero gradient reach it on this example?"""
    flow = {}
    for name, param in model.named_parameters():
        if freeze.DATA_DEPENDENT.fullmatch(name):
            flow[name] = param.grad is not None and bool(param.grad.any())
    return flow


def kernel_table(model, example, cfg, top: int = 30) -> list[tuple[str, float, int]]:
    """CUDA kernels of one forward + backward by total GPU time: engine kernels versus plain aten GEMMs / elementwise (= the pytorch path)."""
    from torch.profiler import ProfilerActivity, profile

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


def load_summary(model, backend: str) -> dict:
    """What the loader chose: the plan (backend per op, fallbacks) and every engine module that does NOT run the requested backend, with the reason."""
    report = getattr(model, "foldforge_load_report", {}) or {}
    plan = report.get("backends", {})
    modules = (report.get("engine_backends") or {}).get("modules", {})
    off = {}
    for name, row in modules.items() if isinstance(modules, dict) else []:
        if isinstance(row, dict) and row.get("reason"):
            key = f"{row.get('class')}:{row.get('op')}"
            off.setdefault(key, {"modules": 0, "backend": row.get("backend"), "requested": row.get("requested"), "reason": row.get("reason")})["modules"] += 1
    return {"backend": backend, "plan": plan, "modules_not_on_requested_backend": off}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--target", default="8ayg-assembly1")
    ap.add_argument("--model", default="protenix1")
    ap.add_argument(
        "--batch-source",
        default=None,
        help="module:callable(cfg, device) -> iterator of TrainBatch",
    )
    ap.add_argument("--steps", type=int, default=None, help="optimizer steps (default 50; 30 with --bench)")
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--num-augment", type=int, default=None)
    ap.add_argument("--max-lr", type=float, default=None)
    ap.add_argument("--min-lr", type=float, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--trunk-lr-scale", type=float, default=None)
    ap.add_argument(
        "--no-confidence",
        action="store_true",
        help="DEBUG ONLY: skip the confidence head (the finetuning is one stage; every parameter trains)",
    )
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument(
        "--profile-phases",
        action="store_true",
        help="per-phase GPU timing (CUDA events) of every step: trunk fwd per recycle, diffusion fwd / bwd, trunk bwd, mini-rollout, confidence fwd / bwd, optimizer + EMA",
    )
    ap.add_argument(
        "--rollout-order",
        choices=("after", "before", "side", "graph"),
        default=None,
        help="where the mini-rollout runs: after the trunk backward (default, least memory), before the diffusion forward, or on a second CUDA stream (side), or graph: the 20-step rollout as ONE CUDA graph per shape, replayed after the trunk backward (train/rollout_graph.py; the noise is drawn inside the graph from the default CUDA generator; measured on the 5I28-size pool the losses and parameters are bitwise those of --rollout-order after)",
    )
    ap.add_argument("--augment-chunk", type=int, default=None, help="diffusion augmentations per forward/backward chunk (0 = all)")
    ap.add_argument("--recycles", type=int, default=None, help="fix the trunk passes (min = max)")
    ap.add_argument("--only", choices=("trunk", "diffusion", "rollout", "confidence"), default=None, help="MEASUREMENT: run one part of the step alone")
    ap.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="recompute the pair-stack blocks in the backward instead of storing their activations",
    )
    ap.add_argument(
        "--ckpt-policy",
        default="all",
        help="which blocks --activation-checkpointing wraps: all (default) | none | pairformer (trunk only) | evoformer | except:<trunk|msa|template|confidence> | every:N / first:K / last:K (those trunk pairformer blocks plus every non-trunk block); 'none' also switches checkpointing off",
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help="start from this .miniworld.pt (default: the registered model's checkpoint)",
    )
    ap.add_argument("--run-dir", default="runs/train_protenix1")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="skip the final checkpoint (measurement runs)")
    ap.add_argument(
        "--backend",
        choices=("pytorch", "cueq", "miniworld"),
        default="miniworld",
        help="engine backend of every module: pytorch (reference), cueq (NVIDIA cuEquivariance where it has the op, the rest falls back to pytorch), miniworld (our kernels: cuda per module class on H100)",
    )
    ap.add_argument("--no-activation-checkpointing", action="store_true", help="--bench turns --activation-checkpointing on (pytorch and cueq OOM an 80 GB H100 at 384 tokens without it); this keeps it off")
    ap.add_argument("--effective-batch", type=int, default=None, help="micro batch 1 x grad_accum x world size; sets grad_accum = N / world size (error if not divisible)")
    ap.add_argument("--dist-backend", default="nccl", help="torch.distributed backend under torchrun (gloo: two processes on one GPU, for tests)")
    ap.add_argument("--seed", type=int, default=None, help="base seed (default: TrainConfig.seed)")
    ap.add_argument("--fused-opt", nargs="?", const="fused", default=None, choices=("foreach", "fused", "triton"), help="clip + Adam + EMA as multi-tensor launches (train/fused_opt.py; capturable Adam, device lr); default: the per-tensor path of train/optim.py")
    ap.add_argument("--bench", action="store_true", help="speed measurement of --steps real optimizer steps (implies --no-save --no-profile; no phase-0 micro step / coverage / kernel table)")
    ap.add_argument("--bench-pool", default="default", help="'default' (targets near --bench-tokens) or comma separated FoldBench targets; STAND-IN examples")
    ap.add_argument("--bench-tokens", type=int, default=bench.RECIPE_TOKENS, help="token count of the default pool (MiniWorld recipe crop: 384)")
    ap.add_argument("--micro-log", action="store_true", help="--bench: print one MICRO line per micro step on every rank (global index, example, recycles, loss) to compare runs micro step by micro step")
    ap.add_argument("--bench-pool-count", type=int, default=8, help="targets of the default pool")
    ap.add_argument("--bench-pool-cache", default=None, help="build the pool once (rank 0) into this file and load it everywhere: identical stand-in labels for every backend")
    ap.add_argument("--step-graph", action="store_true", help="the WHOLE micro step (trunk recycles + diffusion + confidence + backward, gradient accumulation into persistent .grad) as ONE CUDA graph per (recycles, shape signature), replayed (train/step_graph.py); the mini-rollout is recorded inline; needs --bench-bucket (one shape) to stay at <= 4 graphs; default off")
    ap.add_argument("--bench-bucket", action="store_true", help="pad every example to ONE shape (train/bucket_batch.py: --bench-tokens tokens, --bucket-atoms atoms, --bucket-msa-pool raw MSA rows, --bucket-templates), so a CUDA graph of the micro step has one signature; also applies to --batch-source streams; default off")
    ap.add_argument("--bucket-atoms", type=int, default=4096, help="--bench-bucket: flat atoms (MiniWorld crop.max_atoms 4096)")
    ap.add_argument("--bucket-msa-pool", type=int, default=16384, help="--bench-bucket: raw MSA pool rows before the per-recycle draw (16384 = the largest FoldForge MSA bucket: nothing is cut)")
    ap.add_argument("--bucket-templates", type=int, default=4, help="--bench-bucket: templates")
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

    def say(*a, **k) -> None:  # only rank 0 prints
        if world.is_main:
            print(*a, **k)

    cfg = TrainConfig()
    if args.effective_batch is not None:
        if args.grad_accum is not None:
            raise SystemExit("--effective-batch and --grad-accum are exclusive")
        try:
            args.grad_accum = dist.accum_for_effective_batch(args.effective_batch, world.world_size)
        except ValueError as error:
            raise SystemExit(str(error)) from None
    if args.bench and not args.no_activation_checkpointing:
        args.activation_checkpointing = True
    if args.bench and args.augment_chunk is None:
        # No chunking: the recipe runs all augmentations in one forward / backward. (An earlier --bench default of 8 was never asked for; it made
        # eager ~60% slower at 384 tokens.) With activation checkpointing (the --bench default) 48 augmentations at 384 peak 36 GiB on miniworld.
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
        ("rollout_steps", args.rollout_steps),
        ("rollout", args.rollout_order),
        ("step_graph", True if args.step_graph else None),
        ("augment_chunk", args.augment_chunk),
        ("only", args.only),
        ("recycles_min", args.recycles),
        ("recycles_max", args.recycles),
    ):
        if value is not None:
            changes[key] = value
    if args.no_confidence:
        changes["confidence"] = False
    cfg = replace(cfg, **changes)
    run_dir = Path(args.run_dir)
    if world.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    dist.init_process_group(world, args.dist_backend)
    # legacy streams (smoke / --batch-source): rank r seeds seed + r (rank 0 = the single-process run); --bench draws every micro step from
    # bench.micro_plan(seed, global micro index) instead, which does not depend on the order or the rank count
    rng = prepare.seed_everything(cfg.seed + world.rank)

    # dead modules must run in training (their parameters train): do not flag them at load
    pred = ff.load_model(
        args.model,
        config={
            "backend": args.backend,
            "engine": {"skip_dead_modules": False, "unsupported": "fallback"},
        },  # atom_blocks: engine default (2.5.3 fixed the sm90 atom-DiT backward, R20)
        checkpoint=args.ckpt,
        device=None if world.world_size == 1 else torch.device("cuda", torch.cuda.current_device()),
    )
    model = pred._network
    device = pred.device
    say("LOAD_REPORT", json.dumps(load_summary(model, args.backend), default=str), flush=True)
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
        bench.FIXED_LENGTH = max(bench.FIXED_LENGTH, args.bucket_atoms)  # present-atom index of one length for every example (else 512-multiples per example split the step-graph key)
    bucket_spec = (
        bucket_batch.BucketSpec(tokens=args.bench_tokens, atoms=args.bucket_atoms, msa_pool_rows=args.bucket_msa_pool, templates=args.bucket_templates)
        if args.bench_bucket
        else None
    )
    pool: list[TrainBatch] = []
    if args.bench:
        targets = bench.parse_pool(args.bench_pool, args.bench_tokens, args.bench_pool_count, max_atoms=args.bucket_atoms)
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
        say("POOL targets=%s tokens=%s build_s=%.1f" % (",".join(targets), [bench.real_tokens(e) for e in pool], time.time() - t_pool), flush=True)
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
        raise SystemExit("give --smoke, --bench or --batch-source module:callable")
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
    prep_info = prepare.enable_training(model, cfg)
    dist.broadcast_parameters(list(model.parameters()) + [b for b in model.buffers() if b.is_floating_point()], world)
    if args.activation_checkpointing:
        say("ACT_CKPT policy=%s blocks=%d" % (args.ckpt_policy, prepare.enable_activation_checkpointing(model, args.ckpt_policy)), flush=True)
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
    for frozen_name, rule in freeze.frozen_parameters(model).items():
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
        in_loss = lambda k: cfg.confidence or not k.startswith("confidence_head")  # noqa: E731
        bad = {
            k: v for k, v in cover.items() if any(v[column[c]] for c in defects) and in_loss(k)
        }
        say(
            "GRAD_COVERAGE groups=%d  groups_with_defects=%d  (defect = no gradient / zero gradient on a live weight / nonfinite; frozen, data-dependent and underflow are classified, not defects)"
            % (len(cover), len(bad)),
            flush=True,
        )
        sums = [0] * len(COVERAGE_COLUMNS)
        for key, row in sorted(cover.items()):
            if in_loss(key):
                sums = [a + b for a, b in zip(sums, row)]
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
                + " ".join(f"{c}={v:4d}" for c, v in zip(COVERAGE_COLUMNS, row))
                + flag,
                flush=True,
            )
        say(
            "GRAD_SUM " + " ".join(f"{c}={v}" for c, v in zip(COVERAGE_COLUMNS, sums)),
            flush=True,
        )
        for name, flowing in sorted(data_dependent_flow(model).items()):
            say(f"DATA_DEPENDENT {name} gradient_flows={flowing}", flush=True)
        tmask = first.features.get("template_atom_mask")
        if tmask is not None:
            say(
                "TEMPLATES real_templates=%d real_atoms=%d"
                % (int(tmask.reshape(tmask.shape[0], -1).any(-1).sum()), int(tmask.sum())),
                flush=True,
            )
        if args.smoke and not args.no_profile:
            for name, ms, count in kernel_table(model, first, cfg):
                say(f"KERNEL {ms:10.2f} ms  x{count:<6d} {name[:150]}", flush=True)

    # -- training loop
    fopt = fused_opt.build_fused(model, cfg, args.fused_opt) if args.fused_opt else None
    optimizer = fopt.optimizer if fopt else optim.build_optimizer(model, cfg)
    ema = fopt.ema if fopt else (optim.EMA(model, cfg.ema_decay) if cfg.use_ema else None)
    params = [p for g in optimizer.param_groups for p in g["params"]]
    history: list[float] = []
    bench_walls: list[float] = []
    bench_nonfinite = 0
    bench_micro = 0
    bench_tokens = 0
    bench_peak = [0.0, 0.0]
    diverged_printed = False
    accum = cfg.grad_accum_steps
    micro_scale = 1.0 / (accum * world.world_size)  # the SUM over ranks of the scaled gradients is the mean over accum x world micro steps
    from foldforge.modules.engine.serving import report_observations, reset_observations

    reset_observations(model)  # the dispatch table counts the training steps only (not the load / pool build)
    t0 = time.time()
    for it in range(args.steps):
        lr = fopt.set_lr(it, cfg) if fopt else optim.set_lr(optimizer, it, cfg)
        optimizer.zero_grad(set_to_none=not cfg.step_graph)  # a step graph accumulates into the SAME .grad tensors: zeroed in place
        totals = defaultdict(float)
        timer = PhaseTimer(args.profile_phases)
        if args.profile_phases or args.bench:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t_step = time.time()
        for micro in range(accum):
            if args.bench:
                plan = bench.micro_plan(cfg.seed, bench.global_micro_index(it, micro, accum, world.rank, world.world_size), len(pool), cfg)
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
                print("MICRO rank=%d g=%d example=%s passes=%d loss=%.7f diffusion=%.7f" % (world.rank, bench.global_micro_index(it, micro, accum, world.rank, world.world_size), example.name, passes, out["loss"], out["diffusion"]), flush=True)
            for k, v in out.items():
                totals[k] += v / accum
            if it == 0 and micro < 2:
                for line in bench.format_table(bench.engine_table(report_observations(model)), f"ENGINE_AFTER_MICRO{micro + 1}"):
                    say(line, flush=True)
        with timer.phase("allreduce"):
            dist.allreduce_gradients(params, world)
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
            peak, reserved = torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30
            bench_peak = [max(bench_peak[0], peak), max(bench_peak[1], reserved)]
            finite = totals["loss"] == totals["loss"] and abs(totals["loss"]) < 1e6 and norm == norm and abs(norm) != float("inf")
            bench_nonfinite += 0 if finite else 1
            say(
                "BENCHSTEP it=%d wall_s=%.3f loss=%.4f lr=%.3e peak_gib=%.2f micro_steps=%d gnorm=%.6f"
                % (it, wall, totals["loss"], lr, peak, accum * world.world_size, norm),
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
            say("PHASES %4d %s" % (it, json.dumps({k: round(v, 2) for k, v in phases.items()})), flush=True)
            if it == 0:
                gib = 2**30
                say(
                    "PERSISTENT",
                    json.dumps(
                        {
                            "params_fp32_gib": round(sum(p.numel() * p.element_size() for p in model.parameters()) / gib, 2),
                            "grads_gib": round(sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None) / gib, 2),
                            "adam_states_gib": round(tensor_bytes([list(s.values()) for s in optimizer.state.values()]) / gib, 2),
                            "ema_gib": round(tensor_bytes(list(ema.shadow.values())) / gib, 2) if ema is not None else 0.0,
                        }
                    ),
                    flush=True,
                )
        history.append(totals["loss"])
        say(
            "STEP %4d lr=%.2e loss=%.4f diffusion=%.4f bond=%.4f distogram=%.4f plddt=%.4f pde=%.4f pae=%.4f gnorm=%.3f t=%.1fs"
            % (
                it,
                lr,
                totals["loss"],
                totals["diffusion"],
                totals["bond"],
                totals["distogram"],
                totals["plddt"],
                totals["pde"],
                totals["pae"],
                norm,
                time.time() - t0,
            ),
            flush=True,
        )
        if not (totals["loss"] == totals["loss"]) or abs(totals["loss"]) > 1e6:
            if args.bench:  # keep timing: NaN does not change the step-time class
                if not diverged_printed:
                    say("DIVERGED at step", it, "(--bench keeps timing)", flush=True)
                    diverged_printed = True
            else:
                say("DIVERGED at step", it, flush=True)
                break
        if args.save_every and not args.bench and world.is_main and (it + 1) % args.save_every == 0:
            checkpoint.save(
                model,
                run_dir / f"step{it + 1}.miniworld.pt",
                family=args.model,
                source="train",
                ema=ema,
                optimizer=optimizer,
                step=it + 1,
            )

    if history:
        n = max(1, min(10, len(history) // 2))
        say(
            "CURVE first%d_mean=%.4f last%d_mean=%.4f min=%.4f"
            % (n, sum(history[:n]) / n, n, sum(history[-n:]) / n, min(history)),
            flush=True,
        )

    checksum = dist.param_checksum(params)
    agree = dist.ranks_agree(checksum, world, device)
    say("PARAM_CHECKSUM sum_abs=%.10f ranks_identical=%s" % (checksum, agree), flush=True)
    if args.bench:
        import importlib.metadata

        import miniworld_engine

        engine_version = importlib.metadata.version("miniworld-engine")

        table = bench.engine_table(report_observations(model))
        for line in bench.format_table(table, "ENGINE_FINAL"):
            say(line, flush=True)
        if "_foldforge_step_graph" in model.__dict__:
            say("STEP_GRAPH", json.dumps(model.__dict__["_foldforge_step_graph"].report()), flush=True)
        if "_foldforge_rollout_graph" in model.__dict__:  # --rollout-order graph: captures, replays, cache entries, bytes held
            say("ROLLOUT_GRAPH", json.dumps(model.__dict__["_foldforge_rollout_graph"].report()), flush=True)
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
            "seconds_per_micro_step_per_gpu": stats["total_wall_s"] / (args.steps * accum),
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
            "engine_file": list(miniworld_engine.__path__)[0],
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
        step=len(history),
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
        "max_abs_diff(first50)=%.3g" % worst,
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except torch.OutOfMemoryError as error:
        print("OOM", str(error)[:240].replace("\n", " "), "max_allocated_gib=%.1f" % (torch.cuda.max_memory_allocated() / 2**30), flush=True)
