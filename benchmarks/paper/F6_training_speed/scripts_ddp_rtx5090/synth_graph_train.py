"""MiniWorld phase-1 production trainer (random_recycle_graph_trainer: CUDA graphs per recycle, bucketed NCCL all-reduce of the
gradients once per optimizer step) with the BioMolDB loader replaced by synthetic batches. Nothing in the MiniWorld tree is edited:
this wrapper patches the loader class, the trainer entry (diagnostic_steps / validate) and adds DDP-sync / memory / backend reports.

  torchrun --standalone --nproc_per_node=N synth_graph_train.py --config CFG [--steps S] [--no-validate] [--variants K] -- OVERRIDES...

Synthetic data: MiniWorld's own dense synthetic batch (run_miniworld_distogram_train._build_precompile_batch, the batch of its
train.synth_data path) at the config's bucket shape, made into K variants per rank (random token types, MSA rows with 15 % mutations,
random-walk 3.8 A chain coordinates, templates = the noisy chain): every rank sees different data, the loss can fall.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

MW = Path(os.environ.get("MW_ROOT", "/home/psk6950/mwe/agents/mw_ddp/MiniWorld"))
sys.path.insert(0, str(MW / "scripts"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--steps", type=int, default=30, help="optimizer steps (diagnostic_steps)")
p.add_argument("--no-validate", action="store_true", help="skip the trainer's graph-vs-eager validation")
p.add_argument("--variants", type=int, default=4)
p.add_argument("--out", default=None, help="json summary path (rank 0)")
p.add_argument("overrides", nargs="*")
args = p.parse_args()

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL = int(os.environ.get("LOCAL_RANK", "0"))

# ---------------------------------------------------------------- Triton launch counter (engine Triton connected = bug)
TRITON = Counter()
try:
    import triton.runtime.jit as _tj

    _orig_run = _tj.JITFunction.run

    def _counting_run(self, *a, **k):
        TRITON[getattr(getattr(self, "fn", None), "__name__", "?")] += 1
        return _orig_run(self, *a, **k)

    _tj.JITFunction.run = _counting_run
except Exception as exc:  # noqa: BLE001
    print(f"[mw_ddp] triton counter not installed: {exc}", flush=True)

import random_recycle_graph_trainer as G  # noqa: E402
import run_miniworld_distogram_train as R  # noqa: E402

import miniworld.data.dataloader.dataloader as DL  # noqa: E402


def _ceil(x, m):
    return x if not m else -(-x // m) * m


def make_variant(base, gen, n_tokens, n_atoms, msa_depth, n_templates):
    import copy

    b = copy.deepcopy(base)
    tt = torch.randint(0, 20, (n_tokens,), generator=gen)
    b.sequence.token_type.copy_(tt.unsqueeze(0))
    mut = torch.rand(msa_depth, n_tokens, generator=gen) < 0.15
    rows = torch.where(mut, torch.randint(0, 20, (msa_depth, n_tokens), generator=gen), tt.unsqueeze(0))
    rows[0] = tt
    b.msa.aligned_sequences.copy_(rows.unsqueeze(0).to(b.msa.aligned_sequences.dtype))
    prof = torch.nn.functional.one_hot(tt, num_classes=b.msa.profile.shape[-1]).float()
    b.msa.profile.copy_(prof.unsqueeze(0).to(b.msa.profile.dtype))
    step = torch.randn(n_tokens, 3, generator=gen)
    step = step / step.norm(dim=-1, keepdim=True) * 3.8
    center = step.cumsum(0)
    center = center - center.mean(0)
    a2t = b.scheme.atom_to_token_idx_map[0].long()
    atom = center[a2t] + 0.7 * torch.randn(n_atoms, 3, generator=gen)
    b.structure.atom_pos.copy_(atom.unsqueeze(0))
    if n_templates:
        b.template.res_type.copy_(tt.view(1, 1, -1).expand_as(b.template.res_type))
        cb = center.unsqueeze(0) + 0.5 * torch.randn(n_templates, n_tokens, 3, generator=gen)
        b.template.cb_xyz.copy_(cb.unsqueeze(0))
        off = torch.tensor([[-0.55, 0.10, 0.0], [0.0, 0.0, 0.0], [0.65, -0.10, 0.0]])
        b.template.bb_xyz.copy_((cb.unsqueeze(-2) + off).unsqueeze(0))
    b.name = [f"synth_r{RANK}"]
    return b


class _Sampler:
    def set_epoch(self, epoch):
        pass


class _Loader:
    sampler = _Sampler()

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return itertools.cycle(self.batches)


class SynthBioMolData:
    class BioMolConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def __init__(self, config):
        self.config = config

    def set_epoch(self, epoch):
        pass

    def create_ddp_dataloader(self, **kw):
        crop, msa = self.config.crop_config, self.config.msa_config
        n_tokens = _ceil(crop.max_tokens, kw.get("bucket_token_multiple"))
        n_atoms = _ceil(crop.max_atoms, kw.get("bucket_atom_multiple"))
        msa_depth = _ceil(msa.max_msa_depth, kw.get("bucket_msa_multiple"))
        n_templates = kw.get("bucket_template_multiple") or 0
        base = R._build_precompile_batch(
            device=torch.device("cpu"), msa_depth=msa_depth, n_tokens=n_tokens, n_atoms=n_atoms,
            n_templates=n_templates, num_res_class=32,
        )
        gen = torch.Generator().manual_seed(1000 + 17 * kw["rank"])
        batches = [make_variant(base, gen, n_tokens, n_atoms, msa_depth, n_templates) for _ in range(args.variants)]
        if kw["rank"] == 0:
            print(f"[mw_ddp] synthetic batches: {args.variants}/rank tokens={n_tokens} atoms={n_atoms} msa={msa_depth} "
                  f"templates={n_templates}", flush=True)
        return _Loader(batches)


DL.BioMolData = SynthBioMolData

# ---------------------------------------------------------------- hooks: model handle, per-rank losses, comm timing, report
STATE = {"model": None, "losses": [], "t0": time.time(), "comm_ms": [], "replay_n": 0}
_orig_init = G.RecycleGraphs.__init__


def _init(self, model, *a, **k):
    STATE["model"] = model
    _orig_init(self, model, *a, **k)


G.RecycleGraphs.__init__ = _init
_orig_replay = G.RecycleGraphs.replay


def _replay(self, r):
    out = _orig_replay(self, r)
    STATE["losses"].append(out.detach().reshape(()).float().clone())
    STATE["replay_n"] += 1
    return out


G.RecycleGraphs.replay = _replay

def backend_table(model):
    rows = {}
    for name, m in model.named_modules():
        mod = type(m).__module__
        if not mod.startswith("miniworld_engine"):
            continue
        be = getattr(m, "backend_name", None) or getattr(m, "backend", None)
        if be is None:
            continue
        r = rows.setdefault(type(m).__name__, Counter())
        r[str(be)] += 1
    return {k: dict(v) for k, v in sorted(rows.items())}


def report():
    model = STATE["model"]
    dev = torch.device("cuda", LOCAL)
    losses = torch.stack(STATE["losses"]).cpu() if STATE["losses"] else torch.zeros(0)
    n = len(losses)
    k = max(1, min(16, n // 4))
    first = float(losses[:k].mean()) if n else float("nan")
    last = float(losses[-k:].mean()) if n else float("nan")
    finite = bool(torch.isfinite(losses).all()) if n else False
    with torch.no_grad():
        ps = [p for p in model.parameters()]
        flat = torch.cat([p.detach().float().reshape(-1) for p in ps])
        checksum = torch.stack([flat.double().sum(), flat.double().abs().sum(), (flat.double() ** 2).sum()])
    mine = {
        "rank": RANK, "host_gpu": torch.cuda.get_device_name(dev), "micro_losses": n, "loss_first": first, "loss_last": last,
        "finite": finite, "peak_alloc_gib": torch.cuda.max_memory_allocated(dev) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(dev) / 2**30,
        "checksum": [float(x) for x in checksum], "cpu_affinity": sorted(os.sched_getaffinity(0))[:4] + ["..."],
        "triton_launches": dict(TRITON),
    }
    gathered = [None] * WORLD
    if WORLD > 1:
        dist.all_gather_object(gathered, mine)
        # exact parameter agreement: max |p_r - p_0| over all parameters
        ref = flat.clone()
        dist.broadcast(ref, src=0)
        diff = torch.tensor([(flat - ref).abs().max().item()], device=dev)
        dist.all_reduce(diff, op=dist.ReduceOp.MAX)
        max_param_diff = float(diff)
    else:
        gathered = [mine]
        max_param_diff = 0.0
    if RANK == 0:
        table = backend_table(model)
        print("[mw_ddp] ENGINE backend table (class: backend counts)", flush=True)
        for cls, c in table.items():
            print(f"[mw_ddp]   {cls:40s} {c}", flush=True)
        for g in gathered:
            print(f"[mw_ddp] RANK {g['rank']} loss first/last {g['loss_first']:.4f} -> {g['loss_last']:.4f} finite={g['finite']} "
                  f"peak alloc {g['peak_alloc_gib']:.2f} GiB reserved {g['peak_reserved_gib']:.2f} GiB triton={g['triton_launches']}",
                  flush=True)
        print(f"[mw_ddp] DDP sync: max |param_r - param_0| = {max_param_diff:.3e}", flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(dict(ranks=gathered, max_param_diff=max_param_diff, engine_table=table,
                                                       overrides=args.overrides, config=args.config, world=WORLD), indent=2))


_orig_destroy = dist.destroy_process_group


def _destroy(*a, **k):
    try:
        report()
    finally:
        _orig_destroy(*a, **k)


dist.destroy_process_group = _destroy

_orig_train = G.train


def _train(cfg, ckpt, run_dir=None, **kw):
    kw["diagnostic_steps"] = args.steps
    kw["validate"] = not args.no_validate
    out = _orig_train(cfg, ckpt, run_dir, **kw)
    if WORLD == 1:
        report()
    return out


G.train = _train

if __name__ == "__main__":
    sys.argv = ["run_miniworld_distogram_train.py", "train", "--config", args.config, *args.overrides]
    R.cli()
