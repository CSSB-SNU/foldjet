"""The diffusion transformers as FoldForge builds them: engine ``TokenDiT*`` / ``AtomDiT*`` BLOCKS, the stack assembled here.

The engine owns one block per model (``TokenDiT(model=...)``, ``AtomDiT(model=...)``: the model's own class, so a state dict has that
class's keys); the stack is FoldForge's: a ``ModuleList`` ``blocks`` run in order. Each block hoists its own pair bias from the pair
(``attention.ln_pair`` + ``attention.to_bias``), so there is no shared pair norm / projection in the engine layout: the released AF3
stack's shared parameters are broadcast / sliced per block by the offline converter (:mod:`foldforge.models.checkpoints.legacy.released`).

Shapes (engine layout): ``single`` / ``cond`` ``[A, B, L, d]`` (A = diffusion samples), token ``pair`` ``[B, L, L, d_pair]``, atom
``pair`` the trunked ``[B, windows, 32, 128, d_pair]``, ``mask`` the bool key mask ``[B, L]`` (or ``[A, B, L]``), ``uid`` ``[B, N]``
(Chai-1's conformer ids).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from miniworld_engine.modules import AtomDiT, TokenDiT
from miniworld_engine.modules.atom_dit import KEY_WINDOWS
from torch import nn

from foldforge.modules.engine.configure import engine_activation
from foldforge.modules.engine.modules import Precision, finish, model_keys

if TYPE_CHECKING:
    from torch import Tensor

_UNBATCHED_RANK = 2


class TokenDiTStack(nn.Module):
    """``blocks``: the engine token DiT blocks of one family, in order (state-dict keys ``blocks.{i}.<engine names>``).

    Called like the dense token transformer: ``act`` ``[S, L, d]`` (or ``[L, d]``), ``mask`` ``[L]``, ``single_cond`` ``[L, d_cond]``
    (broadcast over the samples, or ``[S, L, d_cond]``), ``pair_cond`` ``[L, L, d_pair]``; ``key_mask`` ``[1, L]`` bool (made once per fold
    by the caller; made here when absent); ``extra_pair_bias`` ``[L, L]``: OpenDDE's bias, added to every head of every block."""

    def __init__(self, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self._pair_source: tuple[Tensor, int] | None = None
        self._pair_engine: Tensor | None = None

    def forward(
        self,
        act: Tensor,
        mask: Tensor,
        single_cond: Tensor,
        pair_cond: Tensor,
        extra_pair_bias: Tensor | None = None,
        key_mask: Tensor | None = None,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Every block in order (each owns its residuals)."""
        kw: dict[str, Any] = {}
        if compute_dtype is not None:
            kw["compute_dtype"] = compute_dtype
        if extra_pair_bias is not None:
            kw["extra_pair_bias"] = extra_pair_bias
        if key_mask is None:
            key_mask = (mask > 0)[None]
        unbatched = act.ndim == _UNBATCHED_RANK
        out_dtype = act.dtype
        # the engine's master blocks take bf16 activations: cast once here (a block would re-round fp32 at its entry on every call)
        act, single_cond = engine_activation(act), engine_activation(single_cond)
        cond = single_cond.expand(*act.shape[:-1], single_cond.shape[-1])
        x, cond = (
            (act[None, None], cond[None, None])
            if unbatched
            else (act[:, None], cond[:, None])
        )
        pair = self._engine_pair(pair_cond)
        for block in self.blocks:
            x = block(x, cond, pair, key_mask, **kw)
        x = x[0, 0] if unbatched else x[:, 0]
        return x if x.dtype == out_dtype else x.to(out_dtype)

    def engine_pair_cast(self) -> tuple[Tensor, Tensor] | None:
        """``(pair_cond, its engine-dtype copy)`` when the cached engine pair is a cast COPY of the pair (fp32 pair, bf16 blocks), else ``None``.

        A sampler graph that serves a new fold from static buffers (``models.sampler_graph``) refills ``pair_cond`` in place; this copy was made
        outside the graph (at the warm-up) and is read by the engine's hoisted pair bias, so it is rewritten from the new pair
        (``copy.copy_(pair_cond[None])`` rounds exactly like the cast) before the engine's tables are refreshed."""
        source, engine = self._pair_source, self._pair_engine
        if (
            source is None
            or engine is None
            or engine.data_ptr() == source[0].data_ptr()
        ):
            return None
        return source[0], engine

    def fold_state_tensors(self) -> list[Tensor]:
        """The tensors `release_fold` drops: the kept pair and its engine-dtype cast."""
        source, engine = self._pair_source, self._pair_engine
        return [
            t
            for t in (source[0] if source is not None else None, engine)
            if t is not None
        ]

    def release_fold(self) -> None:
        """Drop the kept pair (and its engine-dtype cast): the previous fold's conditioning, ~1 GiB per 1000 tokens squared, and the key the
        engine keeps its hoisted pair bias under."""
        self._pair_source = self._pair_engine = None

    def _engine_pair(self, pair_cond: Tensor) -> Tensor:
        """``pair_cond[None]`` in the engine's activation dtype, the SAME tensor object for every call that gets the same ``pair_cond``
        (one fold: every block of every step): the engine hoists the pair bias (pair LayerNorm + bias GEMM) once per pair object, so a fresh
        cast per call would recompute it per block per step. One entry, kept with its source (identity + in-place version); not kept
        when the pair requires grad (a cast carries an autograd graph that must not outlive its step)."""
        source = self._pair_source
        if (
            source is not None
            and source[0] is pair_cond
            and source[1] == pair_cond._version
        ):
            return self._pair_engine
        pair = engine_activation(pair_cond)[None]
        if torch.is_grad_enabled() and pair_cond.requires_grad:
            return pair
        self._pair_source, self._pair_engine = (pair_cond, pair_cond._version), pair
        return pair


class AtomDiTStack(nn.Module):
    """``blocks``: the engine atom DiT blocks of one family, in order (keys ``blocks.{i}.<engine names>``).

    Called like the dense atom transformer on the windowed layout: ``queries_act`` ``[..., windows, 32, d]``, ``queries_mask``
    ``[windows, 32]``, ``queries_single_cond`` ``[windows, 32, d]``, ``pair_cond`` ``[windows, 32, 128, d_pair]``. The engine block derives the
    128-key window itself (from the key mask: the real atoms come first), so ``queries_to_keys`` / ``keys_mask`` / ``keys_single_cond`` are
    not read; ``engine_inputs = (pair [1, windows, 32, 128, d_pair], key_mask [1, atoms], uid)`` is the per-fold layout the caller
    made once (the engine keys its pair bias on the tensor object); ``uid`` only for Chai-1's blocks."""

    def __init__(self, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        #: ESMFold2's block (``AtomDiT("esmfold2_pair")``): the atom pair AND rotary q / k (per-atom tables), the ``+/- 64`` rank window inside
        self.rotary = type(blocks[0]).__name__ == "AtomDiTESMFold2Pair"

    def forward(
        self,
        queries_act: Tensor,
        queries_mask: Tensor,
        queries_to_keys: object,
        keys_mask: Tensor,
        queries_single_cond: Tensor,
        keys_single_cond: Tensor,
        pair_cond: Tensor,
        pair_mask: Tensor | None = None,
        rope_q: object = None,
        rope_k: object = None,
        engine_inputs: tuple[Tensor | None, ...] | None = None,
    ) -> Tensor:
        """Every block in order."""
        if (rope_q is not None or rope_k is not None) and not self.rotary:
            raise ValueError(
                "a rotary atom transformer is not an engine AtomDiT block with a pair"
            )
        windows, queries, channels = queries_act.shape[-3:]
        atoms = windows * queries
        lead = queries_act.shape[:-3]
        pair, key_mask, uid = (
            (*engine_inputs, None, None, None) if engine_inputs else (None, None, None)
        )[:3]
        if (
            pair_mask is not None and uid is None and not self.rotary
        ):  # the rotary block builds its own rank window
            raise ValueError(
                "a pair mask (same-conformer attention) needs the conformer ids in engine_inputs"
            )
        if (
            uid is not None and uid.ndim == 1
        ):  # the encoder keeps the flat ids [N]; the engine takes [B, N]
            uid = uid[None]
        if pair is None:
            pair = pair_cond.reshape(1, windows, queries, -1, pair_cond.shape[-1])
        if key_mask is None:
            key_mask = (queries_mask > 0).reshape(1, atoms)
        x = engine_activation(
            queries_act.reshape(-1, 1, atoms, channels)
        )  # the engine's master blocks take bf16 activations, not fp32
        cond = engine_activation(
            queries_single_cond.reshape(1, 1, atoms, channels)
        ).expand(x.shape[0], 1, atoms, channels)
        pair = engine_activation(pair)
        if self.rotary:
            if rope_q is None:
                raise ValueError(
                    "the ESMFold2 atom block rotates q and k: the caller passes the rope tables"
                )
            # the per-atom tables of the queries layout [windows, 32, r] (a key takes its own atom's row: the block gathers them itself)
            rope = tuple(
                table.reshape(1, atoms, table.shape[-1]).to(torch.float32)
                for table in rope_q
            )
            for block in self.blocks:
                x = block(x, cond, pair, key_mask, rope=rope)
            return x.reshape(*lead, windows, queries, channels)
        for block in self.blocks:
            x = (
                block(x, cond, pair, key_mask, uid)
                if uid is not None
                else block(x, cond, pair, key_mask)
            )
        return x.reshape(*lead, windows, queries, channels)


def engine_key_window(window: str) -> tuple[str, str | None]:
    """``(engine key_window, note)`` of a FoldForge ``atom_key_window``. The engine has ``pad`` / ``slide`` / ``slide_qblock``; Chai-1's
    exact ``circular`` was removed (it becomes ``pad`` + the conformer-id mask ``uid``; output may differ: ``docs/ENGINE-UNSUPPORTED.md``)."""
    if window in KEY_WINDOWS:
        return window, None
    if window == "circular":
        return (
            "pad",
            "circular key window is not an engine key window: pad + uid (outputs may differ)",
        )
    raise ValueError(f"unknown atom key window {window!r}")


def token_dit_stack(
    family: str,
    *,
    n_blocks: int,
    d_single: int,
    d_cond: int,
    d_pair: int,
    n_head: int,
    n: int = 2,
    adaln_eps: float | None = None,
    precision: Precision = "master",
) -> TokenDiTStack:
    """``n_blocks`` ``TokenDiT(model=...)`` blocks of ``family``. ``adaln_eps``: Chai-1's AdaLN epsilon (engine default 0.1, the release;
    FoldForge's fast mode 1e-5)."""
    key = model_keys(family).token_dit
    if key is None:
        raise ValueError(f"family {family!r} has no engine token DiT")
    extra = {"adaln_eps": adaln_eps} if adaln_eps is not None else {}
    blocks = [
        TokenDiT(key, d_single, d_cond, d_pair, n_head, n, **extra)
        for _ in range(n_blocks)
    ]
    return finish(TokenDiTStack(blocks), precision)  # type: ignore[return-value]


def atom_dit_stack(
    family: str,
    *,
    n_blocks: int,
    d_single: int = 128,
    d_cond: int = 128,
    d_pair: int = 16,
    n_head: int = 4,
    n: int = 2,
    key_window: str | None = None,
    adaln_eps: float | None = None,
    precision: Precision = "master",
) -> AtomDiTStack:
    """``n_blocks`` ``AtomDiT(model=...)`` blocks of ``family``. ``key_window``: the FoldForge fast modes pass ``"slide"``
    (:func:`engine_key_window` for the spec's own word); ``None`` keeps the class's default. ``adaln_eps``: Chai-1's."""
    key = model_keys(family).atom_dit
    if key is None:
        raise ValueError(
            f"family {family!r} has no engine atom DiT with a pair (see docs/ENGINE-UNSUPPORTED.md)"
        )
    extra: dict[str, Any] = {"key_window": key_window} if key_window is not None else {}
    if adaln_eps is not None:
        extra["adaln_eps"] = adaln_eps
    if (
        key == "esmfold2_pair"
    ):  # one window (192 keys, the +/- 64 rank band) and its own AdaLN epsilon: nothing to choose
        extra = {}
    blocks = [
        AtomDiT(key, d_single, d_cond, d_pair, n_head, n, **extra)
        for _ in range(n_blocks)
    ]
    stack = AtomDiTStack(blocks)
    stack.key_dim_per_head = (
        d_single // n_head
    )  # the per-head width the caller builds the atom rotary tables for (the dense stack has the same attribute)
    return finish(stack, precision)  # type: ignore[return-value]
