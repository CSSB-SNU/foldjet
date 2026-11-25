"""MiniWorld cropped sample -> FoldJet dense ``TrainBatch`` tensors (CPU, numpy), built with FoldJet's / AF3's OWN featurisation code.

The structure side is not re-implemented: the cropped CIFMol's atoms become an AF3 flat ``AtomLayout`` and AF3's ``tokenizer`` + ``features.*``
classes (the ones ``foldjet.models.io.dense_atoms`` runs for inference) build every layout-dependent tensor (token features, atom-attention
gathers, reference conformers, bonds, pseudo-beta, frames), then FoldJet's ``dense_conventions`` of the protenix1 ``DenseSpec`` are applied. What
MiniWorld contributes is what AF3's inference featuriser takes from the *input*: the crop (which atoms / residues), the sampled MSA, the sampled
templates, the ground-truth coordinates. MSA and templates are translated from MiniWorld's arrays into AF3's alphabet / dense layout (see
``msa_rows`` and ``template_arrays``).
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mwconfig  # noqa: F401  (import shims)
import numpy as np

logging.getLogger("absl").setLevel(
    logging.ERROR
)  # AF3 logs a warning per atom-token without a pseudo-beta atom

ATOMS_PER_TOKEN = 24

# MiniWorld residue alphabet (miniworld.data.constants.mapping.ResidueMapping: protein A R N D C Q E G H I L K M F P S T W Y V X = 0..20,
# RNA A U G C N = 21..25, DNA A T G C N = 26..30, gap 31) -> AF3's (alphafold3.constants.residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP:
# protein 0..20, gap 21, RNA A G C U = 22..25, DNA DA DG DC DT = 26..29, N = 30).
MW_TO_AF3 = np.array(
    [*range(21), 22, 25, 23, 24, 30, 26, 29, 27, 28, 30, 21], dtype=np.int64
)  # index = MiniWorld id
AF3_GAP = 21
AF3_UNK = 20
AF3_NUM_CLASSES = 31
DETERMINISTIC_FRAMES_SEED = 12312837  # alphafold3.model.pipeline.pipeline._DETERMINISTIC_FRAMES_RANDOM_SEED (frames of ligand tokens come from this conformer)


class ConversionError(ValueError):
    """The sample cannot be expressed in FoldJet's layout (counted and skipped by the loader)."""


@dataclass
class Converted:
    name: str
    features: dict[str, np.ndarray]
    atom_positions: np.ndarray  # [tokens, 24, 3] float32, centred on the resolved atoms
    atom_mask: np.ndarray  # [tokens, 24] bool (resolved)
    bond_pairs: np.ndarray  # [P, 4] int64, -1 padded
    present_index: np.ndarray
    present_valid: np.ndarray
    info: dict[str, Any]
    tok_res: np.ndarray | None = (
        None  # residue index (into the cropped CIFMol) of every AF3 token
    )


class Converter:
    def __init__(
        self,
        *,
        family: str = "protenix1",
        mode: str = "fast",
        tokens: int = 384,
        atoms: int = 4096,
        msa_pool: int = 8192,
        templates: int = 4,
        bond_rows: int = 128,
        fixed_atoms: int = 4096,
        ccd_root: str | Path | None = None,
    ) -> None:
        from foldjet import paths
        from foldjet.data.ccd import CCDDatabase

        self.tokens, self.atoms, self.msa_pool, self.templates = (
            tokens,
            atoms,
            msa_pool,
            templates,
        )
        self._install_fast_ref()
        self.bond_rows, self.fixed_atoms = bond_rows, fixed_atoms
        self.db = CCDDatabase(ccd_root or paths.ccd_db())
        self.ccd = self.db.af3_ccd(None)
        from foldjet.models import entry
        from foldjet.modules.dense.spec import spec_for

        spec = spec_for(entry(family).family or "alphafold3", mode)
        # drop_atoms removes OXT / OP3 from the layouts of the INFERENCE featurisation; MiniWorld's loader already removed them from the cropped
        # CIFMol (``remove_terminal_oxygen``), so the flat layout never has them and the layouts are the post-drop ones. Needs layouts that this
        # path does not keep as objects.
        self.spec_full = spec
        self.spec = dataclasses.replace(spec, drop_atoms=())

    _parallel_embed = False

    def _install_fast_ref(self) -> None:
        """Cached bit-identical get_reference (see fast_ref.py); FT_FAST_REF=0 disables it, FT_CONF_CACHE is the conformer cache directory."""
        import os

        from alphafold3.model import features

        if os.environ.get("FT_FAST_REF", "1") == "0":
            return
        import fast_ref

        fast_ref.install(features)
        self._parallel_embed = os.environ.get("FT_PARALLEL_EMBED", "1") == "1"
        cache = os.environ.get(
            "FT_CONF_CACHE", str(Path(__file__).resolve().parent.parent / "conf_cache")
        )
        if os.path.isdir(cache) or os.path.isfile(cache):
            fast_ref.load_cache(cache)

    # ------------------------------------------------------------------------------------------------------------ layouts
    def flat_layout(self, raw) -> tuple[Any, np.ndarray]:
        """AF3 flat layout of the cropped CIFMol atoms (and the per-atom chain type)."""
        from alphafold3.model.atom_layout import atom_layout

        c = raw.cifmol
        atom_to_res = np.asarray(c.index_table.atom_to_res).astype(np.int64)
        res_to_chain = np.asarray(c.index_table.res_to_chain).astype(np.int64)
        chain_ids = np.asarray(c.chains.chain_id.value).astype(str)
        chain_types = np.asarray(c.chains.entity_type.value).astype(str)
        res_names = np.char.strip(np.asarray(c.residues.chem_comp_id.value).astype(str))
        res_ids = np.asarray(c.residues.cif_idx.value).astype(np.int64)
        atom_chain = res_to_chain[atom_to_res]
        known = {
            "polypeptide(L)",
            "polypeptide(D)",
            "polyribonucleotide",
            "polydeoxyribonucleotide",
            "polydeoxyribonucleotide/polyribonucleotide hybrid",
            "non-polymer",
            "branched",
            "macrolide",
            "peptide nucleic acid",
            "cyclic-pseudo-peptide",
            "other",
        }
        bad = sorted(set(chain_types[atom_chain]) - known)
        if bad:
            raise ConversionError(f"unsupported chain type(s) {bad}")
        n = len(atom_to_res)
        layout = atom_layout.AtomLayout(
            atom_name=np.asarray(c.atoms.id.value).astype(str).astype(object),
            atom_element=np.char.capitalize(
                np.asarray(c.atoms.element.value).astype(str)
            ).astype(object),
            res_name=res_names[atom_to_res].astype(object),
            res_id=res_ids[atom_to_res].astype(int),
            chain_id=chain_ids[atom_chain].astype(object),
            chain_type=chain_types[atom_chain].astype(object),
        )
        assert layout.shape == (n,)
        return layout, chain_types[atom_chain]

    def _bond_layouts(self, raw, flat) -> tuple[Any, Any]:
        """(polymer_ligand_bonds, ligand_ligand_bonds) AtomLayouts [n, 2] from the crop's covalent ``struct_conn`` (inter-residue); polymer-polymer
        bonds are dropped (AF3 ``remove_polymer_polymer_bonds``); intra-ligand bonds come from the CCD inside ``RefStructure.compute_features``.
        """
        from alphafold3.constants import mmcif_names
        from alphafold3.model.atom_layout import atom_layout

        try:
            sc = raw.cifmol.atoms.struct_conn
            kind = np.asarray(sc.value)[:, 0]
            src = np.asarray(sc.src).astype(np.int64)
            dst = np.asarray(sc.dst).astype(np.int64)
        except Exception:  # no struct_conn edges in this crop
            return None, None
        keep = kind == "covale"
        src, dst = src[keep], dst[keep]
        poly_types = mmcif_names.POLYMER_CHAIN_TYPES
        ctype = flat.chain_type
        rows_pl, rows_ll = [], []
        for a, b in zip(src.tolist(), dst.tolist(), strict=True):
            if not (0 <= a < flat.shape[0] and 0 <= b < flat.shape[0]):
                continue
            if (flat.chain_id[a], flat.res_id[a]) == (flat.chain_id[b], flat.res_id[b]):
                continue
            pa, pb = ctype[a] in poly_types, ctype[b] in poly_types
            if pa and pb:
                continue
            (rows_pl if (pa or pb) else rows_ll).append((a, b))

        def layout(rows):
            if not rows:
                return None
            idx = np.asarray(rows, dtype=np.int64)
            return atom_layout.AtomLayout(
                atom_name=np.stack(
                    [flat.atom_name[idx[:, 0]], flat.atom_name[idx[:, 1]]], axis=1
                ).astype(object),
                res_id=np.stack(
                    [flat.res_id[idx[:, 0]], flat.res_id[idx[:, 1]]], axis=1
                ).astype(int),
                chain_id=np.stack(
                    [flat.chain_id[idx[:, 0]], flat.chain_id[idx[:, 1]]], axis=1
                ).astype(object),
                chain_type=np.stack(
                    [flat.chain_type[idx[:, 0]], flat.chain_type[idx[:, 1]]], axis=1
                ).astype(object),
            )

        return layout(rows_pl), layout(rows_ll)

    # ------------------------------------------------------------------------------------------------------------ MSA / templates
    def msa_rows(
        self, raw, tok_res: np.ndarray, token_is_ligand: np.ndarray, t_real: int
    ) -> dict[str, np.ndarray]:
        """AF3-format MSA arrays [pool, tokens] from MiniWorld's RESIDUE-level MSA sample (``raw.msa``), expanded to AF3's tokens by ``tok_res``
        (residue of each AF3 token: a flattened non-standard residue repeats its column, as in AF3's ``standard_token_idxs``).

        rows: translated alphabet (``MW_TO_AF3``); true non-polymer (ligand) token columns are all-gap like AF3's ``Msa.from_empty`` (the Protenix
        convention then fills them with the token's own type). deletion counts are recovered from MiniWorld's ``2 arctan(d / 3) / pi`` encoding by
        inversion + rounding (exact for integer d). profile: MiniWorld's [R, 32] columns re-indexed to AF3's 31.
        """
        m = raw.msa
        rows = m.aligned_sequences[0].numpy().astype(np.int64)[:, tok_res]  # [N, T]
        n, t = rows.shape
        if t != t_real:
            raise ConversionError(f"msa has {t} columns, tokens {t_real}")
        if n > self.msa_pool:
            raise ConversionError(f"msa depth {n} above the pool {self.msa_pool}")
        valid = m.mask[0].numpy().astype(bool)
        if not valid.all():
            raise ConversionError("msa mask has holes")
        rows = MW_TO_AF3[np.clip(rows, 0, 31)]
        dv = m.deletion_value[0].numpy().astype(np.float64)[:, tok_res]
        hd = (m.has_deletion[0].numpy() > 0)[:, tok_res]
        dele = np.where(
            hd, np.rint(3.0 * np.tan(0.5 * np.pi * np.clip(dv, 0.0, 0.999999))), 0.0
        )
        rows[:, token_is_ligand] = AF3_GAP
        dele[:, token_is_ligand] = 0
        prof_mw = m.profile[0].numpy().astype(np.float64)[tok_res]  # [T, 32]
        prof = np.zeros((t, AF3_NUM_CLASSES), dtype=np.float64)
        for mw_id in range(prof_mw.shape[1]):
            prof[:, MW_TO_AF3[mw_id]] += prof_mw[:, mw_id]
        prof[token_is_ligand] = 0.0
        prof[token_is_ligand, AF3_GAP] = 1.0
        out = {
            "msa": np.zeros((self.msa_pool, self.tokens), np.int8),
            "msa_mask": np.zeros((self.msa_pool, self.tokens), bool),
            "deletion_matrix": np.zeros((self.msa_pool, self.tokens), np.int8),
            "profile": np.zeros((self.tokens, AF3_NUM_CLASSES), np.float32),
            "deletion_mean": np.zeros((self.tokens,), np.float32),
            "num_alignments": np.array(n, dtype=np.int32),
        }
        out["msa"][:n, :t] = rows.astype(np.int8)
        out["msa_mask"][:n, :t] = True
        out["deletion_matrix"][:n, :t] = np.clip(dele, -128, 127).astype(np.int8)
        out["profile"][:t] = prof.astype(np.float32)
        out["deletion_mean"][:t] = (
            m.deletion_mean[0].numpy().astype(np.float32)[tok_res]
        )
        out["deletion_mean"][:t][token_is_ligand] = 0.0
        return out

    def template_arrays(
        self, raw, tok_res: np.ndarray, is_protein: np.ndarray, t_real: int
    ) -> dict[str, np.ndarray]:
        """AF3 template arrays: aatype [T, tokens], atom_positions [T, tokens, 24, 3], atom_mask [T, tokens, 24] in AF3's per-residue-type DENSE
        atom order (``atom_types.DENSE_ATOM``: N CA C O CB ...; GLY has no CB). MiniWorld stores the backbone N, CA, C and one CB (CA for glycine)
        per template residue, which are the only atoms the template embedding reads (pseudo-beta for the distogram, N-CA-C for the unit vector);
        the O slot and the side chains stay masked.
        """
        tm = (
            raw.templates
        )  # miniworld ProteinTemplate, residue level, numpy, no batch axis
        mask_t = np.asarray(tm.mask).astype(bool)  # [Tmpl]
        out = {
            "template_aatype": np.zeros((self.templates, self.tokens), np.int32),
            "template_atom_positions": np.zeros(
                (self.templates, self.tokens, ATOMS_PER_TOKEN, 3), np.float32
            ),
            "template_atom_mask": np.zeros(
                (self.templates, self.tokens, ATOMS_PER_TOKEN), bool
            ),
        }
        n = min(int(mask_t.sum()), self.templates)
        if n == 0:
            return out
        order = np.flatnonzero(mask_t)[:n]
        res_type = np.asarray(tm.res_type)[order][:, tok_res]  # [n, T]
        bb = np.asarray(tm.bb_xyz)[order][:, tok_res].astype(np.float32)  # [n, T, 3, 3]
        bbm = np.asarray(tm.bb_mask)[order][:, tok_res].astype(bool)
        cb = np.asarray(tm.cb_xyz)[order][:, tok_res].astype(np.float32)
        cbm = np.asarray(tm.cb_mask)[order][:, tok_res].astype(bool)
        t = t_real
        aa = MW_TO_AF3[np.clip(res_type, 0, 31)]
        gly = aa == 7
        prot = np.broadcast_to(is_protein[None], aa.shape)
        aa = np.where(prot, aa, 0)
        out["template_aatype"][:n, :t] = aa
        pos = out["template_atom_positions"]
        msk = out["template_atom_mask"]
        ok = prot & bbm
        for slot in range(3):  # N, CA, C -> dense slots 0, 1, 2
            pos[:n, :t, slot] = np.where(ok[..., None], bb[:, :, slot], 0.0)
            msk[:n, :t, slot] = ok
        has_cb = prot & cbm & ~gly
        pos[:n, :t, 4] = np.where(has_cb[..., None], cb, 0.0)
        msk[:n, :t, 4] = has_cb
        return out

    # ------------------------------------------------------------------------------------------------------------ the conversion
    def convert(self, raw, seed: int, conformer_seed: int | None = None) -> Converted:
        from alphafold3.model import features
        from alphafold3.model.atom_layout import atom_layout
        from alphafold3.structure import chemical_components as struc_chem_comps

        from foldjet.data.ccd.database import (
            database_cache,  # noqa: F401  (keeps the import side effects of the inference path)
        )
        from foldjet.data.features import bond_orders, chirality, dense_conventions
        from foldjet.models.io.dense_atoms import _install_reference_date_fallback

        t0 = time.perf_counter()
        _install_reference_date_fallback()
        rs = np.random.SeedSequence(seed)
        conf_seed, pose_seed = (int(x) for x in rs.generate_state(2, dtype=np.uint32))
        if (
            conformer_seed is not None
        ):  # tests: the seed AF3 featurisation of the same structure uses, to compare ref_pos exactly
            conf_seed = int(conformer_seed)
        flat, _ = self.flat_layout(raw)
        name = raw.item_id
        with self.db.activate():
            from alphafold3.constants import chemical_component_sets

            with chemical_component_sets.use_ccd_sets(self.db.chemical_component_sets):
                return self._convert(
                    raw,
                    seed,
                    conf_seed,
                    pose_seed,
                    flat,
                    name,
                    features,
                    atom_layout,
                    struc_chem_comps,
                    bond_orders,
                    chirality,
                    dense_conventions,
                    t0,
                )

    def _convert(
        self,
        raw,
        seed,  # noqa: ARG002 - kept for the call signature
        conf_seed,
        pose_seed,
        flat,
        name,
        features,
        atom_layout,
        struc_chem_comps,
        bond_orders,
        chirality,
        dense_conventions,
        t0,
    ) -> Converted:
        all_tokens, token_atoms, _standard_idx = features.tokenizer(
            flat,
            ccd=self.ccd,
            max_atoms_per_token=ATOMS_PER_TOKEN,
            flatten_non_standard_residues=True,
            logging_name=name,
        )
        t_real = int(all_tokens.shape[0])
        n_flat = int(flat.shape[0])
        # AF3's token of every residue / ligand atom -> residue index of the cropped CIFMol (through the representative atom's uid). AF3 and MiniWorld
        # tokenise slightly differently (AF3: MSE / UNK / unknown nucleotides are ONE token, MiniWorld's atom tokenizer: one token per atom since they
        # are not in its canonical set), so the MiniWorld token axis is NOT used: MSA and templates are taken at residue level and indexed by AF3's tokens.
        uid_to_atom = {
            u: i
            for i, u in enumerate(
                zip(
                    flat.chain_id.tolist(),
                    flat.res_id.tolist(),
                    flat.atom_name.tolist(),
                    strict=True,
                )
            )
        }
        atom_to_res = np.asarray(raw.cifmol.index_table.atom_to_res).astype(np.int64)
        try:
            tok_res = np.array(
                [
                    atom_to_res[uid_to_atom[u]]
                    for u in zip(
                        all_tokens.chain_id.tolist(),
                        all_tokens.res_id.tolist(),
                        all_tokens.atom_name.tolist(),
                        strict=True,
                    )
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ConversionError(
                f"token representative atom not in the layout: {exc}"
            ) from exc
        n_mw_tokens = int(raw.batch.token_length)
        if t_real > self.tokens or n_flat > self.atoms:
            raise ConversionError(f"{t_real} tokens / {n_flat} atoms above the bucket")
        padding = features.PaddingShapes(
            num_tokens=self.tokens,
            msa_size=self.msa_pool,
            num_chains=1000,
            num_templates=self.templates,
            num_atoms=self.atoms,
        )
        cross = features.AtomCrossAtt.compute_features(
            all_token_atoms_layout=token_atoms,
            queries_subset_size=32,
            keys_subset_size=128,
            padding_shapes=padding,
        )
        token_feats = features.TokenFeatures.compute_features(
            all_tokens=all_tokens, padding_shapes=padding
        )
        pred_info = features.PredictedStructureInfo.compute_features(
            all_tokens=all_tokens,
            all_token_atoms_layout=token_atoms,
            padding_shapes=padding,
        )
        # polymer chemistry data for reference conformers (CCD components only: nothing user-defined in the PDB data)
        chem = struc_chem_comps.populate_missing_ccd_data(
            ccd=self.ccd,
            chemical_components_data=struc_chem_comps.ChemicalComponentsData(
                chem_comp={}
            ),
            populate_pdbx_smiles=True,
        )
        pl_bonds, ll_bonds = self._bond_layouts(raw, flat)
        rng = np.random.RandomState(conf_seed)
        ref_kw = {
            "all_token_atoms_layout": token_atoms,
            "ccd": self.ccd,
            "padding_shapes": padding,
            "chemical_components_data": chem,
            "ref_max_modified_date": None,
            "conformer_max_iterations": None,
        }
        import fast_ref as _fr

        if self._parallel_embed and _fr.has_big_component(
            np.unique(np.asarray(token_atoms.res_name)), self.ccd, features
        ):
            par = True
            # Recording passes + one thread pool for all the embeds of the example (bit-identical, see fast_ref.precompute): a rare example with ~50 ligand
            # instances that RDKit cannot embed (chlorophyll: 9-18 s per failing embed) took 60-110 s in one worker and stalled the in-order DataLoader.
            import fast_ref

            fast_ref.record_start()
            try:
                features.RefStructure.compute_features(
                    random_state=np.random.RandomState(conf_seed),
                    ligand_ligand_bonds=ll_bonds,
                    **ref_kw,
                )
            finally:
                rec_sample = fast_ref.record_stop()
            with (
                _frames_cacheable()
            ):  # only the conformer-cache misses of the frames pass are recorded
                fast_ref.record_start()
                try:
                    features.RefStructure.compute_features(
                        random_state=np.random.RandomState(DETERMINISTIC_FRAMES_SEED),
                        ligand_ligand_bonds=None,
                        **ref_kw,
                    )
                finally:
                    rec_frames = fast_ref.record_stop()
            fast_ref.precompute(rec_sample + rec_frames)
        else:
            par = False
        ref, ll_bonds = features.RefStructure.compute_features(
            random_state=rng, ligand_ligand_bonds=ll_bonds, **ref_kw
        )
        with _frames_cacheable():
            det_ref, _ = features.RefStructure.compute_features(
                random_state=np.random.RandomState(DETERMINISTIC_FRAMES_SEED),
                ligand_ligand_bonds=None,
                **ref_kw,
            )
        if par:
            import fast_ref

            fast_ref._PRE.clear()
        pl_info = features.PolymerLigandBondInfo.compute_features(
            all_tokens=all_tokens,
            all_token_atoms_layout=token_atoms,
            bond_layout=pl_bonds,
            padding_shapes=padding,
        )
        with bond_orders.capture(self.ccd) as orders:
            ll_info = features.LigandLigandBondInfo.compute_features(
                all_tokens, ll_bonds, padding
            )
        pseudo = features.PseudoBetaInfo.compute_features(
            all_token_atoms_layout=token_atoms,
            ccd=self.ccd,
            padding_shapes=padding,
            logging_name=name,
        )
        frames = features.Frames.compute_features(
            all_tokens=all_tokens,
            all_token_atoms_layout=token_atoms,
            ref_structure=det_ref,
            padding_shapes=padding,
        )

        example: dict[str, Any] = {}
        for part in (
            cross,
            token_feats,
            pred_info,
            ref,
            pl_info,
            ll_info,
            pseudo,
            frames,
        ):
            example.update(part.as_data_dict())
        is_protein = np.asarray(token_feats.is_protein)[:t_real]
        is_ligand = np.asarray(token_feats.is_ligand)[:t_real]
        aatype = np.asarray(token_feats.aatype)[:t_real]
        # consistency of the MiniWorld query row with AF3's token types (alphabet translation check)
        mw_query = MW_TO_AF3[
            np.clip(raw.msa.aligned_sequences[0, 0].numpy().astype(np.int64), 0, 31)
        ][tok_res]
        poly = ~is_ligand
        if not np.array_equal(mw_query[poly], aatype[poly]):
            bad = np.flatnonzero(poly & (mw_query != aatype))[:5]
            raise ConversionError(
                f"MSA query row != AF3 aatype at tokens {bad.tolist()}: {mw_query[bad].tolist()} vs {aatype[bad].tolist()}"
            )
        example.update(self.msa_rows(raw, tok_res, is_ligand, t_real))
        example.update(self.template_arrays(raw, tok_res, is_protein, t_real))
        example[bond_orders.KEY] = orders[0]

        np.random.seed(
            pose_seed % (2**32)
        )  # random_ref_pose draws from numpy's global state
        example = dense_conventions.apply(example, self.spec)
        if self.spec.atom_chiral_features:
            example.update(
                chirality.chiral_features(
                    {**example, "token_atoms_layout": np.array(token_atoms, object)},
                    self.ccd,
                )
            )
        example = dense_conventions.apply_after_bucketing(example, self.spec)

        # ground truth in the dense layout
        xyz = np.asarray(raw.cifmol.atoms.xyz.value, dtype=np.float64)
        finite = np.isfinite(xyz).all(axis=1)
        gather = atom_layout.compute_gather_idxs(
            source_layout=flat, target_layout=token_atoms
        )
        xyz0 = np.where(finite[:, None], xyz, 0.0)
        pos_dense = atom_layout.convert(gather, xyz0, layout_axes=(0,)).astype(
            np.float64
        )  # [t, 24, 3]
        res_dense = (
            atom_layout.convert(gather, finite.astype(np.float32), layout_axes=(0,))
            > 0.5
        )
        res_dense &= gather.gather_mask
        centre = pos_dense[res_dense].mean(axis=0) if res_dense.any() else np.zeros(3)
        pos_dense = np.where(res_dense[..., None], pos_dense - centre, 0.0)
        t_dense = self.tokens
        positions = np.zeros((t_dense, ATOMS_PER_TOKEN, 3), np.float32)
        positions[:t_real] = pos_dense
        atom_mask = np.zeros((t_dense, ATOMS_PER_TOKEN), bool)
        atom_mask[:t_real] = res_dense
        # bond-loss pairs: MiniWorld's covalent pairs (inside the crop, both resolved, < 2.4 A) as (token, slot, token, slot)
        flat_to_dense = np.full(n_flat, -1, np.int64)
        ti, si = np.nonzero(gather.gather_mask)
        flat_to_dense[gather.gather_idxs[ti, si]] = ti * ATOMS_PER_TOKEN + si
        pairs = (
            raw.batch.structure.bond_atom_pairs[0]
            .numpy()
            .astype(np.int64)
            .reshape(-1, 2)
        )
        bond_pairs = np.full((self.bond_rows, 4), -1, np.int64)
        kept = 0
        for a, b in pairs.tolist():
            da, db = flat_to_dense[a], flat_to_dense[b]
            if da < 0 or db < 0 or kept >= self.bond_rows:
                continue
            bond_pairs[kept] = (
                da // ATOMS_PER_TOKEN,
                da % ATOMS_PER_TOKEN,
                db // ATOMS_PER_TOKEN,
                db % ATOMS_PER_TOKEN,
            )
            kept += 1

        import torch

        from foldjet.train.data import fixed_atom_index

        index, valid = fixed_atom_index(torch.from_numpy(atom_mask), self.fixed_atoms)

        features_np = {
            k: np.ascontiguousarray(v)
            for k, v in example.items()
            if isinstance(v, np.ndarray) and v.dtype.kind in "biuf"
        }
        features_np["deletion_mean"] = features_np["deletion_mean"].astype(np.float32)
        info = {
            "tokens": t_real,
            "mw_tokens": n_mw_tokens,
            "atoms": n_flat,
            "resolved_atoms": int(atom_mask.sum()),
            "msa_rows": int(example["num_alignments"]),
            "templates": int(
                (
                    np.asarray(example["template_atom_mask"])
                    .reshape(self.templates, -1)
                    .any(-1)
                ).sum()
            ),
            "bonds": int(kept),
            "convert_s": time.perf_counter() - t0,
        }
        return Converted(
            name,
            features_np,
            positions,
            atom_mask,
            bond_pairs,
            index.numpy(),
            valid.numpy(),
            info,
            tok_res,
        )


def _frames_cacheable():
    """Context of the deterministic frames pass (fixed seed 12312837): its (CCD, rdkit seed) conformers repeat across examples, see fast_ref.py."""
    import contextlib
    import os

    if os.environ.get("FT_FAST_REF", "1") == "0":
        return contextlib.nullcontext()
    import fast_ref

    return fast_ref.cacheable(True)
