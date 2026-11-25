"""Gradient of the chiral-centre dihedral error, RoseTTAFold3's atom input.

RoseTTAFold3's diffusion encoder adds a projection of d/dx sum (dih - target)^2
over the plane pairs of every chiral centre (``data.features.chirality``),
taken on the scaled noisy coordinates at every step, so the denoiser can tell
a chiral centre from its mirror image. The release computes it in closed form
rather than by autograd, which also keeps it usable under inference mode; this
is that closed form (rf3 ``calc_ddihedralmse_dxyz``), over any leading axes.
"""

from __future__ import annotations

import torch


def _segment_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """Sum ``values`` (..., m, 3) into ``size`` rows by ``index`` (m,), the same way at every call: float32 (..., size, 3).

    The chiral centres overlap (an atom is in several), and ``index_add_`` adds with float atomics in whatever order the threads
    arrive, so two identical calls could differ in the last bits (run-to-run differences of whole structures after the sampler's
    200 steps). Here the rows are sorted by atom (stable), summed by a float64 prefix sum (one fixed order; float64 makes the
    result the correctly rounded float32 sum up to the rare tie, whatever the order of the centres) and cut at the segment bounds.
    Static shapes and no host sync, so it holds under ``torch.compile`` and CUDA-graph capture. Non-finite values are zeroed
    first, as the caller's ``nan_to_num`` did after the sum (without it one NaN would poison every later segment of the prefix sum).
    """
    order = torch.argsort(index, stable=True)
    sorted_index = index[order]
    summed = torch.cumsum(values.nan_to_num()[..., order, :].double(), dim=-2)
    summed = torch.cat([torch.zeros_like(summed[..., :1, :]), summed], dim=-2)
    atoms = torch.arange(size, device=index.device)
    first = torch.searchsorted(sorted_index, atoms)
    last = torch.searchsorted(sorted_index, atoms, right=True)
    return (summed[..., last, :] - summed[..., first, :]).float()


def dihedral_error_gradient(
    positions: torch.Tensor, centres: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """d/dx of the summed squared dihedral error, per atom.

    ``positions`` is (..., n_atoms, 3), ``centres`` (n, 4) flat atom indices,
    ``targets`` (n,) radians. Returns (..., n_atoms, 3); zero with no centres.
    """
    if centres.shape[0] == 0:
        return torch.zeros_like(positions, dtype=torch.float32)
    eps = 1e-6
    x = positions.float()[..., centres, :]  # (..., n, 4, 3)
    a, b, c, d = x.unbind(-2)
    b0, b1, b2 = a - b, c - b, d - c
    b1_norm = torch.linalg.norm(b1, dim=-1, keepdim=True)
    b1n = b1 / (b1_norm + eps)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    px = (v * w).sum(-1)
    py = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
    dih = torch.atan2(py + eps, px + eps)

    eye = torch.eye(3, device=x.device, dtype=x.dtype)
    outer = lambda p, q: p[..., :, None] * q[..., None, :]
    dmse = (2 * (dih - targets.to(x))).unsqueeze(-1)  # (..., n, 1)
    denom = px**2 + py**2 + eps
    ddih_dx = (-py / denom).unsqueeze(-1)
    ddih_dy = (px / denom).unsqueeze(-1)
    dy_dv = -torch.cross(b1n, w, dim=-1)
    dy_dw = torch.cross(b1n, v, dim=-1)
    dx_dv, dx_dw = w, v
    norm2 = (b1_norm**2 + eps).unsqueeze(-1)
    db1n_db1 = (b1_norm + eps).unsqueeze(-1) * eye / norm2 - outer(b1, b1) / norm2
    dw_db1n = -(b2 * b1n).sum(-1)[..., None, None] * eye - outer(b2, b1n)
    dv_db1n = -(b0 * b1n).sum(-1)[..., None, None] * eye - outer(b0, b1n)
    project = eye - outer(b1n, b1n)  # dv/db0 and dw/db2
    row = lambda vec, mat: (vec[..., None, :] @ mat).squeeze(-2)

    def through(dv: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        """The error's gradient given dv/dp and dw/dp for one atom p."""
        dx = row(dx_dv, dv) + row(dx_dw, dw)
        dy = row(dy_dv, dv) + row(dy_dw, dw)
        return dmse * (ddih_dx * dx + ddih_dy * dy)

    zero = torch.zeros_like(project)
    db1n_db = -db1n_db1
    grad_a = through(project, zero)
    grad_b = through(
        -project + dv_db1n.transpose(-1, -2) @ db1n_db,
        dw_db1n.transpose(-1, -2) @ db1n_db,
    )
    grad_c = through(
        dv_db1n.transpose(-1, -2) @ db1n_db1,
        -project + dw_db1n.transpose(-1, -2) @ db1n_db1,
    )
    grad_d = through(zero, project)
    per_centre = torch.stack([grad_a, grad_b, grad_c, grad_d], dim=-2)  # (..., n, 4, 3)
    return _segment_sum(
        per_centre.flatten(-3, -2), centres.reshape(-1), positions.shape[-2]
    )
