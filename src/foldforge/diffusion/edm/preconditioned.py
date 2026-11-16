"""EDM network preconditioning around the common Euler sampler."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from pydantic import Field

from foldforge.diffusion.augmentation import random_rigid_motion
from foldforge.diffusion.edm.sampling import (
    EulerSampler,
    PreconditionedDenoiser,
)
from foldforge.diffusion.edm.schedules import cap_noise_schedule
from foldforge.diffusion.geometry.kabsch import weighted_kabsch_align

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from foldforge.diffusion.base.solver import ModelFn


class PreconditionedEulerSolver:
    """A scaled-coordinate network with optional rigid alignment and sigma cap."""

    class Config(EulerSampler.Config):
        """Configuration for the PreconditionedEulerSolver class."""

        augment_per_step: bool = True
        translation_noise: float = Field(default=1.0, ge=0)
        align_to_denoised: bool = True
        max_sigma: float | None = Field(default=256.0, gt=0)
        clamp_variance: bool = True

    def __init__(self, config: Config, scheduler: Any) -> None:
        self.config, self.scheduler = config, scheduler
        self.sampler = EulerSampler(
            EulerSampler.Config.model_validate(
                {
                    name: getattr(config, name)
                    for name in EulerSampler.Config.model_fields
                }
            )
        )

    def solver_sigmas(
        self, num_steps: int, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """Noise levels of a ``num_steps`` schedule, capped at ``max_sigma``."""
        sigmas = self.scheduler.sampling_time_steps(num_steps).to(device)
        return cap_noise_schedule(sigmas, self.config.max_sigma)

    def _options(
        self, mask: torch.Tensor | None, generator: torch.Generator | None
    ) -> dict[str, Any]:
        def align(noisy: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
            weights = (
                mask.to(denoised.dtype)
                if mask is not None
                else denoised.new_ones(denoised.shape[:-1])
            )
            return weighted_kabsch_align(
                noisy.float(), denoised.float(), weights, None
            ).to(denoised.dtype)

        return {
            "augment": partial(
                random_rigid_motion,
                mask=mask,
                trans_scale=self.config.translation_noise,
                generator=generator,
            )
            if self.config.augment_per_step
            else None,
            "align": align if self.config.align_to_denoised else None,
        }

    def denoise_step(
        self,
        denoise_fn: ModelFn,
        coords: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Advance ``coords`` by one Euler step from ``sigma`` to ``sigma_next``."""
        return self.sampler.step(
            PreconditionedDenoiser(denoise_fn, self.scheduler),
            coords,
            sigma,
            sigma_next,
            generator=generator,
            **self._options(mask, generator),
        )

    def sample(
        self,
        denoise_fn: ModelFn,
        shape: Sequence[int],
        num_steps: int,
        device: torch.device | str | None = None,
        *,
        generator: torch.Generator | None = None,
        mask: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> Any:
        """Sample from noise; with ``return_trajectory`` also the trajectory."""
        result = self.sampler.sample(
            PreconditionedDenoiser(denoise_fn, self.scheduler),
            tuple(shape),
            self.solver_sigmas(num_steps, device),
            device=device,
            generator=generator,
            return_trajectory=return_trajectory,
            **self._options(mask, generator),
        )
        return result[:2] if return_trajectory else result
