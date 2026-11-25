"""Legacy AF3 interface over the shared EDM sampler.

The constructor's historical seed/backend settings remain for existing callers.
New applications use EulerSampler or PreconditionedEulerSolver with an explicit
Generator; neither changes process-global settings.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from foldjet.diffusion.augmentation import scipy_rigid_motion
from foldjet.diffusion.base.solver import DiffusionSolver
from foldjet.diffusion.edm.sampling import (
    EulerSampler,
    PreconditionedDenoiser,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from foldjet.diffusion.base.scheduler import DiffusionScheduler
    from foldjet.diffusion.base.solver import ModelFn


class AF3Solver(DiffusionSolver):
    """AF3 sampler (Algorithm 18) over the shared Euler sampler."""

    def __init__(
        self, config: DiffusionSolver.SolverConfig, scheduler: DiffusionScheduler
    ) -> None:
        super().__init__(config, scheduler)
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self._lambda = 1.003
        self.step_scale = 1.5
        self.aug_translation = 1.0

    def _sampler(self) -> EulerSampler:
        return EulerSampler(
            EulerSampler.Config(
                gamma_0=self.gamma_0,
                gamma_min=self.gamma_min,
                noise_scale=self._lambda,
                step_scale=self.step_scale,
            )
        )

    def _denoiser(self, model_fn: ModelFn) -> PreconditionedDenoiser:
        return PreconditionedDenoiser(
            model_fn, self.scheduler, condition_noise=True, batch_noise=False
        )

    @torch.no_grad()
    def _centre_random_augmentation(
        self, coords: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return scipy_rigid_motion(coords, mask, self.aug_translation)

    def step(
        self,
        model_fn: ModelFn,
        x: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One Euler step from ``time_steps[t_index]`` to the next time step."""
        return self._sampler().step(
            self._denoiser(model_fn),
            x,
            self.scheduler.sampling_schedule(time_steps[t_index]),
            self.scheduler.sampling_schedule(time_steps[t_index + 1]),
            augment=partial(self._centre_random_augmentation, mask=mask),
        )

    def sample(
        self,
        model_fn: ModelFn,
        shape: Sequence[int],
        num_steps: int,
        device: torch.device | str | None,
        *,
        mask: torch.Tensor | None = None,
        return_intermediate: bool = False,
    ) -> Any:
        """Sample from noise; with ``return_intermediate`` also the trajectory."""
        times = self.scheduler.sampling_time_steps(num_steps).to(device)
        sigmas = torch.stack([self.scheduler.sampling_schedule(t) for t in times])
        initial = torch.randn(shape, device=device) * sigmas[0]
        return self._sampler().integrate(
            self._denoiser(model_fn),
            initial,
            sigmas,
            augment=partial(self._centre_random_augmentation, mask=mask),
            return_trajectory=return_intermediate,
        )
