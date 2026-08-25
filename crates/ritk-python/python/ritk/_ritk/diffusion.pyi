"""Type stubs for the ``_ritk.diffusion`` submodule (tensor fitting)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ritk._ritk.image import Image

class DiffusionMaps:
    """Fitted tensor field and the scalar maps derived from it.

    Every accessor returns a fresh array shaped ``[Z, Y, X]``, matching the
    volumes the fit was given; the eigenvector field is ``[Z, Y, X, 3]``.

    Voxels that were not fitted read zero in every map, so ``mask`` is the only
    thing that separates them from a voxel genuinely measured as isotropic.
    """

    @property
    def fitted_count(self) -> int:
        """Voxels that yielded a physically admissible tensor."""
        ...

    def __len__(self) -> int:
        """Voxels in the volume."""
        ...

    def mask(self) -> npt.NDArray[np.bool_]:
        """Which voxels were fitted, shaped ``[Z, Y, X]``."""
        ...

    def fractional_anisotropy(self) -> npt.NDArray[np.float32]:
        """Fractional anisotropy in ``[0, 1]``, shaped ``[Z, Y, X]``."""
        ...

    def mean_diffusivity(self) -> npt.NDArray[np.float32]:
        """Mean diffusivity in mm²/s, shaped ``[Z, Y, X]``."""
        ...

    def axial_diffusivity(self) -> npt.NDArray[np.float32]:
        """Axial diffusivity ``λ₁`` in mm²/s, shaped ``[Z, Y, X]``."""
        ...

    def radial_diffusivity(self) -> npt.NDArray[np.float32]:
        """Radial diffusivity ``(λ₂ + λ₃) / 2`` in mm²/s, shaped ``[Z, Y, X]``."""
        ...

    def principal_eigenvector(self) -> npt.NDArray[np.float32]:
        """Local fibre orientation, shaped ``[Z, Y, X, 3]``.

        Unit length wherever a tensor was fitted, exactly zero elsewhere. The
        vector carries no sign: ``v`` and ``-v`` describe the same fibre.
        """
        ...

def fit_tensor_maps(
    volumes: Sequence[Image],
    bvals: Sequence[float],
    bvecs: Sequence[Sequence[float]],
    background_fraction: float | None = None,
) -> DiffusionMaps:
    """Fit one diffusion tensor per voxel and derive its scalar maps.

    Raises:
        ValueError: if the volume, b-value and direction counts disagree, the
            volumes do not share a grid, the scheme has no ``b = 0`` reference,
            or ``background_fraction`` is not a usable number.
    """
    ...
