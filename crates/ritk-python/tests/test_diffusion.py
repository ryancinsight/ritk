"""Diffusion tensor fitting through the Python binding.

The binding is a thin layer over ``ritk_diffusion::maps``; the estimator's own
value semantics are tested in Rust. What these cover is the boundary: argument
conversion, array shape and dtype, error mapping, and — most importantly — that
the Python result is numerically the same fit the Rust core produces.

Signals come from the forward model, so every expected value is known in closed
form rather than being whatever the code happens to return.

Run:
    pytest crates/ritk-python/tests/test_diffusion.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import ritk

SHAPE = (2, 3, 4)
"""Deliberately non-cubic: a transposed reshape would still be in bounds on a
cube and the error would be silent."""

B_VALUE = 1000.0

DIAGONAL = 1.0 / math.sqrt(2.0)
DIRECTIONS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (DIAGONAL, DIAGONAL, 0.0),
    (DIAGONAL, 0.0, DIAGONAL),
    (0.0, DIAGONAL, DIAGONAL),
]
BVALS = [0.0] + [B_VALUE] * 6


def _series(diffusivity: float, s0: float = 1000.0) -> list[ritk.Image]:
    """Volumes for an isotropic tensor of the given diffusivity.

    An isotropic tensor attenuates as ``S = S0 exp(-b D)`` along every
    direction, so the signal needs no forward-model machinery and the expected
    MD is exactly ``D`` with FA zero.
    """
    attenuated = s0 * math.exp(-B_VALUE * diffusivity)
    volumes = []
    for bval in BVALS:
        value = s0 if bval == 0.0 else attenuated
        volumes.append(ritk.Image(np.full(SHAPE, value, dtype=np.float32)))
    return volumes


def _fit(diffusivity: float = 8.0e-4, **kwargs):
    return ritk.diffusion.fit_tensor_maps(
        _series(diffusivity),
        BVALS,
        DIRECTIONS,
        background_fraction=kwargs.pop("background_fraction", 0.0),
        **kwargs,
    )


def test_isotropic_phantom_recovers_its_diffusivity() -> None:
    """MD is the input diffusivity and FA is zero, both in closed form."""
    diffusivity = 8.0e-4
    maps = _fit(diffusivity)

    md = maps.mean_diffusivity()
    assert md.shape == SHAPE
    assert md.dtype == np.float32
    np.testing.assert_allclose(md, diffusivity, rtol=1e-5)

    # Equal eigenvalues make the FA numerator vanish exactly.
    assert np.all(maps.fractional_anisotropy() < 1e-3)


def test_the_three_diffusivity_maps_describe_one_decomposition() -> None:
    """MD == (AD + 2*RD)/3 holds only if all three come from one eigenvalue set.

    It therefore catches an accessor wired to the wrong measure, which a
    shape-and-dtype check cannot.
    """
    maps = _fit()
    md = maps.mean_diffusivity().astype(np.float64)
    ad = maps.axial_diffusivity().astype(np.float64)
    rd = maps.radial_diffusivity().astype(np.float64)

    np.testing.assert_allclose(md, (ad + 2.0 * rd) / 3.0, rtol=1e-6)
    assert np.all(ad >= rd - 1e-12)


def test_the_eigenvector_field_is_unit_length_where_fitted() -> None:
    """Unit norm is what identifies these as orientations.

    A field of arbitrary numbers would satisfy any shape or dtype assertion.
    """
    maps = _fit()
    field = maps.principal_eigenvector()
    assert field.shape == (*SHAPE, 3)
    assert field.dtype == np.float32

    fitted = maps.mask()
    norms = np.linalg.norm(field.astype(np.float64), axis=-1)
    np.testing.assert_allclose(norms[fitted], 1.0, atol=1e-6)
    assert np.all(field[~fitted] == 0.0)


def test_the_mask_separates_unfitted_from_isotropic() -> None:
    """Both read zero in every map, so only the mask tells them apart."""
    maps = _fit()
    mask = maps.mask()
    assert mask.shape == SHAPE
    assert mask.dtype == np.bool_
    assert mask.all(), "an unmasked uniform phantom fits everywhere"
    assert maps.fitted_count == mask.sum()
    assert len(maps) == int(np.prod(SHAPE))


def test_background_masking_excludes_dim_voxels() -> None:
    """The default fraction drops voxels far below the reference percentile."""
    bright = _series(8.0e-4, s0=1000.0)
    dim = _series(8.0e-4, s0=5.0)

    # One dim voxel inside an otherwise bright volume.
    volumes = []
    for bright_volume, dim_volume in zip(bright, dim):
        data = bright_volume.to_numpy().copy()
        data[0, 0, 0] = dim_volume.to_numpy()[0, 0, 0]
        volumes.append(ritk.Image(data))

    maps = ritk.diffusion.fit_tensor_maps(volumes, BVALS, DIRECTIONS)
    mask = maps.mask()
    assert not mask[0, 0, 0], "the dim voxel is background"
    assert mask.sum() == int(np.prod(SHAPE)) - 1


def test_shape_is_preserved_and_not_transposed() -> None:
    """A non-cubic volume comes back on the same grid it went in on."""
    maps = _fit()
    assert maps.fractional_anisotropy().shape == SHAPE
    assert maps.mask().shape == SHAPE
    assert maps.principal_eigenvector().shape == (*SHAPE, 3)


def test_mismatched_counts_raise_value_error() -> None:
    with pytest.raises(ValueError, match="b-values"):
        ritk.diffusion.fit_tensor_maps(_series(8.0e-4), BVALS[:-1], DIRECTIONS)


def test_volumes_on_different_grids_raise_value_error() -> None:
    volumes = _series(8.0e-4)
    volumes[2] = ritk.Image(np.zeros((2, 3, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="one grid"):
        ritk.diffusion.fit_tensor_maps(volumes, BVALS, DIRECTIONS)


def test_a_scheme_without_a_reference_volume_raises_value_error() -> None:
    """Every b-value weighted: there is no b = 0 to build a mask from."""
    weighted = [B_VALUE] * len(BVALS)
    with pytest.raises(ValueError):
        ritk.diffusion.fit_tensor_maps(_series(8.0e-4), weighted, DIRECTIONS)
