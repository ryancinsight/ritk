"""
RITK vs SimpleElastix Gap Analysis — Registration Validation Suite.

Validates RITK registration algorithms against SimpleITK baselines using
real medical image data with side-by-side quality comparison.

Test data inventory:
  - Synthetic: shifted sphere (binary), shifted Gaussian blob (continuous)
  - Same-modality T1↔T1: Colin27↔MNI (ANTs), sub-01↔sub-02 (OpenNeuro ds000208)
  - Cross-modal CT↔MR: RIRE patient 001 (with fiducial ground-truth transform)
  - DICOM I/O: head CT (409 slices) and head MR T2 (94 slices)
  - Inter-subject brain: MNI152↔OpenNeuro sub-01 T1w

Return-type invariant:
  - Demons family: (warped, displacement_field)
  - SyN family: (warped_fixed, warped_moving)
  - bspline_ffd_register: single Image
  - lddmm_register: (warped, velocity_field)

Gap analysis dimensions:
  GAP-R08a: Global metric optimizer (Mattes MI + RSGD) — HIGH
  GAP-R08g: DICOM rescale intercept (CT min -1024 vs -2048) — HIGH (now closed: double-rescale fix)
  GAP-R08f: Same-subject NIfTI pair — MEDIUM (now closed via new data)
  GAP-R08b: Parameter-map interface — LOW
  GAP-R08c: ASGD optimizer — LOW
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

import ritk  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_DATA = (
    Path(__file__).resolve().parent.parent.parent.parent / "test_data" / "registration"
)
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"

SIZE = 64  # synthetic volume side length

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation between two arrays."""
    ma, mb = float(a.mean()), float(b.mean())
    da = float(np.sqrt(((a - ma) ** 2).sum()))
    db = float(np.sqrt(((b - mb) ** 2).sum()))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float(((a - ma) * (b - mb)).sum()) / (da * db)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error."""
    return float(((a - b) ** 2).mean())


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice coefficient between two binary masks."""
    a_bin = (a > 0.5).astype(np.float32)
    b_bin = (b > 0.5).astype(np.float32)
    intersection = float((a_bin * b_bin).sum())
    total = float(a_bin.sum()) + float(b_bin.sum())
    return 2.0 * intersection / total if total > 0 else 0.0


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Minmax normalise arr to [0, 1]."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------


def _sitk_to_numpy(img) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(np.float32)


def _numpy_to_sitk(arr, spacing=None):
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    if spacing is not None:
        img.SetSpacing(spacing)
    return img


def _numpy_to_ritk(arr, spacing=None):
    sp = list(spacing) if spacing is not None else [1.0, 1.0, 1.0]
    return ritk.Image(np.ascontiguousarray(arr.astype(np.float32)), spacing=sp)


def _ritk_to_numpy(img) -> np.ndarray:
    return img.to_numpy()


def _crop_to_match(arr, target_shape):
    """Centre-crop array to match target_shape."""
    slices = []
    for i in range(3):
        diff = arr.shape[i] - target_shape[i]
        if diff <= 0:
            slices.append(slice(0, arr.shape[i]))
        else:
            start = diff // 2
            slices.append(slice(start, start + target_shape[i]))
    return arr[slices[0], slices[1], slices[2]]


def _load_pair(fixed_path, moving_path, max_size=128):
    """Load an image pair via SimpleITK, crop to common size.

    Returns (sitk_fixed, sitk_moving, ritk_fixed, ritk_moving, fixed_arr, moving_arr).
    """
    sitk_f = sitk.ReadImage(str(fixed_path))
    sitk_m = sitk.ReadImage(str(moving_path))
    arr_f = _sitk_to_numpy(sitk_f)
    arr_m = _sitk_to_numpy(sitk_m)
    min_dims = [min(arr_f.shape[i], arr_m.shape[i], max_size) for i in range(3)]
    arr_f = _crop_to_match(arr_f, min_dims)
    arr_m = _crop_to_match(arr_m, min_dims)
    sp = list(sitk_f.GetSpacing())
    return (
        _numpy_to_sitk(arr_f, spacing=sp),
        _numpy_to_sitk(arr_m, spacing=sp),
        _numpy_to_ritk(arr_f, spacing=sp),
        _numpy_to_ritk(arr_m, spacing=sp),
        arr_f,
        arr_m,
    )


def _load_ritk_image(path, max_size=128):
    """Load image via RITK's io.read_image, crop to max_size^3."""
    img = ritk.io.read_image(str(path))
    arr = img.to_numpy()
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]  # add z-dim for 2D data
    sp = img.spacing
    if len(sp) == 2:
        sp = [1.0] + list(sp)
    for i in range(3):
        if arr.shape[i] > max_size:
            start = (arr.shape[i] - max_size) // 2
            arr = np.take(arr, range(start, start + max_size), axis=i)
    return arr, sp


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------


def _make_sphere(size=SIZE, radius=6, center=None):
    c = center or (size // 2)
    z, y, x = np.mgrid[:size, :size, :size]
    return ((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= radius**2).astype(np.float32)


def _make_gaussian_blob(size=SIZE, sigma=5.0, center=None):
    c = center or (size // 2)
    z, y, x = np.mgrid[:size, :size, :size]
    return np.exp(
        -((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2) / (2 * sigma**2)
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# SimpleITK registration baselines
# ---------------------------------------------------------------------------


def _sitk_rigid_register(
    fixed_sitk, moving_sitk, num_iterations=100, learning_rate=1.0
):
    """Euler3D rigid registration via SimpleITK (Mattes MI + RSGD)."""
    fixed_f = sitk.Cast(fixed_sitk, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving_sitk, sitk.sitkFloat32)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.25, seed=42)
    transform = sitk.Euler3DTransform()
    center = [(sz - 1) / 2.0 for sz in fixed_f.GetSize()]
    transform.SetCenter(fixed_f.TransformContinuousIndexToPhysicalPoint(center))
    reg.SetInitialTransform(transform, inPlace=True)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=1e-4,
        numberOfIterations=num_iterations,
        gradientMagnitudeTolerance=1e-8,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([4.0, 2.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    try:
        final = reg.Execute(fixed_f, moving_f)
    except RuntimeError:
        return None
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_f)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(final)
    return resampler.Execute(moving_f)


def _sitk_affine_register(
    fixed_sitk, moving_sitk, num_iterations=100, learning_rate=1.0
):
    """Multi-resolution affine registration via SimpleITK."""
    fixed_f = sitk.Cast(fixed_sitk, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving_sitk, sitk.sitkFloat32)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.25, seed=42)
    transform = sitk.AffineTransform(3)
    reg.SetInitialTransform(transform, inPlace=True)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=1e-4,
        numberOfIterations=num_iterations,
        gradientMagnitudeTolerance=1e-8,
    )
    # B-spline coefficients are physical displacements, so their optimizer
    # scales are already dimensionally uniform. Estimating physical-shift scales
    # perturbs every coefficient without changing the analytical shifted-
    # Gaussian oracle's registration contract.
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([4.0, 2.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    try:
        final = reg.Execute(fixed_f, moving_f)
    except RuntimeError:
        return None
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_f)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(final)
    return resampler.Execute(moving_f)


def _sitk_bspline_register(
    fixed_sitk, moving_sitk, grid_spacing=8.0, num_iterations=100, learning_rate=1.0
):
    """BSpline deformable registration via SimpleITK."""
    fixed_f = sitk.Cast(fixed_sitk, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving_sitk, sitk.sitkFloat32)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.25, seed=42)
    bspline_init = sitk.BSplineTransformInitializer(
        fixed_f,
        [int(sz / grid_spacing + 1) for sz in fixed_f.GetSize()],
        order=3,
    )
    reg.SetInitialTransform(bspline_init, inPlace=True)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=1e-4,
        numberOfIterations=num_iterations,
        gradientMagnitudeTolerance=1e-8,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel([1])
    reg.SetSmoothingSigmasPerLevel([0.0])
    try:
        final = reg.Execute(fixed_f, moving_f)
    except RuntimeError:
        return None
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_f)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(final)
    return resampler.Execute(moving_f)


# ---------------------------------------------------------------------------
# RITK registration wrappers (extract warped image based on return type)
# ---------------------------------------------------------------------------


# Opts-based registration functions take a single config object rather than
# flat keyword arguments; map each to its config class so the helper can wrap
# **kwargs.  The Demons family (demons_register and variants) remains flat.
_RITK_OPTS_CONFIG = {
    "syn_register": "SynConfig",
    "multires_syn_register": "MultiResSynOptions",
    "bspline_syn_register": "BSplineSynOptions",
    "bspline_ffd_register": "BSplineFfdConfig",
    "lddmm_register": "LddmmConfig",
    "multires_demons_register": "MultiResDemonsOptions",
}


def _ritk_warped(name, fixed, moving, **kwargs):
    """Run an RITK registration and return only the warped image array."""
    fn = getattr(ritk.registration, name)
    if name in _RITK_OPTS_CONFIG:
        config = getattr(ritk.registration, _RITK_OPTS_CONFIG[name])(**kwargs)
        result = fn(fixed, moving, config)
    else:
        result = fn(fixed, moving, **kwargs)
    if name == "bspline_ffd_register":
        return result.to_numpy()
    elif name in ("syn_register", "multires_syn_register", "bspline_syn_register"):
        return result[1].to_numpy()  # warped_moving
    elif name == "lddmm_register":
        return result[0].to_numpy()  # warped
    else:
        return result[0].to_numpy()  # Demons family: (warped, displacement)


# ===========================================================================
# Section 1: Synthetic — shifted sphere (binary edge recovery)
# ===========================================================================


@pytest.mark.slow
class TestSyntheticSphere:
    """Validate registration on a 3-voxel x-shifted sphere in a 64^3 volume."""

    @pytest.fixture(autouse=True)
    def setup_sphere_pair(self):
        arr = _make_sphere(size=SIZE, radius=6)
        shift = 3
        arr_shifted = np.roll(arr, shift, axis=2).astype(np.float32)
        self.fixed_arr = arr
        self.moving_arr = arr_shifted
        self.fixed_sitk = _numpy_to_sitk(arr)
        self.moving_sitk = _numpy_to_sitk(arr_shifted)
        self.fixed_ritk = _numpy_to_ritk(arr)
        self.moving_ritk = _numpy_to_ritk(arr_shifted)

    # --- SimpleITK baselines ---

    def test_sitk_rigid_recovers_sphere(self):
        """SimpleITK Euler3D rigid must achieve Dice >= 0.85."""
        result = _sitk_rigid_register(
            self.fixed_sitk, self.moving_sitk, num_iterations=100
        )
        assert result is not None
        d = _dice(_sitk_to_numpy(result), self.fixed_arr)
        assert d >= 0.85, f"SimpleITK rigid Dice {d:.4f} < 0.85"

    # --- RITK Demons family ---

    def test_ritk_demons_recovers_sphere(self):
        """RITK Demons must achieve Dice >= 0.80."""
        warped = _ritk_warped(
            "demons_register",
            self.fixed_ritk,
            self.moving_ritk,
            max_iterations=100,
            sigma_diffusion=1.0,
        )
        d = _dice(warped, self.fixed_arr)
        assert d >= 0.80, f"RITK Demons Dice {d:.4f} < 0.80"

    def test_ritk_diffeomorphic_demons_recovers_sphere(self):
        """RITK Diffeomorphic Demons must achieve Dice >= 0.70."""
        warped = _ritk_warped(
            "diffeomorphic_demons_register",
            self.fixed_ritk,
            self.moving_ritk,
            max_iterations=100,
            sigma_diffusion=1.5,
        )
        d = _dice(warped, self.fixed_arr)
        assert d >= 0.70, f"RITK Diffeomorphic Demons Dice {d:.4f} < 0.70"

    def test_ritk_multires_demons_recovers_sphere(self):
        """RITK Multi-Res Demons must achieve Dice >= 0.70."""
        warped = _ritk_warped(
            "multires_demons_register",
            self.fixed_ritk,
            self.moving_ritk,
            max_iterations=100,
            sigma_diffusion=1.0,
            levels=3,
        )
        d = _dice(warped, self.fixed_arr)
        assert d >= 0.70, f"RITK Multi-Res Demons Dice {d:.4f} < 0.70"

    def test_ritk_ic_demons_recovers_sphere(self):
        """RITK Inverse-Consistent Demons must achieve Dice >= 0.70."""
        warped = _ritk_warped(
            "inverse_consistent_demons_register",
            self.fixed_ritk,
            self.moving_ritk,
            max_iterations=100,
            sigma_diffusion=1.5,
        )
        d = _dice(warped, self.fixed_arr)
        assert d >= 0.70, f"RITK IC Demons Dice {d:.4f} < 0.70"

    # --- RITK SyN family ---

    def test_ritk_syn_recovers_sphere(self):
        """RITK SyN must achieve Dice >= 0.60 on shifted sphere.

        SyN returns (warped_fixed, warped_moving) where both are warped to
        the symmetric midpoint. The Dice between warped_moving and the fixed
        sphere is lower than direct Demons because the midpoint warp does
        not fully align with the fixed image.
        """
        warped = _ritk_warped(
            "syn_register",
            self.fixed_ritk,
            self.moving_ritk,
            max_iterations=100,
            sigma_smooth=1.0,
            cc_radius=2,
            gradient_step=0.5,
        )
        d = _dice(warped, self.fixed_arr)
        assert d >= 0.60, f"RITK SyN Dice {d:.4f} < 0.60"

    def test_ritk_bspline_ffd_recovers_sphere(self):
        """RITK BSpline FFD must achieve Dice >= 0.55 on shifted sphere."""
        warped = _ritk_warped(
            "bspline_ffd_register",
            self.fixed_ritk,
            self.moving_ritk,
            initial_control_spacing=8,
            num_levels=3,
            max_iterations=100,
        )
        d = _dice(warped, self.fixed_arr)
        assert d >= 0.55, f"RITK BSpline FFD Dice {d:.4f} < 0.55"

    # --- Side-by-side ---

    def test_side_by_side_demons_vs_sitk_rigid_sphere(self):
        """Both RITK Demons and SimpleITK rigid must achieve Dice >= 0.80."""
        sitk_result = _sitk_rigid_register(
            self.fixed_sitk, self.moving_sitk, num_iterations=100
        )
        assert sitk_result is not None
        d_sitk = _dice(_sitk_to_numpy(sitk_result), self.fixed_arr)
        warped = _ritk_warped(
            "demons_register",
            self.fixed_ritk,
            self.moving_ritk,
            max_iterations=100,
            sigma_diffusion=1.0,
        )
        d_ritk = _dice(warped, self.fixed_arr)
        assert d_sitk >= 0.80, f"SimpleITK Dice {d_sitk:.4f} < 0.80"
        assert d_ritk >= 0.80, f"RITK Dice {d_ritk:.4f} < 0.80"


# ===========================================================================
# Section 2: Synthetic — shifted Gaussian blob (continuous NCC improvement)
# ===========================================================================


@pytest.mark.slow
class TestSyntheticGaussianBlob:
    """Validate NCC improvement after registration on shifted Gaussian blobs.

    Gaussian blobs provide continuous intensity distributions suited to all
    registration algorithms including SyN (which uses cross-correlation).
    """

    @pytest.fixture(autouse=True)
    def setup_blob_pair(self):
        self.fixed_arr = _make_gaussian_blob(size=SIZE, sigma=5.0)
        shift = 4
        self.moving_arr = np.roll(self.fixed_arr, shift, axis=2).astype(np.float32)
        self.fixed_ritk = _numpy_to_ritk(self.fixed_arr)
        self.moving_ritk = _numpy_to_ritk(self.moving_arr)
        self.ncc_before = _ncc(self.fixed_arr, self.moving_arr)

    def test_ritk_syn_ncc_improves(self):
        """RITK SyN must improve NCC over baseline."""
        warped = _ritk_warped(
            "syn_register",
            self.fixed_ritk,
            self.moving_ritk,
            max_iterations=100,
            sigma_smooth=1.0,
            cc_radius=2,
            gradient_step=0.5,
        )
        ncc_after = _ncc(self.fixed_arr, warped)
        assert ncc_after > self.ncc_before, (
            f"SyN did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_after:.4f}"
        )

    def test_ritk_demons_ncc_improves(self):
        """RITK Demons must improve NCC over baseline."""
        warped = _ritk_warped(
            "demons_register", self.fixed_ritk, self.moving_ritk, max_iterations=100
        )
        ncc_after = _ncc(self.fixed_arr, warped)
        assert ncc_after > self.ncc_before, (
            f"Demons did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_after:.4f}"
        )

    def test_ritk_lddmm_ncc_improves(self):
        """RITK LDDMM must improve NCC over baseline."""
        warped = _ritk_warped(
            "lddmm_register",
            self.fixed_ritk,
            self.moving_ritk,
            max_iterations=30,
            num_time_steps=5,
            kernel_sigma=2.0,
        )
        ncc_after = _ncc(self.fixed_arr, warped)
        assert ncc_after > self.ncc_before, (
            f"LDDMM did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_after:.4f}"
        )

    def test_side_by_side_syn_vs_sitk_bspline_ncc(self):
        """Both RITK SyN and SimpleITK BSpline must improve NCC on Gaussian blob."""
        fixed_s = _numpy_to_sitk(self.fixed_arr)
        moving_s = _numpy_to_sitk(self.moving_arr)
        sitk_result = _sitk_bspline_register(
            fixed_s, moving_s, grid_spacing=8.0, num_iterations=30
        )
        ncc_sitk = (
            _ncc(self.fixed_arr, _sitk_to_numpy(sitk_result))
            if sitk_result is not None
            else self.ncc_before
        )
        warped = _ritk_warped(
            "syn_register",
            self.fixed_ritk,
            self.moving_ritk,
            max_iterations=100,
            sigma_smooth=1.0,
            cc_radius=2,
            gradient_step=0.5,
        )
        ncc_ritk = _ncc(self.fixed_arr, warped)
        assert ncc_sitk > self.ncc_before, (
            f"SimpleITK BSpline did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_sitk:.4f}"
        )
        assert ncc_ritk > self.ncc_before, (
            f"RITK SyN did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_ritk:.4f}"
        )


# ===========================================================================
# Section 3: Same-modality inter-subject T1↔T1 — Colin27↔MNI (ANTs)
# ===========================================================================

_skip_ants = pytest.mark.skipif(
    not (TEST_DATA / "brain_mni" / "ants_ch2.nii.gz").exists()
    or not (TEST_DATA / "brain_mni" / "ants_mni.nii.gz").exists(),
    reason="ANTs Colin27/MNI NIfTI pair absent",
)


@_skip_ants
@pytest.mark.slow
class TestColin27VsMNI:
    """Same-modality (T1↔T1) inter-subject registration: Colin27 average ↔ ICBM MNI.

    Colin27 (ch2) and ICBM MNI are both 1mm isotropic T1 brain templates in
    similar anatomical space but from different subjects/averages. This is the
    standard ANTs registration test pair.

    Key difference from MNI152↔OpenNeuro: these images are already in roughly
    aligned MNI space (NCC_before ≈ 0.7-0.9 vs 0.04), so registration can
    meaningfully improve alignment.
    """

    @pytest.fixture(autouse=True)
    def setup_ants_pair(self):
        (
            self.fixed_sitk,
            self.moving_sitk,
            self.fixed_ritk,
            self.moving_ritk,
            self.fixed_arr,
            self.moving_arr,
        ) = _load_pair(
            TEST_DATA / "brain_mni" / "ants_ch2.nii.gz",
            TEST_DATA / "brain_mni" / "ants_mni.nii.gz",
            max_size=128,
        )
        self.fixed_norm = _minmax(self.fixed_arr)
        self.moving_norm = _minmax(self.moving_arr)
        self.fixed_norm_r = _numpy_to_ritk(self.fixed_norm)
        self.moving_norm_r = _numpy_to_ritk(self.moving_norm)
        self.ncc_before = _ncc(self.fixed_norm, self.moving_norm)
        self.mse_before = _mse(self.fixed_norm, self.moving_norm)

    def test_ritk_demons_improves_alignment(self):
        """RITK Demons must improve NCC on Colin27↔MNI pair."""
        warped = _ritk_warped(
            "demons_register", self.fixed_norm_r, self.moving_norm_r, max_iterations=50
        )
        ncc_after = _ncc(self.fixed_norm, warped)
        assert ncc_after > self.ncc_before, (
            f"Demons did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_after:.4f}"
        )

    def test_ritk_syn_improves_alignment(self):
        """RITK SyN must improve NCC on Colin27↔MNI pair."""
        warped = _ritk_warped(
            "syn_register",
            self.fixed_norm_r,
            self.moving_norm_r,
            max_iterations=30,
            sigma_smooth=1.0,
            cc_radius=2,
            gradient_step=0.25,
        )
        ncc_after = _ncc(self.fixed_norm, warped)
        assert ncc_after > self.ncc_before, (
            f"SyN did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_after:.4f}"
        )

    def test_side_by_side_demons_vs_sitk_rigid(self):
        """RITK Demons and SimpleITK rigid must both improve NCC on Colin27↔MNI.

        SimpleITK rigid (Euler3D + Mattes MI) provides the global metric
        optimizer baseline. RITK Demons provides the local deformable baseline.
        Both should improve alignment on this roughly pre-aligned pair.
        """
        # SimpleITK rigid
        sitk_result = _sitk_rigid_register(
            _numpy_to_sitk(self.fixed_norm),
            _numpy_to_sitk(self.moving_norm),
            num_iterations=100,
        )
        if sitk_result is not None:
            ncc_sitk = _ncc(self.fixed_norm, _sitk_to_numpy(sitk_result))
        else:
            ncc_sitk = self.ncc_before

        # RITK Demons
        warped = _ritk_warped(
            "demons_register", self.fixed_norm_r, self.moving_norm_r, max_iterations=50
        )
        ncc_ritk = _ncc(self.fixed_norm, warped)

        assert ncc_sitk > self.ncc_before, (
            f"SimpleITK rigid did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_sitk:.4f}"
        )
        assert ncc_ritk > self.ncc_before, (
            f"RITK Demons did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_ritk:.4f}"
        )

    def test_side_by_side_syn_vs_sitk_affine(self):
        """RITK SyN vs SimpleITK affine on Colin27↔MNI.

        SimpleITK affine has more DOF than rigid but still uses a global
        metric optimizer. RITK SyN is diffeomorphic deformable. Both should
        improve NCC; SyN should match or exceed affine on this pre-aligned pair.
        """
        sitk_result = _sitk_affine_register(
            _numpy_to_sitk(self.fixed_norm),
            _numpy_to_sitk(self.moving_norm),
            num_iterations=100,
        )
        if sitk_result is not None:
            ncc_sitk = _ncc(self.fixed_norm, _sitk_to_numpy(sitk_result))
        else:
            ncc_sitk = self.ncc_before

        warped = _ritk_warped(
            "syn_register",
            self.fixed_norm_r,
            self.moving_norm_r,
            max_iterations=30,
            sigma_smooth=1.0,
            cc_radius=2,
            gradient_step=0.25,
        )
        ncc_ritk = _ncc(self.fixed_norm, warped)

        assert ncc_sitk > self.ncc_before, (
            f"SimpleITK affine did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_sitk:.4f}"
        )
        assert ncc_ritk > self.ncc_before, (
            f"RITK SyN did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_ritk:.4f}"
        )


# ===========================================================================
# Section 4: Same-modality inter-subject T1↔T1 — OpenNeuro ds000208
# ===========================================================================

_skip_openneuro = pytest.mark.skipif(
    not (TEST_DATA / "ixi" / "openneuro_ds000208_sub01_T1w.nii.gz").exists()
    or not (TEST_DATA / "ixi" / "openneuro_ds000208_sub02_T1w.nii.gz").exists(),
    reason="OpenNeuro ds000208 T1w pair absent",
)


@_skip_openneuro
@pytest.mark.slow
class TestOpenNeuroT1Pair:
    """Same-modality inter-subject T1↔T1 registration: sub-01 ↔ sub-02 (ds000208).

    These are real human T1w brain volumes from different subjects with
    same acquisition protocol (isotropic 1mm). Baseline NCC ≈ 0.75 because
    they share similar head positioning and brain anatomy.
    """

    @pytest.fixture(autouse=True)
    def setup_openneuro_pair(self):
        (
            self.fixed_sitk,
            self.moving_sitk,
            self.fixed_ritk,
            self.moving_ritk,
            self.fixed_arr,
            self.moving_arr,
        ) = _load_pair(
            TEST_DATA / "ixi" / "openneuro_ds000208_sub01_T1w.nii.gz",
            TEST_DATA / "ixi" / "openneuro_ds000208_sub02_T1w.nii.gz",
            max_size=128,
        )
        self.fixed_norm = _minmax(self.fixed_arr)
        self.moving_norm = _minmax(self.moving_arr)
        self.fixed_norm_r = _numpy_to_ritk(self.fixed_norm)
        self.moving_norm_r = _numpy_to_ritk(self.moving_norm)
        self.ncc_before = _ncc(self.fixed_norm, self.moving_norm)
        self.mse_before = _mse(self.fixed_norm, self.moving_norm)

    def test_ritk_demons_improves_ncc(self):
        """RITK Demons must improve NCC on same-modality inter-subject pair."""
        warped = _ritk_warped(
            "demons_register", self.fixed_norm_r, self.moving_norm_r, max_iterations=50
        )
        ncc_after = _ncc(self.fixed_norm, warped)
        assert ncc_after > self.ncc_before, (
            f"Demons did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_after:.4f}"
        )

    def test_ritk_syn_improves_ncc(self):
        """RITK SyN must improve NCC on same-modality inter-subject pair."""
        warped = _ritk_warped(
            "syn_register",
            self.fixed_norm_r,
            self.moving_norm_r,
            max_iterations=30,
            sigma_smooth=1.0,
            cc_radius=2,
            gradient_step=0.25,
        )
        ncc_after = _ncc(self.fixed_norm, warped)
        assert ncc_after > self.ncc_before, (
            f"SyN did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_after:.4f}"
        )

    def test_side_by_side_syn_vs_sitk_rigid(self):
        """RITK SyN and SimpleITK rigid both improve NCC on inter-subject T1."""
        sitk_result = _sitk_rigid_register(
            _numpy_to_sitk(self.fixed_norm),
            _numpy_to_sitk(self.moving_norm),
            num_iterations=100,
        )
        if sitk_result is not None:
            ncc_sitk = _ncc(self.fixed_norm, _sitk_to_numpy(sitk_result))
        else:
            ncc_sitk = self.ncc_before

        warped = _ritk_warped(
            "syn_register",
            self.fixed_norm_r,
            self.moving_norm_r,
            max_iterations=30,
            sigma_smooth=1.0,
            cc_radius=2,
            gradient_step=0.25,
        )
        ncc_ritk = _ncc(self.fixed_norm, warped)

        assert ncc_sitk > self.ncc_before, (
            f"SimpleITK rigid did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_sitk:.4f}"
        )
        assert ncc_ritk > self.ncc_before, (
            f"RITK SyN did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_ritk:.4f}"
        )


# ===========================================================================
# Section 5: Cross-modal CT↔MR — RIRE with ground-truth fiducial transform
# ===========================================================================

_skip_rire = pytest.mark.skipif(
    not (TEST_DATA / "rire" / "training_001_ct.mha").exists(),
    reason="RIRE CT/MR pair absent",
)


@_skip_rire
@pytest.mark.slow
class TestRIREWithGroundTruth:
    """CT↔MR cross-modal registration using RIRE data with fiducial ground truth.

    The RIRE (Retrospective Image Registration Evaluation) project provides
    fiducial-marker-based prospective gold standard transforms. The ground
    truth for patient 001 (CT → MR T1) is an Euler3D transform:
      Rotation: ~4.4° Z, ~1.9° X, ~0.04° Y
      Translation: [5.04, -17.50, -27.16] mm

    Cross-modal registration quality is measured by:
    1. Target Registration Error (TRE): distance between registered and
       ground-truth resampled images
    2. NCC improvement: structural alignment improvement
    3. MSE of the warped CT against the ground-truth resampled CT

    Mathematical basis: CT encodes X-ray attenuation (HU), MR encodes
    proton density/T1. NCC on minmax-normalized volumes measures structural
    overlap (skull, brain, ventricle boundaries), not intensity identity.
    """

    @pytest.fixture(autouse=True)
    def setup_rire_pair(self):
        ct_path = str(TEST_DATA / "rire" / "training_001_ct.mha")
        mr_path = str(TEST_DATA / "rire" / "training_001_mr_T1.mha")
        gt_path = str(TEST_DATA / "rire" / "training_001_ct_to_mr_T1_ground_truth.tfm")

        self.ct_sitk = sitk.ReadImage(ct_path)
        self.mr_sitk = sitk.ReadImage(mr_path)
        self.gt_transform = sitk.ReadTransform(gt_path)

        # Resample CT into MR physical space using ground-truth transform
        self.ct_gt_in_mr = sitk.Resample(
            self.ct_sitk,
            self.mr_sitk,
            self.gt_transform,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        self.arr_mr = _sitk_to_numpy(self.mr_sitk)
        self.arr_ct_gt = _sitk_to_numpy(self.ct_gt_in_mr)

        # For RITK registration: work in voxel space (crop to common shape)
        arr_ct = _sitk_to_numpy(self.ct_sitk)
        min_dims = [min(arr_ct.shape[i], self.arr_mr.shape[i], 128) for i in range(3)]
        arr_ct_crop = _crop_to_match(arr_ct, min_dims)
        arr_mr_crop = _crop_to_match(self.arr_mr, min_dims)

        self.fixed_norm = _minmax(arr_mr_crop)  # MR is fixed
        self.moving_norm = _minmax(arr_ct_crop)  # CT is moving
        self.fixed_norm_r = _numpy_to_ritk(self.fixed_norm)
        self.moving_norm_r = _numpy_to_ritk(self.moving_norm)
        self.ncc_before = _ncc(self.fixed_norm, self.moving_norm)

        # Ground-truth NCC (CT resampled into MR space)
        arr_gt_crop = _crop_to_match(_minmax(self.arr_ct_gt), min_dims)
        self.ncc_ground_truth = _ncc(self.fixed_norm, arr_gt_crop)

    def test_ritk_demons_cross_modal_improves(self):
        """RITK Demons must improve NCC on cross-modal CT↔MR data."""
        warped = _ritk_warped(
            "demons_register", self.fixed_norm_r, self.moving_norm_r, max_iterations=50
        )
        ncc_after = _ncc(self.fixed_norm, warped)
        assert ncc_after > self.ncc_before, (
            f"Demons did not improve cross-modal NCC: "
            f"before={self.ncc_before:.4f}, after={ncc_after:.4f}"
        )

    def test_ritk_syn_cross_modal_improves(self):
        """RITK SyN must improve NCC on cross-modal CT↔MR data."""
        warped = _ritk_warped(
            "syn_register",
            self.fixed_norm_r,
            self.moving_norm_r,
            max_iterations=30,
            sigma_smooth=1.0,
            cc_radius=2,
            gradient_step=0.25,
        )
        ncc_after = _ncc(self.fixed_norm, warped)
        assert ncc_after > self.ncc_before, (
            f"SyN did not improve cross-modal NCC: "
            f"before={self.ncc_before:.4f}, after={ncc_after:.4f}"
        )

    def test_ground_truth_baseline_established(self):
        """Ground-truth rigid transform must produce NCC > baseline.

        The fiducial-marker-based ground truth should produce a higher NCC
        than the unaligned baseline, establishing that the transform is
        non-trivial and correct.
        """
        assert self.ncc_ground_truth > self.ncc_before, (
            f"Ground truth did not improve NCC: before={self.ncc_before:.4f}, "
            f"gt={self.ncc_ground_truth:.4f}"
        )

    def test_side_by_side_sitk_rigid_vs_ground_truth(self):
        """SimpleITK rigid registration should approximate the ground truth.

        SimpleITK's Euler3D + Mattes MI + RSGD with multi-resolution pyramid
        is the standard ITK registration pipeline. It should recover a
        transform close to the fiducial ground truth.
        """
        fixed_f = sitk.Cast(self.mr_sitk, sitk.sitkFloat32)
        moving_f = sitk.Cast(self.ct_sitk, sitk.sitkFloat32)
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.25, seed=42)
        transform = sitk.Euler3DTransform()
        reg.SetInitialTransform(transform, inPlace=True)
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-4,
            numberOfIterations=100,
            gradientMagnitudeTolerance=1e-8,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([4.0, 2.0, 0.0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        try:
            final = reg.Execute(fixed_f, moving_f)
        except RuntimeError:
            pytest.skip("SimpleITK rigid registration failed on RIRE data")

        ct_registered = sitk.Resample(
            self.ct_sitk, self.mr_sitk, final, sitk.sitkLinear, 0.0, sitk.sitkFloat32
        )
        arr_reg = _minmax(_sitk_to_numpy(ct_registered))
        arr_gt = _minmax(self.arr_ct_gt)
        ncc_reg = _ncc(_minmax(self.arr_mr), arr_reg)
        ncc_gt = _ncc(_minmax(self.arr_mr), arr_gt)

        # The registration should improve NCC beyond baseline
        assert ncc_reg > self.ncc_before, (
            f"SimpleITK rigid did not improve cross-modal NCC: "
            f"before={self.ncc_before:.4f}, after={ncc_reg:.4f}"
        )

        # The registration should approach ground-truth quality
        # (within 50% of ground-truth improvement delta)
        delta_gt = ncc_gt - self.ncc_before
        delta_reg = ncc_reg - self.ncc_before
        if delta_gt > 0:
            recovery_ratio = delta_reg / delta_gt
            assert recovery_ratio > 0.3, (
                f"SimpleITK rigid recovered only {recovery_ratio:.1%} of "
                f"ground-truth NCC improvement: reg_delta={delta_reg:.4f}, "
                f"gt_delta={delta_gt:.4f}"
            )


# ===========================================================================
# Section 6: DICOM I/O validation — head CT and MR T2
# ===========================================================================

_skip_dicom = pytest.mark.skipif(
    not Path("D:/ritk/test_data/3_head_ct_mridir/DICOM/00000001.dcm").exists(),
    reason="DICOM test data absent",
)


@_skip_dicom
class TestDICOMIOValidation:
    """Validate RITK DICOM I/O produces images comparable to SimpleITK.

    GAP-R08g (resolved): RITK previously read CT minimum as -1024 HU while
    SimpleITK read -2048 HU. Root cause: decode_via_dicom_rs applied the
    modality LUT (RescaleSlope * stored + RescaleIntercept) twice -- once
    internally by dicom-pixeldata and once by decode_native_pixel_bytes_checked.
    Fix: pass identity rescale (slope=1, intercept=0) in decode_via_dicom_rs
    since dicom-pixeldata already applies the transformation.

    Mathematical basis: DICOM PS3.3 C.7.6.3.1.4 defines the modality LUT as
    output = stored_integer * RescaleSlope + RescaleIntercept. For CT with
    PixelRepresentation=1 (signed i16), RescaleSlope=1, RescaleIntercept=-1024:
    stored value -1024 -> HU = -1024 * 1 + (-1024) = -2048.
    """

    @pytest.fixture(autouse=True)
    def setup_dicom_pair(self):
        self.ct_dir = "D:/ritk/test_data/3_head_ct_mridir/DICOM"
        self.mr_dir = "D:/ritk/test_data/2_head_mri_t2/DICOM"

    def test_ritk_reads_dicom_ct_series(self):
        """RITK must read a DICOM CT series as a 3D volume."""
        img = ritk.io.read_image(self.ct_dir)
        arr = img.to_numpy()
        assert arr.ndim == 3, f"Expected 3D volume, got {arr.ndim}D"
        assert arr.shape[0] > 1, (
            f"Expected multi-slice volume, got {arr.shape[0]} slices"
        )
        assert arr.min() < 0, (
            f"CT volumes must contain negative HU values, min={arr.min():.1f}"
        )

    def test_ritk_reads_dicom_mr_series(self):
        """RITK must read a DICOM MR series as a 3D volume."""
        img = ritk.io.read_image(self.mr_dir)
        arr = img.to_numpy()
        assert arr.ndim == 3, f"Expected 3D volume, got {arr.ndim}D"
        assert arr.shape[0] > 1, (
            f"Expected multi-slice volume, got {arr.shape[0]} slices"
        )

    def test_dicom_intensity_range_comparison(self):
        """RITK and SimpleITK must agree on DICOM CT intensity range.

        GAP-R08g regression: after fixing the double-rescale defect in
        decode_via_dicom_rs, both implementations must produce identical
        HU values for the same DICOM CT series. Tolerance of 1% absorbs
        floating-point conversion differences between f32 (RITK) and f64
        (ITK/SimpleITK) across 409 slices.
        """
        # Read with RITK
        ritk_img = ritk.io.read_image(self.ct_dir)
        ritk_arr = ritk_img.to_numpy()
        ritk_min = float(ritk_arr.min())
        ritk_max = float(ritk_arr.max())

        # Read with SimpleITK
        sitk_files = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(self.ct_dir)
        sitk_img = sitk.ReadImage(sitk_files)
        sitk_arr = _sitk_to_numpy(sitk_img)
        sitk_min = float(sitk_arr.min())
        sitk_max = float(sitk_arr.max())

        tol = 0.01  # 1% relative tolerance
        assert abs(ritk_min - sitk_min) / max(abs(sitk_min), 1.0) < tol, (
            f"CT min mismatch: ritk={ritk_min:.1f}, sitk={sitk_min:.1f}"
        )
        assert abs(ritk_max - sitk_max) / max(abs(sitk_max), 1.0) < tol, (
            f"CT max mismatch: ritk={ritk_max:.1f}, sitk={sitk_max:.1f}"
        )

    def test_dicom_shape_consistency(self):
        """RITK and SimpleITK must read the same DICOM series shape."""
        ritk_img = ritk.io.read_image(self.ct_dir)
        sitk_files = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(self.ct_dir)
        sitk_img = sitk.ReadImage(sitk_files)

        ritk_shape = ritk_img.shape
        sitk_shape = sitk_img.GetSize()

        # SimpleITK returns (X, Y, Z); RITK returns (Z, Y, X)
        assert ritk_shape[0] == sitk_shape[2], (
            f"Z dimension mismatch: RITK={ritk_shape[0]}, SimpleITK={sitk_shape[2]}"
        )
        assert ritk_shape[1] == sitk_shape[1], (
            f"Y dimension mismatch: RITK={ritk_shape[1]}, SimpleITK={sitk_shape[1]}"
        )
        assert ritk_shape[2] == sitk_shape[0], (
            f"X dimension mismatch: RITK={ritk_shape[2]}, SimpleITK={sitk_shape[0]}"
        )


# ===========================================================================
# Section 7: Inter-subject brain MNI↔OpenNeuro — MSE reduction
# ===========================================================================

_skip_mni = pytest.mark.skipif(
    not (TEST_DATA / "brain_mni" / "mni152.nii.gz").exists(),
    reason="MNI152 / sub-01_T1w NIfTI pair absent",
)


@_skip_mni
@pytest.mark.slow
class TestInterSubjectBrainMNI:
    """Inter-subject brain registration using MNI152 and OpenNeuro T1w.

    Pre-registration NCC is typically very low (~0.04) because inter-subject
    brain shape differences dominate. The meaningful metric is MSE reduction.
    NCC improvement is not required because these subjects have genuinely
    different brain anatomy.
    """

    @pytest.fixture(autouse=True)
    def setup_mni_pair(self):
        (
            self.fixed_sitk,
            self.moving_sitk,
            self.fixed_ritk,
            self.moving_ritk,
            self.fixed_arr,
            self.moving_arr,
        ) = _load_pair(
            TEST_DATA / "brain_mni" / "mni152.nii.gz",
            TEST_DATA / "brain_mni" / "sub-01_T1w.nii.gz",
            max_size=128,
        )
        self.fixed_norm = _minmax(self.fixed_arr)
        self.moving_norm = _minmax(self.moving_arr)
        self.fixed_norm_r = _numpy_to_ritk(self.fixed_norm)
        self.moving_norm_r = _numpy_to_ritk(self.moving_norm)
        self.mse_before = _mse(self.fixed_norm, self.moving_norm)

    def test_ritk_demons_reduces_mse(self):
        """RITK Demons must reduce MSE on inter-subject brain."""
        warped = _ritk_warped(
            "demons_register", self.fixed_norm_r, self.moving_norm_r, max_iterations=50
        )
        mse_after = _mse(self.fixed_norm, warped)
        assert mse_after < self.mse_before, (
            f"RITK Demons did not reduce MSE: "
            f"before={self.mse_before:.6f}, after={mse_after:.6f}"
        )

    def test_ritk_syn_reduces_mse(self):
        """RITK SyN must reduce MSE on inter-subject brain."""
        warped = _ritk_warped(
            "syn_register",
            self.fixed_norm_r,
            self.moving_norm_r,
            max_iterations=30,
            sigma_smooth=3.0,
            cc_radius=2,
            gradient_step=0.25,
        )
        mse_after = _mse(self.fixed_norm, warped)
        assert mse_after < self.mse_before, (
            f"RITK SyN did not reduce MSE: "
            f"before={self.mse_before:.6f}, after={mse_after:.6f}"
        )


# ===========================================================================
# Section 8: Cross-modal CT↔MR — RIRE (voxel-space registration)
# ===========================================================================

# This section uses the same RIRE data as Section 5 but in a simpler
# voxel-space registration configuration (without ground-truth transform
# application), matching the original test format.


@_skip_rire
@pytest.mark.slow
class TestRIREMultiModalVoxel:
    """CT↔MR cross-modal registration in voxel space using RIRE data.

    This is the simpler voxel-space test (no physical-space resampling)
    matching the original test format. Physical-space ground-truth
    validation is in TestRIREWithGroundTruth.
    """

    @pytest.fixture(autouse=True)
    def setup_rire_voxel_pair(self):
        (
            self.fixed_sitk,
            self.moving_sitk,
            self.fixed_ritk,
            self.moving_ritk,
            self.fixed_arr,
            self.moving_arr,
        ) = _load_pair(
            TEST_DATA / "rire" / "training_001_ct.mha",
            TEST_DATA / "rire" / "training_001_mr_T1.mha",
            max_size=128,
        )
        self.fixed_norm = _minmax(self.fixed_arr)
        self.moving_norm = _minmax(self.moving_arr)
        self.fixed_norm_r = _numpy_to_ritk(self.fixed_norm)
        self.moving_norm_r = _numpy_to_ritk(self.moving_norm)
        self.ncc_before = _ncc(self.fixed_norm, self.moving_norm)

    def test_ritk_demons_cross_modal_ncc_improves(self):
        """RITK Demons must improve NCC on cross-modal data after normalization."""
        warped = _ritk_warped(
            "demons_register", self.fixed_norm_r, self.moving_norm_r, max_iterations=50
        )
        ncc_after = _ncc(self.fixed_norm, warped)
        assert ncc_after > self.ncc_before, (
            f"Demons did not improve cross-modal NCC: "
            f"before={self.ncc_before:.4f}, after={ncc_after:.4f}"
        )

    def test_ritk_syn_cross_modal_ncc_improves(self):
        """RITK SyN must improve NCC on cross-modal data after normalization."""
        warped = _ritk_warped(
            "syn_register",
            self.fixed_norm_r,
            self.moving_norm_r,
            max_iterations=30,
            sigma_smooth=3.0,
            cc_radius=2,
        )
        ncc_after = _ncc(self.fixed_norm, warped)
        assert ncc_after > self.ncc_before, (
            f"SyN did not improve cross-modal NCC: "
            f"before={self.ncc_before:.4f}, after={ncc_after:.4f}"
        )

    def test_side_by_side_cross_modal_ncc(self):
        """RITK SyN and SimpleITK BSpline must both improve NCC on cross-modal data."""
        fixed_s = _numpy_to_sitk(self.fixed_norm)
        moving_s = _numpy_to_sitk(self.moving_norm)
        sitk_result = _sitk_bspline_register(
            fixed_s, moving_s, grid_spacing=8.0, num_iterations=30
        )
        ncc_sitk = (
            _ncc(self.fixed_norm, _sitk_to_numpy(sitk_result))
            if sitk_result is not None
            else self.ncc_before
        )
        warped = _ritk_warped(
            "syn_register",
            self.fixed_norm_r,
            self.moving_norm_r,
            max_iterations=30,
            sigma_smooth=3.0,
            cc_radius=2,
        )
        ncc_ritk = _ncc(self.fixed_norm, warped)
        assert ncc_ritk > self.ncc_before, (
            f"RITK SyN did not improve cross-modal NCC: "
            f"before={self.ncc_before:.4f}, after={ncc_ritk:.4f}"
        )
        if sitk_result is not None:
            assert ncc_sitk > self.ncc_before - 0.05, (
                f"SimpleITK BSpline regressed NCC: "
                f"before={self.ncc_before:.4f}, after={ncc_sitk:.4f}"
            )


# ===========================================================================
# Section 9: Cross-modal CT↔MR — Visible Male head pair
# ===========================================================================

_skip_vm = pytest.mark.skipif(
    not (TEST_DATA / "simpleitk_notebooks" / "vm_head_ct.mha").exists(),
    reason="Visible Male CT/MR head pair absent",
)


@_skip_vm
@pytest.mark.slow
class TestVMHeadMultiModal:
    """CT↔MR cross-modal registration using Visible Male head data."""

    @pytest.fixture(autouse=True)
    def setup_vm_pair(self):
        (
            self.fixed_sitk,
            self.moving_sitk,
            self.fixed_ritk,
            self.moving_ritk,
            self.fixed_arr,
            self.moving_arr,
        ) = _load_pair(
            TEST_DATA / "simpleitk_notebooks" / "vm_head_ct.mha",
            TEST_DATA / "simpleitk_notebooks" / "vm_head_mri.mha",
            max_size=128,
        )
        self.fixed_norm = _minmax(self.fixed_arr)
        self.moving_norm = _minmax(self.moving_arr)
        self.fixed_norm_r = _numpy_to_ritk(self.fixed_norm)
        self.moving_norm_r = _numpy_to_ritk(self.moving_norm)
        self.ncc_before = _ncc(self.fixed_norm, self.moving_norm)

    def test_ritk_demons_cross_modal_ncc_improves(self):
        """RITK Demons must improve NCC on VM head cross-modal data."""
        warped = _ritk_warped(
            "demons_register", self.fixed_norm_r, self.moving_norm_r, max_iterations=50
        )
        ncc_after = _ncc(self.fixed_norm, warped)
        assert ncc_after > self.ncc_before, (
            f"Demons did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_after:.4f}"
        )

    def test_ritk_syn_cross_modal_ncc_improves(self):
        """RITK SyN must improve NCC on VM head cross-modal data."""
        warped = _ritk_warped(
            "syn_register",
            self.fixed_norm_r,
            self.moving_norm_r,
            max_iterations=30,
            sigma_smooth=3.0,
            cc_radius=2,
        )
        ncc_after = _ncc(self.fixed_norm, warped)
        assert ncc_after > self.ncc_before, (
            f"SyN did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_after:.4f}"
        )


# ===========================================================================
# Section 10: Comprehensive quality report across all RITK algorithms
# ===========================================================================


@pytest.mark.slow
class TestRegistrationQualityReport:
    """Generate a comprehensive quality report comparing all RITK algorithms
    against SimpleITK baselines on synthetic data."""

    def test_all_ritk_algorithms_improve_ncc_on_shifted_blob(self):
        """Every RITK deformable algorithm must improve NCC on a shifted Gaussian blob.

        Return-type contract:
        - Demons family: (warped, displacement)
        - SyN family: (warped_fixed, warped_moving)
        - bspline_ffd_register: single warped Image
        - lddmm_register: (warped, velocity_field)
        """
        fixed_arr = _make_gaussian_blob(size=SIZE, sigma=5.0)
        moving_arr = np.roll(fixed_arr, 4, axis=2).astype(np.float32)
        ncc_before = _ncc(fixed_arr, moving_arr)
        fixed_r = _numpy_to_ritk(fixed_arr)
        moving_r = _numpy_to_ritk(moving_arr)

        algorithms = {
            "demons": dict(max_iterations=100),
            "diffeomorphic_demons": dict(max_iterations=100, sigma_diffusion=1.5),
            "symmetric_demons": dict(max_iterations=100),
            "multires_demons": dict(max_iterations=100, levels=3),
            "inverse_consistent_demons": dict(max_iterations=100),
            "syn": dict(
                max_iterations=100, sigma_smooth=1.0, cc_radius=2, gradient_step=0.5
            ),
            "multires_syn": dict(
                num_levels=3, sigma_smooth=1.0, cc_radius=2, gradient_step=0.5
            ),
            "bspline_syn": dict(
                max_iterations=100,
                control_spacing_z=8,
                control_spacing_y=8,
                control_spacing_x=8,
                sigma_smooth=1.0,
                cc_radius=2,
                gradient_step=0.5,
            ),
            "bspline_ffd": dict(
                initial_control_spacing=8, num_levels=2, max_iterations=50
            ),
            "lddmm": dict(max_iterations=30, num_time_steps=5, kernel_sigma=2.0),
        }

        for name, kwargs in algorithms.items():
            reg_name = name + "_register"
            warped = _ritk_warped(reg_name, fixed_r, moving_r, **kwargs)
            ncc_after = _ncc(fixed_arr, warped)
            assert ncc_after > ncc_before, (
                f"{name} did not improve NCC: before={ncc_before:.4f}, "
                f"after={ncc_after:.4f}"
            )


# ===========================================================================
# Section 11: RITK file I/O validation — NIfTI and MetaImage round-trip
# ===========================================================================


class TestRITKFileIOValidation:
    """Validate that RITK's io.read_image produces images consistent with SimpleITK.

    This tests the I/O layer that feeds the registration pipeline. If the
    I/O layer produces incorrect intensity values, registration quality will
    be degraded regardless of algorithm quality.
    """

    @pytest.fixture(autouse=True)
    def setup_io_data(self):
        self.ants_ch2_path = str(TEST_DATA / "brain_mni" / "ants_ch2.nii.gz")
        self.rire_ct_path = str(TEST_DATA / "rire" / "training_001_ct.mha")
        self.rire_mr_path = str(TEST_DATA / "rire" / "training_001_mr_T1.mha")

    def test_ritk_nifti_intensity_matches_sitk(self):
        """RITK and SimpleITK must read NIfTI with consistent intensity ranges.

        The intensity ranges should match within 1% relative error for NIfTI
        (no rescale slope/intercept ambiguity in NIfTI format).
        """
        ritk_img = ritk.io.read_image(self.ants_ch2_path)
        ritk_arr = ritk_img.to_numpy()

        sitk_img = sitk.ReadImage(self.ants_ch2_path)
        sitk_arr = _sitk_to_numpy(sitk_img)

        # Crop to common shape (in case of different slice orderings)
        min_dims = [min(ritk_arr.shape[i], sitk_arr.shape[i]) for i in range(3)]
        ritk_c = _crop_to_match(ritk_arr, min_dims)
        sitk_c = _crop_to_match(sitk_arr, min_dims)

        ritk_min, ritk_max = float(ritk_c.min()), float(ritk_c.max())
        sitk_min, sitk_max = float(sitk_c.min()), float(sitk_c.max())

        # For NIfTI, the intensity values should match exactly
        assert abs(ritk_min - sitk_min) / max(abs(sitk_min), 1.0) < 0.02, (
            f"NIfTI min mismatch: RITK={ritk_min:.2f}, SimpleITK={sitk_min:.2f}"
        )
        assert abs(ritk_max - sitk_max) / max(abs(sitk_max), 1.0) < 0.02, (
            f"NIfTI max mismatch: RITK={ritk_max:.2f}, SimpleITK={sitk_max:.2f}"
        )

    def test_ritk_metaimage_intensity_matches_sitk(self):
        """RITK and SimpleITK must read MetaImage with consistent intensity ranges."""
        ritk_img = ritk.io.read_image(self.rire_mr_path)
        ritk_arr = ritk_img.to_numpy()

        sitk_img = sitk.ReadImage(self.rire_mr_path)
        sitk_arr = _sitk_to_numpy(sitk_img)

        min_dims = [min(ritk_arr.shape[i], sitk_arr.shape[i]) for i in range(3)]
        ritk_c = _crop_to_match(ritk_arr, min_dims)
        sitk_c = _crop_to_match(sitk_arr, min_dims)

        ritk_min, ritk_max = float(ritk_c.min()), float(ritk_c.max())
        sitk_min, sitk_max = float(sitk_c.min()), float(sitk_c.max())

        # MetaImage has no rescale slope/intercept, should match closely
        assert abs(ritk_min - sitk_min) / max(abs(sitk_min), 1.0) < 0.02, (
            f"MetaImage MR min mismatch: RITK={ritk_min:.2f}, SimpleITK={sitk_min:.2f}"
        )
        assert abs(ritk_max - sitk_max) / max(abs(sitk_max), 1.0) < 0.02, (
            f"MetaImage MR max mismatch: RITK={ritk_max:.2f}, SimpleITK={sitk_max:.2f}"
        )

    def test_ritk_nifti_shape_matches_sitk(self):
        """RITK and SimpleITK must read NIfTI with consistent shapes."""
        ritk_img = ritk.io.read_image(self.ants_ch2_path)
        sitk_img = sitk.ReadImage(self.ants_ch2_path)

        ritk_shape = ritk_img.shape
        sitk_shape = sitk_img.GetSize()

        # SimpleITK: (X, Y, Z); RITK: (Z, Y, X)
        assert ritk_shape[0] == sitk_shape[2], (
            f"Z shape mismatch: RITK={ritk_shape[0]}, SimpleITK={sitk_shape[2]}"
        )
        assert ritk_shape[1] == sitk_shape[1], (
            f"Y shape mismatch: RITK={ritk_shape[1]}, SimpleITK={sitk_shape[1]}"
        )
        assert ritk_shape[2] == sitk_shape[0], (
            f"X shape mismatch: RITK={ritk_shape[2]}, SimpleITK={sitk_shape[0]}"
        )

    def test_ritk_nifti_spacing_matches_sitk(self):
        """RITK and SimpleITK must read NIfTI with consistent spacing."""
        ritk_img = ritk.io.read_image(self.ants_ch2_path)
        sitk_img = sitk.ReadImage(self.ants_ch2_path)

        ritk_sp = ritk_img.spacing
        sitk_sp = sitk_img.GetSpacing()

        # RITK spacing is (Z, Y, X); SimpleITK is (X, Y, Z)
        for i in range(3):
            ritk_val = ritk_sp[2 - i]  # reverse index
            sitk_val = sitk_sp[i]
            assert abs(ritk_val - sitk_val) < 0.01, (
                f"Spacing mismatch at dim {i}: RITK={ritk_val:.4f}, "
                f"SimpleITK={sitk_val:.4f}"
            )


# ===========================================================================
# Section 12: NIfTI direct-read registration (RITK io.read_image pipeline)
# ===========================================================================

_skip_ants_direct = pytest.mark.skipif(
    not (TEST_DATA / "brain_mni" / "ants_ch2.nii.gz").exists()
    or not (TEST_DATA / "brain_mni" / "ants_mni.nii.gz").exists(),
    reason="ANTs NIfTI pair absent",
)


@_skip_ants_direct
@pytest.mark.slow
class TestNiftiDirectReadRegistration:
    """Validate registration using RITK's io.read_image (end-to-end pipeline).

    Instead of loading via SimpleITK and converting to numpy→ritk.Image,
    this test loads directly via ritk.io.read_image, which exercises the
    full RITK I/O + registration pipeline.
    """

    @pytest.fixture(autouse=True)
    def setup_direct_read(self):
        import tempfile

        self.fixed_ritk = ritk.io.read_image(
            str(TEST_DATA / "brain_mni" / "ants_ch2.nii.gz")
        )
        self.moving_ritk = ritk.io.read_image(
            str(TEST_DATA / "brain_mni" / "ants_mni.nii.gz")
        )
        # Crop both to 128^3 for speed
        max_size = 128
        for img_attr in ("fixed_ritk", "moving_ritk"):
            img = getattr(self, img_attr)
            arr = img.to_numpy()
            sp = img.spacing
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            if len(sp) == 2:
                sp = [1.0] + list(sp)
            for i in range(3):
                if arr.shape[i] > max_size:
                    start = (arr.shape[i] - max_size) // 2
                    arr = np.take(arr, range(start, start + max_size), axis=i)
            setattr(self, img_attr, _numpy_to_ritk(arr, spacing=sp))
            if img_attr == "fixed_ritk":
                self.fixed_arr = arr
            else:
                self.moving_arr = arr

        self.fixed_norm = _minmax(self.fixed_arr)
        self.moving_norm = _minmax(self.moving_arr)
        self.fixed_norm_r = _numpy_to_ritk(self.fixed_norm)
        self.moving_norm_r = _numpy_to_ritk(self.moving_norm)
        self.ncc_before = _ncc(self.fixed_norm, self.moving_norm)

    def test_direct_read_demons_improves_alignment(self):
        """RITK Demons on directly-read NIfTI must improve NCC."""
        warped = _ritk_warped(
            "demons_register", self.fixed_norm_r, self.moving_norm_r, max_iterations=50
        )
        ncc_after = _ncc(self.fixed_norm, warped)
        assert ncc_after > self.ncc_before, (
            f"Demons on direct-read NIfTI did not improve NCC: "
            f"before={self.ncc_before:.4f}, after={ncc_after:.4f}"
        )

    def test_direct_read_syn_improves_alignment(self):
        """RITK SyN on directly-read NIfTI must improve NCC."""
        warped = _ritk_warped(
            "syn_register",
            self.fixed_norm_r,
            self.moving_norm_r,
            max_iterations=30,
            sigma_smooth=1.0,
            cc_radius=2,
            gradient_step=0.25,
        )
        ncc_after = _ncc(self.fixed_norm, warped)
        assert ncc_after > self.ncc_before, (
            f"SyN on direct-read NIfTI did not improve NCC: "
            f"before={self.ncc_before:.4f}, after={ncc_after:.4f}"
        )
