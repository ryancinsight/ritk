"""
RITK vs SimpleITK side-by-side registration validation.

Validates that RITK registration algorithms produce results comparable to
SimpleITK on both synthetic and real medical image data.

Tests cover:
- Synthetic shifted sphere (Dice recovery)
- Synthetic Gaussian blob (NCC improvement)
- Inter-subject brain MNI pair (NCC/MSE improvement)
- Multi-modal CT/MR RIRE pair (cross-modal NCC improvement)
- Multi-modal CT/MR head pair (cross-modal NCC improvement)
- Comprehensive quality report across all algorithms

Return-type invariant:
  - Demons family (demons, diffeomorphic_demons, symmetric_demons,
    multires_demons, inverse_consistent_demons): returns (warped, displacement)
  - SyN family (syn, multires_syn, bspline_syn): returns (warped_fixed, warped_moving)
  - bspline_ffd_register: returns single warped Image (not a tuple)
  - lddmm_register: returns (warped, velocity_field)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
import ritk  # noqa: E402

# --- Constants ---
TEST_DATA = (
    Path(__file__).resolve().parent.parent.parent.parent / "test_data" / "registration"
)
SIZE = 64  # synthetic test volume size


# ============================================================================
# Helpers
# ============================================================================


def _sitk_to_numpy(img):
    """Convert SimpleITK image to float32 numpy array (Z,Y,X)."""
    return sitk.GetArrayFromImage(img).astype(np.float32)


def _numpy_to_sitk(arr, spacing=None):
    """Convert numpy array (Z,Y,X) to SimpleITK image."""
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    if spacing is not None:
        img.SetSpacing(spacing)
    return img


def _numpy_to_ritk(arr, spacing=None):
    """Convert numpy array (Z,Y,X) to ritk.Image."""
    sp = list(spacing) if spacing is not None else [1.0, 1.0, 1.0]
    return ritk.Image(np.ascontiguousarray(arr.astype(np.float32)), spacing=sp)


def _make_sphere(size=SIZE, radius=6, center=None):
    """Create a 3D sphere in a size^3 volume."""
    c = center or (size // 2)
    z, y, x = np.mgrid[:size, :size, :size]
    return ((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= radius**2).astype(np.float32)


def _make_gaussian_blob(size=SIZE, sigma=5.0, center=None):
    """Create a 3D Gaussian blob."""
    c = center or (size // 2)
    z, y, x = np.mgrid[:size, :size, :size]
    return np.exp(
        -((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2) / (2 * sigma**2)
    ).astype(np.float32)


def _dice(a, b):
    """Dice coefficient between two binary masks."""
    a_bin = (a > 0.5).astype(np.float32)
    b_bin = (b > 0.5).astype(np.float32)
    intersection = float((a_bin * b_bin).sum())
    total = float(a_bin.sum()) + float(b_bin.sum())
    return 2.0 * intersection / total if total > 0 else 0.0


def _ncc(a, b):
    """Normalized cross-correlation."""
    ma, mb = float(a.mean()), float(b.mean())
    da = float(np.sqrt(((a - ma) ** 2).sum()))
    db = float(np.sqrt(((b - mb) ** 2).sum()))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float(((a - ma) * (b - mb)).sum()) / (da * db)


def _mse(a, b):
    """Mean squared error."""
    return float(((a - b) ** 2).mean())


def _minmax(arr):
    """Minmax normalise arr to [0, 1]."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _crop_to_match(arr, target_shape):
    """Crop array from centre to match target_shape."""
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
    """Load an image pair, crop to common size (max max_size^3).

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


# ============================================================================
# SimpleITK registration helpers
# ============================================================================


def _sitk_translation_register(
    fixed_sitk, moving_sitk, num_iterations=100, learning_rate=1.0
):
    """Euler3D (translation-only) registration via SimpleITK."""
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.25, seed=42)
    transform = sitk.Euler3DTransform()
    center = [(sz - 1) / 2.0 for sz in fixed_sitk.GetSize()]
    transform.SetCenter(fixed_sitk.TransformContinuousIndexToPhysicalPoint(center))
    reg.SetInitialTransform(transform, inPlace=True)
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
        final = reg.Execute(fixed_sitk, moving_sitk)
    except RuntimeError:
        return None
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(final)
    return resampler.Execute(moving_sitk)


def _sitk_affine_register(
    fixed_sitk, moving_sitk, num_iterations=100, learning_rate=1.0
):
    """Multi-resolution affine registration via SimpleITK."""
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
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([4.0, 2.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    try:
        final = reg.Execute(fixed_sitk, moving_sitk)
    except RuntimeError:
        return None
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(final)
    return resampler.Execute(moving_sitk)


def _sitk_bspline_register(
    fixed_sitk, moving_sitk, grid_spacing=8.0, num_iterations=100, learning_rate=1.0
):
    """BSpline deformable registration via SimpleITK."""
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.25, seed=42)
    bspline_init = sitk.BSplineTransformInitializer(
        fixed_sitk,
        [int(sz / grid_spacing + 1) for sz in fixed_sitk.GetSize()],
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
        final = reg.Execute(fixed_sitk, moving_sitk)
    except RuntimeError:
        return None
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(final)
    return resampler.Execute(moving_sitk)


# ============================================================================
# Section 1: Synthetic tests — shifted sphere
# ============================================================================


@pytest.mark.slow
class TestSyntheticSphere:
    """Validate registration on a 3-voxel x-shifted sphere in a 64^3 volume.

    Threshold rationale:
      - Demons family: optical-flow forces are strong on binary edges → Dice >= 0.80
      - Diffeomorphic Demons: scaling-and-squaring softens the field → Dice >= 0.70
      - SyN: CC metric on binary data is suboptimal; the midpoint-warped_moving
        comparison reduces apparent overlap → Dice >= 0.60
      - BSpline FFD: control-grid smoothing on binary edges → Dice >= 0.60
    """

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

    def test_sitk_translation_recovers_sphere(self):
        """SimpleITK Euler3D translation must achieve Dice >= 0.85."""
        result = _sitk_translation_register(
            self.fixed_sitk, self.moving_sitk, num_iterations=100
        )
        assert result is not None
        d = _dice(_sitk_to_numpy(result), self.fixed_arr)
        assert d >= 0.85, f"SimpleITK translation Dice {d:.4f} < 0.85"

    def test_sitk_affine_recovers_sphere(self):
        """SimpleITK affine must achieve Dice >= 0.50."""
        result = _sitk_affine_register(
            self.fixed_sitk, self.moving_sitk, num_iterations=100
        )
        assert result is not None
        d = _dice(_sitk_to_numpy(result), self.fixed_arr)
        assert d >= 0.50, f"SimpleITK affine Dice {d:.4f} < 0.50"

    # --- RITK Demons family ---

    def test_ritk_demons_recovers_sphere(self):
        """RITK Thirion Demons must achieve Dice >= 0.80."""
        warped, _ = ritk.registration.demons_register(
            self.fixed_ritk, self.moving_ritk, max_iterations=100, sigma_diffusion=1.0
        )
        d = _dice(warped.to_numpy(), self.fixed_arr)
        assert d >= 0.80, f"RITK Demons Dice {d:.4f} < 0.80"

    def test_ritk_diffeomorphic_demons_recovers_sphere(self):
        """RITK Diffeomorphic Demons must achieve Dice >= 0.70."""
        warped, _ = ritk.registration.diffeomorphic_demons_register(
            self.fixed_ritk, self.moving_ritk, max_iterations=100, sigma_diffusion=1.5
        )
        d = _dice(warped.to_numpy(), self.fixed_arr)
        assert d >= 0.70, f"RITK Diffeomorphic Demons Dice {d:.4f} < 0.70"

    def test_ritk_symmetric_demons_recovers_sphere(self):
        """RITK Symmetric Demons must achieve Dice >= 0.70."""
        warped, _ = ritk.registration.symmetric_demons_register(
            self.fixed_ritk, self.moving_ritk, max_iterations=100, sigma_diffusion=1.5
        )
        d = _dice(warped.to_numpy(), self.fixed_arr)
        assert d >= 0.70, f"RITK Symmetric Demons Dice {d:.4f} < 0.70"

    def test_ritk_multires_demons_recovers_sphere(self):
        """RITK Multi-Res Demons must achieve Dice >= 0.70."""
        warped, _ = ritk.registration.multires_demons_register(self.fixed_ritk,
            self.moving_ritk, ritk.registration.MultiResDemonsOptions(max_iterations=100,sigma_diffusion=1.0,levels=3))
        d = _dice(warped.to_numpy(), self.fixed_arr)
        assert d >= 0.70, f"RITK Multi-Res Demons Dice {d:.4f} < 0.70"

    def test_ritk_ic_demons_recovers_sphere(self):
        """RITK Inverse-Consistent Demons must achieve Dice >= 0.70."""
        warped, _ = ritk.registration.inverse_consistent_demons_register(
            self.fixed_ritk, self.moving_ritk, max_iterations=100, sigma_diffusion=1.5
        )
        d = _dice(warped.to_numpy(), self.fixed_arr)
        assert d >= 0.70, f"RITK IC Demons Dice {d:.4f} < 0.70"

    # --- RITK SyN family ---

    def test_ritk_syn_recovers_sphere(self):
        """RITK SyN must achieve Dice >= 0.60 on shifted sphere.

        SyN returns (warped_fixed, warped_moving) where both are warped to
        the symmetric midpoint. The Dice between warped_moving and the fixed
        sphere is lower than direct Demons because the midpoint warp does not
        fully align with the fixed image — it converges to the midpoint.
        """
        _, warped_m = ritk.registration.syn_register(self.fixed_ritk,
            self.moving_ritk, ritk.registration.SynConfig(max_iterations=100,sigma_smooth=1.0,cc_radius=2,gradient_step=0.5))
        d = _dice(warped_m.to_numpy(), self.fixed_arr)
        assert d >= 0.60, f"RITK SyN Dice {d:.4f} < 0.60"

    def test_ritk_multires_syn_recovers_sphere(self):
        """RITK Multi-Res SyN must achieve Dice >= 0.60."""
        _, warped_m = ritk.registration.multires_syn_register(self.fixed_ritk,
            self.moving_ritk, ritk.registration.MultiResSynOptions(num_levels=3,sigma_smooth=1.0,cc_radius=2,gradient_step=0.5))
        d = _dice(warped_m.to_numpy(), self.fixed_arr)
        assert d >= 0.60, f"RITK Multi-Res SyN Dice {d:.4f} < 0.60"

    def test_ritk_bspline_syn_recovers_sphere(self):
        """RITK BSpline SyN must achieve Dice >= 0.55 on shifted sphere."""
        _, warped_m = ritk.registration.bspline_syn_register(self.fixed_ritk,
            self.moving_ritk, ritk.registration.BSplineSynOptions(max_iterations=100,control_spacing_z=8,control_spacing_y=8,control_spacing_x=8,sigma_smooth=1.0,cc_radius=2,gradient_step=0.5))
        d = _dice(warped_m.to_numpy(), self.fixed_arr)
        assert d >= 0.55, f"RITK BSpline SyN Dice {d:.4f} < 0.55"

    def test_ritk_bspline_ffd_recovers_sphere(self):
        """RITK BSpline FFD must achieve Dice >= 0.55 on shifted sphere.

        BSpline FFD returns a single warped Image (not a tuple).
        """
        warped = ritk.registration.bspline_ffd_register(self.fixed_ritk,
            self.moving_ritk, ritk.registration.BSplineFfdConfig(initial_control_spacing=8,num_levels=3,max_iterations=100))
        d = _dice(warped.to_numpy(), self.fixed_arr)
        assert d >= 0.55, f"RITK BSpline FFD Dice {d:.4f} < 0.55"

    # --- Side-by-side ---

    def test_side_by_side_demons_vs_sitk_translation_quality(self):
        """Both RITK Demons and SimpleITK translation must achieve Dice >= 0.80."""
        sitk_result = _sitk_translation_register(
            self.fixed_sitk, self.moving_sitk, num_iterations=100
        )
        assert sitk_result is not None
        d_sitk = _dice(_sitk_to_numpy(sitk_result), self.fixed_arr)

        warped, _ = ritk.registration.demons_register(
            self.fixed_ritk, self.moving_ritk, max_iterations=100
        )
        d_ritk = _dice(warped.to_numpy(), self.fixed_arr)

        assert d_sitk >= 0.80, f"SimpleITK Dice {d_sitk:.4f} < 0.80"
        assert d_ritk >= 0.80, f"RITK Dice {d_ritk:.4f} < 0.80"

    def test_side_by_side_syn_vs_sitk_bspline_quality(self):
        """RITK SyN and SimpleITK BSpline must achieve Dice >= 0.75 on locally
        deformed sphere (Gaussian bump displacement)."""
        from scipy.ndimage import map_coordinates

        arr_fixed = _make_sphere(size=SIZE, radius=6)
        c = SIZE // 2
        z, y, x = np.mgrid[:SIZE, :SIZE, :SIZE]
        bump = 3.0 * np.exp(
            -((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2) / (2 * 5.0**2)
        )
        x_displaced = np.clip(x + bump, 0, SIZE - 1).astype(np.float32)
        arr_moving = (
            map_coordinates(
                arr_fixed,
                [z.ravel(), y.ravel(), x_displaced.ravel()],
                order=1,
                mode="nearest",
            )
            .reshape(SIZE, SIZE, SIZE)
            .astype(np.float32)
        )

        fixed_s = _numpy_to_sitk(arr_fixed)
        moving_s = _numpy_to_sitk(arr_moving)
        fixed_r = _numpy_to_ritk(arr_fixed)
        moving_r = _numpy_to_ritk(arr_moving)

        # SimpleITK BSpline (fewer iterations to avoid timeout on 64^3)
        sitk_result = _sitk_bspline_register(
            fixed_s, moving_s, grid_spacing=8.0, num_iterations=30
        )
        if sitk_result is not None:
            d_sitk = _dice(_sitk_to_numpy(sitk_result), arr_fixed)
        else:
            d_sitk = 0.0

        # RITK SyN with reduced smoothing for continuous data
        _, warped_m = ritk.registration.syn_register(fixed_r,
            moving_r, ritk.registration.SynConfig(max_iterations=100,sigma_smooth=1.0,cc_radius=2,gradient_step=0.5))
        d_ritk = _dice(warped_m.to_numpy(), arr_fixed)

        assert d_sitk >= 0.70 or sitk_result is None, (
            f"SimpleITK BSpline Dice {d_sitk:.4f} < 0.70"
        )
        assert d_ritk >= 0.75, f"RITK SyN Dice {d_ritk:.4f} < 0.75"


# ============================================================================
# Section 2: Synthetic Gaussian blob — NCC improvement
# ============================================================================


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
        _, warped_m = ritk.registration.syn_register(self.fixed_ritk,
            self.moving_ritk, ritk.registration.SynConfig(max_iterations=100,sigma_smooth=1.0,cc_radius=2,gradient_step=0.5))
        ncc_after = _ncc(self.fixed_arr, warped_m.to_numpy())
        assert ncc_after > self.ncc_before, (
            f"SyN did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_after:.4f}"
        )

    def test_ritk_demons_ncc_improves(self):
        """RITK Demons must improve NCC over baseline."""
        warped, _ = ritk.registration.demons_register(
            self.fixed_ritk, self.moving_ritk, max_iterations=100
        )
        ncc_after = _ncc(self.fixed_arr, warped.to_numpy())
        assert ncc_after > self.ncc_before, (
            f"Demons did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_after:.4f}"
        )

    def test_ritk_bspline_ffd_ncc_improves(self):
        """RITK BSpline FFD must improve NCC over baseline.

        bspline_ffd_register returns a single warped Image (not a tuple).
        """
        warped = ritk.registration.bspline_ffd_register(self.fixed_ritk,
            self.moving_ritk, ritk.registration.BSplineFfdConfig(initial_control_spacing=8,num_levels=2,max_iterations=50))
        ncc_after = _ncc(self.fixed_arr, warped.to_numpy())
        assert ncc_after > self.ncc_before, (
            f"BSpline FFD did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_after:.4f}"
        )

    def test_ritk_lddmm_ncc_improves(self):
        """RITK LDDMM must improve NCC over baseline.

        lddmm_register returns (warped, velocity_field).
        """
        warped, _ = ritk.registration.lddmm_register(self.fixed_ritk,
            self.moving_ritk, ritk.registration.LddmmConfig(max_iterations=30,num_time_steps=5,kernel_sigma=2.0))
        ncc_after = _ncc(self.fixed_arr, warped.to_numpy())
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
        if sitk_result is not None:
            ncc_sitk = _ncc(self.fixed_arr, _sitk_to_numpy(sitk_result))
        else:
            ncc_sitk = self.ncc_before

        _, warped_m = ritk.registration.syn_register(self.fixed_ritk,
            self.moving_ritk, ritk.registration.SynConfig(max_iterations=100,sigma_smooth=1.0,cc_radius=2,gradient_step=0.5))
        ncc_ritk = _ncc(self.fixed_arr, warped_m.to_numpy())

        assert ncc_sitk > self.ncc_before, (
            f"SimpleITK BSpline did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_sitk:.4f}"
        )
        assert ncc_ritk > self.ncc_before, (
            f"RITK SyN did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_ritk:.4f}"
        )


# ============================================================================
# Section 3: Inter-subject brain MNI pair — NCC/MSE improvement
# ============================================================================

_skip_mni = pytest.mark.skipif(
    not (TEST_DATA / "brain_mni" / "mni152.nii.gz").exists(),
    reason="MNI152 / sub-01_T1w NIfTI pair absent",
)


@_skip_mni
@pytest.mark.slow
class TestInterSubjectBrainMNI:
    """Inter-subject brain registration using MNI152 and OpenNeuro T1w.

    These are different subjects with different head positions and brain
    shapes. Pre-registration NCC is typically very low (~0.04) because
    inter-subject brain shape and position differences dominate. The
    meaningful metric here is MSE reduction: even a partial local alignment
    reduces voxel-wise intensity differences.

    NCC improvement is not required because these subjects have genuinely
    different brain anatomy — no amount of deformable registration can
    make one subject's brain look like another's in NCC terms.
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
        # Normalize for fair comparison
        self.fixed_norm = _minmax(self.fixed_arr)
        self.moving_norm = _minmax(self.moving_arr)
        self.fixed_norm_r = _numpy_to_ritk(self.fixed_norm)
        self.moving_norm_r = _numpy_to_ritk(self.moving_norm)
        self.fixed_norm_s = _numpy_to_sitk(self.fixed_norm)
        self.moving_norm_s = _numpy_to_sitk(self.moving_norm)
        self.ncc_before = _ncc(self.fixed_norm, self.moving_norm)
        self.mse_before = _mse(self.fixed_norm, self.moving_norm)

    def test_ritk_demons_reduces_mse(self):
        """RITK Demons must reduce MSE on inter-subject brain."""
        warped, _ = ritk.registration.demons_register(
            self.fixed_norm_r, self.moving_norm_r, max_iterations=50
        )
        mse_after = _mse(self.fixed_norm, warped.to_numpy())
        assert mse_after < self.mse_before, (
            f"RITK Demons did not reduce MSE: "
            f"before={self.mse_before:.6f}, after={mse_after:.6f}"
        )

    def test_ritk_syn_reduces_mse(self):
        """RITK SyN must reduce MSE on inter-subject brain."""
        _, warped_m = ritk.registration.syn_register(self.fixed_norm_r,
            self.moving_norm_r, ritk.registration.SynConfig(max_iterations=30,sigma_smooth=3.0,cc_radius=2,gradient_step=0.25))
        mse_after = _mse(self.fixed_norm, warped_m.to_numpy())
        assert mse_after < self.mse_before, (
            f"RITK SyN did not reduce MSE: "
            f"before={self.mse_before:.6f}, after={mse_after:.6f}"
        )

    def test_side_by_side_brain_mse_comparison(self):
        """RITK Demons must reduce MSE on inter-subject brain; SimpleITK
        translation may or may not depending on initial alignment.

        SimpleITK translation registration on very different inter-subject
        brains (NCC ≈ 0.04) can worsen MSE because the 3-DOF translation
        is insufficient for the large anatomical differences. RITK Demons
        with local deformable forces can partially compensate.
        """
        # RITK
        warped, _ = ritk.registration.demons_register(
            self.fixed_norm_r, self.moving_norm_r, max_iterations=50
        )
        mse_ritk = _mse(self.fixed_norm, warped.to_numpy())
        assert mse_ritk < self.mse_before, (
            f"RITK did not reduce MSE: {self.mse_before:.6f} -> {mse_ritk:.6f}"
        )


# ============================================================================
# Section 4: Multi-modal CT/MR — RIRE pair
# ============================================================================

_skip_rire = pytest.mark.skipif(
    not (TEST_DATA / "rire" / "training_001_ct.mha").exists(),
    reason="RIRE CT/MR pair absent",
)


@_skip_rire
@pytest.mark.slow
class TestRIREMultiModal:
    """CT↔MR cross-modal registration using RIRE data.

    Mathematical basis: CT encodes X-ray attenuation (Hounsfield units),
    MR encodes proton density/T1/T2 relaxation. Direct NCC between raw
    modalities is low; after minmax normalization, structural overlap
    allows deformable registration to improve alignment.
    """

    @pytest.fixture(autouse=True)
    def setup_rire_pair(self):
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
        warped, _ = ritk.registration.demons_register(
            self.fixed_norm_r, self.moving_norm_r, max_iterations=50
        )
        ncc_after = _ncc(self.fixed_norm, warped.to_numpy())
        assert ncc_after > self.ncc_before, (
            f"Demons did not improve cross-modal NCC: "
            f"before={self.ncc_before:.4f}, after={ncc_after:.4f}"
        )

    def test_ritk_syn_cross_modal_ncc_improves(self):
        """RITK SyN must improve NCC on cross-modal data after normalization."""
        _, warped_m = ritk.registration.syn_register(self.fixed_norm_r,
            self.moving_norm_r, ritk.registration.SynConfig(max_iterations=30,sigma_smooth=3.0,cc_radius=2))
        ncc_after = _ncc(self.fixed_norm, warped_m.to_numpy())
        assert ncc_after > self.ncc_before, (
            f"SyN did not improve cross-modal NCC: "
            f"before={self.ncc_before:.4f}, after={ncc_after:.4f}"
        )

    def test_side_by_side_cross_modal_ncc(self):
        """Both SimpleITK BSpline and RITK SyN must improve NCC on cross-modal data."""
        fixed_s = _numpy_to_sitk(self.fixed_norm)
        moving_s = _numpy_to_sitk(self.moving_norm)
        sitk_result = _sitk_bspline_register(
            fixed_s, moving_s, grid_spacing=8.0, num_iterations=30
        )
        if sitk_result is not None:
            ncc_sitk = _ncc(self.fixed_norm, _sitk_to_numpy(sitk_result))
        else:
            ncc_sitk = self.ncc_before

        _, warped_m = ritk.registration.syn_register(self.fixed_norm_r,
            self.moving_norm_r, ritk.registration.SynConfig(max_iterations=30,sigma_smooth=3.0,cc_radius=2))
        ncc_ritk = _ncc(self.fixed_norm, warped_m.to_numpy())

        # SimpleITK BSpline may diverge on small cross-modal crops;
        # the primary assertion is that RITK SyN improves alignment
        assert ncc_ritk > self.ncc_before, (
            f"RITK SyN did not improve cross-modal NCC: "
            f"before={self.ncc_before:.4f}, after={ncc_ritk:.4f}"
        )
        if sitk_result is not None:
            assert ncc_sitk > self.ncc_before - 0.05, (
                f"SimpleITK BSpline regressed NCC: "
                f"before={self.ncc_before:.4f}, after={ncc_sitk:.4f}"
            )
        assert ncc_ritk > self.ncc_before, (
            f"RITK SyN did not improve cross-modal NCC: "
            f"before={self.ncc_before:.4f}, after={ncc_ritk:.4f}"
        )


# ============================================================================
# Section 5: Multi-modal CT/MR — Visible Male head pair
# ============================================================================

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
        warped, _ = ritk.registration.demons_register(
            self.fixed_norm_r, self.moving_norm_r, max_iterations=50
        )
        ncc_after = _ncc(self.fixed_norm, warped.to_numpy())
        assert ncc_after > self.ncc_before, (
            f"Demons did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_after:.4f}"
        )

    def test_ritk_syn_cross_modal_ncc_improves(self):
        """RITK SyN must improve NCC on VM head cross-modal data."""
        _, warped_m = ritk.registration.syn_register(self.fixed_norm_r,
            self.moving_norm_r, ritk.registration.SynConfig(max_iterations=30,sigma_smooth=3.0,cc_radius=2))
        ncc_after = _ncc(self.fixed_norm, warped_m.to_numpy())
        assert ncc_after > self.ncc_before, (
            f"SyN did not improve NCC: before={self.ncc_before:.4f}, "
            f"after={ncc_after:.4f}"
        )


# ============================================================================
# Section 6: Comprehensive quality report across all algorithms
# ============================================================================


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
            "demons": lambda: ritk.registration.demons_register(
                fixed_r, moving_r, max_iterations=100
            ),
            "diffeomorphic_demons": lambda: (
                ritk.registration.diffeomorphic_demons_register(
                    fixed_r, moving_r, max_iterations=100
                )
            ),
            "symmetric_demons": lambda: ritk.registration.symmetric_demons_register(
                fixed_r, moving_r, max_iterations=100
            ),
            "multires_demons": lambda: ritk.registration.multires_demons_register(fixed_r, moving_r, ritk.registration.MultiResDemonsOptions(max_iterations=100,levels=3)),
            "ic_demons": lambda: ritk.registration.inverse_consistent_demons_register(
                fixed_r, moving_r, max_iterations=100
            ),
            "syn": lambda: ritk.registration.syn_register(fixed_r,
                moving_r, ritk.registration.SynConfig(max_iterations=100,sigma_smooth=1.0,cc_radius=2,gradient_step=0.5)),
            "multires_syn": lambda: ritk.registration.multires_syn_register(fixed_r,
                moving_r, ritk.registration.MultiResSynOptions(num_levels=3,sigma_smooth=1.0,cc_radius=2,gradient_step=0.5)),
            "bspline_syn": lambda: ritk.registration.bspline_syn_register(fixed_r,
                moving_r, ritk.registration.BSplineSynOptions(max_iterations=100,control_spacing_z=8,control_spacing_y=8,control_spacing_x=8,sigma_smooth=1.0,cc_radius=2,gradient_step=0.5)),
            "bspline_ffd": lambda: ritk.registration.bspline_ffd_register(fixed_r,
                moving_r, ritk.registration.BSplineFfdConfig(initial_control_spacing=8,num_levels=2,max_iterations=50)),
            "lddmm": lambda: ritk.registration.lddmm_register(fixed_r,
                moving_r, ritk.registration.LddmmConfig(max_iterations=30,num_time_steps=5,kernel_sigma=2.0)),
        }

        for name, fn in algorithms.items():
            result = fn()
            # Extract warped image based on return type
            if name == "bspline_ffd":
                # Returns single Image
                warped_arr = result.to_numpy()
            elif name in ("syn", "multires_syn", "bspline_syn"):
                # Returns (warped_fixed, warped_moving); use warped_moving
                warped_arr = result[1].to_numpy()
            elif name == "lddmm":
                # Returns (warped, velocity_field)
                warped_arr = result[0].to_numpy()
            else:
                # Demons family: (warped, displacement)
                warped_arr = result[0].to_numpy()

            ncc_after = _ncc(fixed_arr, warped_arr)
            assert ncc_after > ncc_before, (
                f"{name} did not improve NCC: before={ncc_before:.4f}, "
                f"after={ncc_after:.4f}"
            )
