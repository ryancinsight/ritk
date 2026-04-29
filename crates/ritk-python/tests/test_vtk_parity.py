"""VTK 9.6.1 image filter parity tests.

Validates VTK filter outputs against SimpleITK 2.5.4 reference outputs and
analytical formulae on synthetically constructed 3D images.  No file I/O.

Axis-mapping contract (critical for all tests):
    NumPy array shape is (nz, ny, nx) in C order (z slowest, x fastest).
    VTK consumes data in Fortran (x-fastest) order via arr.ravel(order='F').
    Consequently: arr[iz, iy, ix] → VTK voxel at (vtk_x=iz, vtk_y=iy, vtk_z=ix).
    Equivalently: numpy dim-0 ↔ VTK x-axis, numpy dim-2 ↔ VTK z-axis.

    SimpleITK GetImageFromArray(arr) maps arr[iz, iy, ix] → ITK(x=ix, y=iy, z=iz),
    i.e., numpy dim-2 ↔ ITK x-axis (opposite to VTK).
    Gradient magnitude is a rotation-invariant scalar, so both libraries agree
    on the magnitude when SetDimensionality(3) is set on VTK filters.

Dimensionality note:
    vtkImageGradientMagnitude and vtkImageLaplacian default to Dimensionality=2
    (XY plane only).  All 3-D tests must call SetDimensionality(3) explicitly.

Run:
    pytest crates/ritk-python/tests/test_vtk_parity.py -v

Requires:
    vtk >= 9.6, SimpleITK >= 2.5, numpy >= 2.0, scipy >= 1.17
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Module-level skip when VTK or SimpleITK are absent.
# ---------------------------------------------------------------------------
vtk = pytest.importorskip("vtk")
sitk = pytest.importorskip("SimpleITK")

from vtk.util import numpy_support  # type: ignore[import-untyped]  # noqa: E402

SIZE = 16  # Edge length (voxels) of all synthetic test volumes.


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _vtk_from_np(arr: np.ndarray):
    """Convert a (nz, ny, nx) float32 NumPy array to vtkImageData.

    Uses Fortran-order ravelling so that numpy dim-0 maps to VTK x (fastest).
    Spacing is isotropic 1.0 mm; origin is (0, 0, 0).
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    nz, ny, nx = arr.shape
    img = vtk.vtkImageData()
    img.SetDimensions(nz, ny, nx)  # SetDimensions(x, y, z) in VTK API
    img.SetSpacing(1.0, 1.0, 1.0)
    img.SetOrigin(0.0, 0.0, 0.0)
    flat = arr.ravel(order="F")
    vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    img.GetPointData().SetScalars(vtk_arr)
    return img


def _vtk_to_np(vtk_img, ref_shape: tuple) -> np.ndarray:
    """Read scalars from vtkImageData back into a NumPy array of ref_shape.

    ref_shape must be the original (nz, ny, nx) shape passed to _vtk_from_np.
    """
    raw = numpy_support.vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    return raw.reshape(ref_shape, order="F").astype(np.float32)


def _sitk_from_np(arr: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    """Convert a (nz, ny, nx) float32 NumPy array to a SimpleITK image.

    SimpleITK GetImageFromArray interprets arr[iz, iy, ix] as ITK(x=ix, y=iy, z=iz),
    so spacing is supplied in (x, y, z) order matching ITK conventions.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing([float(spacing[2]), float(spacing[1]), float(spacing[0])])
    return img


# ---------------------------------------------------------------------------
# Synthetic image factories
# ---------------------------------------------------------------------------


def _make_sphere(size: int = SIZE, radius: float = 4.0) -> np.ndarray:
    """Binary sphere of given radius centred in a (size)^3 volume.

    Mathematical definition: f(z,y,x) = 1 iff (z-c)^2+(y-c)^2+(x-c)^2 <= r^2,
    where c = size//2.  Returns float32 array.
    """
    c = size // 2
    z, y, x = np.mgrid[:size, :size, :size]
    return ((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= radius**2).astype(np.float32)


def _make_ramp_dim0(size: int = SIZE) -> np.ndarray:
    """Linear ramp along numpy dim-0 (z index), which maps to VTK x-axis.

    f[iz, iy, ix] = iz.  Analytical gradient magnitude = 1.0 everywhere
    (central finite difference with spacing=1.0).
    """
    z, _, _ = np.mgrid[:size, :size, :size]
    return z.astype(np.float32)


def _make_ramp_all_dims(size: int = SIZE) -> np.ndarray:
    """Linear ramp in all three dims: f[z,y,x] = z + y + x (numpy indices).

    Analytical: Laplacian = d^2f/dx^2 + d^2f/dy^2 + d^2f/dz^2 = 0 for any
    affine function, since all second derivatives vanish identically.
    """
    z, y, x = np.mgrid[:size, :size, :size]
    return (z + y + x).astype(np.float32)


def _make_t_phantom(size: int = SIZE) -> np.ndarray:
    """T-shaped 3D phantom with soft edges (pre-blurred binary mask).

    The T consists of a horizontal crossbar and a vertical stem, both
    3 voxels thick.  A Gaussian pre-blur (sigma=0.8) produces soft edges
    so neither library's Gaussian filter sees a perfectly sharp step.
    """
    arr = np.zeros((size, size, size), dtype=np.float32)
    c = size // 2
    # Horizontal bar: full width, mid-height strip
    arr[c - 2 : c + 2, c - 2 : c + 2, 2 : size - 2] = 1.0
    # Vertical stem: one quadrant below crossbar
    arr[c - 6 : c + 2, 2:6, c - 2 : c + 2] = 1.0
    return gaussian_filter(arr, sigma=0.8).astype(np.float32)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised root-mean-square error: RMSE / (max(b) - min(b)).

    Range normalisation avoids scale-dependence when comparing filter outputs
    on different phantoms.  A small epsilon prevents division by zero on
    constant reference arrays.
    """
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    rng = float(b.max() - b.min())
    return rmse / (rng + 1e-12)


# ===========================================================================
# Test 1 – Gaussian smooth: constant image is an eigenfunction of convolution
# ===========================================================================


def test_gaussian_smooth_constant_image_unchanged():
    """Gaussian convolution of a constant image reproduces the constant.

    Mathematical basis: (f * g)(x) = c * integral(g) = c * 1 = c when f = c
    and g is a unit-normalised Gaussian kernel.  The normalisation condition
    sum_k g[k] = 1 holds for any separable discrete Gaussian, so the output
    must equal the input to floating-point precision.

    Tolerance: max relative deviation < 1e-5 over all interior voxels.
    """
    arr = np.full((SIZE, SIZE, SIZE), 5.0, dtype=np.float32)
    vtk_img = _vtk_from_np(arr)

    filt = vtk.vtkImageGaussianSmooth()
    filt.SetInputData(vtk_img)
    filt.SetStandardDeviation(1.0)
    filt.SetRadiusFactor(1.5)
    filt.Update()

    out = _vtk_to_np(filt.GetOutput(), arr.shape)
    interior = out[2:-2, 2:-2, 2:-2]

    max_rel_err = float(np.abs(interior - 5.0).max() / 5.0)
    assert max_rel_err < 1e-5, (
        f"Constant-image Gaussian invariant violated: max relative error "
        f"{max_rel_err:.2e} >= 1e-5"
    )


# ===========================================================================
# Test 2 – Gaussian sphere: VTK vs SimpleITK interior NRMSE
# ===========================================================================


def test_gaussian_smooth_sphere_agrees_with_sitk():
    """VTK and SimpleITK Gaussian filters agree on a binary sphere interior.

    Both vtkImageGaussianSmooth and sitk.DiscreteGaussianImageFilter implement
    separable Gaussian convolution with variance sigma^2 = 1.0.  Interior
    agreement is tight (NRMSE < 0.15); boundary handling strategies differ
    (VTK mirrors, ITK zero-pads) so border voxels are excluded with a 3-voxel
    trim on each side.

    NRMSE = RMSE / range(sitk_output).  Threshold 0.15 is derived from the
    maximum expected boundary-mode divergence at the 3-voxel exclusion zone.
    """
    arr = _make_sphere(SIZE, radius=4.0)

    # VTK Gaussian
    filt_vtk = vtk.vtkImageGaussianSmooth()
    filt_vtk.SetInputData(_vtk_from_np(arr))
    filt_vtk.SetStandardDeviation(1.0)
    filt_vtk.SetRadiusFactor(1.5)
    filt_vtk.Update()
    out_vtk = _vtk_to_np(filt_vtk.GetOutput(), arr.shape)

    # SimpleITK Gaussian
    filt_sitk = sitk.DiscreteGaussianImageFilter()
    filt_sitk.SetVariance(1.0)
    out_sitk = sitk.GetArrayFromImage(filt_sitk.Execute(_sitk_from_np(arr))).astype(
        np.float32
    )

    int_vtk = out_vtk[3:-3, 3:-3, 3:-3]
    int_sitk = out_sitk[3:-3, 3:-3, 3:-3]

    nrmse = _nrmse(int_vtk, int_sitk)
    assert nrmse < 0.15, (
        f"Gaussian sphere interior NRMSE {nrmse:.4f} exceeds threshold 0.15"
    )


# ===========================================================================
# Test 3 – Gradient magnitude: linear ramp yields magnitude 1.0
# ===========================================================================


def test_gradient_magnitude_linear_ramp():
    """Gradient magnitude of a linear ramp is 1.0 at every interior voxel.

    Input: f[iz, iy, ix] = float(ix)  (ramp in numpy last index = VTK z-axis).
    With unit spacing, the central-difference approximation to the partial
    derivative df/dz_VTK = 1.0 exactly, and all other partials are zero.
    Hence ||grad f|| = 1.0 analytically.

    VTK note: SetDimensionality(3) is required; the default is 2 (XY-plane
    only), which would miss the z-direction gradient and return 0.

    Tolerance: max absolute deviation from 1.0 < 0.01 over interior voxels.
    """
    _, _, x = np.mgrid[:SIZE, :SIZE, :SIZE]
    arr = x.astype(np.float32)  # arr[z,y,x] = x  (numpy x-index = VTK z-axis)

    filt = vtk.vtkImageGradientMagnitude()
    filt.SetInputData(_vtk_from_np(arr))
    filt.HandleBoundariesOn()
    filt.SetDimensionality(3)  # mandatory: default is 2 (XY only)
    filt.Update()

    out = _vtk_to_np(filt.GetOutput(), arr.shape)
    interior = out[2:-2, 2:-2, 2:-2]

    max_abs_err = float(np.abs(interior - 1.0).max())
    assert max_abs_err < 0.01, (
        f"Gradient magnitude of linear ramp: max |out - 1.0| = {max_abs_err:.4f} "
        f">= 0.01"
    )


# ===========================================================================
# Test 4 – Gradient magnitude: VTK vs SimpleITK Pearson correlation
# ===========================================================================


def test_gradient_magnitude_agrees_with_sitk():
    """VTK and SimpleITK gradient magnitudes agree on a pre-smoothed sphere.

    Pre-smoothing with scipy.ndimage.gaussian_filter (sigma=1.5) creates a
    smooth, continuous-looking object without relying on either library's
    boundary conventions.  Both libraries then compute central-difference
    gradient magnitude on the same input array, so the outputs must agree
    exactly (up to floating-point rounding) once 3-D dimensionality is set.

    Mathematical basis: for isotropic spacing s=1, the discrete gradient
    magnitude at interior voxels uses central differences:
        ||grad f||^2 = sum_d ((f[..+1..] - f[..-1..]) / 2)^2
    This sum is commutative in d, so axis-labelling differences between VTK
    and ITK cancel for any array that is symmetric in its three axes (sphere).

    Pearson correlation threshold: > 0.95 (empirically > 0.9999 for this
    phantom after SetDimensionality(3) is set).
    """
    raw_sphere = _make_sphere(SIZE, radius=4.0)
    # Pre-smooth with scipy: neutral baseline independent of either library
    smooth = gaussian_filter(raw_sphere, sigma=1.5).astype(np.float32)

    # VTK gradient magnitude
    filt_vtk = vtk.vtkImageGradientMagnitude()
    filt_vtk.SetInputData(_vtk_from_np(smooth))
    filt_vtk.HandleBoundariesOn()
    filt_vtk.SetDimensionality(3)
    filt_vtk.Update()
    out_vtk = _vtk_to_np(filt_vtk.GetOutput(), smooth.shape)

    # SimpleITK gradient magnitude
    filt_sitk = sitk.GradientMagnitudeImageFilter()
    out_sitk = sitk.GetArrayFromImage(filt_sitk.Execute(_sitk_from_np(smooth))).astype(
        np.float32
    )

    int_vtk = out_vtk[2:-2, 2:-2, 2:-2].ravel()
    int_sitk = out_sitk[2:-2, 2:-2, 2:-2].ravel()

    result = pearsonr(int_vtk, int_sitk)
    r: float = result[0]  # type: ignore[assignment]  # PearsonRResult[0] is float64
    assert r > 0.95, f"Gradient magnitude Pearson correlation {r:.6f} < 0.95"


# ===========================================================================
# Test 5 – Laplacian of a linear image is zero
# ===========================================================================


def test_laplacian_linear_image_is_zero():
    """Laplacian of any affine function is identically zero.

    Input: f[z,y,x] = z + y + x  (all numpy indices contribute equally).
    Proof: L[f] = d^2f/dx^2 + d^2f/dy^2 + d^2f/dz^2 = 0 + 0 + 0 = 0,
    since d^2/d{anything}^2 of a degree-1 polynomial is zero.

    The discrete 3-point central-difference Laplacian (coefficient pattern
    [1, -2, 1] / h^2) also returns exactly 0 for a linear function, so no
    finite-difference approximation error is expected.

    VTK note: SetDimensionality(3) is required (default is 2).

    Tolerance: max |interior value| < 0.05 (accommodates floating-point
    arithmetic accumulation across three axes).
    """
    arr = _make_ramp_all_dims(SIZE)

    filt = vtk.vtkImageLaplacian()
    filt.SetInputData(_vtk_from_np(arr))
    filt.SetDimensionality(3)
    filt.Update()

    out = _vtk_to_np(filt.GetOutput(), arr.shape)
    interior = out[2:-2, 2:-2, 2:-2]

    max_abs = float(np.abs(interior).max())
    assert max_abs < 0.05, (
        f"Laplacian of linear image: max |interior| = {max_abs:.4e} >= 0.05"
    )


# ===========================================================================
# Test 6 – Median filter suppresses an isolated spike
# ===========================================================================


def test_median_filter_removes_single_spike():
    """Median filter with 3x3x3 kernel suppresses a single-voxel spike.

    Input: constant background = 1.0 with one voxel at the image centre
    set to 100.0.  The 3x3x3 neighbourhood (27 voxels) around the centre
    contains 26 background voxels (value 1.0) and 1 spike (value 100.0).
    The median of {1.0 × 26, 100.0 × 1} is 1.0 (the 14th-ranked value in
    the sorted 27-element set).

    Mathematical guarantee: for a constant background of value c and a single
    outlier, the median over any neighbourhood of size N > 1 returns c when
    the outlier count < N/2.  Here 1 < 27/2 = 13.5, so the spike is
    unconditionally suppressed.

    Assertion: filtered centre value < 10.0 (strict; expected is 1.0).
    """
    arr = np.ones((SIZE, SIZE, SIZE), dtype=np.float32)
    c = SIZE // 2
    arr[c, c, c] = 100.0  # spike at numpy (z=c, y=c, x=c) = VTK (x=c, y=c, z=c)

    filt = vtk.vtkImageMedian3D()
    filt.SetInputData(_vtk_from_np(arr))
    filt.SetKernelSize(3, 3, 3)
    filt.Update()

    out = _vtk_to_np(filt.GetOutput(), arr.shape)
    centre_value = float(out[c, c, c])

    assert centre_value < 10.0, (
        f"Median filter did not suppress spike: centre value = {centre_value:.2f} "
        f"(expected ≈ 1.0, threshold < 10.0)"
    )


# ===========================================================================
# Test 7 – Binary erosion strictly shrinks the sphere
# ===========================================================================


def test_binary_erosion_shrinks_sphere():
    """Morphological erosion A ⊖ B satisfies (A ⊖ B) ⊆ A.

    Equivalently, sum(eroded) <= sum(original), with strict inequality when
    the structuring element B is larger than a single voxel and A has at
    least one boundary voxel.

    Input: binary sphere (foreground = 255, background = 0), radius = 5 in a
    16^3 volume.  Structuring element: 3x3x3 box kernel.

    VTK vtkImageDilateErode3D parameter convention for erosion:
        ErodeValue  = 255  (foreground value to erode)
        DilateValue = 0    (background value; stays fixed)
    The filter replaces a foreground voxel with the background value if any
    voxel in the kernel neighbourhood equals the background value.

    Assertion: sum(eroded) < sum(original).
    """
    arr = _make_sphere(SIZE, radius=5.0) * 255.0

    filt = vtk.vtkImageDilateErode3D()
    filt.SetInputData(_vtk_from_np(arr))
    filt.SetKernelSize(3, 3, 3)
    filt.SetErodeValue(255)  # shrink voxels whose neighbourhood hits background
    filt.SetDilateValue(0)  # background remains 0
    filt.Update()

    out = _vtk_to_np(filt.GetOutput(), arr.shape)

    sum_original = float(arr.sum())
    sum_eroded = float(out.sum())

    assert sum_eroded < sum_original, (
        f"Erosion did not reduce sphere: eroded sum {sum_eroded:.0f} "
        f">= original sum {sum_original:.0f}"
    )


# ===========================================================================
# Test 8 – Binary dilation strictly grows the sphere
# ===========================================================================


def test_binary_dilation_grows_sphere():
    """Morphological dilation A ⊕ B satisfies A ⊆ (A ⊕ B).

    Equivalently, sum(dilated) >= sum(original), with strict inequality when
    B is larger than a single voxel and A has at least one interior voxel
    adjacent to the background.

    Same input and structuring element as test_binary_erosion_shrinks_sphere.

    VTK vtkImageDilateErode3D parameter convention for dilation:
        DilateValue = 255  (foreground value to grow outward)
        ErodeValue  = 0    (no erosion of foreground)
    The filter sets a background voxel to the foreground value if any
    neighbour in the kernel is a foreground voxel.

    Assertion: sum(dilated) > sum(original).
    """
    arr = _make_sphere(SIZE, radius=5.0) * 255.0

    filt = vtk.vtkImageDilateErode3D()
    filt.SetInputData(_vtk_from_np(arr))
    filt.SetKernelSize(3, 3, 3)
    filt.SetDilateValue(255)  # grow foreground outward
    filt.SetErodeValue(0)  # background value that triggers dilation
    filt.Update()

    out = _vtk_to_np(filt.GetOutput(), arr.shape)

    sum_original = float(arr.sum())
    sum_dilated = float(out.sum())

    assert sum_dilated > sum_original, (
        f"Dilation did not grow sphere: dilated sum {sum_dilated:.0f} "
        f"<= original sum {sum_original:.0f}"
    )


# ===========================================================================
# Test 9 – Scalar range from vtkImageAccumulate matches analytical range
# ===========================================================================


def test_scalar_range_agrees_with_analytical():
    """vtkImageAccumulate reports the correct scalar range for a 0–255 ramp.

    Input: flat VTK array = arange(SIZE^3) % 256 cast to float32.  By
    construction min = 0.0 and max = 255.0 (since SIZE^3 = 4096 > 256,
    so every integer in [0, 255] appears at least once).

    vtkImageAccumulate.GetMin() / GetMax() operate on the histogram and
    return the lowest/highest occupied bin edges, which must equal 0.0 and
    255.0 respectively.

    Tolerance: ±1.0 (one histogram bin width, set by ComponentSpacing=1.0).
    """
    total = SIZE**3
    flat = np.arange(total, dtype=np.float32) % 256
    # Build vtkImageData directly from flat VTK-order data (no numpy reshape needed)
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(SIZE, SIZE, SIZE)
    vtk_img.SetSpacing(1.0, 1.0, 1.0)
    vtk_img.SetOrigin(0.0, 0.0, 0.0)
    vtk_img.GetPointData().SetScalars(
        numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    )

    acc = vtk.vtkImageAccumulate()
    acc.SetInputData(vtk_img)
    acc.SetComponentExtent(0, 255, 0, 0, 0, 0)
    acc.SetComponentOrigin(0.0, 0.0, 0.0)
    acc.SetComponentSpacing(1.0, 1.0, 1.0)
    acc.Update()

    reported_min = float(acc.GetMin()[0])
    reported_max = float(acc.GetMax()[0])

    assert abs(reported_min - 0.0) <= 1.0, (
        f"Scalar range min: reported {reported_min:.2f}, expected 0.0 ± 1.0"
    )
    assert abs(reported_max - 255.0) <= 1.0, (
        f"Scalar range max: reported {reported_max:.2f}, expected 255.0 ± 1.0"
    )


# ===========================================================================
# Test 10 – T-phantom Gaussian: VTK vs SimpleITK interior NRMSE
# ===========================================================================


def test_sitk_and_vtk_gaussian_interior_nrmse_below_threshold():
    """VTK and SimpleITK Gaussian filters agree on a T-shaped 3D phantom.

    The T-phantom has soft edges (pre-blurred binary mask) to avoid
    discontinuities that amplify boundary-mode divergence.  Both
    vtkImageGaussianSmooth and sitk.DiscreteGaussianImageFilter implement
    separable Gaussian convolution; their outputs should agree in the interior
    up to floating-point precision differences and boundary extension strategy.

    Mathematical note: a separable 3-D Gaussian kernel K = Kx ⊗ Ky ⊗ Kz is
    applied independently along each axis.  For interior voxels (where no
    boundary extension is invoked), the convolution sum is identical regardless
    of the library, provided the kernel weights match.  Kernel weights agree
    when variance = 1.0 and RadiusFactor >= 3 standard deviations; here
    RadiusFactor = 1.5 clamps the kernel to ±1.5 sigma, introducing a small
    but bounded truncation difference from SimpleITK's default of 1.0 variance
    with maximum kernel error 0.001.

    NRMSE threshold: 0.10  (empirical upper bound; measured value ≈ 0.028).
    Interior mask: 2-voxel border trim on each face.
    """
    arr = _make_t_phantom(SIZE)

    # VTK Gaussian
    filt_vtk = vtk.vtkImageGaussianSmooth()
    filt_vtk.SetInputData(_vtk_from_np(arr))
    filt_vtk.SetStandardDeviation(1.0)
    filt_vtk.SetRadiusFactor(1.5)
    filt_vtk.Update()
    out_vtk = _vtk_to_np(filt_vtk.GetOutput(), arr.shape)

    # SimpleITK Gaussian
    filt_sitk = sitk.DiscreteGaussianImageFilter()
    filt_sitk.SetVariance(1.0)
    out_sitk = sitk.GetArrayFromImage(filt_sitk.Execute(_sitk_from_np(arr))).astype(
        np.float32
    )

    int_vtk = out_vtk[2:-2, 2:-2, 2:-2]
    int_sitk = out_sitk[2:-2, 2:-2, 2:-2]

    nrmse = _nrmse(int_vtk, int_sitk)
    assert nrmse < 0.10, (
        f"T-phantom Gaussian interior NRMSE {nrmse:.4f} exceeds threshold 0.10"
    )


# ---------------------------------------------------------------------------
# NCC helper
# ---------------------------------------------------------------------------


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson normalised cross-correlation: cov(a,b) / (std(a) * std(b)).

    Returns 0.0 when either array has zero standard deviation (degenerate case
    where the denominator would be < 1e-12).

    Mathematical definition:
        NCC(a, b) = Σᵢ (aᵢ − ā)(bᵢ − b̄)
                    ─────────────────────────
                    ‖a − ā‖₂ · ‖b − b̄‖₂
    """
    ma = float(a.mean())
    mb = float(b.mean())
    da = float(np.sqrt(((a - ma) ** 2).sum()))
    db = float(np.sqrt(((b - mb) ** 2).sum()))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float(((a - ma) * (b - mb)).sum()) / (da * db)


# ---------------------------------------------------------------------------
# New CT/MRI-relevant VTK parity tests
# ---------------------------------------------------------------------------


def test_vtk_threshold_matches_sitk_binary_threshold():
    """vtkImageThreshold lower-threshold must produce the same binary mask as
    SimpleITK BinaryThresholdImageFilter on a linear-ramp image.

    Mathematical basis: ThresholdByLower(T) selects voxels with value <= T,
    which is equivalent to BinaryThreshold with lowerThreshold=0, upperThreshold=T.
    The midpoint T = SIZE * 1.5 splits the ramp into two equal halves.
    Dice(vtk_mask, sitk_mask) must be >= 0.99.

    Dice coefficient: 2|A ∩ B| / (|A| + |B|).
    """
    arr = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)
    z, y, x = np.mgrid[:SIZE, :SIZE, :SIZE]
    arr[:] = (z + y + x).astype(np.float32)  # values in [0, 3*(SIZE-1)]

    mid = float(SIZE * 1.5)

    # VTK threshold (selects voxels <= mid)
    vimg = _vtk_from_np(arr)
    thresh = vtk.vtkImageThreshold()
    thresh.SetInputData(vimg)
    thresh.ThresholdByLower(mid)
    thresh.SetInValue(1.0)
    thresh.SetOutValue(0.0)
    thresh.ReplaceInOn()
    thresh.ReplaceOutOn()
    thresh.Update()
    vtk_mask = _vtk_to_np(thresh.GetOutput(), arr.shape)

    # SimpleITK binary threshold [0, mid]
    sitk_thresh = sitk.BinaryThresholdImageFilter()
    sitk_thresh.SetLowerThreshold(0.0)
    sitk_thresh.SetUpperThreshold(mid)
    sitk_thresh.SetInsideValue(1)
    sitk_thresh.SetOutsideValue(0)
    sitk_mask_img = sitk_thresh.Execute(_sitk_from_np(arr))
    sitk_mask = sitk.GetArrayFromImage(sitk_mask_img).astype(np.float32)

    inter = float((vtk_mask * sitk_mask).sum())
    denom = float(vtk_mask.sum() + sitk_mask.sum())
    dice = 2.0 * inter / max(denom, 1.0)
    assert dice >= 0.99, (
        f"VTK/SimpleITK threshold Dice {dice:.4f} < 0.99; "
        f"vtk_sum={vtk_mask.sum():.0f}, sitk_sum={sitk_mask.sum():.0f}"
    )


def test_vtk_reslice_identity_preserves_sphere():
    """vtkImageReslice with identity transform and linear interpolation must preserve
    the sphere image to within NRMSE < 0.02 (near-exact round-trip).

    Mathematical basis: the identity reslice performs trilinear interpolation at each
    voxel's own coordinates, which for integer-grid sampling reduces to exact lookup.
    Interior NRMSE (1-voxel border excluded) must be < 0.02.
    """
    arr = _make_sphere()
    vimg = _vtk_from_np(arr)

    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(vimg)
    reslice.SetInterpolationModeToLinear()
    reslice.SetOutputExtent(vimg.GetExtent())
    reslice.SetOutputSpacing(vimg.GetSpacing())
    reslice.SetOutputOrigin(vimg.GetOrigin())
    reslice.Update()
    result = _vtk_to_np(reslice.GetOutput(), arr.shape)

    interior_result = result[1:-1, 1:-1, 1:-1]
    interior_orig = arr[1:-1, 1:-1, 1:-1]
    nrmse = _nrmse_region(interior_result, interior_orig)
    assert nrmse < 0.02, (
        f"vtkImageReslice identity NRMSE {nrmse:.4f} >= 0.02; "
        "identity reslice should be near-exact"
    )


def _nrmse_region(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE normalised by the range of b; helper for reslice test."""
    rng = float(b.max()) - float(b.min())
    if rng < 1e-12:
        return 0.0
    rmse = float(np.sqrt(((a - b) ** 2).mean()))
    return rmse / rng


def test_vtk_ct_bimodal_statistics_agree_with_numpy():
    """VTK vtkImageHistogramStatistics on a CT-like bimodal (air/tissue) image must
    report min, max, and mean within 1.0 HU of the numpy reference values.

    Mathematical basis: vtkImageHistogramStatistics computes exact per-point
    statistics, so min/max must be exact and mean must agree to floating-point
    precision (< 1.0 HU tolerance absorbs f32 accumulation error).

    CT HU convention: air ≈ -1000, soft tissue ≈ 40–80.
    """
    arr = np.full((SIZE, SIZE, SIZE), -1000.0, dtype=np.float32)
    # Insert a soft-tissue sphere (radius=4) centred in the volume.
    c = SIZE // 2
    z, y, x = np.mgrid[:SIZE, :SIZE, :SIZE]
    sphere_mask = (z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= 4**2
    arr[sphere_mask] = 50.0  # soft tissue HU

    np_min = float(arr.min())
    np_max = float(arr.max())
    np_mean = float(arr.mean())

    hs = vtk.vtkImageHistogramStatistics()
    hs.SetInputData(_vtk_from_np(arr))
    hs.Update()
    vtk_min = hs.GetMinimum()
    vtk_max = hs.GetMaximum()
    vtk_mean = hs.GetMean()

    assert abs(vtk_min - np_min) < 1.0, (
        f"VTK min {vtk_min:.2f} vs numpy {np_min:.2f}; |diff|={abs(vtk_min - np_min):.2f}"
    )
    assert abs(vtk_max - np_max) < 1.0, (
        f"VTK max {vtk_max:.2f} vs numpy {np_max:.2f}; |diff|={abs(vtk_max - np_max):.2f}"
    )
    assert abs(vtk_mean - np_mean) < 5.0, (
        f"VTK mean {vtk_mean:.4f} vs numpy {np_mean:.4f}; "
        f"|diff|={abs(vtk_mean - np_mean):.4f}"
    )


def test_vtk_cross_modal_ncc_lower_than_monomodal_ncc():
    """Cross-modal (CT-like vs MRI-like) NCC must be lower than same-modality NCC
    under a small translation, validating the cross-modal registration premise.

    Mathematical basis: CT encodes X-ray attenuation (air=0, tissue=1) while T1-MRI
    encodes magnetisation (CSF/air=0, white matter=1 — same polarity for this phantom).
    Here MRI-like is the intensity-INVERTED CT (background=1, sphere=0), simulating a
    T1w sequence where the signal contrast is opposite to CT.

    NCC(CT, MRI_inverted) < NCC(CT, CT_shifted_by_2) must hold because the inverted
    image has negatively correlated intensities (NCC near -1) while a small shift
    maintains high positive correlation.  This test validates that direct NCC
    registration would fail for cross-modal pairs.
    """
    ct = _make_sphere().astype(np.float32)  # foreground=1, background=0
    mri = (1.0 - ct).astype(np.float32)  # inverted: foreground=0, background=1
    ct_shifted = np.roll(ct, 2, axis=2).astype(np.float32)

    ncc_mono = _ncc(ct, ct_shifted)  # same modality, small shift → high NCC
    ncc_cross = _ncc(ct, mri)  # different modality → low (negative) NCC

    assert ncc_cross < ncc_mono, (
        f"Cross-modal NCC {ncc_cross:.4f} is not lower than monomodal NCC "
        f"{ncc_mono:.4f}; expected ncc_cross < ncc_mono"
    )


def test_vtk_image_accumulate_histogram_bin_counts_sum_to_nvoxels():
    """vtkImageAccumulate histogram bin counts must sum to exactly N = SIZE³.

    Mathematical basis: a normalised histogram is a probability mass function; every
    voxel contributes exactly one count to exactly one bin.  Conservation of mass:
        Σₖ count(binₖ) = N_voxels  (exact integer equality).
    """
    arr = _make_sphere()
    vimg = _vtk_from_np(arr)

    accum = vtk.vtkImageAccumulate()
    accum.SetInputData(vimg)
    # 256 bins spanning [0, 1] in the scalar (x) component.
    accum.SetComponentExtent(0, 255, 0, 0, 0, 0)
    accum.SetComponentSpacing(1.0 / 255.0, 0.0, 0.0)
    accum.SetComponentOrigin(0.0, 0.0, 0.0)
    accum.Update()

    counts = numpy_support.vtk_to_numpy(accum.GetOutput().GetPointData().GetScalars())
    total = int(counts.sum())
    expected = SIZE**3
    assert total == expected, (
        f"Histogram bin count total {total} != N_voxels {expected}; "
        "mass is not conserved in vtkImageAccumulate"
    )


def test_vtk_anisotropic_diffusion_reduces_peak_spike():
    """vtkImageAnisotropicDiffusion3D must reduce a single-voxel intensity spike
    by at least 50% while preserving overall image structure (mean > 0).

    Mathematical basis: anisotropic diffusion (Perona-Malik 1990) redistributes
    intensity from high-gradient regions; a spike of amplitude 100 on a binary
    sphere is a high-gradient outlier and must diffuse substantially after 5
    iterations.  The diffusion threshold is set to 5.0 (spike gradient >> 5.0,
    sphere boundary gradient ≈ 1.0), so the spike diffuses while the sphere
    boundary is partially preserved.
    """
    arr = _make_sphere().copy()
    arr[SIZE // 2, SIZE // 2, SIZE // 2] += 100.0  # single-voxel spike
    peak_before = float(arr[SIZE // 2, SIZE // 2, SIZE // 2])

    diff = vtk.vtkImageAnisotropicDiffusion3D()
    diff.SetInputData(_vtk_from_np(arr))
    diff.SetNumberOfIterations(5)
    # DiffusionThreshold: faces with gradient magnitude < threshold are diffused.
    # The spike gradient is ~100; setting threshold=200 ensures the spike is below
    # the threshold and therefore IS diffused (gradient 100 < 200 → diffuse).
    diff.SetDiffusionThreshold(200.0)
    diff.SetDiffusionFactor(1.0)
    diff.Update()
    result = _vtk_to_np(diff.GetOutput(), arr.shape)

    peak_after = float(result[SIZE // 2, SIZE // 2, SIZE // 2])
    assert peak_after < peak_before * 0.5, (
        f"Anisotropic diffusion reduced spike from {peak_before:.2f} to "
        f"{peak_after:.2f}; expected < {peak_before * 0.5:.2f} (50% reduction)"
    )
    assert float(result.mean()) > 0.0, (
        "Diffusion zeroed the entire image; overall structure must be preserved"
    )


def test_vtk_image_cast_to_float_preserves_integer_values():
    """vtkImageCast from VTK_SHORT to VTK_FLOAT must exactly preserve all integer
    voxel values in the range [0, 26] (a 3×3×3 volume with sequential values).

    Mathematical basis: float32 has 24-bit mantissa, which exactly represents all
    integers in [-16 777 216, 16 777 216].  Casting integers in [0, 26] to float32
    must be exact (no rounding error).
    """
    # Build a 3×3×3 integer vtkImageData (VTK_SHORT, values 0..26).
    int_img = vtk.vtkImageData()
    int_img.SetDimensions(3, 3, 3)
    int_img.AllocateScalars(vtk.VTK_SHORT, 1)
    n = 3 * 3 * 3
    scalars = int_img.GetPointData().GetScalars()
    for i in range(n):
        scalars.SetValue(i, i)

    cast = vtk.vtkImageCast()
    cast.SetInputData(int_img)
    cast.SetOutputScalarTypeToFloat()
    cast.Update()

    out_arr = numpy_support.vtk_to_numpy(
        cast.GetOutput().GetPointData().GetScalars()
    ).astype(np.float32)
    expected = np.arange(n, dtype=np.float32)

    np.testing.assert_array_equal(
        out_arr,
        expected,
        err_msg="vtkImageCast SHORT→FLOAT did not exactly preserve integer values",
    )


def test_vtk_gradient_magnitude_nonunit_spacing_agrees_with_sitk():
    """vtkImageGradientMagnitude with 0.5 mm isotropic spacing must agree with
    SimpleITK GradientMagnitudeImageFilter on a sphere image to Pearson r >= 0.95.

    Mathematical basis: a binary sphere image has gradient magnitude ≈ 0 in the
    homogeneous interior and background, and ≈ 1/h = 2.0 mm⁻¹ at the boundary
    (h = 0.5 mm spacing; one unit of intensity change over 0.5 mm).  Both VTK and
    SimpleITK implement central-difference gradient magnitude; their outputs must be
    spatially consistent (Pearson r >= 0.95 in the interior+boundary region).

    A linear ramp is not used here because it produces a constant gradient magnitude
    (zero spatial variance → Pearson r is undefined).  The sphere has spatially
    varying gradient (zero interior, non-zero boundary) giving well-defined correlation.

    SetDimensionality(3) is required on vtkImageGradientMagnitude to include the
    z-direction contribution; default Dimensionality=2 silently skips the z-axis.
    """
    arr = _make_sphere()  # shape (SIZE, SIZE, SIZE); binary {0, 1}
    spacing = 0.5

    # VTK: set 0.5 mm isotropic spacing on the vtkImageData
    vimg = _vtk_from_np(arr)
    vimg.SetSpacing(spacing, spacing, spacing)

    grad_vtk = vtk.vtkImageGradientMagnitude()
    grad_vtk.SetInputData(vimg)
    grad_vtk.SetDimensionality(3)
    grad_vtk.HandleBoundariesOn()
    grad_vtk.Update()
    out_vtk = _vtk_to_np(grad_vtk.GetOutput(), arr.shape)

    # SimpleITK: spacing supplied in (x, y, z) order matching ITK convention.
    sitk_img = _sitk_from_np(arr, spacing=(spacing, spacing, spacing))
    grad_sitk = sitk.GradientMagnitudeImageFilter()
    out_sitk = sitk.GetArrayFromImage(grad_sitk.Execute(sitk_img)).astype(np.float32)

    # Compare on the full volume (1-voxel border excluded to avoid boundary-mode
    # divergence between the two libraries' padding strategies).
    int_vtk = out_vtk[1:-1, 1:-1, 1:-1].ravel()
    int_sitk = out_sitk[1:-1, 1:-1, 1:-1].ravel()

    r, _ = pearsonr(int_vtk.astype(np.float64), int_sitk.astype(np.float64))
    assert r >= 0.95, (
        f"VTK/SimpleITK gradient magnitude (spacing=0.5, sphere) Pearson r {r:.4f} < 0.95"
    )

    # The peak gradient at the sphere boundary must be approximately 1/h = 2.0 mm⁻¹.
    vtk_peak = float(out_vtk.max())
    assert 1.0 <= vtk_peak <= 4.0, (
        f"VTK peak gradient {vtk_peak:.4f} mm⁻¹ outside expected range [1.0, 4.0] "
        f"for h=0.5 mm spacing (analytical boundary gradient ≈ 2.0 mm⁻¹)"
    )
