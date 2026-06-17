"""ritk-vs-SimpleITK filter parity on the real SimpleITK test corpus (cthead1).

Each test applies the same operation with ritk and with SimpleITK on identical
real-image bytes (the canonical ``cthead1`` CT head SimpleITK ships for its own
regression suite) and asserts agreement to a tolerance derived from the operation.

Crucially, each comparison pins the *exact* SimpleITK calling convention that the
operation expects — these were established empirically and are easy to get wrong:

* ``sitk.Sigmoid`` takes ``outputMaximum`` *before* ``outputMinimum`` in positional
  order; passed here by keyword to avoid the trap.
* ``sitk.Threshold(lower, upper, outside)`` keeps ``[lower, upper]`` and sets the
  rest to ``outside`` — the complement of ritk's ``threshold_below`` (set values
  *below* the threshold to the outside value), so the bounds are mapped accordingly.
* ``GrayscaleDilate``/``WhiteTopHat`` default to a *ball* structuring element;
  ritk uses a flat *box* (cube), so the SimpleITK kernel is forced to ``sitkBox``.
* ``sitk.Bilateral`` ``domainSigma`` is in physical units (mm); ritk's
  ``spatial_sigma`` is in voxels, so the SimpleITK sigma is scaled by the spacing.

Skips cleanly when SimpleITK or the object store is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

import ritk  # noqa: E402

from _sitk_data import fetch  # noqa: E402

IMAGE = "cthead1-Float.mha"  # 256×256 float CT head, spacing 0.352778 mm (isotropic)
SPACING = 0.3527777777778


@pytest.fixture(scope="module")
def images():
    path = fetch(IMAGE)
    ri = ritk.io.read_image(path)
    si = sitk.Cast(sitk.ReadImage(path), sitk.sitkFloat32)
    return ri, si


def _sa(simg) -> np.ndarray:
    a = sitk.GetArrayFromImage(simg).astype(np.float64)
    return a[None] if a.ndim == 2 else a


def _interior_absmax(ra: np.ndarray, sa: np.ndarray, m: int = 3) -> float:
    assert ra.shape == sa.shape, f"shape {ra.shape} != {sa.shape}"
    r = ra[:, m:-m, m:-m]
    s = sa[:, m:-m, m:-m]
    return float(np.abs(r.astype(np.float64) - s.astype(np.float64)).max())


def _rng(sa: np.ndarray) -> float:
    return max(float(sa.max() - sa.min()), 1e-6)


# ── Point / intensity transforms: exact up to f32 rounding ────────────────────


def test_rescale_intensity_matches_sitk(images):
    ri, si = images
    ra = ritk.filter.rescale_intensity(ri, 0.0, 255.0).to_numpy()
    sa = _sa(sitk.RescaleIntensity(si, 0.0, 255.0))
    # Pure affine remap of each sample: agreement is f32 round-off of a value ≤ 255.
    assert _interior_absmax(ra, sa) / _rng(sa) < 1e-4


def test_intensity_windowing_matches_sitk(images):
    ri, si = images
    ra = ritk.filter.intensity_windowing(ri, 50.0, 200.0, 0.0, 255.0).to_numpy()
    sa = _sa(sitk.IntensityWindowing(si, 50.0, 200.0, 0.0, 255.0))
    assert _interior_absmax(ra, sa) / _rng(sa) < 1e-4


def test_sigmoid_matches_sitk(images):
    ri, si = images
    ra = ritk.filter.sigmoid_filter(ri, 20.0, 128.0, 0.0, 255.0).to_numpy()
    # NOTE: sitk positional order is (alpha, beta, outputMaximum, outputMinimum).
    sa = _sa(sitk.Sigmoid(si, alpha=20.0, beta=128.0, outputMinimum=0.0, outputMaximum=255.0))
    assert _interior_absmax(ra, sa) / _rng(sa) < 1e-4


def test_binary_threshold_matches_sitk(images):
    ri, si = images
    ra = ritk.filter.binary_threshold(ri, 50.0, 150.0, 1.0, 0.0).to_numpy()
    sa = _sa(sitk.BinaryThreshold(si, 50.0, 150.0, 1, 0))
    assert _interior_absmax(ra, sa) == 0.0


def test_threshold_below_matches_sitk(images):
    ri, si = images
    # ritk: set values strictly below 100 to 0, keep the rest.
    # sitk.Threshold(lower, upper, outside) keeps [lower, upper]; keep [100, +inf).
    ra = ritk.filter.threshold_below(ri, 100.0, 0.0).to_numpy()
    sa = _sa(sitk.Threshold(si, 100.0, 1e9, 0.0))
    assert _interior_absmax(ra, sa) == 0.0


def test_threshold_above_matches_sitk(images):
    ri, si = images
    ra = ritk.filter.threshold_above(ri, 100.0, 0.0).to_numpy()
    sa = _sa(sitk.Threshold(si, -1e9, 100.0, 0.0))
    assert _interior_absmax(ra, sa) == 0.0


# ── Derivative / spatial filters ──────────────────────────────────────────────


def _full_absmax(ra: np.ndarray, sa: np.ndarray) -> float:
    """Max abs difference over the *entire* image, including the 1-voxel border."""
    assert ra.shape == sa.shape, f"shape {ra.shape} != {sa.shape}"
    return float(np.abs(ra.astype(np.float64) - sa.astype(np.float64)).max())


def test_laplacian_matches_sitk(images):
    """Full-image (boundary-inclusive) parity: ritk's Laplacian uses the same
    central [1,-2,1] stencil with a ZeroFluxNeumann clamp at the border that ITK
    does, so the result agrees to float rounding everywhere — not just interior."""
    ri, si = images
    ra = ritk.filter.laplacian(ri).to_numpy()
    sa = _sa(sitk.Laplacian(si))
    assert _full_absmax(ra, sa) / _rng(sa) < 1e-5


def test_gradient_magnitude_matches_sitk(images):
    """Full-image parity: central differences scaled by physical spacing, with a
    ZeroFluxNeumann boundary matching ITK's GradientMagnitudeImageFilter. The
    border previously used a one-sided difference (≈2× the ITK value there)."""
    ri, si = images
    ra = ritk.filter.gradient_magnitude(ri).to_numpy()
    sa = _sa(sitk.GradientMagnitude(si))
    assert _full_absmax(ra, sa) / _rng(sa) < 1e-5


def test_gaussian_filter_preserves_mean(images):
    """Regression: GaussianFilter convolved the size-1 z-axis with a wide kernel
    under zero padding, scaling a z=1 image by the kernel centre weight (≈0.2)."""
    ri, _ = images
    rin = ri.to_numpy().astype(np.float64)
    rg = ritk.filter.gaussian_filter(ri, 2.0).to_numpy().astype(np.float64)
    # Gaussian smoothing conserves the image mean (up to boundary zero-padding).
    assert abs(rg.mean() - rin.mean()) / rin.mean() < 0.02


def test_laplacian_of_gaussian_matches_sitk(images):
    """LoG is float-exact to sitk LaplacianRecursiveGaussian: ritk now computes
    Σ_d ∂²/∂x_d²(G_σ*I) via the per-axis Deriche recursion (second-order along d,
    zero-order along the others), matching ITK's recursive-Gaussian second
    derivative rather than a discrete Gaussian + finite-difference Laplacian.
    Interior relative residual is at the f32 round-off floor (~5e-8)."""
    ri, si = images
    ra = ritk.filter.laplacian_of_gaussian(ri, sigma=2.0).to_numpy()
    sa = _sa(sitk.LaplacianRecursiveGaussian(si, sigma=2.0))
    assert _interior_absmax(ra, sa) / _rng(sa) < 1e-6


# ── Morphology (flat box structuring element) ─────────────────────────────────


def test_grayscale_dilation_matches_sitk(images):
    ri, si = images
    ra = ritk.filter.grayscale_dilation(ri, 1).to_numpy()
    sa = _sa(sitk.GrayscaleDilate(si, [1, 1]))  # default box kernel for vector radius
    assert _interior_absmax(ra, sa) == 0.0


def test_grayscale_erosion_matches_sitk(images):
    ri, si = images
    ra = ritk.filter.grayscale_erosion(ri, 1).to_numpy()
    sa = _sa(sitk.GrayscaleErode(si, [1, 1]))
    assert _interior_absmax(ra, sa) == 0.0


def test_white_top_hat_matches_sitk_box_kernel(images):
    ri, si = images
    ra = ritk.filter.white_top_hat(ri, 3).to_numpy()
    f = sitk.WhiteTopHatImageFilter()
    f.SetKernelType(sitk.sitkBox)
    f.SetKernelRadius(3)
    sa = _sa(f.Execute(si))
    # White top hat = I − opening(I); box SE matched. Residual is boundary handling.
    assert _interior_absmax(ra, sa) / _rng(sa) < 0.02


def test_black_top_hat_matches_sitk_box_kernel(images):
    ri, si = images
    ra = ritk.filter.black_top_hat(ri, 3).to_numpy()
    f = sitk.BlackTopHatImageFilter()
    f.SetKernelType(sitk.sitkBox)
    f.SetKernelRadius(3)
    sa = _sa(f.Execute(si))
    assert _interior_absmax(ra, sa) / _rng(sa) < 0.02


def _recon_inputs(images):
    """Marker = (image − 30) clamped ≥ 0; mask = image. Shared geometry."""
    ri, si = images
    img = _sa(si)
    marker_np = np.clip(img - 30.0, 0.0, None).astype(np.float32)
    spacing = list(ri.spacing)
    rmarker = ritk.Image(np.ascontiguousarray(marker_np), spacing=spacing)
    smarker = sitk.GetImageFromArray(marker_np[0] if marker_np.shape[0] == 1 else marker_np)
    smarker.CopyInformation(si)
    return rmarker, ri, smarker, si


def test_morphological_reconstruction_face_matches_sitk(images):
    """Default face connectivity matches ITK's FullyConnectedOff (its default)."""
    rmarker, rmask, smarker, smask = _recon_inputs(images)
    ra = ritk.filter.morphological_reconstruction(
        rmarker, rmask, mode="dilation", fully_connected=False
    ).to_numpy()
    sa = _sa(sitk.ReconstructionByDilation(smarker, smask, fullyConnected=False))
    assert _full_absmax(ra, sa) / _rng(sa) < 1e-5


def test_morphological_reconstruction_full_matches_sitk(images):
    """fully_connected=True matches ITK's FullyConnectedOn (26/8-connectivity)."""
    rmarker, rmask, smarker, smask = _recon_inputs(images)
    ra = ritk.filter.morphological_reconstruction(
        rmarker, rmask, mode="dilation", fully_connected=True
    ).to_numpy()
    sa = _sa(sitk.ReconstructionByDilation(smarker, smask, fullyConnected=True))
    assert _full_absmax(ra, sa) / _rng(sa) < 1e-5


def test_morphological_reconstruction_default_is_face(images):
    """The default (no fully_connected kwarg) equals face connectivity, so it
    diverges from ITK's full-connectivity result — guarding the default choice."""
    rmarker, rmask, smarker, smask = _recon_inputs(images)
    ra_default = ritk.filter.morphological_reconstruction(rmarker, rmask).to_numpy()
    ra_face = ritk.filter.morphological_reconstruction(
        rmarker, rmask, fully_connected=False
    ).to_numpy()
    assert _full_absmax(ra_default, ra_face) == 0.0


def test_bilateral_matches_sitk(images):
    ri, si = images
    # ritk spatial_sigma is in voxels; sitk domainSigma is physical (mm).
    ra = ritk.filter.bilateral_filter(ri, 2.0, 50.0).to_numpy()
    sa = _sa(sitk.Bilateral(si, 2.0 * SPACING, 50.0))
    assert _interior_absmax(ra, sa) / _rng(sa) < 0.01


def test_gradient_anisotropic_diffusion_matches_sitk(images):
    """ITK-exact gradient anisotropic diffusion (exponential kind) vs SimpleITK.

    Regression for the simplified Perona-Malik scheme that previously diverged
    ~2.6%/iteration (face-gradient conductance + average-gradient-magnitude K
    rescaling now matched).
    """
    ri, si = images
    ra = ritk.filter.anisotropic_diffusion(
        ri, iterations=5, conductance=3.0, time_step=0.0625
    ).to_numpy()
    sa = _sa(
        sitk.GradientAnisotropicDiffusion(
            si, timeStep=0.0625, conductanceParameter=3.0, numberOfIterations=5
        )
    )
    # ritk accumulates in f64, ITK in f32; residual is f32 round-off over iterations.
    assert _interior_absmax(ra, sa) / _rng(sa) < 2e-3


def test_curvature_anisotropic_diffusion_matches_sitk(images):
    """ITK-exact curvature MCDE vs SimpleITK (was ~44% off)."""
    ri, si = images
    # Time step below ITK's stability threshold (~0.044 for 0.35 mm spacing).
    ra = ritk.filter.curvature_anisotropic_diffusion(
        ri, iterations=5, time_step=0.04, conductance=3.0
    ).to_numpy()
    sa = _sa(
        sitk.CurvatureAnisotropicDiffusion(
            si, timeStep=0.04, conductanceParameter=3.0, numberOfIterations=5
        )
    )
    assert _interior_absmax(ra, sa) / _rng(sa) < 2e-3


# ── Distance transform (unsigned Euclidean, Danielsson semantics) ─────────────


def test_distance_transform_matches_sitk_danielsson():
    """ritk's Meijster exact Euclidean DT matches ITK's Danielsson map on a binary image."""
    path = fetch("2th_cthead1.png")
    ri = ritk.io.read_image(path)
    si = sitk.ReadImage(path)
    ra = ritk.filter.distance_transform(ri, foreground_threshold=0.5).to_numpy()
    # Danielsson: distance to nearest non-zero voxel, in physical units, no squaring.
    binary = sitk.Cast(si > 0, sitk.sitkUInt8)
    sa = _sa(sitk.DanielssonDistanceMap(binary, inputIsBinary=True, squaredDistance=False, useImageSpacing=True))
    # Two exact-Euclidean DTs differ only at the half-voxel boundary convention.
    assert _interior_absmax(ra, sa) / _rng(sa) < 0.02


# ── Geometric resampling (z = 1 anisotropic grid) ─────────────────────────────


def test_resample_downsample_matches_sitk(images):
    """2× linear downsample of the z = 1 cthead1 slice matches sitk's resampler.

    Regression guard for the resample axis-order defect: ``indices_to_physical``
    paired innermost-first index columns with axis-major spacing by position, so a
    z = 1 (2-D promoted) anisotropic grid collapsed every output row to a constant.
    """
    ri, si = images
    ns = SPACING * 2.0  # 0.7056 mm → 256 → 128 in plane; z stays 1.
    rr = ritk.filter.resample_image(ri, 1.0, ns, ns, "linear")
    ra = rr.to_numpy().astype(np.float64)
    rf = sitk.ResampleImageFilter()
    rf.SetOutputSpacing((ns, ns))
    rf.SetSize([ra.shape[2], ra.shape[1]])
    rf.SetOutputOrigin(si.GetOrigin())
    rf.SetOutputDirection(si.GetDirection())
    rf.SetInterpolator(sitk.sitkLinear)
    rf.SetDefaultPixelValue(0.0)
    sa = _sa(rf.Execute(si))
    assert ra.shape[0] == 1, "z = 1 slice must stay a single output plane"
    # Identical output grid + linear interpolation: agreement is f32 sampling
    # round-off, far below any structural axis error (which gave rel ≈ 1).
    assert _interior_absmax(ra, sa) / _rng(sa) < 1e-3


def test_resample_identity_reproduces_input(images):
    """Resampling to the same spacing reproduces the input (no off-by-axis shift)."""
    ri, _ = images
    rid = ritk.filter.resample_image(ri, 1.0, SPACING, SPACING, "linear").to_numpy()
    ia = ri.to_numpy()
    assert rid.shape == ia.shape
    # Each output voxel centre maps to an input voxel centre; residual is f32
    # round-off in the world↔index round trip.
    assert _interior_absmax(rid.astype(np.float64), ia.astype(np.float64)) / _rng(ia.astype(np.float64)) < 1e-3


def test_resample_bspline_downsample_matches_sitk(images):
    """2× cubic B-spline downsample of the z=1 cthead1 slice matches sitk.

    Guards the B-spline coefficient prefiltering: without it the interpolator
    smooths instead of interpolates (and the z=1 grid collapsed to zero).
    """
    ri, si = images
    ns = SPACING * 2.0
    rr = ritk.filter.resample_image(ri, 1.0, ns, ns, "bspline")
    ra = rr.to_numpy().astype(np.float64)
    rf = sitk.ResampleImageFilter()
    rf.SetOutputSpacing((ns, ns))
    rf.SetSize([ra.shape[2], ra.shape[1]])
    rf.SetOutputOrigin(si.GetOrigin())
    rf.SetOutputDirection(si.GetDirection())
    rf.SetInterpolator(sitk.sitkBSpline)
    rf.SetDefaultPixelValue(0.0)
    sa = _sa(rf.Execute(si))
    assert ra.shape[0] == 1
    # Both prefilter to coefficients and use mirror boundary; residual is f32 +
    # the prefilter horizon (ritk exact mirror vs ITK's 1e-10-truncated).
    assert _interior_absmax(ra, sa) / _rng(sa) < 5e-3


def test_resample_bspline_identity_reproduces_input(images):
    """Identity B-spline resample reproduces the input — the interpolation
    property that prefiltering provides (the old smoothing path lost ~73%)."""
    ri, _ = images
    rid = ritk.filter.resample_image(ri, 1.0, SPACING, SPACING, "bspline").to_numpy()
    ia = ri.to_numpy()
    assert rid.shape == ia.shape
    assert _interior_absmax(rid.astype(np.float64), ia.astype(np.float64)) / _rng(ia.astype(np.float64)) < 5e-3


# ── Automatic threshold-selection algorithms ──────────────────────────────────
#
# Each computes a scalar threshold from a 256-bin intensity histogram. ritk
# defaults to 256 bins, so SimpleITK's calculators are forced to 256 bins too
# (its own defaults are 128). The tolerance is the histogram bin width: range/255
# ≈ 1.0 intensity unit for cthead1; two correct implementations can disagree by up
# to one bin from the bin-centre mapping and argmax ties.


def _sitk_threshold(filt, simg, bins=256):
    filt.SetNumberOfHistogramBins(bins)
    filt.Execute(simg)
    return filt.GetThreshold()


def test_otsu_threshold_matches_sitk(images):
    ri, si = images
    rthr, _ = ritk.segmentation.otsu_threshold(ri)
    sthr = _sitk_threshold(sitk.OtsuThresholdImageFilter(), si)
    assert abs(rthr - sthr) < 2.0, f"otsu ritk {rthr} vs sitk {sthr}"


def test_yen_threshold_matches_sitk(images):
    """Regression for the degenerate Yen criterion (−log(P1sq+P2sq) is constant)."""
    ri, si = images
    rthr, _ = ritk.segmentation.yen_threshold(ri)
    sthr = _sitk_threshold(sitk.YenThresholdImageFilter(), si)
    assert abs(rthr - sthr) < 2.0, f"yen ritk {rthr} vs sitk {sthr}"


def test_li_threshold_matches_sitk(images):
    """Regression for Li computed via ISODATA (arithmetic mean) instead of the
    minimum-cross-entropy logarithmic mean."""
    ri, si = images
    rthr, _ = ritk.segmentation.li_threshold(ri)
    sthr = _sitk_threshold(sitk.LiThresholdImageFilter(), si)
    assert abs(rthr - sthr) < 2.0, f"li ritk {rthr} vs sitk {sthr}"


def test_triangle_threshold_matches_sitk(images):
    """Regression for the triangle method: ritk used the first/last non-empty
    bins as the peak→tail line endpoints and a peak≤N/2 side heuristic, giving
    3.0 here against ITK's 4.48 (and 13.7 % off on skewed 3-D histograms). It now
    follows itk::TriangleThresholdCalculator exactly — 1st/99th-percentile
    endpoints, longer-side selection, +1 bin shift, bin-centre output — matching
    to well under one histogram bin."""
    ri, si = images
    rthr, _ = ritk.segmentation.triangle_threshold(ri)
    sthr = _sitk_threshold(sitk.TriangleThresholdImageFilter(), si)
    assert abs(rthr - sthr) < 0.5, f"triangle ritk {rthr} vs sitk {sthr}"


# ── Binary morphology on a z=1 (2-D promoted) Otsu mask ───────────────────────
#
# These guard the degenerate z-axis handling: a 3×3×3 box structuring element on
# a z=1 slice reaches the out-of-bounds z±1 neighbours, which previously eroded
# every voxel (collapsing the mask to zero) and made hole-filling treat every
# voxel as a z-face border. SimpleITK is forced to a box SE to match ritk.


@pytest.fixture(scope="module")
def masks(images):
    ri, si = images
    _, rmask = ritk.segmentation.otsu_threshold(ri)
    smask = sitk.Cast(sitk.OtsuThreshold(si, 0, 1, 256), sitk.sitkUInt8)
    return rmask, smask


def _bin_mismatch(ra: np.ndarray, sa: np.ndarray) -> int:
    return int(((ra > 0.5) != (sa > 0.5)).sum())


def test_binary_erosion_matches_sitk(masks):
    """Regression: a 3×3×3 SE on a z=1 mask eroded every voxel to zero."""
    rmask, smask = masks
    ra = ritk.segmentation.binary_erosion(rmask, 1).to_numpy()
    sa = _sa(sitk.BinaryErode(smask, [1, 1], sitk.sitkBox))
    assert _bin_mismatch(ra, sa) == 0


def test_binary_dilation_matches_sitk(masks):
    rmask, smask = masks
    ra = ritk.segmentation.binary_dilation(rmask, 1).to_numpy()
    sa = _sa(sitk.BinaryDilate(smask, [1, 1], sitk.sitkBox))
    assert _bin_mismatch(ra, sa) == 0


def test_binary_opening_matches_sitk(masks):
    rmask, smask = masks
    ra = ritk.segmentation.binary_opening(rmask, 1).to_numpy()
    sa = _sa(sitk.BinaryMorphologicalOpening(smask, [1, 1], sitk.sitkBox))
    assert _bin_mismatch(ra, sa) == 0


def test_binary_closing_matches_sitk(masks):
    rmask, smask = masks
    ra = ritk.segmentation.binary_closing(rmask, 1).to_numpy()
    sa = _sa(sitk.BinaryMorphologicalClosing(smask, [1, 1], sitk.sitkBox))
    assert _bin_mismatch(ra, sa) == 0


def test_binary_fill_holes_matches_sitk(masks):
    """Regression: z=1 made every voxel a z-face border, so no holes were filled."""
    rmask, smask = masks
    ra = ritk.segmentation.binary_fill_holes(rmask).to_numpy()
    sa = _sa(sitk.BinaryFillhole(smask))
    assert _bin_mismatch(ra, sa) == 0


def test_dice_coefficient_matches_sitk(masks):
    """Dice of the mask vs its erosion; previously 0 because erosion was empty."""
    rmask, smask = masks
    rmask2 = ritk.segmentation.binary_erosion(rmask, 1)
    smask2 = sitk.BinaryErode(smask, [1, 1], sitk.sitkBox)
    rdice = ritk.statistics.dice_coefficient(rmask, rmask2)
    f = sitk.LabelOverlapMeasuresImageFilter()
    f.Execute(smask, smask2)
    assert abs(rdice - f.GetDiceCoefficient()) < 1e-4, f"dice {rdice} vs {f.GetDiceCoefficient()}"


def test_connected_components_match_sitk(masks):
    """ritk's labelling partitions the Otsu mask identically to ITK (face-conn)."""
    rmask, smask = masks
    rcc = ritk.segmentation.connected_components(rmask, connectivity=6)[0].to_numpy()
    scc = _sa(sitk.ConnectedComponent(smask, False))

    def sizes(lbl):
        lbl = lbl.astype(np.int64).ravel()
        c = np.bincount(lbl[lbl > 0])
        return sorted(c[c > 0].tolist(), reverse=True)

    assert sizes(rcc) == sizes(scc)


# ── Full-image resample parity across all interpolators ───────────────────────
#
# The corpus tests above compare only the interior (axis-order regression). These
# exercise the *whole* output, including the out-of-FOV halo, on a synthetic 3-D
# volume resampled to a mixed up/down spacing that pushes trailing samples past
# the input buffer. Guards the boundary fix: ITK's ResampleImageFilter emits the
# default pixel value where the mapped continuous index leaves [-0.5, N-0.5);
# ritk previously edge-clamped that halo instead (rel ≈ 0.9).

_SXYZ = (1.0, 1.2, 0.8)   # sitk x,y,z input spacing (anisotropic)
_OXYZ = (5.0, -3.0, 2.0)


def _synthetic_pair():
    nz, ny, nx = 8, 10, 12
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    rng = np.random.default_rng(0)
    vol = (np.sin(0.3 * xx) * np.cos(0.4 * yy) * (1 + 0.1 * zz)
           + 0.05 * rng.standard_normal((nz, ny, nx))).astype(np.float32)
    si = sitk.GetImageFromArray(np.ascontiguousarray(vol))
    si.SetSpacing(_SXYZ)
    si.SetOrigin(_OXYZ)
    ri = ritk.Image(np.ascontiguousarray(vol),
                    spacing=[_SXYZ[2], _SXYZ[1], _SXYZ[0]],
                    origin=[_OXYZ[2], _OXYZ[1], _OXYZ[0]])
    return ri, si, vol


def _sitk_resample(si, new_xyz, interp):
    in_sz = np.array(si.GetSize(), float)
    in_sp = np.array(si.GetSpacing(), float)
    out_sp = np.array(new_xyz, float)
    out_sz = np.maximum(1, np.round(in_sz * in_sp / out_sp)).astype(int)
    rf = sitk.ResampleImageFilter()
    rf.SetOutputSpacing([float(s) for s in out_sp])
    rf.SetSize([int(s) for s in out_sz])
    rf.SetOutputOrigin(si.GetOrigin())
    rf.SetOutputDirection(si.GetDirection())
    rf.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    rf.SetInterpolator(interp)
    rf.SetDefaultPixelValue(0.0)
    return _sa(rf.Execute(si))


# spacing ratios chosen non-commensurate so no continuous index lands on an exact
# half-integer tie (where f32 geometry vs ITK's f64 can round to opposite voxels).
@pytest.mark.parametrize("mode,sinterp,tol", [
    ("linear",  sitk.sitkLinear,              1e-4),
    ("bspline", sitk.sitkBSpline,             1e-4),
    ("lanczos", sitk.sitkLanczosWindowedSinc, 1e-4),
])
def test_resample_full_image_matches_sitk(mode, sinterp, tol):
    """Whole-output parity (incl. out-of-FOV halo) with sitk.Resample."""
    ri, si, _ = _synthetic_pair()
    new_xyz = (1.37, 0.71, 0.91)
    ro = ritk.filter.resample_image(ri, spacing_z=new_xyz[2], spacing_y=new_xyz[1],
                                    spacing_x=new_xyz[0], mode=mode)
    ra = ro.to_numpy().astype(np.float64)
    sa = _sitk_resample(si, new_xyz, sinterp)
    assert ra.shape == sa.shape
    assert np.abs(ra - sa).max() / _rng(sa) < tol


def test_resample_nearest_matches_sitk_no_ties():
    """Nearest-neighbour resampling matches sitk exactly when no continuous index
    is an exact half-integer (the tie case is f32-vs-f64 sensitive: ritk's f32
    geometry can round 0.5 to the opposite voxel of ITK's double precision)."""
    ri, si, _ = _synthetic_pair()
    new_xyz = (1.37, 0.71, 0.91)   # non-commensurate → no exact ties
    ro = ritk.filter.resample_image(ri, spacing_z=new_xyz[2], spacing_y=new_xyz[1],
                                    spacing_x=new_xyz[0], mode="nearest")
    ra = ro.to_numpy().astype(np.float64)
    sa = _sitk_resample(si, new_xyz, sitk.sitkNearestNeighbor)
    assert ra.shape == sa.shape
    assert np.abs(ra - sa).max() == 0.0
