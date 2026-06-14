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


def test_laplacian_matches_sitk(images):
    ri, si = images
    ra = ritk.filter.laplacian(ri).to_numpy()
    sa = _sa(sitk.Laplacian(si))
    assert _interior_absmax(ra, sa) / _rng(sa) < 1e-4


def test_gradient_magnitude_matches_sitk(images):
    ri, si = images
    ra = ritk.filter.gradient_magnitude(ri).to_numpy()
    sa = _sa(sitk.GradientMagnitude(si))
    # Both compute |∇I| with central differences scaled by physical spacing.
    assert _interior_absmax(ra, sa) / _rng(sa) < 1e-4


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


def test_bilateral_matches_sitk(images):
    ri, si = images
    # ritk spatial_sigma is in voxels; sitk domainSigma is physical (mm).
    ra = ritk.filter.bilateral_filter(ri, 2.0, 50.0).to_numpy()
    sa = _sa(sitk.Bilateral(si, 2.0 * SPACING, 50.0))
    assert _interior_absmax(ra, sa) / _rng(sa) < 0.01


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
