"""Systematic ritk-vs-SimpleITK filter parity corpus on the canonical SimpleITK
test image (``cthead1``).

This is a single data-driven table mirroring the SimpleITK cmake filter-test
pattern: each row applies the *same* operation with ritk and with SimpleITK on
identical real-image bytes (the ``cthead1`` CT head SimpleITK ships for its own
regression suite) and asserts interior agreement to a tolerance derived from the
operation. It consolidates the per-op differential checks established across the
parity campaign into a permanent, exhaustive regression battery.

Each row pins the exact SimpleITK calling convention (these are easy to get
wrong — see the module-level notes in ``test_sitk_filter_corpus_parity``):

* float-exact (rel ``< 1e-6``): the recursive/discrete Gaussian family, the
  intensity transforms, the discrete Laplacian / gradient magnitude.
* bit-exact (rel ``== 0``): image arithmetic, median, grayscale box morphology.
* small derived tolerances for the analytically-different but convergent ops
  (anisotropic diffusion, which depends on per-iteration f32 accumulation).

Skips cleanly when SimpleITK or the object store is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

import ritk  # noqa: E402

from _sitk_data import fetch  # noqa: E402

IMAGE = "cthead1-Float.mha"  # 256×256 float CT head, spacing 0.352778 mm (isotropic)


@pytest.fixture(scope="module")
def images():
    path = fetch(IMAGE)
    ri = ritk.io.read_image(path)
    si = sitk.Cast(sitk.ReadImage(path), sitk.sitkFloat32)
    return ri, si


def _np3(a) -> np.ndarray:
    """SimpleITK array → 3-D float64 (promote a 2-D slice to z=1)."""
    arr = sitk.GetArrayFromImage(a).astype(np.float64) if not isinstance(a, np.ndarray) else a.astype(np.float64)
    return arr[None] if arr.ndim == 2 else arr


def _r3(a) -> np.ndarray:
    arr = np.asarray(a.to_numpy(), np.float64)
    return arr[None] if arr.ndim == 2 else arr


def _rel_interior(ra: np.ndarray, sa: np.ndarray, margin: int) -> float:
    ra, sa = _r3(ra) if not isinstance(ra, np.ndarray) else ra, _np3(sa)
    assert ra.shape == sa.shape, f"shape {ra.shape} != {sa.shape}"
    m = margin
    sl = (slice(None), slice(m, -m), slice(m, -m)) if ra.shape[0] == 1 else (slice(m, -m),) * 3
    return float(np.abs(ra[sl] - sa[sl]).max() / max(np.abs(sa[sl]).max(), 1e-9))


# Each case: (id, ritk(ri) -> PyImage, sitk(si) -> sitk.Image, rel_tol, margin)
# `ri`/`si` are the cthead1 ritk/sitk images from the `images` fixture.
CASES = [
    # ── Image arithmetic — bit-exact ───────────────────────────────────────────
    ("add_images", lambda ri, si: ritk.filter.add_images(ri, ri), lambda si: sitk.Add(si, si), 0.0, 2),
    ("subtract_images", lambda ri, si: ritk.filter.subtract_images(ri, ri), lambda si: sitk.Subtract(si, si), 0.0, 2),
    ("multiply_images", lambda ri, si: ritk.filter.multiply_images(ri, ri), lambda si: sitk.Multiply(si, si), 0.0, 2),
    ("maximum_images", lambda ri, si: ritk.filter.maximum_images(ri, ri), lambda si: sitk.Maximum(si, si), 0.0, 2),
    ("minimum_images", lambda ri, si: ritk.filter.minimum_images(ri, ri), lambda si: sitk.Minimum(si, si), 0.0, 2),
    # ── Intensity transforms — float-exact ─────────────────────────────────────
    ("sigmoid", lambda ri, si: ritk.filter.sigmoid_filter(ri, alpha=50.0, beta=100.0),
     lambda si: sitk.Sigmoid(si, alpha=50.0, beta=100.0, outputMaximum=1.0, outputMinimum=0.0), 1e-6, 4),
    ("rescale_intensity", lambda ri, si: ritk.filter.rescale_intensity(ri, 0.0, 255.0),
     lambda si: sitk.RescaleIntensity(si, 0.0, 255.0), 1e-6, 2),
    ("intensity_windowing", lambda ri, si: ritk.filter.intensity_windowing(ri, 50.0, 150.0, 0.0, 255.0),
     lambda si: sitk.IntensityWindowing(si, 50.0, 150.0, 0.0, 255.0), 1e-6, 2),
    # ── Discrete edge operators — float-exact ──────────────────────────────────
    ("laplacian", lambda ri, si: ritk.filter.laplacian(ri), lambda si: sitk.Laplacian(si), 1e-6, 4),
    ("gradient_magnitude", lambda ri, si: ritk.filter.gradient_magnitude(ri), lambda si: sitk.GradientMagnitude(si), 1e-6, 4),
    # ── Gaussian family — float-exact ──────────────────────────────────────────
    ("discrete_gaussian", lambda ri, si: ritk.filter.discrete_gaussian(ri, 4.0),
     lambda si: sitk.DiscreteGaussian(si, 4.0), 1e-6, 4),
    ("smoothing_recursive_gaussian", lambda ri, si: ritk.filter.recursive_gaussian(ri, sigma=2.0, order=0),
     lambda si: sitk.SmoothingRecursiveGaussian(si, sigma=2.0), 1e-6, 4),
    ("gradient_magnitude_recursive_gaussian", lambda ri, si: ritk.filter.recursive_gaussian(ri, sigma=2.0, order=1),
     lambda si: sitk.GradientMagnitudeRecursiveGaussian(si, sigma=2.0), 1e-6, 4),
    ("laplacian_recursive_gaussian", lambda ri, si: ritk.filter.laplacian_of_gaussian(ri, sigma=2.0),
     lambda si: sitk.LaplacianRecursiveGaussian(si, sigma=2.0), 1e-6, 4),
    ("recursive_gaussian_directional_o1_x", lambda ri, si: ritk.filter.recursive_gaussian_directional(ri, 2.0, 1, 2),
     lambda si: sitk.RecursiveGaussian(si, 2.0, False, 1, 0), 1e-6, 5),
    ("recursive_gaussian_directional_o2_y", lambda ri, si: ritk.filter.recursive_gaussian_directional(ri, 2.0, 2, 1),
     lambda si: sitk.RecursiveGaussian(si, 2.0, False, 2, 1), 1e-6, 5),
    ("unsharp_mask", lambda ri, si: ritk.filter.unsharp_mask(ri, 1.0, 0.5, 0.0),
     lambda si: sitk.UnsharpMask(si, [1.0, 1.0], 0.5, 0.0), 1e-6, 4),
    # ── Anisotropic diffusion — convergent f32 accumulation ────────────────────
    ("gradient_anisotropic_diffusion", lambda ri, si: ritk.filter.anisotropic_diffusion(ri, 5, 3.0, 0.0625),
     lambda si: sitk.GradientAnisotropicDiffusion(si, timeStep=0.0625, conductanceParameter=3.0, numberOfIterations=5), 1e-5, 4),
    ("curvature_anisotropic_diffusion", lambda ri, si: ritk.filter.curvature_anisotropic_diffusion(ri, 5, 0.0625, 3.0),
     lambda si: sitk.CurvatureAnisotropicDiffusion(si, timeStep=0.0625, conductanceParameter=3.0, numberOfIterations=5), 1e-4, 4),
    # ── Grayscale box morphology — bit-exact ───────────────────────────────────
    ("grayscale_dilation", lambda ri, si: ritk.filter.grayscale_dilation(ri, 1), lambda si: sitk.GrayscaleDilate(si, [1, 1]), 0.0, 2),
    ("grayscale_erosion", lambda ri, si: ritk.filter.grayscale_erosion(ri, 1), lambda si: sitk.GrayscaleErode(si, [1, 1]), 0.0, 2),
    ("median", lambda ri, si: ritk.filter.median_filter(ri, 1), lambda si: sitk.Median(si, [1, 1]), 0.0, 2),
    # ── Grayscale top-hat (box kernel) — bit-exact ─────────────────────────────
    ("white_top_hat", lambda ri, si: ritk.filter.white_top_hat(ri, 2),
     lambda si: sitk.WhiteTopHat(si, [2, 2], sitk.sitkBox), 0.0, 3),
    ("black_top_hat", lambda ri, si: ritk.filter.black_top_hat(ri, 2),
     lambda si: sitk.BlackTopHat(si, [2, 2], sitk.sitkBox), 0.0, 3),
    # ── Binary threshold — bit-exact ───────────────────────────────────────────
    ("binary_threshold", lambda ri, si: ritk.filter.binary_threshold(ri, 50.0, 150.0, 1.0, 0.0),
     lambda si: sitk.Cast(sitk.BinaryThreshold(si, 50.0, 150.0, 1, 0), sitk.sitkFloat32), 0.0, 2),
    # ── Bin-shrink downsampling — bit-exact ────────────────────────────────────
    ("bin_shrink", lambda ri, si: ritk.filter.bin_shrink(ri, 1, 2, 2),
     lambda si: sitk.BinShrink(si, [2, 2]), 0.0, 1),
]


@pytest.mark.parametrize("name,rfn,sfn,tol,margin", CASES, ids=[c[0] for c in CASES])
def test_filter_matches_sitk(images, name, rfn, sfn, tol, margin):
    ri, si = images
    rel = _rel_interior(rfn(ri, si), sfn(si), margin)
    if tol == 0.0:
        assert rel == 0.0, f"{name}: expected bit-exact, got rel {rel:.2e}"
    else:
        assert rel < tol, f"{name}: rel {rel:.2e} >= {tol:.0e}"


# ── 3-D ops that the 2-D cthead slice cannot exercise ──────────────────────────

@pytest.fixture(scope="module")
def vol3d():
    rng = np.random.default_rng(3)
    arr = (rng.standard_normal((12, 16, 20)) * 50 + 100).astype(np.float32)
    return ritk.Image(np.ascontiguousarray(arr)), sitk.GetImageFromArray(arr), arr


# (id, ritk(ri, ax)->PyImage, sitk(si, sdim)->Image, ritk_axis, sitk_dim, tol)
# ritk projection axis is (z,y,x); sitk projectionDimension is (x,y,z) ⇒ z is dim 2.
_PROJ = [
    ("max_projection", ritk.filter.max_intensity_projection, sitk.MaximumProjection, 0.0),
    ("min_projection", ritk.filter.min_intensity_projection, sitk.MinimumProjection, 0.0),
    ("mean_projection", ritk.filter.mean_intensity_projection, sitk.MeanProjection, 1e-5),
    ("sum_projection", ritk.filter.sum_intensity_projection, sitk.SumProjection, 1e-5),
]


@pytest.mark.parametrize("name,rfn,sfn,tol", _PROJ, ids=[c[0] for c in _PROJ])
@pytest.mark.parametrize("rax,sdim", [(0, 2), (1, 1), (2, 0)], ids=["z", "y", "x"])
def test_projection_matches_sitk(vol3d, name, rfn, sfn, tol, rax, sdim):
    ri, si, _ = vol3d
    r = np.squeeze(np.asarray(rfn(ri, rax).to_numpy(), np.float64))
    s = np.squeeze(sitk.GetArrayFromImage(sfn(si, sdim)).astype(np.float64))
    assert r.shape == s.shape, f"{name}[{rax}]: shape {r.shape} != {s.shape}"
    rel = np.abs(r - s).max() / max(np.abs(s).max(), 1e-9)
    assert rel <= tol, f"{name}[axis={rax}]: rel {rel:.2e} > {tol:.0e}"


def test_distance_transform_matches_scipy(vol3d):
    # Spacing-aware Euclidean distance transform vs scipy's exact reference
    # (ritk: foreground -> 0, background -> distance to nearest foreground).
    ndimage = pytest.importorskip("scipy.ndimage")
    _, _, arr = vol3d
    mask = (arr > 100).astype(np.float32)
    for spacing in [(1.0, 1.0, 1.0), (2.0, 1.0, 0.5)]:
        ri = ritk.Image(np.ascontiguousarray(mask), spacing=list(spacing))
        r = np.asarray(ritk.filter.distance_transform(ri, foreground_threshold=0.5).to_numpy(), np.float64)
        ref = ndimage.distance_transform_edt(1.0 - mask, sampling=spacing)
        rel = np.abs(r - ref).max() / max(np.abs(ref).max(), 1e-9)
        assert rel < 1e-6, f"distance_transform spacing={spacing}: rel {rel:.2e}"


# ── Resampling transforms: ritk vs the equivalent SimpleITK Resample ───────────
# ritk transforms move *content*; ITK transforms map output->input coordinates.
# The pinned-convention equivalences below are float/bit-exact and lock the sign
# and rotation-center conventions against drift.

@pytest.fixture(scope="module")
def slice2d():
    rng = np.random.default_rng(7)
    arr = (rng.standard_normal((1, 40, 48)) * 40 + 100).astype(np.float32)
    return ritk.Image(np.ascontiguousarray(arr)), sitk.GetImageFromArray(arr[0])


def _rel2d(r, s, m=8):
    r = np.squeeze(np.asarray(r.to_numpy(), np.float64))
    s = sitk.GetArrayFromImage(s).astype(np.float64)
    return np.abs(r[m:-m, m:-m] - s[m:-m, m:-m]).max() / max(np.abs(s).max(), 1e-9)


def test_shift_matches_sitk_translation(slice2d):
    # ritk shifts content by +3 in x; ITK TranslationTransform maps out->in, so
    # the same image results from translating the sampling coordinate by -3.
    ri, si = slice2d
    out = ritk.filter.shift_image(ri, shift_x=3.0, mode="linear")
    ref = sitk.Resample(si, si, sitk.TranslationTransform(2, (-3.0, 0.0)),
                        sitk.sitkLinear, 0.0)
    assert _rel2d(out, ref) == 0.0


def test_rotate_matches_sitk_euler2d(slice2d):
    import math
    ri, si = slice2d
    out = ritk.filter.rotate_image(ri, angle_z=math.pi / 2, mode="linear")
    center = si.TransformContinuousIndexToPhysicalPoint(
        [(si.GetWidth() - 1) / 2.0, (si.GetHeight() - 1) / 2.0])
    ref = sitk.Resample(si, si, sitk.Euler2DTransform(center, math.pi / 2),
                        sitk.sitkLinear, 0.0)
    assert _rel2d(out, ref) == 0.0


def test_zoom_matches_sitk_magnify(slice2d):
    # 2x zoom == resample onto a 2x-finer grid (half spacing), identity transform.
    ri, si = slice2d
    out = ritk.filter.zoom_image(ri, zoom_x=2.0, zoom_y=2.0)
    ref_grid = sitk.Image([si.GetWidth() * 2, si.GetHeight() * 2], sitk.sitkFloat32)
    ref_grid.SetSpacing([s / 2.0 for s in si.GetSpacing()])
    ref_grid.SetOrigin(si.GetOrigin())
    ref = sitk.Resample(si, ref_grid, sitk.Transform(), sitk.sitkLinear, 0.0)
    assert _rel2d(out, ref) < 1e-6


# ── Automatic threshold-selection family vs the ITK histogram calculators ──────
# Every ritk auto-threshold reproduces the corresponding ITK calculator's value
# under ITK's 256-bin histogram geometry (MarginalScale=100 upper-edge margin;
# Otsu/multi-Otsu report a bin edge, Li/Yen/Kapur/Triangle a bin centre).

def _sitk_threshold(filter_cls, si):
    f = filter_cls()
    f.SetInsideValue(1)
    f.SetOutsideValue(0)
    f.SetNumberOfHistogramBins(256)
    f.Execute(si)
    return f.GetThreshold()


_AUTO_THRESHOLDS = [
    ("otsu", ritk.segmentation.otsu_threshold, sitk.OtsuThresholdImageFilter),
    ("li", ritk.segmentation.li_threshold, sitk.LiThresholdImageFilter),
    ("yen", ritk.segmentation.yen_threshold, sitk.YenThresholdImageFilter),
    ("triangle", ritk.segmentation.triangle_threshold, sitk.TriangleThresholdImageFilter),
    ("kapur", ritk.segmentation.kapur_threshold, sitk.MaximumEntropyThresholdImageFilter),
]


@pytest.mark.parametrize("name,rfn,sfilt", _AUTO_THRESHOLDS, ids=[c[0] for c in _AUTO_THRESHOLDS])
def test_auto_threshold_matches_sitk(images, name, rfn, sfilt):
    ri, si = images
    rt = rfn(ri)[0]
    st = _sitk_threshold(sfilt, si)
    # f32 round-trip in the histogram/entropy accumulation; the bin selection and
    # geometry are identical, so the residual is pure single-precision noise.
    rel = abs(rt - st) / max(abs(st), 1.0)
    assert rel < 1e-4, f"{name}: ritk={rt} sitk={st} rel={rel:.2e}"


def test_multi_otsu_matches_sitk(images):
    ri, si = images
    rt = ritk.segmentation.multi_otsu_threshold(ri, num_classes=3)
    rv = rt[0] if isinstance(rt, tuple) else rt
    f = sitk.OtsuMultipleThresholdsImageFilter()
    f.SetNumberOfThresholds(2)
    f.SetNumberOfHistogramBins(256)
    f.Execute(si)
    sv = f.GetThresholds()
    assert len(rv) == len(sv) == 2
    for r, s in zip(sorted(rv), sorted(sv)):
        assert abs(r - s) / max(abs(s), 1.0) < 1e-4, f"multi_otsu ritk={rv} sitk={sv}"
