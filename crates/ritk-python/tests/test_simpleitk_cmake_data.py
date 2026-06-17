"""ritk parity on SimpleITK's cmake test cases using the **real upstream input
data**.

This is the high-fidelity companion to ``test_simpleitk_cmake_cases.py``. Where
that module reuses the locally-available ``cthead1`` with upstream parameter
values, this one fetches the *exact* input images the upstream cmake tests use
(``RA-Float.nrrd``, ``WhiteDots.png``, ``Ramp-Zero-One-Float.nrrd``, …) from
SimpleITK's content-addressed ExternalData store, then drives ritk and live
SimpleITK with the parameters pinned in each filter's
``Code/BasicFilters/yaml/<Filter>.yaml`` ``tests:`` entry.

Inputs are resolved by SHA-512 from the committed manifest
``externals/sitk_input_manifest.json`` (harvested from the upstream
``Testing/Data/Input/<name>.sha512`` files) and downloaded once into a local
cache (``externals/sitk_data``, git-ignored). The download is hash-verified;
tests skip cleanly if the object store is unreachable, so the suite stays
hermetic offline.

Because SimpleITK computes the comparison output via the same ITK code path that
produced each upstream md5 baseline, asserting ritk == SimpleITK here is a
faithful re-implementation of the cmake regression — the same input bytes, the
same parameters, the same oracle.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

import ritk  # noqa: E402

_HERE = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# Manifest (input name -> sha512) is committed next to this test; the fetched
# data is cached under the git-ignored externals/ tree.
_MANIFEST = os.path.join(_HERE, "sitk_input_manifest.json")
_CACHE = os.path.join(_REPO_ROOT, "externals", "sitk_data")
_CDN = "https://data.kitware.com/api/v1/file/hashsum/sha512/{}/download"


def _manifest():
    if not os.path.exists(_MANIFEST):
        pytest.skip("SimpleITK input manifest not present")
    with open(_MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


def fetch_input(name: str) -> str:
    """Return a local path to the upstream input `name`, downloading + verifying
    it from ExternalData on first use. Skips the test if unavailable."""
    sha = _manifest().get(name)
    if sha is None:
        pytest.skip(f"{name} not in manifest")
    os.makedirs(_CACHE, exist_ok=True)
    dst = os.path.join(_CACHE, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    try:
        urllib.request.urlretrieve(_CDN.format(sha), dst)
    except Exception as exc:  # offline / store down
        pytest.skip(f"could not fetch {name}: {exc}")
    if hashlib.sha512(open(dst, "rb").read()).hexdigest() != sha:
        os.remove(dst)
        pytest.fail(f"{name}: sha512 mismatch after download")
    return dst


def _pair(name):
    path = fetch_input(name)
    return ritk.io.read_image(path), sitk.Cast(sitk.ReadImage(path), sitk.sitkFloat32)


def _rel(r, s, m=3):
    r = np.asarray((r[0] if isinstance(r, tuple) else r).to_numpy(), np.float64)
    s = sitk.GetArrayFromImage(s).astype(np.float64)
    if r.ndim == 2:
        r = r[None]
    if s.ndim == 2:
        s = s[None]
    return np.abs(r[:, m:-m, m:-m] - s[:, m:-m, m:-m]).max() / max(np.abs(s).max(), 1e-9)


# Each case: (id mirroring <Filter>.yaml::tag, input name, ritk fn, sitk fn, tol).
_CASES = [
    ("DiscreteGaussian/bigG", "WhiteDots.png",
     lambda ri: ritk.filter.discrete_gaussian(ri, 100.0),
     lambda si: _dg(si, 100.0, 64), 1e-6),
    ("DiscreteGaussian/float", "RA-Float.nrrd",
     lambda ri: ritk.filter.discrete_gaussian(ri, 1.0),
     lambda si: sitk.DiscreteGaussian(si, 1.0), 1e-6),
    ("DiscreteGaussian/short", "RA-Slice-Short.nrrd",
     lambda ri: ritk.filter.discrete_gaussian(ri, 1.0),
     lambda si: sitk.DiscreteGaussian(si, 1.0), 1e-6),
    ("IntensityWindowing/3dFloat", "RA-Float.nrrd",
     lambda ri: ritk.filter.intensity_windowing(ri, 0.0, 255.0, 0.0, 255.0),
     lambda si: sitk.IntensityWindowing(si, 0.0, 255.0, 0.0, 255.0), 1e-6),
    ("Median/defaults", "RA-Float.nrrd",
     lambda ri: ritk.filter.median_filter(ri, 1),
     lambda si: sitk.Median(si, [1, 1, 1]), 0.0),
    ("Mean/defaults", "RA-Float.nrrd",
     lambda ri: ritk.filter.mean_filter(ri, 1),
     lambda si: sitk.Mean(si, [1, 1, 1]), 1e-6),
    ("GradientMagnitude/default", "RA-Float.nrrd",
     lambda ri: ritk.filter.gradient_magnitude(ri),
     lambda si: sitk.GradientMagnitude(si), 1e-6),
    ("Laplacian/default", "RA-Float.nrrd",
     lambda ri: ritk.filter.laplacian(ri),
     lambda si: sitk.Laplacian(si), 1e-6),
    ("SmoothingRecursiveGaussian/default", "RA-Float.nrrd",
     lambda ri: ritk.filter.recursive_gaussian(ri, sigma=1.0, order=0),
     lambda si: sitk.SmoothingRecursiveGaussian(si, 1.0), 1e-6),
    ("RescaleIntensity/3d", "RA-Float.nrrd",
     lambda ri: ritk.filter.rescale_intensity(ri, 0.0, 255.0),
     lambda si: sitk.RescaleIntensity(si, 0.0, 255.0), 1e-6),
    ("Normalize/defaults", "Ramp-Up-Short.nrrd",
     lambda ri: ritk.filter.normalize_image(ri),
     lambda si: sitk.Normalize(si), 1e-6),
    ("Sigmoid/defaults", "Ramp-Zero-One-Float.nrrd",
     lambda ri: ritk.filter.sigmoid_filter(ri, alpha=1.0, beta=0.0, min_output=0.0, max_output=1.0),
     lambda si: sitk.Sigmoid(si, 1.0, 0.0, 1.0, 0.0), 1e-6),
    # bit-exact:
    ("BinaryThreshold/NarrowThreshold", "RA-Short.nrrd",
     lambda ri: ritk.filter.binary_threshold(ri, 10.0, 100.0, 255.0, 0.0),
     lambda si: sitk.BinaryThreshold(si, 10.0, 100.0, 255, 0), 0.0),
    ("Threshold/Threshold1", "RA-Slice-Short.nrrd",
     lambda ri: ritk.filter.threshold_outside(ri, 25000.0, 65535.0),
     lambda si: sitk.Threshold(si, 25000.0, 65535.0, 0.0), 0.0),
    ("Clamp/default", "RA-Short.nrrd",
     lambda ri: ritk.filter.clamp_image(ri, 0.0, 20000.0),
     lambda si: sitk.Clamp(si, sitk.sitkFloat32, 0.0, 20000.0), 0.0),
    ("InvertIntensity/default", "RA-Short.nrrd",
     lambda ri: ritk.filter.invert_intensity(ri, 255.0),
     lambda si: sitk.InvertIntensity(si, 255.0), 0.0),
    # GrayscaleDilate/Erode: upstream uses a radius-1 ball SE, which equals a
    # radius-1 box (ritk's flat SE), so this is bit-exact despite the box/ball
    # convention difference at larger radii.
    ("GrayscaleDilate/GrayscaleDilate", "STAPLE1.png",
     lambda ri: ritk.filter.grayscale_dilation(ri, 1),
     lambda si: sitk.GrayscaleDilate(si, [1, 1], sitk.sitkBall), 0.0),
    ("GrayscaleErode/GrayscaleErode", "STAPLE1.png",
     lambda ri: ritk.filter.grayscale_erosion(ri, 1),
     lambda si: sitk.GrayscaleErode(si, [1, 1], sitk.sitkBall), 0.0),
    ("GradientMagnitudeRecursiveGaussian/default", "RA-Float.nrrd",
     lambda ri: ritk.filter.recursive_gaussian(ri, sigma=1.0, order=1),
     lambda si: sitk.GradientMagnitudeRecursiveGaussian(si, 1.0), 1e-6),
    ("LaplacianRecursiveGaussian/default", "RA-Float.nrrd",
     lambda ri: ritk.filter.laplacian_of_gaussian(ri, sigma=1.0),
     lambda si: sitk.LaplacianRecursiveGaussian(si, 1.0), 1e-6),
    ("BinShrink/by4", "RA-Float.nrrd",
     lambda ri: ritk.filter.bin_shrink(ri, 4, 4, 4),
     lambda si: sitk.BinShrink(si, [4, 4, 4]), 0.0),
    ("WhiteTopHat/WhiteTopHatErode", "STAPLE1.png",
     lambda ri: ritk.filter.white_top_hat(ri, 1),
     lambda si: sitk.WhiteTopHat(si, [1, 1], sitk.sitkBall), 0.0),
    ("BlackTopHat/BlackTopHapErode", "STAPLE1.png",
     lambda ri: ritk.filter.black_top_hat(ri, 1),
     lambda si: sitk.BlackTopHat(si, [1, 1], sitk.sitkBall), 0.0),
    # Anisotropic diffusion: per-iteration f32 accumulation gives a small derived
    # tolerance (matches the corpus diffusion convention). Upstream TimeStep=0.01,
    # default 5 iterations / conductance 1.0.
    ("GradientAnisotropicDiffusion/defaults", "RA-Float.nrrd",
     lambda ri: ritk.filter.anisotropic_diffusion(ri, 5, 1.0, 0.01),
     lambda si: sitk.GradientAnisotropicDiffusion(si, 0.01, 1.0, 5), 1e-3),
    ("CurvatureAnisotropicDiffusion/defaults", "RA-Float.nrrd",
     lambda ri: ritk.filter.curvature_anisotropic_diffusion(ri, 5, 0.01, 1.0),
     lambda si: sitk.CurvatureAnisotropicDiffusion(si, 0.01, 1.0, 5), 2e-3),
    ("GradientAnisotropicDiffusion/longer", "RA-Float.nrrd",
     lambda ri: ritk.filter.anisotropic_diffusion(ri, 10, 1.0, 0.01),
     lambda si: sitk.GradientAnisotropicDiffusion(si, 0.01, 1.0, 10), 2e-2),
]

# Two-input image-arithmetic cases (<Filter>.yaml::tag with two inputs).
_BINARY_CASES = [
    ("Add/3d", "RA-Short.nrrd", "RA-Short.nrrd",
     lambda a, b: ritk.filter.add_images(a, b), lambda a, b: sitk.Add(a, b), 0.0),
    ("Subtract/3D", "RA-Short.nrrd", "RA-Short.nrrd",
     lambda a, b: ritk.filter.subtract_images(a, b), lambda a, b: sitk.Subtract(a, b), 0.0),
    ("Subtract/2D", "RA-Slice-Float.nrrd", "RA-Slice-Float.nrrd",
     lambda a, b: ritk.filter.subtract_images(a, b), lambda a, b: sitk.Subtract(a, b), 0.0),
    ("Multiply/defaults", "Ramp-Zero-One-Float.nrrd", "Ramp-One-Zero-Float.nrrd",
     lambda a, b: ritk.filter.multiply_images(a, b), lambda a, b: sitk.Multiply(a, b), 0.0),
    ("Divide/defaults", "Ramp-Up-Short.nrrd", "Ramp-Down-Short.nrrd",
     lambda a, b: ritk.filter.divide_images(a, b), lambda a, b: sitk.Divide(a, b), 0.0),
    ("Add/2d", "STAPLE1.png", "STAPLE2.png",
     lambda a, b: ritk.filter.add_images(a, b), lambda a, b: sitk.Add(a, b), 0.0),
    ("SquaredDifference/3d", "Ramp-Up-Short.nrrd", "Ramp-Down-Short.nrrd",
     lambda a, b: ritk.filter.squared_difference_images(a, b),
     lambda a, b: sitk.SquaredDifference(a, b), 0.0),
    ("AbsoluteValueDifference/3d", "Ramp-Up-Short.nrrd", "Ramp-Down-Short.nrrd",
     lambda a, b: ritk.filter.absolute_value_difference_images(a, b),
     lambda a, b: sitk.AbsoluteValueDifference(a, b), 0.0),
    # Transcendental: ritk computes in f32, sitk in double then narrows — tol
    # is the f32-vs-double evaluation gap, not a fudge factor.
    ("Atan2/defaults", "Ramp-Zero-One-Float.nrrd", "Ramp-One-Zero-Float.nrrd",
     lambda a, b: ritk.filter.atan2_images(a, b),
     lambda a, b: sitk.Atan2(a, b), 1e-6),
    ("Pow/defaults", "Ramp-Zero-One-Float.nrrd", "Ramp-One-Zero-Float.nrrd",
     lambda a, b: ritk.filter.pow_images(a, b),
     lambda a, b: sitk.Pow(a, b), 1e-6),
]


def _dg(si, variance, mkw):
    f = sitk.DiscreteGaussianImageFilter()
    f.SetVariance(variance)
    f.SetMaximumKernelWidth(mkw)
    return f.Execute(si)


@pytest.mark.parametrize("tag,name,rfn,sfn,tol", _CASES, ids=[c[0] for c in _CASES])
def test_cmake_case_on_upstream_data(tag, name, rfn, sfn, tol):
    ri, si = _pair(name)
    rel = _rel(rfn(ri), sfn(si))
    if tol == 0.0:
        assert rel == 0.0, f"{tag}: expected bit-exact on {name}, got rel {rel:.2e}"
    else:
        assert rel < tol, f"{tag}: rel {rel:.2e} >= {tol:.0e} on {name}"


@pytest.mark.parametrize("tag,na,nb,rfn,sfn,tol", _BINARY_CASES, ids=[c[0] for c in _BINARY_CASES])
def test_cmake_binary_case_on_upstream_data(tag, na, nb, rfn, sfn, tol):
    ra, sa = _pair(na)
    rb, sb = _pair(nb)
    rel = _rel(rfn(ra, rb), sfn(sa, sb))
    if tol == 0.0:
        assert rel == 0.0, f"{tag}: expected bit-exact on {na},{nb}, got rel {rel:.2e}"
    else:
        assert rel < tol, f"{tag}: rel {rel:.2e} >= {tol:.0e} on {na},{nb}"


# Auto-threshold mask cases (<Filter>.yaml::tag "default" on RA-Short). The
# upstream baseline is the segmented mask. ITK marks `inside` = below-threshold;
# ritk marks foreground = at-or-above-threshold, so the masks are exact
# complements when the threshold value matches (it does — see the corpus
# auto-threshold value tests). This pins the *mask* output bit-exactly.
_THRESHOLD_MASK = [
    ("OtsuThreshold/default", ritk.segmentation.otsu_threshold, sitk.OtsuThresholdImageFilter),
    ("LiThreshold/default", ritk.segmentation.li_threshold, sitk.LiThresholdImageFilter),
    ("YenThreshold/default", ritk.segmentation.yen_threshold, sitk.YenThresholdImageFilter),
    ("TriangleThreshold/default", ritk.segmentation.triangle_threshold, sitk.TriangleThresholdImageFilter),
    ("MaximumEntropyThreshold/default", ritk.segmentation.kapur_threshold, sitk.MaximumEntropyThresholdImageFilter),
]


# Unary math filters (<Filter>.yaml "defaults"). Exercised on Ramp-Zero-One-Float
# whose values lie in [0,1] — a safe domain for log/sqrt/asin/acos.
_UNARY_MATH = [
    ("AbsImageFilter", ritk.filter.abs_image, sitk.Abs, 0.0),
    ("SqrtImageFilter", ritk.filter.sqrt_image, sitk.Sqrt, 0.0),
    ("SquareImageFilter", ritk.filter.square_image, sitk.Square, 0.0),
    ("LogImageFilter", ritk.filter.log_image, sitk.Log, 0.0),
    ("Log10ImageFilter", ritk.filter.log10_image, sitk.Log10, 0.0),
    ("ExpImageFilter", ritk.filter.exp_image, sitk.Exp, 1e-6),
    ("ExpNegativeImageFilter", ritk.filter.exp_negative_image, sitk.ExpNegative, 1e-6),
    ("SinImageFilter", ritk.filter.sin_image, sitk.Sin, 1e-6),
    ("CosImageFilter", ritk.filter.cos_image, sitk.Cos, 1e-6),
    ("TanImageFilter", ritk.filter.tan_image, sitk.Tan, 1e-6),
    ("AsinImageFilter", ritk.filter.asin_image, sitk.Asin, 1e-6),
    ("AcosImageFilter", ritk.filter.acos_image, sitk.Acos, 1e-6),
    ("AtanImageFilter", ritk.filter.atan_image, sitk.Atan, 1e-6),
]


@pytest.mark.parametrize("tag,rfn,sfn,tol", _UNARY_MATH, ids=[c[0] for c in _UNARY_MATH])
def test_cmake_unary_math_on_upstream_data(tag, rfn, sfn, tol):
    ri, si = _pair("Ramp-Zero-One-Float.nrrd")
    rel = _rel(rfn(ri), sfn(si), m=2)
    if tol == 0.0:
        assert rel == 0.0, f"{tag}: expected bit-exact, got rel {rel:.2e}"
    else:
        assert rel < tol, f"{tag}: rel {rel:.2e} >= {tol:.0e}"


@pytest.mark.parametrize("inp", ["RA-Short.nrrd", "Ramp-Zero-One-Float.nrrd"],
                         ids=["default", "default_on_float"])
@pytest.mark.parametrize("tag,rfn,sfilt", _THRESHOLD_MASK, ids=[c[0] for c in _THRESHOLD_MASK])
def test_cmake_threshold_mask_on_upstream_data(tag, rfn, sfilt, inp):
    ri, si = _pair(inp)
    r = np.squeeze(np.asarray(rfn(ri)[1].to_numpy(), np.float64))
    f = sfilt()
    f.SetInsideValue(1)
    f.SetOutsideValue(0)
    f.SetNumberOfHistogramBins(256)
    inside = np.squeeze(sitk.GetArrayFromImage(f.Execute(si)).astype(np.float64))
    # ritk foreground == ITK outside == 1 - inside.
    assert np.array_equal(r, 1.0 - inside), f"{tag}: mask differs from sitk complement"


@pytest.mark.parametrize("negated", [False, True], ids=["Mask", "MaskNegated"])
def test_cmake_mask_on_upstream_data(negated):
    # MaskImageFilter / MaskNegatedImageFilter on RA-Short with a thresholded mask.
    ri, si = _pair("RA-Short.nrrd")
    arr = sitk.GetArrayFromImage(si).astype(np.float64)
    mbin = (arr > 10000).astype(np.float32)
    rim = ritk.Image(np.ascontiguousarray(mbin))
    sim = sitk.GetImageFromArray((arr > 10000).astype(np.uint8))
    sim.CopyInformation(si)
    if negated:
        r = ritk.filter.mask_negated_image(ri, rim, 0.0)
        s = sitk.MaskNegated(si, sim, 0.0)
    else:
        r = ritk.filter.mask_image(ri, rim, 0.0)
        s = sitk.Mask(si, sim, 0.0)
    assert _rel(r, s, m=2) == 0.0


def test_cmake_rgb_median_on_upstream_data():
    # MedianImageFilter on the upstream RGB image (VM1111Shrink-RGB). ITK applies
    # the scalar median per component on a vector image; ritk's color_median does
    # the same via the per-component adaptor — bit-exact.
    path = fetch_input("VM1111Shrink-RGB.png")
    si = sitk.ReadImage(path)
    if si.GetNumberOfComponentsPerPixel() != 3:
        pytest.skip("expected a 3-component RGB input")
    arr = sitk.GetArrayFromImage(si).astype(np.float32)  # (H, W, 3)
    ci = ritk.ColorImage(np.ascontiguousarray(arr[None]))  # (1, H, W, 3)
    r = np.squeeze(np.asarray(ritk.filter.color_median(ci, 1).to_numpy()))  # (H, W, 3)
    s = sitk.GetArrayFromImage(
        sitk.Median(sitk.Cast(si, sitk.sitkVectorFloat32), [1, 1])
    ).astype(np.float64)
    assert np.array_equal(r[2:-2, 2:-2], s[2:-2, 2:-2]), "RGB median differs from sitk vector median"


def test_cmake_rgb_mean_on_upstream_data():
    # MeanImageFilter on the upstream RGB image, per-component.
    path = fetch_input("VM1111Shrink-RGB.png")
    si = sitk.ReadImage(path)
    if si.GetNumberOfComponentsPerPixel() != 3:
        pytest.skip("expected a 3-component RGB input")
    arr = sitk.GetArrayFromImage(si).astype(np.float32)
    ci = ritk.ColorImage(np.ascontiguousarray(arr[None]))
    r = np.squeeze(np.asarray(ritk.filter.color_mean(ci, 1).to_numpy()))
    s = sitk.GetArrayFromImage(sitk.Mean(sitk.Cast(si, sitk.sitkVectorFloat32), [1, 1])).astype(np.float64)
    m = 2
    rel = np.abs(r[m:-m, m:-m] - s[m:-m, m:-m]).max() / max(np.abs(s).max(), 1e-9)
    assert rel < 1e-6, f"RGB mean rel {rel:.2e}"


def test_cmake_rgb_smoothing_recursive_gaussian_on_upstream_data():
    # SmoothingRecursiveGaussianImageFilter on the upstream RGB image — ITK
    # filters each component independently; ritk's color_smoothing_recursive_gaussian
    # matches float-exact (the per-component Deriche IIR).
    path = fetch_input("VM1111Shrink-RGB.png")
    si = sitk.ReadImage(path)
    if si.GetNumberOfComponentsPerPixel() != 3:
        pytest.skip("expected a 3-component RGB input")
    arr = sitk.GetArrayFromImage(si).astype(np.float32)
    ci = ritk.ColorImage(np.ascontiguousarray(arr[None]))
    r = np.squeeze(np.asarray(ritk.filter.color_smoothing_recursive_gaussian(ci, 2.0).to_numpy()))
    s = sitk.GetArrayFromImage(
        sitk.SmoothingRecursiveGaussian(sitk.Cast(si, sitk.sitkVectorFloat32), 2.0)
    ).astype(np.float64)
    m = 6
    rel = np.abs(r[m:-m, m:-m] - s[m:-m, m:-m]).max() / max(np.abs(s).max(), 1e-9)
    assert rel < 1e-6, f"RGB smoothing recursive gaussian rel {rel:.2e}"


def test_cmake_histogram_matching_matches_sitk():
    # HistogramMatchingImageFilter (256 levels, 7 match points, threshold at mean).
    path = fetch_input("RA-Float.nrrd")
    si = sitk.Cast(sitk.ReadImage(path), sitk.sitkFloat32)
    ri = ritk.io.read_image(path)
    ref_s = sitk.DiscreteGaussian(si, 4.0)
    ref_r = ritk.filter.discrete_gaussian(ri, 4.0)
    rh = ritk.statistics.histogram_match(ri, ref_r, num_bins=256, num_match_points=7,
                                         threshold_at_mean=True)
    sh = sitk.HistogramMatching(si, ref_s, numberOfHistogramLevels=256,
                                numberOfMatchPoints=7, thresholdAtMeanIntensity=True)
    r = np.asarray(rh.to_numpy(), np.float64)
    s = sitk.GetArrayFromImage(sh).astype(np.float64)
    m = 4
    rel = np.abs(r[m:-m, m:-m, m:-m] - s[m:-m, m:-m, m:-m]).max() / max(np.abs(s).max(), 1e-9)
    assert rel < 1e-6, f"HistogramMatching rel {rel:.2e}"


def test_cmake_label_shape_statistics_match_sitk():
    # LabelShapeStatisticsImageFilter on a single connected region. ritk centroid
    # is [z, y, x]; sitk centroid is physical (x, y, z) — reverse to compare.
    # (ritk labels by connected component; a single solid blob gives one label,
    # matching sitk's single label value.)
    m = np.zeros((20, 30, 40), np.float32)
    m[5:15, 8:22, 10:30] = 1.0
    rm = ritk.Image(np.ascontiguousarray(m))
    sm = sitk.GetImageFromArray(m.astype(np.uint8))
    rs = ritk.segmentation.label_shape_statistics(rm)
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(sm)
    lab = f.GetLabels()[0]
    assert len(rs) == 1
    assert rs[0]["voxel_count"] == f.GetNumberOfPixels(lab)
    rc = rs[0]["centroid"][::-1]  # [z,y,x] -> (x,y,z)
    sc = f.GetCentroid(lab)
    assert max(abs(a - b) for a, b in zip(rc, sc)) < 1e-6


def test_cmake_statistics_matches_sitk():
    # StatisticsImageFilter (min/max/mean/variance) on an upstream image.
    path = fetch_input("RA-Float.nrrd")
    si = sitk.Cast(sitk.ReadImage(path), sitk.sitkFloat32)
    ri = ritk.io.read_image(path)
    st = ritk.statistics.compute_statistics(ri)
    f = sitk.StatisticsImageFilter()
    f.Execute(si)
    assert abs(st["mean"] - f.GetMean()) / max(abs(f.GetMean()), 1e-9) < 1e-5
    assert st["min"] == f.GetMinimum()
    assert st["max"] == f.GetMaximum()
    assert abs(st["std"] - f.GetVariance() ** 0.5) / max(f.GetVariance() ** 0.5, 1e-9) < 1e-5


def _staple_pair():
    path = fetch_input("RA-Float.nrrd")
    si = sitk.Cast(sitk.ReadImage(path), sitk.sitkFloat32)
    a = sitk.GetArrayFromImage(si).astype(np.float32)
    m1 = (a > 25000).astype(np.float32)
    m2 = (a > 26000).astype(np.float32)
    return (
        ritk.Image(np.ascontiguousarray(m1)),
        ritk.Image(np.ascontiguousarray(m2)),
        sitk.GetImageFromArray(m1.astype(np.uint8)),
        sitk.GetImageFromArray(m2.astype(np.uint8)),
    )


def test_cmake_dice_matches_sitk():
    # LabelOverlapMeasuresImageFilter Dice coefficient.
    r1, r2, s1, s2 = _staple_pair()
    rd = ritk.statistics.dice_coefficient(r1, r2)
    f = sitk.LabelOverlapMeasuresImageFilter()
    f.Execute(s1, s2)
    assert abs(rd - f.GetDiceCoefficient()) < 1e-5, f"dice ritk={rd} sitk={f.GetDiceCoefficient()}"


def test_cmake_label_overlap_measures_match_sitk():
    # LabelOverlapMeasuresImageFilter: Jaccard + volume similarity.
    r1, r2, s1, s2 = _staple_pair()
    lo = ritk.statistics.label_overlap_measures(r1, r2)[0]
    f = sitk.LabelOverlapMeasuresImageFilter()
    f.Execute(s1, s2)
    assert abs(lo["jaccard"] - f.GetJaccardCoefficient()) < 1e-5
    assert abs(lo["volume_similarity"] - f.GetVolumeSimilarity()) < 1e-5


def test_cmake_hausdorff_matches_sitk():
    # HausdorffDistanceImageFilter.
    r1, r2, s1, s2 = _staple_pair()
    rh = ritk.statistics.hausdorff_distance(r1, r2)
    f = sitk.HausdorffDistanceImageFilter()
    f.Execute(s1, s2)
    assert abs(rh - f.GetHausdorffDistance()) / max(f.GetHausdorffDistance(), 1e-9) < 1e-5


def test_cmake_registration_meansquares_metric_matches_sitk():
    # Registration-suite parity (deterministic component): the MeanSquares image
    # metric evaluated at the identity transform. SimpleITK's
    # ImageRegistrationMethod MeanSquares == ritk.metrics.compute_mse, float-exact.
    # (The full optimiser pipeline is convergence-dependent and not exact-matchable;
    # the metric value at a fixed transform is deterministic and is.)
    path = fetch_input("RA-Float.nrrd")
    si = sitk.Cast(sitk.ReadImage(path), sitk.sitkFloat32)
    sm = sitk.DiscreteGaussian(si, 4.0)
    ri = ritk.io.read_image(path)
    rm = ritk.filter.discrete_gaussian(ri, 4.0)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares()
    reg.SetMetricSamplingStrategy(reg.NONE)
    reg.SetInitialTransform(sitk.Transform(3, sitk.sitkIdentity))
    reg.SetInterpolator(sitk.sitkLinear)
    sitk_ms = reg.MetricEvaluate(si, sm)
    ritk_ms = ritk.metrics.compute_mse(ri, rm)
    rel = abs(ritk_ms - sitk_ms) / max(abs(sitk_ms), 1e-9)
    assert rel < 1e-6, f"MeanSquares metric ritk={ritk_ms} sitk={sitk_ms} rel={rel:.2e}"


_PROJECTIONS = [
    ("MinimumProjection", ritk.filter.min_intensity_projection, sitk.MinimumProjection, 0.0),
    ("SumProjection", ritk.filter.sum_intensity_projection, sitk.SumProjection, 0.0),
    ("StandardDeviationProjection", ritk.filter.stddev_intensity_projection,
     sitk.StandardDeviationProjection, 1e-6),
]


@pytest.mark.parametrize("tag,rfn,sfn,tol", _PROJECTIONS, ids=[c[0] for c in _PROJECTIONS])
def test_cmake_projection_on_upstream_data(tag, rfn, sfn, tol):
    # ritk projection axis (z,y,x) 0 == sitk projectionDimension (x,y,z) 2.
    ri, si = _pair("RA-Float.nrrd")
    r = np.squeeze(np.asarray(rfn(ri, 0).to_numpy(), np.float64))
    s = np.squeeze(sitk.GetArrayFromImage(sfn(si, 2)).astype(np.float64))
    assert r.shape == s.shape, f"{tag}: {r.shape} != {s.shape}"
    rel = np.abs(r - s).max() / max(np.abs(s).max(), 1e-9)
    if tol == 0.0:
        assert rel == 0.0, f"{tag}: expected bit-exact, got {rel:.2e}"
    else:
        assert rel < tol, f"{tag}: rel {rel:.2e}"


def test_cmake_squared_difference_on_upstream_data():
    # SquaredDifferenceImageFilter: (a - b)^2. ritk composes square(subtract).
    ri, si = _pair("RA-Float.nrrd")
    ref_r = ritk.filter.discrete_gaussian(ri, 4.0)
    ref_s = sitk.DiscreteGaussian(si, 4.0)
    r = ritk.filter.square_image(ritk.filter.subtract_images(ri, ref_r))
    s = sitk.SquaredDifference(si, ref_s)
    assert _rel(r, s) < 1e-6


def test_cmake_fft_roundtrip_on_upstream_data():
    # Forward + inverse FFT recovers the input (SimpleITK's ForwardFFT/InverseFFT
    # round-trip test pattern) on an upstream float slice.
    ri, _ = _pair("RA-Slice-Float.nrrd")
    rt = np.asarray(ritk.filter.inverse_fft(ritk.filter.forward_fft(ri)).to_numpy(), np.float64)
    inp = np.asarray(ri.to_numpy(), np.float64)
    rel = np.abs(rt - inp).max() / max(np.abs(inp).max(), 1e-9)
    assert rel < 1e-5, f"FFT round-trip rel {rel:.2e}"


_AUTO_THRESHOLD_VALUES = [
    ("IsoDataThreshold", ritk.segmentation.isodata_threshold, sitk.IsoDataThresholdImageFilter),
    ("MomentsThreshold", ritk.segmentation.moments_threshold, sitk.MomentsThresholdImageFilter),
    ("HuangThreshold", ritk.segmentation.huang_threshold, sitk.HuangThresholdImageFilter),
    ("IntermodesThreshold", ritk.segmentation.intermodes_threshold, sitk.IntermodesThresholdImageFilter),
    ("ShanbhagThreshold", ritk.segmentation.shanbhag_threshold, sitk.ShanbhagThresholdImageFilter),
    ("KittlerIllingworthThreshold", ritk.segmentation.kittler_illingworth_threshold,
     sitk.KittlerIllingworthThresholdImageFilter),
    ("RenyiEntropyThreshold", ritk.segmentation.renyi_entropy_threshold,
     sitk.RenyiEntropyThresholdImageFilter),
    ("LiThreshold", ritk.segmentation.li_threshold, sitk.LiThresholdImageFilter),
    ("YenThreshold", ritk.segmentation.yen_threshold, sitk.YenThresholdImageFilter),
    ("KapurThreshold", ritk.segmentation.kapur_threshold, sitk.MaximumEntropyThresholdImageFilter),
    ("TriangleThreshold", ritk.segmentation.triangle_threshold, sitk.TriangleThresholdImageFilter),
]


@pytest.mark.parametrize("tag,rfn,sfilt", _AUTO_THRESHOLD_VALUES, ids=[c[0] for c in _AUTO_THRESHOLD_VALUES])
def test_cmake_auto_threshold_value_on_upstream_data(tag, rfn, sfilt):
    # IsoData / Moments threshold *value* on RA-Short vs the ITK calculator.
    ri, si = _pair("RA-Short.nrrd")
    rt = rfn(ri)[0]
    f = sfilt()
    f.SetNumberOfHistogramBins(256)
    f.Execute(si)
    st = f.GetThreshold()
    assert abs(rt - st) / max(abs(st), 1.0) < 1e-4, f"{tag}: ritk={rt} sitk={st}"


def test_cmake_multi_otsu_on_upstream_data():
    # OtsuMultipleThresholdsImageFilter (2 thresholds / 3 classes) on RA-Short.
    ri, si = _pair("RA-Short.nrrd")
    rt = ritk.segmentation.multi_otsu_threshold(ri, num_classes=3)
    rv = rt[0] if isinstance(rt, tuple) else rt
    f = sitk.OtsuMultipleThresholdsImageFilter()
    f.SetNumberOfThresholds(2)
    f.SetNumberOfHistogramBins(256)
    f.SetReturnBinMidpoint(False)
    f.Execute(si)
    sv = f.GetThresholds()
    for r, s in zip(sorted(rv), sorted(sv)):
        assert abs(r - s) / max(abs(s), 1.0) < 1e-4, f"multi_otsu ritk={rv} sitk={sv}"


def test_cmake_zero_crossing_on_upstream_data():
    # ZeroCrossingImageFilter/defaults on the upstream 2th_cthead1_distance image.
    ri, si = _pair("2th_cthead1_distance.nrrd")
    r = np.squeeze(np.asarray(ritk.filter.zero_crossing_image(ri).to_numpy(), np.float64))
    s = np.squeeze(sitk.GetArrayFromImage(sitk.ZeroCrossing(si)).astype(np.float64))
    assert np.array_equal(r, s), "zero_crossing differs from sitk.ZeroCrossing"


def _staple1_mask():
    """Binarise the upstream STAPLE1 label image (foreground = nonzero)."""
    _, si = _pair("STAPLE1.png")
    mask = (sitk.GetArrayFromImage(si).astype(np.float64) > 0).astype(np.float32)
    return ritk.Image(np.ascontiguousarray(mask[None])), sitk.GetImageFromArray(mask.astype(np.uint8))


# Binary morphology on the upstream STAPLE1 (radius-1 box SE), bit-exact interior.
_BINARY_MORPH_CMAKE = [
    ("BinaryMorphologicalOpening/BinaryMorphologicalOpening",
     lambda m: ritk.segmentation.binary_opening(m, 1),
     lambda m: sitk.BinaryMorphologicalOpening(m, [1, 1], sitk.sitkBox)),
    ("BinaryMorphologicalClosing/BinaryMorphologicalClosing",
     lambda m: ritk.segmentation.binary_closing(m, 1),
     lambda m: sitk.BinaryMorphologicalClosing(m, [1, 1], sitk.sitkBox)),
    ("BinaryFillhole/BinaryFillhole",
     lambda m: ritk.segmentation.binary_fill_holes(m),
     lambda m: sitk.BinaryFillhole(m)),
    ("BinaryErode/BinaryErode",
     lambda m: ritk.segmentation.binary_erosion(m, 1),
     lambda m: sitk.BinaryErode(m, [1, 1], sitk.sitkBox, 0.0, 1.0)),
    ("BinaryDilate/BinaryDilate",
     lambda m: ritk.segmentation.binary_dilation(m, 1),
     lambda m: sitk.BinaryDilate(m, [1, 1], sitk.sitkBox, 0.0, 1.0)),
]


@pytest.mark.parametrize("tag,rfn,sfn", _BINARY_MORPH_CMAKE, ids=[c[0] for c in _BINARY_MORPH_CMAKE])
def test_cmake_binary_morph_on_upstream_data(tag, rfn, sfn):
    rim, sim = _staple1_mask()
    r = np.squeeze(np.asarray(rfn(rim).to_numpy(), np.float64))
    s = np.squeeze(sitk.GetArrayFromImage(sfn(sim)).astype(np.float64))
    rel = np.abs(r[2:-2, 2:-2] - s[2:-2, 2:-2]).max() / max(np.abs(s).max(), 1e-9)
    assert rel == 0.0, f"{tag}: rel {rel:.2e}"


# H-transform grayscale morphology (<Filter>.yaml). Reconstruction-based, so
# bit-exact to SimpleITK on the upstream cthead1 grayscale image.
_H_TRANSFORM_CMAKE = [
    ("HMaxima/HMaxima", lambda i, h: ritk.filter.h_maxima(i, h),
     lambda i, h: sitk.HMaxima(i, h)),
    ("HMinima/HMinima", lambda i, h: ritk.filter.h_minima(i, h),
     lambda i, h: sitk.HMinima(i, h)),
    ("HConvex/HConvex", lambda i, h: ritk.filter.h_convex(i, h),
     lambda i, h: sitk.HConvex(i, h)),
    ("HConcave/HConcave", lambda i, h: ritk.filter.h_concave(i, h),
     lambda i, h: sitk.HConcave(i, h)),
]


@pytest.mark.parametrize("height", [20.0, 50.0], ids=["h20", "h50"])
@pytest.mark.parametrize("tag,rfn,sfn", _H_TRANSFORM_CMAKE, ids=[c[0] for c in _H_TRANSFORM_CMAKE])
def test_cmake_h_transform_on_upstream_data(tag, rfn, sfn, height):
    ri, si = _pair("cthead1.png")
    si = sitk.Cast(si, sitk.sitkFloat32)
    r = np.squeeze(np.asarray(rfn(ri, height).to_numpy(), np.float64))
    s = np.squeeze(sitk.GetArrayFromImage(sfn(si, height)).astype(np.float64))
    assert np.array_equal(r, s), f"{tag} (h={height}): differs from sitk"


# Regional-extrema grayscale morphology (<Filter>.yaml). Flat-zone flood, so
# bit-exact to SimpleITK on the upstream cthead1 grayscale image.
_REGIONAL_EXTREMA_CMAKE = [
    ("RegionalMaxima/RegionalMaxima", lambda i: ritk.filter.regional_maxima(i),
     lambda i: sitk.RegionalMaxima(i)),
    ("RegionalMinima/RegionalMinima", lambda i: ritk.filter.regional_minima(i),
     lambda i: sitk.RegionalMinima(i)),
    ("ValuedRegionalMaxima/ValuedRegionalMaxima", lambda i: ritk.filter.valued_regional_maxima(i),
     lambda i: sitk.ValuedRegionalMaxima(i)),
    ("ValuedRegionalMinima/ValuedRegionalMinima", lambda i: ritk.filter.valued_regional_minima(i),
     lambda i: sitk.ValuedRegionalMinima(i)),
]


@pytest.mark.parametrize("tag,rfn,sfn", _REGIONAL_EXTREMA_CMAKE,
                         ids=[c[0] for c in _REGIONAL_EXTREMA_CMAKE])
def test_cmake_regional_extrema_on_upstream_data(tag, rfn, sfn):
    ri, si = _pair("cthead1.png")
    si = sitk.Cast(si, sitk.sitkFloat32)
    r = np.squeeze(np.asarray(rfn(ri).to_numpy(), np.float64))
    s = np.squeeze(sitk.GetArrayFromImage(sfn(si)).astype(np.float64))
    assert np.array_equal(r, s), f"{tag}: differs from sitk"


# Opening/closing-by-reconstruction grayscale morphology (<Filter>.yaml). Box SE,
# bit-exact to SimpleITK on the upstream cthead1 image.
_RECON_OPEN_CLOSE_CMAKE = [
    ("OpeningByReconstruction/OpeningByReconstruction",
     lambda i, r: ritk.filter.opening_by_reconstruction(i, r),
     lambda i, r: sitk.OpeningByReconstruction(i, [r, r], sitk.sitkBox)),
    ("ClosingByReconstruction/ClosingByReconstruction",
     lambda i, r: ritk.filter.closing_by_reconstruction(i, r),
     lambda i, r: sitk.ClosingByReconstruction(i, [r, r], sitk.sitkBox)),
]


@pytest.mark.parametrize("radius", [2, 3], ids=["r2", "r3"])
@pytest.mark.parametrize("tag,rfn,sfn", _RECON_OPEN_CLOSE_CMAKE,
                         ids=[c[0] for c in _RECON_OPEN_CLOSE_CMAKE])
def test_cmake_recon_open_close_on_upstream_data(tag, rfn, sfn, radius):
    ri, si = _pair("cthead1.png")
    si = sitk.Cast(si, sitk.sitkFloat32)
    r = np.squeeze(np.asarray(rfn(ri, radius).to_numpy(), np.float64))
    s = np.squeeze(sitk.GetArrayFromImage(sfn(si, radius)).astype(np.float64))
    assert np.array_equal(r, s), f"{tag} (r={radius}): differs from sitk"


# Grayscale fill-hole / grind-peak (no SE) — bit-exact to sitk on cthead1.
@pytest.mark.parametrize("tag,rfn,sfn", [
    ("GrayscaleFillhole/GrayscaleFillhole", lambda i: ritk.filter.grayscale_fillhole(i),
     lambda i: sitk.GrayscaleFillhole(i)),
    ("GrayscaleGrindPeak/GrayscaleGrindPeak", lambda i: ritk.filter.grayscale_grind_peak(i),
     lambda i: sitk.GrayscaleGrindPeak(i)),
], ids=["GrayscaleFillhole", "GrayscaleGrindPeak"])
def test_cmake_fillhole_grindpeak_on_upstream_data(tag, rfn, sfn):
    ri, si = _pair("cthead1.png")
    si = sitk.Cast(si, sitk.sitkFloat32)
    r = np.squeeze(np.asarray(rfn(ri).to_numpy(), np.float64))
    s = np.squeeze(sitk.GetArrayFromImage(sfn(si)).astype(np.float64))
    assert np.array_equal(r, s), f"{tag}: differs from sitk"


# Grayscale morphological opening/closing (box SE) — bit-exact to sitk on cthead1.
@pytest.mark.parametrize("radius", [2, 3], ids=["r2", "r3"])
@pytest.mark.parametrize("tag,rfn,sfn", [
    ("GrayscaleMorphologicalClosing", lambda i, r: ritk.filter.grayscale_closing(i, r),
     lambda i, r: sitk.GrayscaleMorphologicalClosing(i, [r, r], sitk.sitkBox)),
    ("GrayscaleMorphologicalOpening", lambda i, r: ritk.filter.grayscale_opening(i, r),
     lambda i, r: sitk.GrayscaleMorphologicalOpening(i, [r, r], sitk.sitkBox)),
], ids=["GrayscaleClosing", "GrayscaleOpening"])
def test_cmake_grayscale_open_close_on_upstream_data(tag, rfn, sfn, radius):
    ri, si = _pair("cthead1.png")
    si = sitk.Cast(si, sitk.sitkFloat32)
    r = np.squeeze(np.asarray(rfn(ri, radius).to_numpy(), np.float64))
    s = np.squeeze(sitk.GetArrayFromImage(sfn(si, radius)).astype(np.float64))
    assert np.array_equal(r, s), f"{tag} (r={radius}): differs from sitk"


# Geometric transforms (FlipImageFilter, {Constant,Mirror,Wrap}PadImageFilter,
# RegionOfInterestImageFilter, PermuteAxesImageFilter). ritk uses tensor-axis
# order [z,y,x]; sitk uses [x,y,z], so size/index/axis tuples reverse. All
# bit-exact to SimpleITK on the upstream cthead1 image.
def _eq(r, s):
    ra = np.squeeze(np.asarray(r.to_numpy(), np.float64))
    sa = np.squeeze(sitk.GetArrayFromImage(s).astype(np.float64))
    return ra.shape == sa.shape and np.array_equal(ra, sa)


@pytest.mark.parametrize("tag,rfn,sfn", [
    ("Flip/x", lambda i: ritk.filter.flip(i, False, False, True),
     lambda i: sitk.Flip(i, [True, False, False])),
    ("Flip/y", lambda i: ritk.filter.flip(i, False, True, False),
     lambda i: sitk.Flip(i, [False, True, False])),
    ("Flip/xy", lambda i: ritk.filter.flip(i, False, True, True),
     lambda i: sitk.Flip(i, [True, True, False])),
    # pads: ritk (z,y,x) lower/upper -> sitk [x,y,z]
    ("ConstantPad", lambda i: ritk.filter.constant_pad(i, (0, 3, 5), (0, 2, 4), 7.0),
     lambda i: sitk.ConstantPad(i, [5, 3, 0], [4, 2, 0], 7.0)),
    ("MirrorPad", lambda i: ritk.filter.mirror_pad(i, (0, 3, 5), (0, 2, 4)),
     lambda i: sitk.MirrorPad(i, [5, 3, 0], [4, 2, 0])),
    ("WrapPad", lambda i: ritk.filter.wrap_pad(i, (0, 3, 5), (0, 2, 4)),
     lambda i: sitk.WrapPad(i, [5, 3, 0], [4, 2, 0])),
    # ROI: ritk start (z,y,x)=(0,10,20) size (1,40,50) -> sitk size [50,40,1] index [20,10,0]
    ("RegionOfInterest", lambda i: ritk.filter.region_of_interest(i, (0, 10, 20), (1, 40, 50)),
     lambda i: sitk.RegionOfInterest(i, [50, 40, 1], [20, 10, 0])),
    # Permute: ritk tensor order (0,2,1) swaps y,x <-> sitk PermuteAxes [1,0,2]
    ("PermuteAxes", lambda i: ritk.filter.permute_axes(i, (0, 2, 1)),
     lambda i: sitk.PermuteAxes(i, [1, 0, 2])),
], ids=["Flip-x", "Flip-y", "Flip-xy", "ConstantPad", "MirrorPad", "WrapPad",
        "RegionOfInterest", "PermuteAxes"])
def test_cmake_geometry_on_upstream_data(tag, rfn, sfn):
    ri, si = _pair("cthead1.png")
    si = sitk.Cast(si, sitk.sitkFloat32)
    assert _eq(rfn(ri), sfn(si)), f"{tag}: differs from sitk"


def test_cmake_paste_on_upstream_data():
    # Paste a cropped 40×50 region back into the image at (z,y,x)=(0,60,70).
    # ITK Parity: PasteImageFilter (sitk.Paste, destination index [x,y,z]).
    ri, si = _pair("cthead1.png")
    si = sitk.Cast(si, sitk.sitkFloat32)
    rsrc = ritk.filter.region_of_interest(ri, (0, 10, 20), (1, 40, 50))
    ssrc = sitk.RegionOfInterest(si, [50, 40, 1], [20, 10, 0])
    r = ritk.filter.paste(ri, rsrc, (0, 60, 70))
    s = sitk.Paste(si, ssrc, [50, 40, 1], [0, 0, 0], [70, 60, 0])
    assert _eq(r, s), "Paste: differs from sitk"
