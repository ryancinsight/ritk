"""Real value-semantic differential tests vs sitk for the filters that are bound
but do not yet match SimpleITK.

The cmake-coverage generator counts a filter "covered" when ``sitk.<Name>``
appears in any test, so the existence-only structural tests elsewhere mark these
as covered while their output does **not** match sitk. These tests assert actual
value parity. The ones currently wrong are ``xfail(strict=False)`` with the
measured discrepancy, so:

* they document the real gap in the suite (not just in a doc), and
* they flip to **xpass** automatically when the underlying port is corrected,
  signalling the fix without anyone re-checking by hand.

See ``SITK_BLOCKED_FILTERS_REFERENCE.md`` for the correct algorithm per filter.
"""

import numpy as np
import pytest

import ritk

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover
    pytest.skip("SimpleITK not available", allow_module_level=True)


def _sq(o):
    return np.squeeze(np.asarray(o.to_numpy()))


def test_contour_extractor_2d_matches_sitk():
    """CORRECT: iso-contour vertices are set-equal to sitk's GetContour."""
    img = np.zeros((12, 12), np.float32)
    img[3:9, 3:9] = 100.0
    f = sitk.ContourExtractor2DImageFilter()
    f.SetContourValue(50.0)
    f.Execute(sitk.GetImageFromArray(img))
    sv = set()
    for i in range(f.GetNumberOfCountours()):
        c = f.GetContour(i)
        for k in range(0, len(c), 2):
            sv.add((round(c[k], 3), round(c[k + 1], 3)))
    contours = ritk.filter.contour_extractor_2d(
        ritk.Image(np.ascontiguousarray(img[None])), 50.0
    )
    rv = {(round(x, 3), round(y, 3)) for c in contours for (y, x) in c}
    assert sv == rv


@pytest.mark.xfail(
    reason="ritk uses threshold-connectivity, not ITK's hierarchical watershed "
    "Segmenter on the gradient; 0.0 label match. Needs itk::watershed::Segmenter.",
    strict=False,
)
def test_isolated_watershed_matches_sitk():
    img = np.array(
        [[1, 1, 2, 5, 2, 1, 1], [1, 1, 2, 5, 2, 1, 1], [2, 2, 3, 6, 3, 2, 2],
         [5, 5, 6, 9, 6, 5, 5], [2, 2, 3, 6, 3, 2, 2], [1, 1, 2, 5, 2, 1, 1],
         [1, 1, 2, 5, 2, 1, 1]], np.float32)
    ref = sitk.GetArrayFromImage(
        sitk.IsolatedWatershed(sitk.GetImageFromArray(img), seed1=[1, 3], seed2=[5, 3],
                               threshold=0.0, upperValueLimit=1.0,
                               isolatedValueTolerance=0.001, replaceValue1=1,
                               replaceValue2=2)).astype(int)
    out = ritk.segmentation.isolated_watershed_segment(
        ritk.Image(np.ascontiguousarray(img[None])), [0, 3, 1], [0, 3, 5])
    assert (_sq(out).astype(int) == ref).mean() > 0.999


@pytest.mark.xfail(
    reason="ritk does not reproduce the Awate-Whitaker entropy update + seeded "
    "GaussianRandomSpatialNeighborSubsampler; 25.1 max abs error.",
    strict=False,
)
def test_patch_based_denoising_matches_sitk():
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    np.random.seed(1)
    im = (np.random.rand(12, 12) * 60 + 40).astype(np.float32)
    ref = sitk.GetArrayFromImage(
        sitk.PatchBasedDenoising(sitk.GetImageFromArray(im), 400.0, 2, 1, 200, 400.0))
    out = ritk.filter.patch_based_denoising(ritk.Image(np.ascontiguousarray(im[None])))
    assert float(np.abs(_sq(out) - ref).max()) < 1e-3


@pytest.mark.xfail(
    reason="ritk does not reproduce the dense multiphase level-set evolution "
    "(SharedData means + adaptive dt + RMS stop); 0.19 segmentation match.",
    strict=False,
)
def test_scalar_chan_and_vese_matches_sitk():
    feat = np.zeros((24, 24), np.float32)
    feat[8:16, 8:16] = 1.0
    yy, xx = np.mgrid[0:24, 0:24]
    phi = -(np.sqrt((yy - 11.5) ** 2 + (xx - 11.5) ** 2) - 6.0).astype(np.float32)
    ref = sitk.GetArrayFromImage(
        sitk.ScalarChanAndVeseDenseLevelSet(
            sitk.GetImageFromArray(phi), sitk.GetImageFromArray(feat),
            numberOfIterations=100))
    out = ritk.filter.scalar_chan_and_vese_dense_level_set(
        ritk.Image(np.ascontiguousarray(phi[None])),
        ritk.Image(np.ascontiguousarray(feat[None])))
    assert (((_sq(out) > 0.5) == (ref > 0.5)).mean()) > 0.99


@pytest.mark.xfail(
    reason="ritk output is inverted ±1 (no curvature-flow smoothing) vs sitk's "
    "±3 level set; corr -0.90. Needs SparseFieldLevelSet evolution.",
    strict=False,
)
def test_anti_alias_binary_matches_sitk():
    b = np.zeros((24, 24), np.int16)
    b[6:18, 6:18] = 1
    ref = sitk.GetArrayFromImage(sitk.AntiAliasBinary(sitk.GetImageFromArray(b), 0.07, 50))
    out = ritk.filter.anti_alias_binary(
        ritk.Image(np.ascontiguousarray(b[None].astype(np.float32))), 0.07, 50)
    assert float(np.abs(_sq(out) - ref).max()) < 1e-2
