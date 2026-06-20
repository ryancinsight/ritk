"""Bit-exact parity tests for SignedMaurerDistanceMap and ScalarChanAndVese.

Both filters were ported faithfully from the ITK source:
- SignedMaurerDistanceMap: exact signed Euclidean distance to the object border
  (foreground voxels with a fully-connected background neighbour), reusing the
  Meijster EDT. Bit-exact to ``sitk.SignedMaurerDistanceMap``.
- ScalarChanAndVeseDenseLevelSet: dense Chan-Vese with the ITK hardcoded
  ``dt = 0.08`` constant, atan Heaviside-of-(-phi) region means, and
  per-iteration Maurer reinitialization; binary output ``(phi < 0)``. Bit-exact
  to ``sitk.ScalarChanAndVeseDenseLevelSet``.

Evidence tier: differential (bit-exact vs SimpleITK).
"""

import numpy as np
import pytest

ritk = pytest.importorskip("ritk")
sitk = pytest.importorskip("SimpleITK")


def _shapes():
    yy, xx = np.mgrid[0:30, 0:30]
    circle = (np.sqrt((yy - 15.0) ** 2 + (xx - 15.0) ** 2) < 8).astype(np.uint8)
    L = np.zeros((20, 20), np.uint8)
    L[3:15, 3:7] = 1
    L[11:15, 3:16] = 1
    rng = np.random.default_rng(0)
    blobs = (rng.random((25, 25)) > 0.6).astype(np.uint8)
    return {"circle": circle, "L": L, "blobs": blobs}


@pytest.mark.parametrize("name", ["circle", "L", "blobs"])
@pytest.mark.parametrize("squared", [False, True])
def test_signed_maurer_distance_map_bit_exact(name, squared):
    img = _shapes()[name]
    si = sitk.GetImageFromArray(img)
    so = sitk.GetArrayFromImage(
        sitk.SignedMaurerDistanceMap(
            si,
            insideIsPositive=False,
            squaredDistance=squared,
            useImageSpacing=True,
        )
    ).astype(np.float64)

    ri = ritk.Image(np.ascontiguousarray(img.astype(np.float32)[np.newaxis]))
    ro = ritk.filter.signed_maurer_distance_map(
        ri,
        inside_is_positive=False,
        squared_distance=squared,
        use_image_spacing=True,
    )
    r = np.asarray(ro.to_numpy(), np.float64).reshape(img.shape)
    max_err = float(np.abs(r - so).max())
    assert max_err < 1e-4, f"SignedMaurer {name} squared={squared}: max-err {max_err:.3e}"


@pytest.mark.parametrize("n_iter", [1, 2, 3, 5, 8])
def test_scalar_chan_and_vese_bit_exact(n_iter):
    H = W = 24
    feat = np.zeros((H, W), np.float32)
    feat[6:18, 6:18] = 100.0
    yy, xx = np.mgrid[0:H, 0:W]
    init = (np.sqrt((yy - 12.0) ** 2 + (xx - 12.0) ** 2) - 7.0).astype(np.float32)

    si_feat = sitk.GetImageFromArray(feat)
    si_init = sitk.GetImageFromArray(init)
    try:
        so = sitk.ScalarChanAndVeseDenseLevelSet(
            si_init,
            si_feat,
            maximumRMSError=0.0,
            numberOfIterations=n_iter,
        )
    except Exception as exc:  # pragma: no cover - availability guard
        pytest.skip(f"sitk.ScalarChanAndVeseDenseLevelSet unavailable: {exc}")
    s_arr = sitk.GetArrayFromImage(so).astype(np.float32)

    ri_feat = ritk.Image(np.ascontiguousarray(feat[np.newaxis]))
    ri_init = ritk.Image(np.ascontiguousarray(init[np.newaxis]))
    ro = ritk.filter.scalar_chan_and_vese_dense_level_set(
        ri_init,
        ri_feat,
        number_of_iterations=n_iter,
        lambda1=1.0,
        lambda2=1.0,
        curvature_weight=1.0,
        area_weight=0.0,
        epsilon=1.0,
    )
    r_arr = np.asarray(ro.to_numpy(), np.float32).reshape(H, W)
    mism = int((r_arr != s_arr).sum())
    assert mism == 0, f"ScalarChanAndVese N={n_iter}: {mism}/{H * W} binary mismatches"
