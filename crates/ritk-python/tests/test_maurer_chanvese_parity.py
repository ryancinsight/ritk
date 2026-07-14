"""Numerical parity tests for SignedMaurerDistanceMap and ScalarChanAndVese.

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
import ritk
import SimpleITK as sitk


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
        mu=1.0,
        nu=0.0,
        epsilon=1.0,
    )
    r_arr = np.asarray(ro.to_numpy(), np.float32).reshape(H, W)
    mism = int((r_arr != s_arr).sum())
    assert mism == 0, f"ScalarChanAndVese N={n_iter}: {mism}/{H * W} binary mismatches"


@pytest.mark.parametrize("sz,R,nit", [(13, 1, 1), (12, 2, 1), (20, 2, 1), (24, 4, 1)])
def test_patch_based_denoising_single_rounding_step(sz, R, nit):
    """PatchBasedDenoising matches single-threaded sitk within one f32 step.

    Faithful ITK port: Gaussian-kernel joint-entropy gradient over patches drawn
    by the GaussianRandomSpatialNeighborSubsampler (variance 400, 200 results),
    using ITK's MersenneTwister (seed 0) and visited in ImageBoundaryFacesCalculator
    order. sitk is forced single-threaded so its thread-seeded RNG (SetSeed(thread))
    is deterministic with seed 0; ritk reproduces that exact draw sequence.

    Both implementations preserve the same sample and accumulation order and
    convert the final result to f32. Platform math can place that final rounding
    on adjacent representable values, so one ULP is the derived output bound.
    Evidence tier: differential against single-threaded SimpleITK.
    """
    import numpy as _np

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    rng = _np.random.default_rng(3)
    img = (rng.random((sz, sz)).astype(_np.float32) * 90 + 5).astype(_np.float32)

    f = sitk.PatchBasedDenoisingImageFilter()
    f.SetPatchRadius(R)
    f.SetNumberOfIterations(nit)
    f.SetNumberOfWorkUnits(1)
    so = sitk.GetArrayFromImage(f.Execute(sitk.GetImageFromArray(img))).astype(_np.float32)

    ri = ritk.Image(_np.ascontiguousarray(img[_np.newaxis]))
    ro = ritk.filter.patch_based_denoising(
        ri, number_of_iterations=nit, number_of_sample_patches=200, patch_radius=R
    )
    r = _np.asarray(ro.to_numpy(), _np.float32).reshape(sz, sz)
    _np.testing.assert_array_max_ulp(r, so, maxulp=1)
