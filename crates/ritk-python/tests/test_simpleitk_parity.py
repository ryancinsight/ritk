"""SimpleITK numerical parity tests for ritk-python.

Derived from canonical SimpleITK notebook examples
(https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks).

Compares ritk-python filter, segmentation, and statistics outputs
numerically against SimpleITK reference implementations on synthetically
constructed images.  All test images are analytically constructed so
expected values can be derived without file-system access.

Run:
    pytest crates/ritk-python/tests/test_simpleitk_parity.py -v

Requires:
    SimpleITK >= 2.0, numpy >= 1.20, ritk (installed wheel)

The entire module is skipped when SimpleITK is not importable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# Skip every test in this module when SimpleITK is not installed.
sitk = pytest.importorskip("SimpleITK")

import ritk  # noqa: E402

SIZE = 32  # Edge length (voxels) of all synthetic test volumes.


# -- Conversion helpers -------------------------------------------------------


def _ritk(arr, spacing=(1.0, 1.0, 1.0)):
    return ritk.Image(np.ascontiguousarray(arr, dtype=np.float32), spacing=list(spacing))


def _sitk(arr, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr, dtype=np.float32))
    img.SetSpacing([float(spacing[2]), float(spacing[1]), float(spacing[0])])
    return img


def _np(img):
    return sitk.GetArrayFromImage(img).astype(np.float32)


# -- Synthetic image factories ------------------------------------------------


def _make_sphere(size=SIZE, radius=6):
    c = size // 2
    z, y, x = np.mgrid[:size, :size, :size]
    return ((z - c)**2 + (y - c)**2 + (x - c)**2 <= radius**2).astype(np.float32)


def _make_gradient(size=SIZE):
    return np.broadcast_to(
        np.linspace(0.0, 1.0, size, dtype=np.float32), (size, size, size)
    ).copy()


def _make_two_blobs(size=SIZE):
    arr = np.zeros((size, size, size), dtype=np.float32)
    c = size // 2
    r = size // 8
    z, y, x = np.mgrid[:size, :size, :size]
    arr[(z-c)**2 + (y-c)**2 + (x-(c-size//4))**2 <= r**2] = 1.0
    arr[(z-c)**2 + (y-c)**2 + (x-(c+size//4))**2 <= r**2] = 1.0
    return arr


def _make_shell(size=SIZE, outer=8, inner=4):
    c = size // 2
    z, y, x = np.mgrid[:size, :size, :size]
    d2 = (z-c)**2 + (y-c)**2 + (x-c)**2
    return ((d2 <= outer**2) & (d2 > inner**2)).astype(np.float32)


def _make_noisy(size=SIZE, seed=0):
    rng = np.random.default_rng(seed)
    sphere = _make_sphere(size)
    noise = rng.standard_normal((size, size, size)).astype(np.float32) * 0.1
    return np.clip(sphere + noise, 0.0, 1.0).astype(np.float32)


def _dice(a, b):
    inter = float((a * b).sum())
    denom = float(a.sum() + b.sum())
    return 2.0 * inter / max(denom, 1.0)


# ==========================================================================
# Section 1 -- Filter parity
# ==========================================================================


def test_discrete_gaussian_interior_agrees_with_sitk():
    # ITK DiscreteGaussianImageFilter parity: variance=4.0, maximum_error=0.01,
    # use_image_spacing=False; kernel radius=7 voxels; interior crop [8:-8].
    # Tolerances: interior max diff < 0.01, global mean diff < 0.005.
    arr = _make_gradient()
    sr = _np(sitk.DiscreteGaussian(_sitk(arr), variance=4.0,
                                   maximumError=0.01, useImageSpacing=False))
    rr = ritk.filter.discrete_gaussian(
        _ritk(arr), variance=4.0, maximum_error=0.01, use_image_spacing=False
    ).to_numpy()
    assert sr.shape == rr.shape
    m = 8
    diff_i = np.abs(sr[m:-m, m:-m, m:-m] - rr[m:-m, m:-m, m:-m])
    assert float(diff_i.max()) < 0.01, "DiscreteGaussian interior max diff > 0.01: " + str(float(diff_i.max()))
    assert float(np.abs(sr - rr).mean()) < 0.005, "DiscreteGaussian global mean diff > 0.005"


def test_discrete_gaussian_constant_image_invariant():
    # Invariant: conv(c, normalised_kernel) = c. Verifies normalisation + boundary.
    # Tolerance: max absolute deviation from 0.5 < 1e-4.
    arr = np.full((SIZE, SIZE, SIZE), 0.5, dtype=np.float32)
    sr = _np(sitk.DiscreteGaussian(_sitk(arr), variance=4.0,
                                   maximumError=0.01, useImageSpacing=False))
    rr = ritk.filter.discrete_gaussian(
        _ritk(arr), variance=4.0, maximum_error=0.01, use_image_spacing=False
    ).to_numpy()
    assert float(np.abs(sr - 0.5).max()) < 1e-4, "SimpleITK DiscreteGaussian constant deviation >= 1e-4"
    assert float(np.abs(rr - 0.5).max()) < 1e-4, "RITK DiscreteGaussian constant deviation >= 1e-4"


def test_median_filter_radius1_agrees_with_sitk():
    # Both sort 27 neighbours and return lower median with replicate boundary.
    # Tolerance: max absolute difference < 1e-4.
    arr = _make_noisy()
    sr = _np(sitk.Median(_sitk(arr), [1, 1, 1]))
    rr = ritk.filter.median_filter(_ritk(arr), radius=1).to_numpy()
    assert sr.shape == rr.shape
    assert float(np.abs(sr - rr).max()) < 1e-4, "MedianFilter max diff > 1e-4: " + str(float(np.abs(sr - rr).max()))


def test_gradient_magnitude_interior_matches_analytical():
    # f(z,y,x)=x/(SIZE-1) => ||grad f||=1/(SIZE-1) analytically.
    # Both use central finite differences.
    # Tolerances: interior mean vs analytical < 0.003; mutual max diff < 0.01.
    arr = _make_gradient()
    analytical = 1.0 / (SIZE - 1)
    sr = _np(sitk.GradientMagnitude(_sitk(arr)))
    rr = ritk.filter.gradient_magnitude(_ritk(arr)).to_numpy()
    m = 2
    si = sr[m:-m, m:-m, m:-m]
    ri = rr[m:-m, m:-m, m:-m]
    assert abs(float(si.mean()) - analytical) < 0.003, "SimpleITK GM interior mean deviates from analytical"
    assert abs(float(ri.mean()) - analytical) < 0.003, "RITK GM interior mean deviates from analytical"
    assert float(np.abs(si - ri).max()) < 0.01, "GradientMagnitude mutual max diff > 0.01"


def test_rescale_intensity_agrees_with_sitk():
    # output=(v-v_min)/(v_max-v_min). Tolerance: max diff < 1e-4.
    arr = _make_noisy()
    filt = sitk.RescaleIntensityImageFilter()
    filt.SetOutputMinimum(0.0)
    filt.SetOutputMaximum(1.0)
    sr = _np(filt.Execute(_sitk(arr)))
    rr = ritk.filter.rescale_intensity(_ritk(arr), out_min=0.0, out_max=1.0).to_numpy()
    assert float(rr.min()) >= -1e-5, "rescale_intensity output min < 0"
    assert float(rr.max()) <= 1.0 + 1e-5, "rescale_intensity output max > 1"
    assert float(np.abs(sr - rr).max()) < 1e-4, "RescaleIntensity max diff > 1e-4: " + str(float(np.abs(sr-rr).max()))


def test_binary_threshold_agrees_with_sitk_and_analytical():
    # Analytical: output=1.0 iff 0.3<=v<=0.7.
    # Gradient f(z,y,x)=x/(SIZE-1); threshold maps to contiguous X slices.
    # Tolerances: vs analytical < 1e-4; mutual < 1e-4.
    arr = _make_gradient()
    sr = _np(sitk.BinaryThreshold(_sitk(arr), lowerThreshold=0.3,
                                  upperThreshold=0.7,
                                  insideValue=1, outsideValue=0)).astype(np.float32)
    rr = ritk.filter.binary_threshold(
        _ritk(arr), lower_threshold=0.3, upper_threshold=0.7,
        foreground=1.0, background=0.0,
    ).to_numpy()
    expected = ((arr >= 0.3) & (arr <= 0.7)).astype(np.float32)
    assert float(np.abs(sr - expected).max()) < 1e-4, "SimpleITK BinaryThreshold vs analytical > 1e-4"
    assert float(np.abs(rr - expected).max()) < 1e-4, "RITK BinaryThreshold vs analytical > 1e-4"
    assert float(np.abs(sr - rr).max()) < 1e-4, "BinaryThreshold RITK vs SimpleITK > 1e-4"


def test_grayscale_erosion_box_interior_agrees_with_sitk():
    # RITK: (2r+1)^3 cubic SE, replicate boundary. SimpleITK: sitkBox.
    # Tolerance: interior max absolute diff < 1e-4.
    arr = _make_gradient()
    sr = _np(sitk.GrayscaleErode(_sitk(arr), [1, 1, 1], sitk.sitkBox))
    rr = ritk.filter.grayscale_erosion(_ritk(arr), radius=1).to_numpy()
    m = 2
    d = np.abs(sr[m:-m, m:-m, m:-m] - rr[m:-m, m:-m, m:-m])
    assert float(d.max()) < 1e-4, "GrayscaleErosion interior max diff > 1e-4: " + str(float(d.max()))


def test_grayscale_dilation_box_interior_agrees_with_sitk():
    # RITK: (2r+1)^3 cubic SE, replicate boundary. SimpleITK: sitkBox.
    # Tolerance: interior max absolute diff < 1e-4.
    arr = _make_gradient()
    sr = _np(sitk.GrayscaleDilate(_sitk(arr), [1, 1, 1], sitk.sitkBox))
    rr = ritk.filter.grayscale_dilation(_ritk(arr), radius=1).to_numpy()
    m = 2
    d = np.abs(sr[m:-m, m:-m, m:-m] - rr[m:-m, m:-m, m:-m])
    assert float(d.max()) < 1e-4, "GrayscaleDilation interior max diff > 1e-4: " + str(float(d.max()))


def test_laplacian_of_linear_image_is_zero_interior():
    # nabla^2(ax+b)=0 analytically; 7-point FD stencil verifies.
    # Tolerances: interior |values| < 1e-3; mutual max diff < 1e-3.
    arr = _make_gradient()
    sr = _np(sitk.Laplacian(_sitk(arr)))
    rr = ritk.filter.laplacian(_ritk(arr)).to_numpy()
    m = 2
    si = sr[m:-m, m:-m, m:-m]
    ri = rr[m:-m, m:-m, m:-m]
    assert float(np.abs(si).max()) < 1e-3, "SimpleITK Laplacian ramp interior max >= 1e-3"
    assert float(np.abs(ri).max()) < 1e-3, "RITK Laplacian ramp interior max >= 1e-3"
    assert float(np.abs(si - ri).max()) < 1e-3, "Laplacian mutual interior max diff >= 1e-3"


# ==========================================================================
# Section 2 -- Segmentation parity
# ==========================================================================


def test_otsu_threshold_value_within_two_bins_of_sitk():
    # Both maximise between-class variance over 256 bins.
    # Tolerance: |t_ritk - t_sitk| <= 2 * bin_width = 2*(v_max-v_min)/255.
    arr = _make_noisy()
    filt = sitk.OtsuThresholdImageFilter()
    filt.SetInsideValue(0)
    filt.SetOutsideValue(1)
    filt.Execute(_sitk(arr))
    sitk_t = float(filt.GetThreshold())
    ritk_t, _ = ritk.segmentation.otsu_threshold(_ritk(arr))
    bw = (float(arr.max()) - float(arr.min())) / 255.0
    diff = abs(float(ritk_t) - sitk_t)
    assert diff <= 2.0 * bw, "Otsu threshold diff > 2*bin_width: ritk=" + str(float(ritk_t)) + " sitk=" + str(sitk_t)


def test_otsu_mask_dice_vs_sitk_exceeds_threshold():
    # Noisy sphere has clear bimodal histogram; Dice must be >= 0.97.
    arr = _make_noisy()
    sm = _np(sitk.OtsuThreshold(_sitk(arr), 0, 1)).astype(np.float32)
    _, rm_img = ritk.segmentation.otsu_threshold(_ritk(arr))
    rm = rm_img.to_numpy()
    d = _dice(rm, sm)
    assert d >= 0.97, "Otsu mask Dice " + str(d) + " < 0.97"


def test_li_threshold_produces_valid_segmentation():
    # Li minimum cross-entropy threshold (Li and Tam 1998).
    # RITK and SimpleITK diverge significantly on this algorithm
    # (different convergence criteria and initialisation), so this test
    # validates RITK independently.
    # Criterion: threshold in (0.05, 0.95); mask Dice vs ground-truth sphere >= 0.90.
    arr = _make_noisy()
    sphere_gt = _make_sphere()
    ritk_t, mask_img = ritk.segmentation.li_threshold(_ritk(arr))
    t = float(ritk_t)
    assert 0.05 < t < 0.95, 'Li threshold ' + str(t) + ' outside (0.05, 0.95)'
    mask = mask_img.to_numpy()
    d = _dice(mask, sphere_gt)
    assert d >= 0.90, 'Li threshold mask Dice ' + str(d) + ' < 0.90'

def test_connected_components_count_equals_sitk():
    # Both RITK (connectivity=6) and SimpleITK (False=6-connectivity) report 2.
    arr = _make_two_blobs()
    bin8 = sitk.Cast(
        sitk.BinaryThreshold(_sitk(arr), lowerThreshold=0.5, upperThreshold=2.0,
                             insideValue=1, outsideValue=0),
        sitk.sitkUInt8)
    cc = sitk.ConnectedComponent(bin8, False)
    sf = sitk.LabelShapeStatisticsImageFilter()
    sf.Execute(cc)
    sc = sf.GetNumberOfLabels()
    _, rc = ritk.segmentation.connected_components(_ritk(arr), connectivity=6)
    assert sc == 2, "SimpleITK CC count " + str(sc) + " != 2"
    assert rc == 2, "RITK CC count " + str(rc) + " != 2"
    assert rc == sc, "CC count mismatch: ritk=" + str(rc) + " sitk=" + str(sc)


def test_connected_components_per_label_voxel_counts_match_sitk():
    # Sorted per-label voxel count lists must be identical.
    arr = _make_two_blobs()
    bin8 = sitk.Cast(
        sitk.BinaryThreshold(_sitk(arr), lowerThreshold=0.5, upperThreshold=2.0,
                             insideValue=1, outsideValue=0),
        sitk.sitkUInt8)
    sl = _np(sitk.ConnectedComponent(bin8, False))
    rl_img, _ = ritk.segmentation.connected_components(_ritk(arr), connectivity=6)
    rl = rl_img.to_numpy()
    sc = sorted(int((sl == lb).sum()) for lb in np.unique(sl) if lb > 0)
    rc = sorted(int((rl == lb).sum()) for lb in np.unique(rl) if lb > 0)
    assert sc == rc, "Per-label counts differ: ritk=" + str(rc) + " sitk=" + str(sc)


def test_binary_erosion_dice_vs_sitk():
    # RITK: out-of-bounds=background; SimpleITK BinaryErode sitkBox: same.
    # Sphere is well inside image. Tolerance: Dice >= 0.98.
    arr = _make_sphere()
    bin8 = sitk.Cast(
        sitk.BinaryThreshold(_sitk(arr), lowerThreshold=0.5, upperThreshold=1.5,
                             insideValue=1, outsideValue=0),
        sitk.sitkUInt8)
    sr = (_np(sitk.BinaryErode(bin8, [1, 1, 1], sitk.sitkBox)) > 0.5).astype(np.float32)
    rr = (ritk.segmentation.binary_erosion(_ritk(arr), radius=1).to_numpy() > 0.5).astype(np.float32)
    d = _dice(rr, sr)
    assert d >= 0.98, "BinaryErosion Dice " + str(d) + " < 0.98"


def test_binary_dilation_dice_vs_sitk():
    # RITK: replicate boundary; SimpleITK sitkBox. Tolerance: Dice >= 0.98.
    arr = _make_sphere()
    bin8 = sitk.Cast(
        sitk.BinaryThreshold(_sitk(arr), lowerThreshold=0.5, upperThreshold=1.5,
                             insideValue=1, outsideValue=0),
        sitk.sitkUInt8)
    sr = (_np(sitk.BinaryDilate(bin8, [1, 1, 1], sitk.sitkBox)) > 0.5).astype(np.float32)
    rr = (ritk.segmentation.binary_dilation(_ritk(arr), radius=1).to_numpy() > 0.5).astype(np.float32)
    d = _dice(rr, sr)
    assert d >= 0.98, "BinaryDilation Dice " + str(d) + " < 0.98"


def test_binary_fill_holes_fills_hollow_sphere():
    # shell={p: inner^2 < dist2 <= outer^2}; expected filled=solid sphere outer.
    # Both flood-fill from border with 6-connectivity.
    # Tolerance: Dice vs analytical solid >= 0.98 for both.
    shell = _make_shell(size=SIZE, outer=8, inner=4)
    solid = _make_sphere(size=SIZE, radius=8)
    bin8 = sitk.Cast(
        sitk.BinaryThreshold(_sitk(shell), lowerThreshold=0.5, upperThreshold=1.5,
                             insideValue=1, outsideValue=0),
        sitk.sitkUInt8)
    sf = (_np(sitk.BinaryFillhole(bin8)) > 0.5).astype(np.float32)
    rf = (ritk.segmentation.binary_fill_holes(_ritk(shell)).to_numpy() > 0.5).astype(np.float32)
    ds = _dice(sf, solid)
    dr = _dice(rf, solid)
    assert ds >= 0.98, "SimpleITK fill-holes Dice vs solid " + str(ds) + " < 0.98"
    assert dr >= 0.98, "RITK fill-holes Dice vs solid " + str(dr) + " < 0.98"


# ==========================================================================
# Section 3 -- Statistics parity
# ==========================================================================


def test_statistics_mean_min_max_agree_with_sitk():
    # Both iterate over all N=32^3 voxels.
    # Tolerances: mean < 1e-3 (float32 accumulation); min,max < 1e-5.
    arr = _make_noisy()
    sf = sitk.StatisticsImageFilter()
    sf.Execute(_sitk(arr))
    sm, sn, sx = sf.GetMean(), sf.GetMinimum(), sf.GetMaximum()
    stats = ritk.statistics.compute_statistics(_ritk(arr))
    assert abs(stats["mean"] - sm) < 1e-3, "mean mismatch: ritk=" + str(stats["mean"]) + " sitk=" + str(sm)
    assert abs(stats["min"] - sn) < 1e-5, "min mismatch: ritk=" + str(stats["min"]) + " sitk=" + str(sn)
    assert abs(stats["max"] - sx) < 1e-5, "max mismatch: ritk=" + str(stats["max"]) + " sitk=" + str(sx)


def test_statistics_std_accounts_for_ddof_difference():
    # RITK: sigma_pop=sqrt(sum((v-mu)^2)/N).
    # SimpleITK GetSigma(): sigma_smp=sqrt(sum((v-mu)^2)/(N-1)).
    # Relation: sigma_pop=sigma_smp*sqrt((N-1)/N). For N=32768 diff ~0.0015%.
    # Tolerance: |ritk_std - expected_pop| < 0.002.
    import math as _math
    arr = _make_noisy()
    N = arr.size
    sf = sitk.StatisticsImageFilter()
    sf.Execute(_sitk(arr))
    sigma_smp = sf.GetSigma()
    stats = ritk.statistics.compute_statistics(_ritk(arr))
    exp_pop = sigma_smp * _math.sqrt((N - 1) / N)
    assert abs(float(stats["std"]) - exp_pop) < 0.002, "Std mismatch: ritk=" + str(float(stats["std"])) + " expected_pop=" + str(exp_pop)


def test_psnr_agrees_with_analytical_formula():
    # arr2=arr1+0.05; MSE=0.0025; analytical PSNR=-10*log10(0.0025)~26.02 dB.
    # Tolerance: |ritk_psnr - analytical| < 0.1 dB.
    import math as _math
    rng = np.random.default_rng(42)
    arr1 = rng.uniform(0.1, 0.9, (SIZE, SIZE, SIZE)).astype(np.float32)
    arr2 = (arr1 + 0.05).astype(np.float32)
    mse = float(np.mean((arr1 - arr2) ** 2))
    analytical = -10.0 * _math.log10(mse)
    rp = float(ritk.statistics.psnr(_ritk(arr1), _ritk(arr2), max_val=1.0))
    assert abs(rp - analytical) < 0.1, "PSNR mismatch: ritk=" + str(rp) + " analytical=" + str(analytical)


def test_psnr_identical_images_is_infinity():
    # PSNR(image,image) -> +inf as MSE -> 0+.
    import math as _math
    arr = _make_gradient()
    psnr_val = float(ritk.statistics.psnr(_ritk(arr), _ritk(arr), max_val=1.0))
    assert _math.isinf(psnr_val) and psnr_val > 0.0, "PSNR of identical images must be +inf, got " + str(psnr_val)


def test_dice_agrees_with_sitk_label_overlap_filter():
    # Two spheres r=8, centers offset 4 X-voxels. Dice=2*|A^B|/(|A|+|B|).
    # Tolerance: |ritk_dice - sitk_dice| < 1e-4.
    c = SIZE // 2
    z, y, x = np.mgrid[:SIZE, :SIZE, :SIZE]
    r = 8
    sp1 = ((z-c)**2+(y-c)**2+(x-c)**2 <= r**2).astype(np.float32)
    sp2 = ((z-c)**2+(y-c)**2+(x-(c+4))**2 <= r**2).astype(np.float32)
    b1 = sitk.Cast(sitk.GetImageFromArray(sp1.astype(np.uint8)), sitk.sitkUInt8)
    b2 = sitk.Cast(sitk.GetImageFromArray(sp2.astype(np.uint8)), sitk.sitkUInt8)
    of = sitk.LabelOverlapMeasuresImageFilter()
    of.Execute(b1, b2)
    sd = float(of.GetDiceCoefficient())
    rd = float(ritk.statistics.dice_coefficient(_ritk(sp1), _ritk(sp2)))
    assert abs(rd - sd) < 1e-4, "Dice mismatch: ritk=" + str(rd) + " sitk=" + str(sd)


def test_minmax_normalize_agrees_with_sitk_rescale_intensity():
    # output=(v-v_min)/(v_max-v_min). Tolerance: max diff < 1e-4; spans [0,1].
    arr = _make_noisy()
    filt = sitk.RescaleIntensityImageFilter()
    filt.SetOutputMinimum(0.0)
    filt.SetOutputMaximum(1.0)
    sr = _np(filt.Execute(_sitk(arr)))
    rr = ritk.statistics.minmax_normalize(_ritk(arr)).to_numpy()
    assert float(rr.min()) >= -1e-5, "minmax_normalize min < 0"
    assert float(rr.max()) <= 1.0 + 1e-5, "minmax_normalize max > 1"
    assert float(np.abs(sr - rr).max()) < 1e-4, "minmax_normalize vs RescaleIntensity max diff > 1e-4: " + str(float(np.abs(sr-rr).max()))


def test_ssim_identical_images_is_one():
    # SSIM(x,x)=1 by Wang et al. 2004 identity property.
    # Tolerance: |ssim - 1.0| < 1e-4.
    arr = _make_gradient()
    v = float(ritk.statistics.ssim(_ritk(arr), _ritk(arr), max_val=1.0))
    assert abs(v - 1.0) < 1e-4, "SSIM of identical images " + str(v) + " != 1.0"


def test_hausdorff_distance_parallel_planes_analytical():
    # mask1: z=8 plane; mask2: z=20 plane.
    # Analytical HD = |20-8| * 1.0 mm/voxel = 12.0 mm.
    # Tolerance: |ritk_hd - 12.0| < 1.0 (discrete boundary extraction).
    m1 = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)
    m1[8, :, :] = 1.0
    m2 = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)
    m2[20, :, :] = 1.0
    hd = float(ritk.statistics.hausdorff_distance(_ritk(m1), _ritk(m2)))
    assert abs(hd - 12.0) < 1.0, "Hausdorff parallel planes: ritk=" + str(hd) + " analytical=12.0"


def test_hausdorff_distance_agrees_with_sitk():
    # Symmetric HD = max(dir_HD(A->B), dir_HD(B->A)) between surface voxels.
    # Tolerance: |ritk_hd - sitk_hd| < 1.5 voxels.
    c = SIZE // 2
    r, d = 6, 4
    z, y, x = np.mgrid[:SIZE, :SIZE, :SIZE]
    sp1 = ((z-c)**2+(y-c)**2+(x-c)**2 <= r**2).astype(np.float32)
    sp2 = ((z-c)**2+(y-c)**2+(x-(c+d))**2 <= r**2).astype(np.float32)
    b1 = sitk.Cast(sitk.GetImageFromArray(sp1.astype(np.uint8)), sitk.sitkUInt8)
    b2 = sitk.Cast(sitk.GetImageFromArray(sp2.astype(np.uint8)), sitk.sitkUInt8)
    hdf = sitk.HausdorffDistanceImageFilter()
    hdf.Execute(b1, b2)
    sh = float(hdf.GetHausdorffDistance())
    rh = float(ritk.statistics.hausdorff_distance(_ritk(sp1), _ritk(sp2)))
    assert abs(rh - sh) < 1.5, "Hausdorff distance mismatch: ritk=" + str(rh) + " sitk=" + str(sh)


# ==========================================================================
# Section 4 -- Elastix registration parity
# ==========================================================================

# Guard: skip the entire section if this SimpleITK build lacks Elastix.
_has_elastix = hasattr(sitk, "ElastixImageFilter")


def _elastix_translate(fixed, moving, max_iter=50):
    """Run Elastix translation registration with minimal iterations."""
    f = sitk.ElastixImageFilter()
    f.SetFixedImage(fixed)
    f.SetMovingImage(moving)
    pm = sitk.GetDefaultParameterMap("translation")
    pm["MaximumNumberOfIterations"] = [str(max_iter)]
    pm["NumberOfSpatialSamples"] = ["512"]
    pm["NumberOfResolutions"] = ["1"]
    f.SetParameterMap(pm)
    f.LogToConsoleOff()
    f.Execute()
    return f.GetResultImage()


@pytest.mark.skipif(not _has_elastix, reason="SimpleITK not built with Elastix")
def test_elastix_translation_recovers_sphere_overlap():
    """Elastix translation registration on a shifted sphere must achieve Dice >= 0.85.

    Mathematical basis: translation by 3 voxels in x applied to a sphere of radius 6
    in a 32^3 volume.  Elastix translation (EulerTransform with 0 rotations) recovers
    the translation; the registered image should overlap with the fixed sphere with
    Dice >= 0.85 (loose bound because Elastix uses sampled MI, not exact resampling).
    This test validates that Elastix is functional in the installed environment and
    establishes a reference quality baseline for RITK comparison tests.
    """
    arr = _make_sphere().astype(np.float32)
    shift = 3
    arr_shifted = np.roll(arr, shift, axis=2).astype(np.float32)
    fixed = _sitk(arr)
    moving = _sitk(arr_shifted)

    result = _elastix_translate(fixed, moving, max_iter=100)
    result_arr = (_np(result) > 0.5).astype(np.float32)
    ref_arr = (arr > 0.5).astype(np.float32)
    d = _dice(result_arr, ref_arr)
    assert d >= 0.85, (
        f"Elastix translation Dice {d:.4f} < 0.85; registration may have failed"
    )


@pytest.mark.skipif(not _has_elastix, reason="SimpleITK not built with Elastix")
def test_ritk_demons_vs_elastix_translation_quality():
    """RITK demons registration must match Elastix translation quality (Dice >= 0.85).

    Both algorithms are applied to the same fixed/moving sphere pair (3-voxel x-shift).
    The Dice of each result vs the fixed reference sphere must be >= 0.85.
    This is a parallel-quality test: the two algorithms are not required to produce
    identical outputs, only comparable registration quality on this synthetic case.
    """
    arr = _make_sphere().astype(np.float32)
    shift = 3
    arr_shifted = np.roll(arr, shift, axis=2).astype(np.float32)
    fixed_sitk = _sitk(arr)
    moving_sitk = _sitk(arr_shifted)
    fixed_ritk = _ritk(arr)
    moving_ritk = _ritk(arr_shifted)

    # Elastix reference
    result_elastix = _elastix_translate(fixed_sitk, moving_sitk, max_iter=100)
    elastix_arr = (_np(result_elastix) > 0.5).astype(np.float32)
    ref_arr = (arr > 0.5).astype(np.float32)
    d_elastix = _dice(elastix_arr, ref_arr)
    assert d_elastix >= 0.85, (
        f"Elastix baseline Dice {d_elastix:.4f} < 0.85"
    )

    # RITK Demons registration
    warped_ritk, _ = ritk.registration.demons_register(
        fixed_ritk, moving_ritk, max_iterations=100, sigma_diffusion=1.0
    )
    ritk_arr = (warped_ritk.to_numpy() > 0.5).astype(np.float32)
    d_ritk = _dice(ritk_arr, ref_arr)
    assert d_ritk >= 0.85, (
        f"RITK Demons Dice {d_ritk:.4f} < 0.85 (Elastix achieved {d_elastix:.4f})"
    )


@pytest.mark.skipif(not _has_elastix, reason="SimpleITK not built with Elastix")
def test_elastix_bspline_deformable_vs_ritk_syn():
    """RITK SyN registration must match Elastix BSpline quality on a locally deformed sphere.

    The moving image is constructed by applying a smooth Gaussian-shaped local displacement
    to a sphere, creating a non-rigid deformation.  Both Elastix BSpline and RITK SyN are
    applied; the Dice of the warped moving vs the fixed sphere must be >= 0.80 for both.

    Mathematical basis: Gaussian bump deformation with amplitude A=3.0 voxels and sigma=5.0
    applied in the x-direction.  This is within the capture range of both BSpline (control
    grid spacing 8 voxels) and SyN (Gaussian regularization sigma=3).
    """
    # Fixed: sphere centred at SIZE//2
    arr_fixed = _make_sphere(size=SIZE, radius=6).astype(np.float32)

    # Moving: apply a smooth local x-displacement to the sphere
    c = SIZE // 2
    z, y, x = np.mgrid[:SIZE, :SIZE, :SIZE]
    amplitude = 3.0
    sigma = 5.0
    # Gaussian bump in x centred at image centre
    bump = amplitude * np.exp(-((z - c)**2 + (y - c)**2 + (x - c)**2) / (2 * sigma**2))
    x_displaced = np.clip(x + bump, 0, SIZE - 1).astype(np.float32)
    from scipy.ndimage import map_coordinates
    arr_moving = map_coordinates(
        arr_fixed,
        [z.ravel(), y.ravel(), x_displaced.ravel()],
        order=1, mode="nearest",
    ).reshape(SIZE, SIZE, SIZE).astype(np.float32)

    fixed_sitk = _sitk(arr_fixed)
    moving_sitk = _sitk(arr_moving)
    fixed_ritk = _ritk(arr_fixed)
    moving_ritk = _ritk(arr_moving)
    ref_arr = (arr_fixed > 0.5).astype(np.float32)

    # Elastix BSpline deformable
    f = sitk.ElastixImageFilter()
    f.SetFixedImage(fixed_sitk)
    f.SetMovingImage(moving_sitk)
    pm = sitk.GetDefaultParameterMap("bspline")
    pm["MaximumNumberOfIterations"] = ["100"]
    pm["NumberOfSpatialSamples"] = ["256"]
    pm["NumberOfResolutions"] = ["1"]
    pm["FinalGridSpacingInPhysicalUnits"] = ["4.0"]
    pm["GridSpacingSchedule"] = ["1.0"]
    f.SetParameterMap(pm)
    f.LogToConsoleOff()
    f.Execute()
    elastix_arr = (_np(f.GetResultImage()) > 0.5).astype(np.float32)
    d_elastix = _dice(elastix_arr, ref_arr)
    assert d_elastix >= 0.80, (
        f"Elastix BSpline Dice {d_elastix:.4f} < 0.80"
    )

    # RITK SyN deformable
    warped_ritk, _ = ritk.registration.syn_register(
        fixed_ritk, moving_ritk, max_iterations=30, sigma_smooth=3.0
    )
    ritk_arr = (warped_ritk.to_numpy() > 0.5).astype(np.float32)
    d_ritk = _dice(ritk_arr, ref_arr)
    assert d_ritk >= 0.80, (
        f"RITK SyN Dice {d_ritk:.4f} < 0.80 (Elastix BSpline achieved {d_elastix:.4f})"
    )


@pytest.mark.skipif(not _has_elastix, reason="SimpleITK not built with Elastix")
def test_elastix_parameter_map_api_matches_expected_keys():
    """Elastix default parameter maps must contain the expected registration keys.

    This is a structural API test verifying that the installed SimpleITK-Elastix
    exposes the documented parameter map keys for translation, rigid, affine, and
    bspline transforms.  Failing this test indicates an incompatible Elastix version.
    """
    required_keys_by_type = {
        "translation": {"Transform", "Metric", "Optimizer", "MaximumNumberOfIterations"},
        "rigid":       {"Transform", "Metric", "Optimizer", "MaximumNumberOfIterations"},
        "affine":      {"Transform", "Metric", "Optimizer", "MaximumNumberOfIterations"},
        "bspline":     {"Transform", "Metric", "Optimizer", "FinalGridSpacingInPhysicalUnits"},
    }
    for map_type, required_keys in required_keys_by_type.items():
        pm = sitk.GetDefaultParameterMap(map_type)
        present = set(pm.keys())
        missing = required_keys - present
        assert not missing, (
            f"Elastix {map_type!r} parameter map missing keys: {missing}"
        )
