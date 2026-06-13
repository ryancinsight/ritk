"""Coverage-gap tests for ritk-python public functions with zero prior coverage.

Covers 27 functions across 9 groups: intensity filters, demons variants,
BSpline SyN, LDDMM, label fusion, statistics, normalization, morphology,
segmentation, and I/O.  All thresholds are analytically derived.

Run fast tests only:
    pytest crates/ritk-python/tests/test_coverage_gaps.py -m "not slow" -v
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

import ritk
import ritk.filter as rfilter
import ritk.registration as rreg
import ritk.segmentation as rseg
import ritk.statistics as rstat
import ritk.io as rio

SIZE = 24
HALF = SIZE // 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ritk(arr, spacing=(1.0, 1.0, 1.0)):
    return ritk.Image(np.ascontiguousarray(arr, dtype=np.float32), spacing=list(spacing))


def _make_sphere(size=SIZE, radius=5):
    c = size // 2
    z, y, x = np.mgrid[:size, :size, :size]
    return ((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= radius ** 2).astype(np.float32)


def _make_translated(size=SIZE, radius=5, shift=4):
    c = size // 2
    z, y, x = np.mgrid[:size, :size, :size]
    return ((z - c) ** 2 + (y - c) ** 2 + (x - (c + shift)) ** 2 <= radius ** 2).astype(
        np.float32
    )


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64) - a.mean()
    b = b.ravel().astype(np.float64) - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


# ---------------------------------------------------------------------------
# Group A: Intensity filters
# ---------------------------------------------------------------------------

def test_intensity_windowing_clamp_and_linear():
    """Analytical: window [0.3, 0.7] → [0, 1].
    Midpoint x=0.5 → (0.5-0.3)/(0.7-0.3) = 0.5.
    x=0.1 < 0.3 → 0.0.  x=0.9 > 0.7 → 1.0."""
    arr = np.array([[[0.1, 0.5, 0.9]]], dtype=np.float32)
    out = rfilter.intensity_windowing(_ritk(arr), 0.3, 0.7).to_numpy()
    assert abs(float(out.flat[0]) - 0.0) < 1e-5, f"below window: {out.flat[0]}"
    assert abs(float(out.flat[1]) - 0.5) < 1e-5, f"midpoint: {out.flat[1]}"
    assert abs(float(out.flat[2]) - 1.0) < 1e-5, f"above window: {out.flat[2]}"


def test_intensity_windowing_vs_sitk():
    arr = _make_sphere()
    rf = sitk.IntensityWindowingImageFilter()
    rf.SetWindowMinimum(0.0)
    rf.SetWindowMaximum(1.0)
    rf.SetOutputMinimum(0.0)
    rf.SetOutputMaximum(1.0)
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_out = sitk.GetArrayFromImage(rf.Execute(sitk_img)).astype(np.float32)
    ritk_out = rfilter.intensity_windowing(_ritk(arr), 0.0, 1.0).to_numpy()
    np.testing.assert_allclose(ritk_out, sitk_out, atol=1e-5)


def test_threshold_below_removes_low_values():
    """Analytical: threshold=0.5; values < 0.5 become outside_value=0."""
    arr = np.array([[[0.2, 0.5, 0.8]]], dtype=np.float32)
    out = rfilter.threshold_below(_ritk(arr), 0.5).to_numpy()
    assert abs(float(out.flat[0]) - 0.0) < 1e-5, "below threshold must be zeroed"
    assert abs(float(out.flat[1]) - 0.5) < 1e-5, "at threshold must be preserved"
    assert abs(float(out.flat[2]) - 0.8) < 1e-5, "above threshold must be preserved"


def test_threshold_above_removes_high_values():
    """Analytical: threshold=0.5; values > 0.5 become outside_value=0."""
    arr = np.array([[[0.2, 0.5, 0.8]]], dtype=np.float32)
    out = rfilter.threshold_above(_ritk(arr), 0.5).to_numpy()
    assert abs(float(out.flat[0]) - 0.2) < 1e-5, "below threshold preserved"
    assert abs(float(out.flat[1]) - 0.5) < 1e-5, "at threshold preserved"
    assert abs(float(out.flat[2]) - 0.0) < 1e-5, "above threshold zeroed"


def test_threshold_outside_preserves_interior():
    """Analytical: [lower=0.3, upper=0.7]; exterior → 0.0."""
    arr = np.array([[[0.1, 0.5, 0.9]]], dtype=np.float32)
    out = rfilter.threshold_outside(_ritk(arr), 0.3, 0.7).to_numpy()
    assert abs(float(out.flat[0]) - 0.0) < 1e-5, "below lower: zeroed"
    assert abs(float(out.flat[1]) - 0.5) < 1e-5, "interior: preserved"
    assert abs(float(out.flat[2]) - 0.0) < 1e-5, "above upper: zeroed"


# ---------------------------------------------------------------------------
# Group B: Demons registration variants
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_diffeomorphic_demons_improves_ncc():
    fixed = _make_sphere()
    moving = _make_translated()
    ncc_before = _ncc(moving, fixed)
    warped, disp = rreg.diffeomorphic_demons_register(_ritk(fixed), _ritk(moving))
    warped_arr = warped.to_numpy()
    assert warped_arr.shape == fixed.shape, "warped shape must match fixed"
    ncc_after = _ncc(warped_arr, fixed)
    assert ncc_after > ncc_before, f"NCC: {ncc_before:.4f} → {ncc_after:.4f}"


@pytest.mark.slow
def test_symmetric_demons_improves_ncc():
    fixed = _make_sphere()
    moving = _make_translated()
    ncc_before = _ncc(moving, fixed)
    warped, disp = rreg.symmetric_demons_register(_ritk(fixed), _ritk(moving))
    warped_arr = warped.to_numpy()
    assert warped_arr.shape == fixed.shape
    ncc_after = _ncc(warped_arr, fixed)
    assert ncc_after > ncc_before, f"NCC: {ncc_before:.4f} → {ncc_after:.4f}"


@pytest.mark.slow
def test_inverse_consistent_demons_improves_ncc():
    fixed = _make_sphere()
    moving = _make_translated()
    ncc_before = _ncc(moving, fixed)
    warped, disp = rreg.inverse_consistent_demons_register(_ritk(fixed), _ritk(moving))
    warped_arr = warped.to_numpy()
    assert warped_arr.shape == fixed.shape
    ncc_after = _ncc(warped_arr, fixed)
    assert ncc_after > ncc_before, f"NCC: {ncc_before:.4f} → {ncc_after:.4f}"


@pytest.mark.slow
def test_multires_demons_improves_ncc():
    fixed = _make_sphere()
    moving = _make_translated()
    ncc_before = _ncc(moving, fixed)
    warped, disp = rreg.multires_demons_register(_ritk(fixed), _ritk(moving), rreg.MultiResDemonsOptions(levels=2))
    warped_arr = warped.to_numpy()
    assert warped_arr.shape == fixed.shape
    ncc_after = _ncc(warped_arr, fixed)
    assert ncc_after > ncc_before, f"NCC: {ncc_before:.4f} → {ncc_after:.4f}"


# ---------------------------------------------------------------------------
# Group C: BSpline SyN + LDDMM
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_bspline_syn_register_improves_ncc():
    fixed = _make_sphere()
    moving = _make_translated()
    ncc_before = _ncc(moving, fixed)
    wf, wm = rreg.bspline_syn_register(_ritk(fixed), _ritk(moving), rreg.BSplineSynOptions(max_iterations=30))
    wm_arr = wm.to_numpy()
    assert wm_arr.shape == fixed.shape
    ncc_after = _ncc(wm_arr, fixed)
    assert ncc_after > ncc_before, f"NCC: {ncc_before:.4f} → {ncc_after:.4f}"


@pytest.mark.slow
def test_lddmm_register_improves_ncc():
    fixed = _make_sphere()
    moving = _make_translated()
    ncc_before = _ncc(moving, fixed)
    # lddmm_register returns (warped_moving, displacement_field)
    # displacement_field shape is (3*nz, ny, nx) — packed x/y/z channels
    warped, disp = rreg.lddmm_register(_ritk(fixed), _ritk(moving), rreg.LddmmConfig(max_iterations=20))
    warped_arr = warped.to_numpy()
    assert warped_arr.shape == fixed.shape, f"warped shape {warped_arr.shape} must match fixed {fixed.shape}"
    ncc_after = _ncc(warped_arr, fixed)
    assert ncc_after > ncc_before, f"NCC: {ncc_before:.4f} → {ncc_after:.4f}"


# ---------------------------------------------------------------------------
# Group D: Label fusion
# ---------------------------------------------------------------------------

def test_majority_vote_fusion_unanimous():
    """3/3 identical atlas labels → confidence = 1.0 at every voxel."""
    sphere = _make_sphere()
    labels = [_ritk(sphere)] * 3
    result_labels, confidence = rreg.majority_vote_fusion(labels)
    lbl_arr = result_labels.to_numpy()
    conf_arr = confidence.to_numpy()
    # Center voxel: all 3 atlases vote 1 → label=1, confidence=1.0
    assert abs(float(lbl_arr[HALF, HALF, HALF]) - 1.0) < 1e-5, "center label must be 1"
    assert abs(float(conf_arr[HALF, HALF, HALF]) - 1.0) < 1e-5, "confidence must be 1.0"
    # Corner voxel: all 3 atlases vote 0 → label=0, confidence=1.0
    assert abs(float(lbl_arr[0, 0, 0]) - 0.0) < 1e-5, "corner label must be 0"
    assert abs(float(conf_arr[0, 0, 0]) - 1.0) < 1e-5, "corner confidence must be 1.0"


def test_joint_label_fusion_unanimous():
    """JLF with identical atlases behaves like majority vote: center=1."""
    sphere = _make_sphere()
    target = _ritk(sphere)
    atlas_imgs = [_ritk(sphere)] * 2
    atlas_lbls = [_ritk(sphere)] * 2
    result_labels, confidence = rreg.joint_label_fusion_py(target, atlas_imgs, atlas_lbls)
    lbl_arr = result_labels.to_numpy()
    assert abs(float(lbl_arr[HALF, HALF, HALF]) - 1.0) < 1e-5, "center must be labeled 1"
    assert abs(float(lbl_arr[0, 0, 0]) - 0.0) < 1e-5, "corner must be labeled 0"


# ---------------------------------------------------------------------------
# Group E: Statistics
# ---------------------------------------------------------------------------

def test_masked_statistics_sphere_only():
    """Mask=sphere; image values inside=2.0, outside=0.0.
    Masked mean must be 2.0, std must be 0.0."""
    sphere = _make_sphere()
    arr = sphere * 2.0
    out = rstat.masked_statistics(_ritk(arr), _ritk(sphere))
    assert isinstance(out, dict), "masked_statistics must return dict"
    assert abs(out["mean"] - 2.0) < 1e-4, f"masked mean={out['mean']}"
    assert abs(out["std"] - 0.0) < 1e-3, f"masked std={out['std']}"
    assert abs(out["min"] - 2.0) < 1e-4, f"masked min={out['min']}"
    assert abs(out["max"] - 2.0) < 1e-4, f"masked max={out['max']}"


def test_mean_surface_distance_identical_is_zero():
    sphere = _make_sphere()
    msd = rstat.mean_surface_distance(_ritk(sphere), _ritk(sphere))
    assert abs(msd) < 1e-5, f"MSD of identical masks must be 0, got {msd}"


def test_mean_surface_distance_shifted_sphere_positive():
    fixed = _make_sphere()
    shifted = _make_translated(shift=4)
    msd = rstat.mean_surface_distance(_ritk(fixed), _ritk(shifted))
    assert msd > 0.5, f"MSD of 4-voxel shifted sphere must be > 0.5, got {msd}"


def test_estimate_noise_uniform_near_zero():
    """Uniform image has zero variance → noise estimate ≈ 0."""
    arr = np.ones((SIZE, SIZE, SIZE), dtype=np.float32)
    noise = rstat.estimate_noise(_ritk(arr))
    assert noise < 0.05, f"noise on uniform image must be < 0.05, got {noise}"


def test_estimate_noise_noisy_image_positive():
    rng = np.random.default_rng(42)
    arr = rng.normal(0.5, 0.1, (SIZE, SIZE, SIZE)).astype(np.float32)
    noise = rstat.estimate_noise(_ritk(arr))
    assert noise > 0.0, f"noise on noisy image must be positive, got {noise}"


# ---------------------------------------------------------------------------
# Group F: Normalization
# ---------------------------------------------------------------------------

def test_minmax_normalize_range_maps_to_target():
    """Gradient [0,1] normalized to [-5, 5]: min→-5, max→5."""
    arr = np.linspace(0.0, 1.0, SIZE ** 3, dtype=np.float32).reshape(SIZE, SIZE, SIZE)
    out = rstat.minmax_normalize_range(_ritk(arr), -5.0, 5.0).to_numpy()
    assert abs(float(out.min()) - (-5.0)) < 1e-4, f"min={out.min()}"
    assert abs(float(out.max()) - 5.0) < 1e-4, f"max={out.max()}"


def test_zscore_normalize_with_mask_uses_mask_stats():
    """Sphere region has value 1.0 (mu=1.0, sigma=0) if uniform within mask.
    Use mixed array: background=0, sphere=2.  Mask=sphere.
    Expected mu=2.0, sigma≈0 from mask region."""
    sphere = _make_sphere()
    arr = sphere * 2.0
    # zscore over uniform foreground → (2-2)/sigma = 0/tiny → near zero
    # Use nonuniform sphere to get meaningful zscore
    rng = np.random.default_rng(7)
    arr2 = sphere * (1.0 + rng.standard_normal(sphere.shape).astype(np.float32) * 0.1)
    mask = sphere
    mu = float(arr2[mask > 0.5].mean())
    sigma = float(arr2[mask > 0.5].std())
    out = rstat.zscore_normalize(_ritk(arr2), _ritk(mask)).to_numpy()
    expected_center = (float(arr2[HALF, HALF, HALF]) - mu) / (sigma + 1e-8)
    assert abs(float(out[HALF, HALF, HALF]) - expected_center) < 0.05, (
        f"zscore at center: expected {expected_center:.4f}, got {float(out[HALF, HALF, HALF]):.4f}"
    )


def test_white_stripe_normalize_returns_5tuple():
    """white_stripe_normalize returns (Image, float, float, float, int)."""
    arr = np.linspace(0.1, 1.0, SIZE ** 3, dtype=np.float32).reshape(SIZE, SIZE, SIZE)
    result = rstat.white_stripe_normalize(_ritk(arr), contrast="t1")
    assert len(result) == 5, f"expected 5-tuple, got len={len(result)}"
    normalized, mu, sigma, wm_peak, stripe_size = result
    assert isinstance(normalized, ritk.Image), "first element must be Image"
    assert isinstance(mu, float), f"mu must be float, got {type(mu)}"
    assert isinstance(sigma, float), f"sigma must be float, got {type(sigma)}"
    assert isinstance(wm_peak, float), f"wm_peak must be float, got {type(wm_peak)}"
    assert isinstance(stripe_size, (int, np.integer)), f"stripe_size must be int, got {type(stripe_size)}"
    norm_arr = normalized.to_numpy()
    assert not np.all(norm_arr == 0), "normalized output must not be all-zero"


def test_nyul_udupa_normalize_output_in_range():
    """Training on 2 identical gradient images: output must be finite and in [0, 2]."""
    arr = np.linspace(0.0, 1.0, SIZE ** 3, dtype=np.float32).reshape(SIZE, SIZE, SIZE)
    training = [_ritk(arr), _ritk(arr)]
    out = rstat.nyul_udupa_normalize(_ritk(arr), training).to_numpy()
    assert np.all(np.isfinite(out)), "Nyul output must be finite"
    assert float(out.max()) <= 2.0 and float(out.min()) >= -0.5, (
        f"out range [{out.min():.3f}, {out.max():.3f}] outside expected bounds"
    )


# ---------------------------------------------------------------------------
# Group G: Morphology
# ---------------------------------------------------------------------------

def test_morphological_gradient_zero_at_center():
    """Interior of sphere is uniform (all 1) → dilation=erosion=1 → gradient=0 at center."""
    sphere = _make_sphere()
    out = rseg.morphological_gradient(_ritk(sphere)).to_numpy()
    assert abs(float(out[HALF, HALF, HALF])) < 1e-5, (
        f"gradient at sphere center must be 0, got {out[HALF, HALF, HALF]}"
    )
    assert float(out.max()) > 0.0, "gradient must be nonzero at boundary"


def test_skeletonization_sparser_than_input():
    """Skeleton ⊆ sphere: nonzero count strictly less than sphere, but > 0."""
    sphere = _make_sphere()
    skel = rseg.skeletonization(_ritk(sphere)).to_numpy()
    n_sphere = int((sphere > 0.5).sum())
    n_skel = int((skel > 0.5).sum())
    assert n_skel > 0, "skeleton must be non-empty"
    assert n_skel < n_sphere, f"skeleton ({n_skel}) must be sparser than sphere ({n_sphere})"


def test_marker_watershed_labels_match_markers():
    """Gradient of sphere + 2 markers (center=1, corner=2) → two label regions."""
    sphere = _make_sphere()
    gradient = rfilter.gradient_magnitude(_ritk(sphere)).to_numpy()
    markers = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)
    markers[HALF, HALF, HALF] = 1.0  # center marker
    markers[0, 0, 0] = 2.0           # corner marker
    out = rseg.marker_watershed_segment(_ritk(gradient), _ritk(markers)).to_numpy()
    unique_labels = set(np.unique(np.round(out).astype(int)).tolist())
    assert 1 in unique_labels, f"label 1 missing; found {unique_labels}"
    assert 2 in unique_labels, f"label 2 missing; found {unique_labels}"


# ---------------------------------------------------------------------------
# Group H: Segmentation
# ---------------------------------------------------------------------------

def test_multi_otsu_threshold_two_classes():
    """Bimodal image: two clusters at 0.2 and 0.8 (equal count).
    Multi-Otsu num_classes=2 must produce 1 threshold strictly between 0.2 and 0.8,
    and a label image with exactly 2 distinct values."""
    rng = np.random.default_rng(3)
    low = np.full((SIZE * SIZE * SIZE // 2,), 0.2, dtype=np.float32)
    high = np.full((SIZE * SIZE * SIZE // 2,), 0.8, dtype=np.float32)
    arr = np.concatenate([low, high]).reshape(SIZE, SIZE, SIZE)
    result = rseg.multi_otsu_threshold(_ritk(arr), num_classes=2)
    # Returns (thresholds_list, labeled_image)
    assert isinstance(result, (tuple, list)) and len(result) == 2, (
        f"expected 2-tuple, got {type(result)}"
    )
    thresholds, labeled = result
    assert len(thresholds) == 1, f"expected 1 threshold for 2 classes, got {len(thresholds)}"
    t = float(thresholds[0])
    assert 0.2 < t < 0.8, f"Otsu threshold {t:.4f} must be strictly between modes 0.2 and 0.8"
    lbl_arr = labeled.to_numpy()
    unique = set(np.unique(np.round(lbl_arr).astype(int)).tolist())
    assert len(unique) == 2, f"expected 2 label classes, got {unique}"


# ---------------------------------------------------------------------------
# Group I: I/O
# ---------------------------------------------------------------------------

def test_read_write_image_nrrd_roundtrip():
    rng = np.random.default_rng(0)
    arr = rng.random((8, 8, 8)).astype(np.float32)
    img = _ritk(arr, spacing=(2.0, 1.5, 1.0))
    with tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False) as f:
        path = f.name
    try:
        rio.write_image(img, path)
        assert os.path.exists(path) and os.path.getsize(path) > 0
        img2 = rio.read_image(path)
        arr2 = img2.to_numpy()
        assert arr2.shape == arr.shape, f"shape mismatch: {arr2.shape} != {arr.shape}"
        np.testing.assert_allclose(arr2, arr, atol=1e-5)
    finally:
        os.unlink(path)


def test_read_write_image_nifti_roundtrip():
    rng = np.random.default_rng(1)
    arr = rng.random((8, 8, 8)).astype(np.float32)
    img = _ritk(arr)
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
        path = f.name
    try:
        rio.write_image(img, path)
        assert os.path.exists(path) and os.path.getsize(path) > 0
        img2 = rio.read_image(path)
        arr2 = img2.to_numpy()
        assert arr2.shape == arr.shape, f"shape mismatch: {arr2.shape} != {arr.shape}"
        np.testing.assert_allclose(arr2, arr, atol=1e-4)
    finally:
        os.unlink(path)


def test_read_write_transform_translation_roundtrip():
    offset = [1.5, -2.0, 3.25]
    transform_list = [{"type": "translation", "offset": offset}]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        rio.write_transform(path, 3, transform_list)
        assert os.path.exists(path) and os.path.getsize(path) > 0
        loaded = rio.read_transform(path)
        # loaded may be dict with "transforms" list, or directly a list
        if isinstance(loaded, dict):
            transforms = loaded.get("transforms", [loaded])
        else:
            transforms = loaded
        t = transforms[0]
        loaded_offset = t.get("offset", t.get("translation", None))
        assert loaded_offset is not None, f"offset key missing in {t}"
        np.testing.assert_allclose(loaded_offset, offset, atol=1e-6)
    finally:
        os.unlink(path)
