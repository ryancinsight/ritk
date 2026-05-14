"""Sprint 212: ritk.metrics parity tests against NumPy reference implementations.

All expected values are computed analytically or via NumPy (no SimpleITK dependency).
Tests cover: MSE, NCC, MI (mattes/standard/normalized), shape-mismatch guards,
variant-validation guards, and real-world brain MRI monotonicity properties.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

import ritk
from ritk.metrics import (
    compute_entropy,
    compute_joint_entropy,
    compute_mse,
    compute_mutual_information,
    compute_ncc,
    compute_symmetric_uncertainty,
)

# ── helpers ───────────────────────────────────────────────────────────────────

BRAIN_MNI = Path(__file__).parent.parent.parent.parent / "test_data" / "registration" / "brain_mni"
MNI152 = BRAIN_MNI / "mni152.nii.gz"
SUBJ_T1 = BRAIN_MNI / "single_subj_T1.nii.gz"


def make_image(arr: np.ndarray) -> ritk.Image:
    return ritk.Image(arr.astype(np.float32))


def numpy_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def numpy_ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    mean_a, mean_b = a.mean(), b.mean()
    da, db = a - mean_a, b - mean_b
    cov = (da * db).sum()
    std_a = np.sqrt((da ** 2).mean())
    std_b = np.sqrt((db ** 2).mean())
    n = a.size
    return float(cov / (n * (std_a * std_b + 1e-10)))


def numpy_mi_standard(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    joint_hist, _, _ = np.histogram2d(a, b, bins=bins)
    joint_prob = joint_hist / joint_hist.sum()
    hist_a = joint_prob.sum(axis=1)
    hist_b = joint_prob.sum(axis=0)

    def entropy(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    h_a = entropy(hist_a)
    h_b = entropy(hist_b)
    h_ab = entropy(joint_prob.ravel())
    return h_a + h_b - h_ab


# ── MSE parity ────────────────────────────────────────────────────────────────

def test_mse_zero_for_identical_image():
    arr = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    img = make_image(arr)
    result = compute_mse(img, img)
    assert result == pytest.approx(0.0, abs=1e-10), f"MSE(A,A) must be 0, got {result}"


def test_mse_known_pair_matches_numpy():
    # A = uniform [0..1), B = A + 0.5 (clamped); expected MSE = 0.25
    rng = np.random.default_rng(42)
    a = rng.uniform(0.0, 1.0, size=(4, 5, 6)).astype(np.float32)
    b = (a + 0.5).clip(0.0, 2.0).astype(np.float32)
    expected = numpy_mse(a, b)
    result = compute_mse(make_image(a), make_image(b))
    assert result == pytest.approx(expected, rel=1e-5), f"MSE mismatch: ritk={result}, numpy={expected}"


def test_mse_shape_mismatch_raises():
    a = make_image(np.zeros((2, 3, 4), dtype=np.float32))
    b = make_image(np.zeros((2, 3, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_mse(a, b)


def test_mse_symmetric():
    rng = np.random.default_rng(7)
    a = rng.standard_normal((3, 4, 5)).astype(np.float32)
    b = rng.standard_normal((3, 4, 5)).astype(np.float32)
    assert compute_mse(make_image(a), make_image(b)) == pytest.approx(
        compute_mse(make_image(b), make_image(a)), rel=1e-10
    ), "MSE must be symmetric"


# ── NCC parity ────────────────────────────────────────────────────────────────

def test_ncc_identical_image_returns_one():
    arr = np.arange(1, 61, dtype=np.float32).reshape(3, 4, 5)
    img = make_image(arr)
    result = compute_ncc(img, img)
    assert result == pytest.approx(1.0, abs=1e-8), f"NCC(A,A) must be 1.0, got {result}"


def test_ncc_anti_correlated_returns_negative_one():
    n = 60
    a = np.arange(1, n + 1, dtype=np.float32).reshape(3, 4, 5)
    b = np.arange(n, 0, -1, dtype=np.float32).reshape(3, 4, 5)
    result = compute_ncc(make_image(a), make_image(b))
    assert result == pytest.approx(-1.0, abs=1e-8), f"NCC(A, reverse(A)) must be -1.0, got {result}"


def test_ncc_matches_numpy_reference():
    rng = np.random.default_rng(99)
    a = rng.standard_normal((4, 5, 6)).astype(np.float32)
    b = rng.standard_normal((4, 5, 6)).astype(np.float32)
    expected = numpy_ncc(a, b)
    result = compute_ncc(make_image(a), make_image(b))
    assert result == pytest.approx(expected, abs=1e-5), f"NCC mismatch: ritk={result}, numpy={expected}"


def test_ncc_shape_mismatch_raises():
    a = make_image(np.zeros((2, 3, 4), dtype=np.float32))
    b = make_image(np.zeros((2, 3, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_ncc(a, b)


def test_ncc_bounded_between_negative_one_and_one():
    rng = np.random.default_rng(13)
    a = rng.standard_normal((5, 6, 7)).astype(np.float32)
    b = rng.standard_normal((5, 6, 7)).astype(np.float32)
    result = compute_ncc(make_image(a), make_image(b))
    assert -1.0 - 1e-8 <= result <= 1.0 + 1e-8, f"NCC must be in [-1,1], got {result}"


# ── MI parity ─────────────────────────────────────────────────────────────────

def test_mi_standard_matches_numpy_histogram():
    rng = np.random.default_rng(55)
    a = (rng.standard_normal((4, 5, 6)) * 50 + 128).clip(0, 255).astype(np.float32)
    b = (rng.standard_normal((4, 5, 6)) * 50 + 128).clip(0, 255).astype(np.float32)
    expected = numpy_mi_standard(a, b, bins=32)
    result = compute_mutual_information(make_image(a), make_image(b), num_bins=32, variant="standard")
    # Standard MI uses nearest-bin which matches np.histogram2d; allow 5% relative tolerance
    # due to boundary bin treatment differences.
    assert result == pytest.approx(expected, rel=0.05), (
        f"MI standard mismatch: ritk={result}, numpy={expected}"
    )


def test_mi_self_exceeds_noise():
    # MI(A, A) = H(A) ≥ MI(A, A + independent_noise) for any noise.
    rng = np.random.default_rng(77)
    a = (rng.standard_normal((4, 5, 6)) * 30 + 100).astype(np.float32)
    noise = (rng.standard_normal((4, 5, 6)) * 30).astype(np.float32)
    b = (a + noise).astype(np.float32)
    img_a = make_image(a)
    mi_self = compute_mutual_information(img_a, img_a, num_bins=32, variant="mattes")
    mi_noisy = compute_mutual_information(img_a, make_image(b), num_bins=32, variant="mattes")
    assert mi_self > mi_noisy, (
        f"MI(A,A)={mi_self:.4f} must exceed MI(A, A+noise)={mi_noisy:.4f}"
    )


def test_mi_constant_image_is_zero():
    # MI(A, constant) = 0: H(B)=0 and H(A,B)=H(A), so MI = H(A)+0−H(A) = 0.
    a = np.arange(1, 33, dtype=np.float32).reshape(2, 4, 4)
    b = np.full((2, 4, 4), 50.0, dtype=np.float32)
    result = compute_mutual_information(make_image(a), make_image(b), num_bins=16, variant="standard")
    assert abs(result) < 1e-8, f"MI(A,constant) must be 0, got {result}"


def test_mi_normalized_in_unit_interval():
    rng = np.random.default_rng(33)
    a = (rng.standard_normal((4, 5, 6)) * 40 + 100).astype(np.float32)
    b = (rng.standard_normal((4, 5, 6)) * 40 + 100).astype(np.float32)
    result = compute_mutual_information(make_image(a), make_image(b), num_bins=32, variant="normalized")
    assert 0.0 <= result <= 1.0 + 1e-8, f"Normalized MI must be in [0,1], got {result}"


def test_mi_normalized_self_equals_one():
    # NMI(A,A) = 2·MI(A,A)/(H(A)+H(A)) = 2·H(A)/(2·H(A)) = 1 for non-constant A.
    a = np.arange(1, 33, dtype=np.float32).reshape(2, 4, 4)
    img = make_image(a)
    result = compute_mutual_information(img, img, num_bins=16, variant="normalized")
    assert result == pytest.approx(1.0, abs=1e-6), f"NMI(A,A) must be 1.0, got {result}"


def test_mi_unknown_variant_raises():
    a = make_image(np.ones((2, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="unknown variant"):
        compute_mutual_information(a, a, num_bins=16, variant="cosine")


def test_mi_shape_mismatch_raises():
    a = make_image(np.zeros((2, 3, 4), dtype=np.float32))
    b = make_image(np.zeros((2, 3, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_mutual_information(a, b, num_bins=16, variant="mattes")


# ── real-world brain MRI tests ─────────────────────────────────────────────────

@pytest.mark.skipif(not MNI152.exists(), reason="brain_mni test data not available")
def test_mse_same_image_is_zero_on_brain():
    img = ritk.io.read_image(str(MNI152))
    result = compute_mse(img, img)
    assert result == pytest.approx(0.0, abs=1e-8), f"MSE(brain,brain) must be 0, got {result}"


@pytest.mark.skipif(not MNI152.exists(), reason="brain_mni test data not available")
def test_ncc_same_image_is_one_on_brain():
    img = ritk.io.read_image(str(MNI152))
    result = compute_ncc(img, img)
    assert result == pytest.approx(1.0, abs=1e-6), f"NCC(brain,brain) must be 1.0, got {result}"


@pytest.mark.skipif(not MNI152.exists(), reason="brain_mni test data not available")
def test_mi_self_is_positive_on_brain():
    img = ritk.io.read_image(str(MNI152))
    result = compute_mutual_information(img, img, num_bins=64, variant="mattes")
    assert result > 0.0, f"MI(brain,brain) must be positive, got {result}"


@pytest.mark.skipif(
    not MNI152.exists() or not SUBJ_T1.exists(),
    reason="brain_mni test data not available",
)
def test_mi_self_exceeds_cross_subject_on_brain():
    # mni152 and single_subj_T1 have different shapes; crop to common 91×109×91 region.
    mni = ritk.io.read_image(str(MNI152))
    subj = ritk.io.read_image(str(SUBJ_T1))
    sz, sy, sx = subj.shape  # (91, 109, 91) — smaller shape sets the crop
    mni_arr = mni.to_numpy()[:sz, :sy, :sx].astype(np.float32)
    subj_arr = subj.to_numpy()
    assert mni_arr.shape == subj_arr.shape, "cropped shapes must match for comparison"
    img_mni_crop = make_image(mni_arr)
    img_subj = make_image(subj_arr)
    mi_self = compute_mutual_information(img_mni_crop, img_mni_crop, num_bins=64, variant="mattes")
    mi_cross = compute_mutual_information(img_mni_crop, img_subj, num_bins=64, variant="mattes")
    assert mi_self > mi_cross, (
        f"MI(mni,mni)={mi_self:.4f} must exceed MI(mni,subj)={mi_cross:.4f}"
    )


# ── compute_entropy parity ────────────────────────────────────────────────────


def numpy_marginal_entropy(arr: np.ndarray, bins: int = 64) -> float:
    """H(X) = -∑ p·log(p) via histogram binning."""
    a = arr.astype(np.float64).ravel()
    hist, _ = np.histogram(a, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def test_entropy_zero_for_constant_image():
    # H(constant) = 0: all probability mass in one bin.
    arr = np.full((3, 4, 5), 42.0, dtype=np.float32)
    result = compute_entropy(make_image(arr), num_bins=16)
    assert result == pytest.approx(0.0, abs=1e-8), f"H(constant) must be 0, got {result}"


def test_entropy_positive_for_nonconstant_image():
    rng = np.random.default_rng(11)
    arr = rng.standard_normal((4, 5, 6)).astype(np.float32)
    result = compute_entropy(make_image(arr), num_bins=32)
    assert result > 0.0, f"H(random) must be positive, got {result}"


def test_entropy_matches_numpy_reference():
    # Reference: NumPy histogram-based marginal entropy.
    rng = np.random.default_rng(22)
    arr = (rng.standard_normal((4, 5, 6)) * 30 + 100).astype(np.float32)
    expected = numpy_marginal_entropy(arr, bins=32)
    result = compute_entropy(make_image(arr), num_bins=32)
    assert result == pytest.approx(expected, rel=0.05), (
        f"H(X) mismatch: ritk={result:.6f}, numpy={expected:.6f}"
    )


# ── compute_joint_entropy parity ──────────────────────────────────────────────


def numpy_joint_entropy(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """H(X,Y) = -∑ p(x,y)·log(p(x,y)) via 2-D histogram."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    joint_hist, _, _ = np.histogram2d(a, b, bins=bins)
    p = joint_hist / joint_hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def test_joint_entropy_self_equals_marginal_entropy():
    # H(X,X) = H(X): conditioning on itself adds no information.
    rng = np.random.default_rng(33)
    arr = rng.standard_normal((4, 5, 6)).astype(np.float32)
    img = make_image(arr)
    h_x = compute_entropy(img, num_bins=32)
    h_xx = compute_joint_entropy(img, img, num_bins=32)
    assert h_xx == pytest.approx(h_x, rel=0.01), (
        f"H(X,X)={h_xx:.6f} must equal H(X)={h_x:.6f}"
    )


def test_joint_entropy_geq_marginal():
    # H(X,Y) ≥ H(X) and H(X,Y) ≥ H(Y) for independent X, Y.
    rng = np.random.default_rng(44)
    a = rng.standard_normal((4, 5, 6)).astype(np.float32)
    b = rng.standard_normal((4, 5, 6)).astype(np.float32)
    img_a, img_b = make_image(a), make_image(b)
    h_xy = compute_joint_entropy(img_a, img_b, num_bins=32)
    h_x = compute_entropy(img_a, num_bins=32)
    h_y = compute_entropy(img_b, num_bins=32)
    assert h_xy >= h_x - 1e-6, f"H(X,Y)={h_xy:.4f} must be >= H(X)={h_x:.4f}"
    assert h_xy >= h_y - 1e-6, f"H(X,Y)={h_xy:.4f} must be >= H(Y)={h_y:.4f}"


def test_joint_entropy_matches_numpy_reference():
    rng = np.random.default_rng(55)
    a = (rng.standard_normal((4, 5, 6)) * 30 + 100).astype(np.float32)
    b = (rng.standard_normal((4, 5, 6)) * 30 + 100).astype(np.float32)
    expected = numpy_joint_entropy(a, b, bins=32)
    result = compute_joint_entropy(make_image(a), make_image(b), num_bins=32)
    assert result == pytest.approx(expected, rel=0.05), (
        f"H(X,Y) mismatch: ritk={result:.6f}, numpy={expected:.6f}"
    )


# ── compute_symmetric_uncertainty parity ──────────────────────────────────────


def numpy_symmetric_uncertainty(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """SU(X,Y) = 2·MI(X,Y) / (H(X) + H(Y))."""
    h_x = numpy_marginal_entropy(a, bins)
    h_y = numpy_marginal_entropy(b, bins)
    h_xy = numpy_joint_entropy(a, b, bins)
    mi = h_x + h_y - h_xy
    denom = h_x + h_y
    if denom < 1e-12:
        return 0.0
    return float(2.0 * mi / denom)


def test_symmetric_uncertainty_self_is_one():
    # SU(X,X) = 2·MI(X,X)/(H(X)+H(X)) = 2·H(X)/(2·H(X)) = 1 for non-constant X.
    rng = np.random.default_rng(66)
    arr = rng.standard_normal((4, 5, 6)).astype(np.float32)
    img = make_image(arr)
    result = compute_symmetric_uncertainty(img, img, num_bins=32)
    assert result == pytest.approx(1.0, abs=1e-5), (
        f"SU(X,X) must be 1.0, got {result}"
    )


def test_symmetric_uncertainty_in_zero_one():
    # SU ∈ [0, 1] by definition.
    rng = np.random.default_rng(77)
    a = rng.standard_normal((4, 5, 6)).astype(np.float32)
    b = rng.standard_normal((4, 5, 6)).astype(np.float32)
    result = compute_symmetric_uncertainty(make_image(a), make_image(b), num_bins=32)
    assert 0.0 - 1e-8 <= result <= 1.0 + 1e-8, f"SU must be in [0,1], got {result}"


def test_symmetric_uncertainty_matches_numpy_reference():
    rng = np.random.default_rng(88)
    a = (rng.standard_normal((4, 5, 6)) * 30 + 100).astype(np.float32)
    b = (rng.standard_normal((4, 5, 6)) * 30 + 100).astype(np.float32)
    expected = numpy_symmetric_uncertainty(a, b, bins=32)
    result = compute_symmetric_uncertainty(make_image(a), make_image(b), num_bins=32)
    assert result == pytest.approx(expected, rel=0.05), (
        f"SU mismatch: ritk={result:.6f}, numpy={expected:.6f}"
    )
