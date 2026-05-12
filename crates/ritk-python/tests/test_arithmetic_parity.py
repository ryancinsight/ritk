"""Arithmetic image operation parity tests: RITK vs SimpleITK.

Covers blend_images, add_images, subtract_images, multiply_images,
divide_images, minimum_images, maximum_images.

All thresholds are analytically derived.  Each test validates:
  (a) the mathematical contract of the operation on known inputs, and
  (b) numerical equivalence with SimpleITK's counterpart filter.

Run:
    pytest crates/ritk-python/tests/test_arithmetic_parity.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

import ritk
import ritk.filter as rfilter

SIZE = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ritk(arr: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> ritk.Image:
    return ritk.Image(np.ascontiguousarray(arr, dtype=np.float32), spacing=list(spacing))


def _sitk(arr: np.ndarray) -> sitk.Image:
    return sitk.GetImageFromArray(arr.astype(np.float32))


def _from_sitk(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(np.float32)


def _rng_pair(seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.random((SIZE, SIZE, SIZE)).astype(np.float32)
    b = rng.random((SIZE, SIZE, SIZE)).astype(np.float32) + 0.1  # avoid 0 for divide
    return a, b


# ---------------------------------------------------------------------------
# blend_images
# ---------------------------------------------------------------------------

class TestBlendImages:
    def test_alpha_zero_returns_first_image(self):
        """Analytical: (1-0)*A + 0*B = A exactly."""
        a, b = _rng_pair(0)
        out = rfilter.blend_images(_ritk(a), _ritk(b), alpha=0.0).to_numpy()
        np.testing.assert_allclose(out, a, atol=1e-6)

    def test_alpha_one_returns_second_image(self):
        """Analytical: (1-1)*A + 1*B = B exactly."""
        a, b = _rng_pair(1)
        out = rfilter.blend_images(_ritk(a), _ritk(b), alpha=1.0).to_numpy()
        np.testing.assert_allclose(out, b, atol=1e-6)

    def test_alpha_half_is_arithmetic_mean(self):
        """Analytical: 0.5*A + 0.5*B = (A+B)/2."""
        a, b = _rng_pair(2)
        out = rfilter.blend_images(_ritk(a), _ritk(b), alpha=0.5).to_numpy()
        expected = 0.5 * a + 0.5 * b
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_blend_vs_sitk_weighted_sum(self):
        """Numerical parity: (1-α)*A + α*B matches SimpleITK Add(Multiply(A,1-α), Multiply(B,α))."""
        a, b = _rng_pair(3)
        alpha = 0.3
        ritk_out = rfilter.blend_images(_ritk(a), _ritk(b), alpha=alpha).to_numpy()

        sa, sb = _sitk(a), _sitk(b)
        sitk_out = _from_sitk(
            sitk.Add(
                sitk.Multiply(sa, 1.0 - alpha),
                sitk.Multiply(sb, alpha),
            )
        )
        np.testing.assert_allclose(ritk_out, sitk_out, atol=1e-5)

    def test_blend_default_alpha_is_half(self):
        """Default alpha=0.5: result equals (A+B)/2."""
        a, b = _rng_pair(4)
        out = rfilter.blend_images(_ritk(a), _ritk(b)).to_numpy()
        expected = 0.5 * a + 0.5 * b
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_blend_shape_mismatch_raises(self):
        """Shape mismatch between a and b raises RuntimeError."""
        a = _ritk(np.ones((4, 4, 4), dtype=np.float32))
        b = _ritk(np.ones((4, 4, 8), dtype=np.float32))
        with pytest.raises(RuntimeError, match="shape mismatch"):
            rfilter.blend_images(a, b, alpha=0.5)


# ---------------------------------------------------------------------------
# add_images
# ---------------------------------------------------------------------------

class TestAddImages:
    def test_analytical_sum(self):
        """Analytical: [1,2,3] + [4,5,6] = [5,7,9]."""
        a = _ritk(np.array([[[1.0, 2.0, 3.0]]]))
        b = _ritk(np.array([[[4.0, 5.0, 6.0]]]))
        out = rfilter.add_images(a, b).to_numpy()
        np.testing.assert_allclose(out.ravel(), [5.0, 7.0, 9.0], atol=1e-6)

    def test_add_zero_image_is_identity(self):
        """Analytical: A + 0 = A."""
        a, _ = _rng_pair(5)
        z = np.zeros_like(a)
        out = rfilter.add_images(_ritk(a), _ritk(z)).to_numpy()
        np.testing.assert_allclose(out, a, atol=1e-6)

    def test_add_vs_sitk(self):
        """Numerical parity with sitk.Add."""
        a, b = _rng_pair(6)
        ritk_out = rfilter.add_images(_ritk(a), _ritk(b)).to_numpy()
        sitk_out = _from_sitk(sitk.Add(_sitk(a), _sitk(b)))
        np.testing.assert_allclose(ritk_out, sitk_out, atol=1e-5)

    def test_add_commutativity(self):
        """A + B == B + A."""
        a, b = _rng_pair(7)
        ab = rfilter.add_images(_ritk(a), _ritk(b)).to_numpy()
        ba = rfilter.add_images(_ritk(b), _ritk(a)).to_numpy()
        np.testing.assert_allclose(ab, ba, atol=1e-6)


# ---------------------------------------------------------------------------
# subtract_images
# ---------------------------------------------------------------------------

class TestSubtractImages:
    def test_analytical_difference(self):
        """Analytical: [4,5,6] - [1,2,3] = [3,3,3]."""
        a = _ritk(np.array([[[4.0, 5.0, 6.0]]]))
        b = _ritk(np.array([[[1.0, 2.0, 3.0]]]))
        out = rfilter.subtract_images(a, b).to_numpy()
        np.testing.assert_allclose(out.ravel(), [3.0, 3.0, 3.0], atol=1e-6)

    def test_self_subtraction_is_zero(self):
        """Analytical: A - A = 0."""
        a, _ = _rng_pair(8)
        out = rfilter.subtract_images(_ritk(a), _ritk(a)).to_numpy()
        np.testing.assert_allclose(out, np.zeros_like(a), atol=1e-6)

    def test_subtract_vs_sitk(self):
        """Numerical parity with sitk.Subtract."""
        a, b = _rng_pair(9)
        ritk_out = rfilter.subtract_images(_ritk(a), _ritk(b)).to_numpy()
        sitk_out = _from_sitk(sitk.Subtract(_sitk(a), _sitk(b)))
        np.testing.assert_allclose(ritk_out, sitk_out, atol=1e-5)

    def test_subtract_add_inverse(self):
        """(A + B) - B == A."""
        a, b = _rng_pair(10)
        s = rfilter.add_images(_ritk(a), _ritk(b))
        out = rfilter.subtract_images(s, _ritk(b)).to_numpy()
        np.testing.assert_allclose(out, a, atol=1e-5)


# ---------------------------------------------------------------------------
# multiply_images
# ---------------------------------------------------------------------------

class TestMultiplyImages:
    def test_analytical_product(self):
        """Analytical: [2,3,4] * [3,4,5] = [6,12,20]."""
        a = _ritk(np.array([[[2.0, 3.0, 4.0]]]))
        b = _ritk(np.array([[[3.0, 4.0, 5.0]]]))
        out = rfilter.multiply_images(a, b).to_numpy()
        np.testing.assert_allclose(out.ravel(), [6.0, 12.0, 20.0], atol=1e-6)

    def test_multiply_by_ones_is_identity(self):
        """Analytical: A * 1 = A."""
        a, _ = _rng_pair(11)
        ones = np.ones_like(a)
        out = rfilter.multiply_images(_ritk(a), _ritk(ones)).to_numpy()
        np.testing.assert_allclose(out, a, atol=1e-6)

    def test_multiply_by_zero_is_zero(self):
        """Analytical: A * 0 = 0."""
        a, _ = _rng_pair(12)
        z = np.zeros_like(a)
        out = rfilter.multiply_images(_ritk(a), _ritk(z)).to_numpy()
        np.testing.assert_allclose(out, np.zeros_like(a), atol=1e-6)

    def test_multiply_vs_sitk(self):
        """Numerical parity with sitk.Multiply."""
        a, b = _rng_pair(13)
        ritk_out = rfilter.multiply_images(_ritk(a), _ritk(b)).to_numpy()
        sitk_out = _from_sitk(sitk.Multiply(_sitk(a), _sitk(b)))
        np.testing.assert_allclose(ritk_out, sitk_out, atol=1e-5)

    def test_multiply_commutativity(self):
        """A * B == B * A."""
        a, b = _rng_pair(14)
        ab = rfilter.multiply_images(_ritk(a), _ritk(b)).to_numpy()
        ba = rfilter.multiply_images(_ritk(b), _ritk(a)).to_numpy()
        np.testing.assert_allclose(ab, ba, atol=1e-6)


# ---------------------------------------------------------------------------
# divide_images
# ---------------------------------------------------------------------------

class TestDivideImages:
    def test_analytical_quotient(self):
        """Analytical: [10,20,30] / [2,4,5] = [5,5,6]."""
        a = _ritk(np.array([[[10.0, 20.0, 30.0]]]))
        b = _ritk(np.array([[[2.0, 4.0, 5.0]]]))
        out = rfilter.divide_images(a, b).to_numpy()
        np.testing.assert_allclose(out.ravel(), [5.0, 5.0, 6.0], atol=1e-5)

    def test_divide_by_zero_yields_zero(self):
        """Division by zero at a voxel must return 0, not NaN or inf."""
        a = _ritk(np.array([[[1.0, 2.0, 3.0]]]))
        b = _ritk(np.array([[[0.0, 1.0, 0.0]]]))
        out = rfilter.divide_images(a, b).to_numpy()
        assert float(out.flat[0]) == pytest.approx(0.0, abs=1e-6), "div/0 must be 0"
        assert float(out.flat[1]) == pytest.approx(2.0, abs=1e-6)
        assert float(out.flat[2]) == pytest.approx(0.0, abs=1e-6), "div/0 must be 0"

    def test_divide_by_ones_is_identity(self):
        """Analytical: A / 1 = A."""
        a, _ = _rng_pair(15)
        ones = np.ones_like(a)
        out = rfilter.divide_images(_ritk(a), _ritk(ones)).to_numpy()
        np.testing.assert_allclose(out, a, atol=1e-6)

    def test_divide_vs_sitk_no_zeros(self):
        """Numerical parity with sitk.Divide on inputs guaranteed non-zero."""
        a, b = _rng_pair(16)  # b has +0.1 offset so b > 0 everywhere
        ritk_out = rfilter.divide_images(_ritk(a), _ritk(b)).to_numpy()
        sitk_out = _from_sitk(sitk.Divide(_sitk(a), _sitk(b)))
        np.testing.assert_allclose(ritk_out, sitk_out, atol=1e-5)

    def test_multiply_divide_roundtrip(self):
        """(A * B) / B == A when B > 0."""
        a, b = _rng_pair(17)
        ab = rfilter.multiply_images(_ritk(a), _ritk(b))
        out = rfilter.divide_images(ab, _ritk(b)).to_numpy()
        np.testing.assert_allclose(out, a, atol=1e-5)


# ---------------------------------------------------------------------------
# minimum_images / maximum_images
# ---------------------------------------------------------------------------

class TestMinimumMaximumImages:
    def test_minimum_analytical(self):
        """Analytical: min([1,5,3,7], [4,2,6,1]) = [1,2,3,1]."""
        a = _ritk(np.array([[[1.0, 5.0], [3.0, 7.0]]]))
        b = _ritk(np.array([[[4.0, 2.0], [6.0, 1.0]]]))
        out = rfilter.minimum_images(a, b).to_numpy()
        np.testing.assert_allclose(out.ravel(), [1.0, 2.0, 3.0, 1.0], atol=1e-6)

    def test_maximum_analytical(self):
        """Analytical: max([1,5,3,7], [4,2,6,1]) = [4,5,6,7]."""
        a = _ritk(np.array([[[1.0, 5.0], [3.0, 7.0]]]))
        b = _ritk(np.array([[[4.0, 2.0], [6.0, 1.0]]]))
        out = rfilter.maximum_images(a, b).to_numpy()
        np.testing.assert_allclose(out.ravel(), [4.0, 5.0, 6.0, 7.0], atol=1e-6)

    def test_minimum_self_is_identity(self):
        """Analytical: min(A, A) = A."""
        a, _ = _rng_pair(18)
        out = rfilter.minimum_images(_ritk(a), _ritk(a)).to_numpy()
        np.testing.assert_allclose(out, a, atol=1e-6)

    def test_maximum_self_is_identity(self):
        """Analytical: max(A, A) = A."""
        a, _ = _rng_pair(19)
        out = rfilter.maximum_images(_ritk(a), _ritk(a)).to_numpy()
        np.testing.assert_allclose(out, a, atol=1e-6)

    def test_minimum_vs_sitk(self):
        """Numerical parity with sitk.Minimum."""
        a, b = _rng_pair(20)
        ritk_out = rfilter.minimum_images(_ritk(a), _ritk(b)).to_numpy()
        sitk_out = _from_sitk(sitk.Minimum(_sitk(a), _sitk(b)))
        np.testing.assert_allclose(ritk_out, sitk_out, atol=1e-5)

    def test_maximum_vs_sitk(self):
        """Numerical parity with sitk.Maximum."""
        a, b = _rng_pair(21)
        ritk_out = rfilter.maximum_images(_ritk(a), _ritk(b)).to_numpy()
        sitk_out = _from_sitk(sitk.Maximum(_sitk(a), _sitk(b)))
        np.testing.assert_allclose(ritk_out, sitk_out, atol=1e-5)

    def test_min_max_envelope(self):
        """min(A,B) <= max(A,B) element-wise always."""
        a, b = _rng_pair(22)
        mn = rfilter.minimum_images(_ritk(a), _ritk(b)).to_numpy()
        mx = rfilter.maximum_images(_ritk(a), _ritk(b)).to_numpy()
        assert np.all(mn <= mx + 1e-6), "min(A,B) > max(A,B) at some voxel"

    def test_add_equals_min_plus_max(self):
        """A + B == min(A,B) + max(A,B) everywhere."""
        a, b = _rng_pair(23)
        add_out = rfilter.add_images(_ritk(a), _ritk(b)).to_numpy()
        mn = rfilter.minimum_images(_ritk(a), _ritk(b)).to_numpy()
        mx = rfilter.maximum_images(_ritk(a), _ritk(b)).to_numpy()
        np.testing.assert_allclose(add_out, mn + mx, atol=1e-5)
