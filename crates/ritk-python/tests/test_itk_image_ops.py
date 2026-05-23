"""Tests for itk.image_ops module."""

from __future__ import annotations

import numpy as np
import pytest
from itk.image_ops import to_canonical


class TestToCanonical:
    """Tests for the to_canonical function."""

    def test_rgb_to_bgr_numpy_2d(self):
        """Test RGB to BGR conversion for 2D numpy array."""
        # Create RGB image: shape (H, W, 3)
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[:, :, 0] = 100  # R
        rgb[:, :, 1] = 150  # G
        rgb[:, :, 2] = 200  # B

        result = to_canonical(rgb)

        # After conversion to BGR: B, G, R
        assert result.shape == (10, 10, 3)
        assert np.all(result[:, :, 0] == 200)  # B
        assert np.all(result[:, :, 1] == 150)  # G
        assert np.all(result[:, :, 2] == 100)  # R

    def test_rgb_to_bgr_numpy_3d(self):
        """Test RGB to BGR conversion for 3D numpy array."""
        # Create RGB image: shape (D, H, W, 3)
        rgb = np.zeros((5, 10, 10, 3), dtype=np.uint8)
        rgb[:, :, :, 0] = 100  # R
        rgb[:, :, :, 1] = 150  # G
        rgb[:, :, :, 2] = 200  # B

        result = to_canonical(rgb)

        # After conversion to BGR: B, G, R
        assert result.shape == (5, 10, 10, 3)
        assert np.all(result[:, :, :, 0] == 200)  # B
        assert np.all(result[:, :, :, 1] == 150)  # G
        assert np.all(result[:, :, :, 2] == 100)  # R

    def test_rgba_to_bgra_numpy(self):
        """Test RGBA to BGRA conversion for numpy array."""
        # Create RGBA image: shape (H, W, 4)
        rgba = np.zeros((10, 10, 4), dtype=np.uint8)
        rgba[:, :, 0] = 100  # R
        rgba[:, :, 1] = 150  # G
        rgba[:, :, 2] = 200  # B
        rgba[:, :, 3] = 255  # A

        result = to_canonical(rgba)

        # After conversion to BGRA: B, G, R, A
        # RGBA [R, G, B, A] -> BGRA [B, G, R, A]
        assert result.shape == (10, 10, 4)
        assert np.all(result[:, :, 0] == 200)  # B
        assert np.all(result[:, :, 1] == 150)  # G
        assert np.all(result[:, :, 2] == 100)  # R
        assert np.all(result[:, :, 3] == 255)  # A (unchanged)

    def test_float_with_integer_values_to_uint8(self):
        """Test conversion of float array with integer values in [0, 255] to uint8."""
        float_arr = np.array([1.0, 2.0, 3.0, 255.0], dtype=np.float32)
        result = to_canonical(float_arr)

        assert result.dtype == np.uint8
        assert np.array_equal(result, np.array([1, 2, 3, 255], dtype=np.uint8))

    def test_float_with_integer_values_to_uint16(self):
        """Test conversion of float array with integer values in [0, 65535] to uint16."""
        float_arr = np.array([1.0, 1000.0, 65535.0], dtype=np.float32)
        result = to_canonical(float_arr)

        assert result.dtype == np.uint16
        assert np.array_equal(result, np.array([1, 1000, 65535], dtype=np.uint16))

    def test_float_with_integer_values_to_int16(self):
        """Test conversion of float array with negative integer values to int16."""
        float_arr = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
        result = to_canonical(float_arr)

        assert result.dtype == np.int16
        assert np.array_equal(result, np.array([-100, 0, 100], dtype=np.int16))

    def test_float_with_integer_values_to_int32(self):
        """Test conversion of float array with large integer values to int32."""
        float_arr = np.array([-100000.0, 0.0, 100000.0], dtype=np.float32)
        result = to_canonical(float_arr)

        assert result.dtype == np.int32
        assert np.array_equal(result, np.array([-100000, 0, 100000], dtype=np.int32))

    def test_float_with_non_integer_values_preserved(self):
        """Test that float arrays with non-integer values are preserved."""
        float_arr = np.array([1.5, 2.7, 3.14, 0.0], dtype=np.float32)
        result = to_canonical(float_arr)

        assert result.dtype == np.float32
        assert np.allclose(result, float_arr)

    def test_float_with_mixed_values_preserved(self):
        """Test that float arrays with some non-integer values are preserved."""
        float_arr = np.array([1.0, 2.5, 3.0], dtype=np.float32)
        result = to_canonical(float_arr)

        # Should be preserved as float because of 2.5
        assert result.dtype == np.float32
        assert np.allclose(result, float_arr)

    def test_float_with_small_values(self):
        """Test float arrays with very small values."""
        float_arr = np.array([0.0, 0.0001, 0.0002], dtype=np.float32)
        result = to_canonical(float_arr)

        # Small values that are close to integers (0) should be converted
        # 0.0001 and 0.0002 are within tolerance of 0
        assert result.dtype in [np.uint8, np.int32, np.int64]

    def test_integer_array_unchanged(self):
        """Test that integer arrays are unchanged."""
        int_arr = np.array([1, 2, 3], dtype=np.uint8)
        result = to_canonical(int_arr)

        assert result.dtype == np.uint8
        assert np.array_equal(result, int_arr)

    def test_3d_grayscale_unchanged(self):
        """Test that 3D grayscale images are unchanged."""
        gray = np.random.rand(10, 10, 10).astype(np.float32)
        result = to_canonical(gray)

        # 3D grayscale should not be treated as RGB
        # (only last dimension of 3 or 4 is treated as color)
        assert result.shape == (10, 10, 10)
        # Since values are random floats, should remain float
        assert result.dtype == np.float32

    def test_2d_single_channel_unchanged(self):
        """Test that 2D single-channel images are unchanged."""
        gray = np.random.rand(10, 10).astype(np.float32)
        result = to_canonical(gray)

        assert result.shape == (10, 10)
        assert result.dtype == np.float32

    def test_float64_with_integer_values(self):
        """Test float64 arrays with integer values."""
        float_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = to_canonical(float_arr)

        # Should be converted to integer type
        assert result.dtype in [np.uint8, np.int32, np.int64]


class TestToCanonicalWithSimpleITK:
    """Tests for to_canonical with SimpleITK images."""

    @pytest.fixture
    def sitk_available(self):
        """Check if SimpleITK is available."""
        pytest.importorskip("SimpleITK")

    def test_sitk_rgb_to_bgr(self, sitk_available):
        """Test RGB to BGR conversion for SimpleITK image."""
        import SimpleITK as sitk

        # Create RGB image
        rgb_arr = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb_arr[:, :, 0] = 100  # R
        rgb_arr[:, :, 1] = 150  # G
        rgb_arr[:, :, 2] = 200  # B
        rgb_img = sitk.GetImageFromArray(rgb_arr)

        result = to_canonical(rgb_img)
        result_arr = sitk.GetArrayFromImage(result)

        # After conversion to BGR: B, G, R
        assert result_arr.shape == (10, 10, 3)
        assert np.all(result_arr[:, :, 0] == 200)  # B
        assert np.all(result_arr[:, :, 1] == 150)  # G
        assert np.all(result_arr[:, :, 2] == 100)  # R

    def test_sitk_float_to_int(self, sitk_available):
        """Test float to int conversion for SimpleITK image."""
        import SimpleITK as sitk

        # Create float image with integer values
        float_arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        float_img = sitk.GetImageFromArray(float_arr)

        result = to_canonical(float_img)
        result_arr = sitk.GetArrayFromImage(result)

        # Should be converted to integer type
        assert result_arr.dtype in [np.uint8, np.int16, np.int32, np.uint16]

    def test_sitk_preserves_metadata(self, sitk_available):
        """Test that metadata is preserved during conversion."""
        import SimpleITK as sitk

        # Create image with metadata
        img = sitk.Image(10, 10, sitk.sitkFloat32)
        img.SetSpacing([1.0, 2.0])
        img.SetOrigin([10.0, 20.0])

        # Set some pixel values
        img[5, 5] = 1.0
        img[5, 6] = 2.0

        result = to_canonical(img)

        # Check that metadata is preserved
        assert result.GetSpacing() == [1.0, 2.0]
        assert result.GetOrigin() == [10.0, 20.0]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_array(self):
        """Test with empty array."""
        empty = np.array([], dtype=np.float32)
        result = to_canonical(empty)
        assert len(result) == 0

    def test_single_pixel(self):
        """Test with single pixel."""
        single = np.array([128.0], dtype=np.float32)
        result = to_canonical(single)
        assert result.dtype in [np.uint8, np.int32, np.int64]

    def test_unknown_type_returned_as_is(self):
        """Test that unknown types are returned unchanged."""

        class UnknownType:
            pass

        unknown = UnknownType()
        result = to_canonical(unknown)
        assert result is unknown
