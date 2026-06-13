"""
Image operations for ITK compatibility.

This module provides utility functions for converting images to canonical
forms compatible with ITK conventions.
"""

from __future__ import annotations

import numpy as np


def to_canonical(image) -> object:
    """
    Convert an image to canonical ITK form.

    This function ensures that images conform to ITK conventions:
    - RGB images are converted to BGR order (ITK convention)
    - Floating point images with integer-valued pixels are converted to
      integer type (uint8 for [0, 255], uint16 for [0, 65535], int16 for
      signed ranges, etc.)
    - Floating point images with truly floating point pixel values are
      preserved as float

    Parameters
    ----------
    image : object
        An image object. Can be:
        - SimpleITK image
        - numpy ndarray
        - Any object with `GetPixelIDTypeAsString()` and `GetArrayFromImage()` methods
          (SimpleITK-like interface)

    Returns
    -------
    object
        The image in canonical ITK form. Returns the same type as input
        (SimpleITK image for SimpleITK input, numpy array for numpy input).

    Examples
    --------
    >>> import numpy as np
    >>> from itk.image_ops import to_canonical
    >>> # RGB image (will be converted to BGR)
    >>> rgb_img = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> canonical = to_canonical(rgb_img)

    Notes
    -----
    ITK convention uses BGR color order for RGB images, unlike most other
    libraries which use RGB order. This function handles the conversion
    automatically.
    """
    # Handle SimpleITK images
    try:
        import SimpleITK as sitk

        # Check if input is a SimpleITK image
        if isinstance(image, sitk.Image):
            return _to_canonical_sitk(image)
    except ImportError:
        pass

    # Handle numpy arrays
    if isinstance(image, np.ndarray):
        return _to_canonical_numpy(image)

    # Try to handle other ITK-like objects
    try:
        # Check for SimpleITK-like interface
        if hasattr(image, "GetPixelIDTypeAsString") and hasattr(
            image, "GetArrayFromImage"
        ):
            # This looks like a SimpleITK image
            return _to_canonical_sitk_like(image)
    except Exception:
        pass

    # If we can't handle it, return as-is
    return image


def _to_canonical_sitk(image):
    """Convert a SimpleITK image to canonical form.

    Canonicalization is performed on the image's numpy array view so the
    decision logic is shared verbatim with :func:`_to_canonical_numpy` (single
    source of truth).  This avoids keying off ``GetPixelIDTypeAsString()``,
    whose human-readable strings (``"32-bit float"``,
    ``"vector of 8-bit unsigned integer"``) never match the SWIG enum
    identifiers (``"sitkFloat32"`` …) the previous implementation compared
    against — a comparison that silently disabled both conversions.
    """
    import SimpleITK as sitk

    num_components = image.GetNumberOfComponentsPerPixel()
    arr = sitk.GetArrayFromImage(image)
    original_dtype = arr.dtype
    original_shape = arr.shape

    canonical = _to_canonical_numpy(arr)

    # Nothing changed — return the original image untouched to preserve every
    # geometry/metadata field exactly.
    if canonical.dtype == original_dtype and canonical.shape == original_shape:
        if num_components >= 3 or canonical is arr:
            return image
        if np.array_equal(canonical, arr):
            return image

    new_image = sitk.GetImageFromArray(canonical, isVector=num_components >= 3)
    new_image.CopyInformation(image)
    return new_image


def _to_canonical_numpy(arr):
    """Convert a numpy array to canonical form."""
    # Handle RGB/RGBA images (shape: ..., 3 or ..., 4)
    if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
        # Check if the last dimension is RGB/RGBA
        # ITK uses BGR order, so we need to reverse the color channels
        if arr.shape[-1] == 3:
            # RGB to BGR: reverse all 3 channels
            arr = arr[..., ::-1].copy()
        elif arr.shape[-1] == 4:
            # RGBA to BGRA: reverse first 3 channels (RGB -> BGR), keep alpha at end
            # Input: [R, G, B, A] -> Output: [B, G, R, A]
            # We need to reorder: [2, 1, 0, 3] for the last dimension
            arr = arr[..., [2, 1, 0, 3]].copy()

    # Check if floating point with integer values should be converted to integer
    if arr.dtype.kind == "f":
        # Check if all values are integers (within floating point precision)
        if _is_integer_valued(arr):
            # Determine the appropriate integer type based on the range
            arr = _convert_float_to_int_numpy(arr)

    return arr


def _to_canonical_sitk_like(image):
    """Convert a SimpleITK-like image object to canonical form."""
    # Get the array
    try:
        arr = image.GetArrayFromImage()
        pixel_type = image.GetPixelIDTypeAsString()
        num_components = image.GetNumberOfComponentsPerPixel()
    except Exception:
        return image

    # Convert to numpy, process, and convert back
    arr = np.array(arr)

    # Handle RGB/RGBA
    if num_components >= 3:
        if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
            if arr.shape[-1] == 3:
                # RGB to BGR: reverse all 3 channels
                arr = arr[..., ::-1].copy()
            elif arr.shape[-1] == 4:
                # RGBA to BGRA: reverse first 3 channels, keep alpha at end
                arr = arr[..., [2, 1, 0, 3]].copy()

    # Handle float to int conversion
    if arr.dtype.kind == "f" and _is_integer_valued(arr):
        arr = _convert_float_to_int_numpy(arr)

    # Try to create a new image from the processed array
    try:
        import SimpleITK as sitk

        new_image = sitk.GetImageFromArray(arr)
        # Copy metadata
        new_image.CopyInformation(image)
        return new_image
    except Exception:
        return arr


def _is_integer_valued(arr):
    """Check if a floating point array contains only integer values."""
    if arr.dtype.kind != "f":
        return True

    # Handle empty arrays
    if arr.size == 0:
        return True

    # Check if all values are close to integers
    # Use a relative tolerance to handle different scales
    threshold = 1e-5

    # For arrays with very small values, use absolute threshold
    # For arrays with larger values, use relative threshold
    max_val = np.abs(arr).max()
    if max_val < 1e-3:
        # Very small values - check absolute difference
        return np.all(np.abs(arr - np.round(arr)) < threshold)
    else:
        # Use relative threshold
        return np.all(
            np.abs(arr - np.round(arr)) < threshold * np.maximum(1.0, np.abs(arr))
        )


def _convert_float_to_int_numpy(arr):
    """Convert a floating point numpy array with integer values to integer type."""
    # Handle empty arrays
    if arr.size == 0:
        return arr

    # Round the values
    arr_rounded = np.round(arr).astype(np.int64)

    # Determine the appropriate integer type based on the range
    min_val = arr_rounded.min()
    max_val = arr_rounded.max()

    if min_val >= 0:
        if max_val <= 255:
            return arr_rounded.astype(np.uint8)
        elif max_val <= 65535:
            return arr_rounded.astype(np.uint16)
        elif max_val <= 4294967295:
            return arr_rounded.astype(np.uint32)
        else:
            return arr_rounded.astype(np.uint64)
    else:
        if min_val >= -32768 and max_val <= 32767:
            return arr_rounded.astype(np.int16)
        elif min_val >= -2147483648 and max_val <= 2147483647:
            return arr_rounded.astype(np.int32)
        else:
            return arr_rounded.astype(np.int64)
