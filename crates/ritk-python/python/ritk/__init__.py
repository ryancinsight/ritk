"""
ritk — Medical Image Processing Toolkit
========================================

Python bindings for the RITK Rust library, exposing:

- ``ritk.Image``          — Medical image with physical-space metadata (Z×Y×X, f32).
- ``ritk.io``             — Read / write NIfTI, PNG, DICOM, MetaImage, NRRD.
- ``ritk.filter``         — Gaussian, median, bilateral, anisotropic diffusion, edge detection, morphology.
- ``ritk.registration``   — Deformable registration (Demons, SyN, BSpline FFD, LDDMM, atlas building).
- ``ritk.segmentation``   — Thresholding, connected components, clustering, level sets, morphology.
- ``ritk.statistics``     — Descriptive statistics, comparison metrics, normalization.

Quick start
-----------
>>> import ritk
>>> img = ritk.io.read_image("brain.nii.gz")
>>> smoothed = ritk.filter.gaussian_filter(img, sigma=1.0)
>>> threshold, mask = ritk.segmentation.otsu_threshold(smoothed)
>>> ritk.io.write_image(smoothed, "brain_smooth.nii.gz")
"""

from ritk._ritk import (
    filter,  # noqa: F401
    io,  # noqa: F401
    registration,  # noqa: F401
    segmentation,  # noqa: F401
    statistics,  # noqa: F401
)
from ritk._ritk import image as _image_mod  # noqa: F401 (submodule reference)

# Surface the Image class at the top-level namespace so users can write
# ``ritk.Image(array, spacing=...)`` instead of ``ritk.image.Image(...)``.
from ritk._ritk.image import Image  # noqa: F401

__all__ = [
    "Image",
    "io",
    "filter",
    "registration",
    "segmentation",
    "statistics",
]

__version__ = "0.1.0"
