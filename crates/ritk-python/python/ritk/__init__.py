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
image = _image_mod  # expose ritk.image as top-level attribute
Image = _image_mod.Image  # extract from already-imported submodule attribute

import sys as _sys

# Register PyO3 submodule objects in sys.modules so that
# "import ritk.filter", "import ritk.io", etc. work as expected.
# PyO3 add_submodule() exposes submodules as attributes but does not
# automatically register them as importable paths; we do it here.
_sys.modules.setdefault("ritk.filter", filter)
_sys.modules.setdefault("ritk.io", io)
_sys.modules.setdefault("ritk.registration", registration)
_sys.modules.setdefault("ritk.segmentation", segmentation)
_sys.modules.setdefault("ritk.statistics", statistics)
_sys.modules.setdefault("ritk.image", _image_mod)

__all__ = [
    "Image",
    "io",
    "filter",
    "registration",
    "segmentation",
    "statistics",
]

__version__ = "0.12.0"
