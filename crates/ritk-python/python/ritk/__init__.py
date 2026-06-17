"""
ritk — Medical Image Processing Toolkit
========================================

Python bindings for the RITK Rust library, exposing:

- ``ritk.Image``          — Medical image with physical-space metadata (Z×Y×X, f32).
- ``ritk.io``             — Read / write NIfTI, PNG, DICOM, MetaImage, NRRD.
- ``ritk.filter``         — Gaussian, median, bilateral, anisotropic diffusion, edge detection, morphology.
- ``ritk.metrics``        — Image similarity metrics (MSE, NCC, mutual information).
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

from pathlib import Path as _Path
import os as _os

_DLL_DIRECTORY_HANDLES: list[object] = []


def _add_windows_dll_directories() -> None:
    if _os.name != "nt" or not hasattr(_os, "add_dll_directory"):
        return

    package_dir = _Path(__file__).resolve().parent
    candidate_dirs: list[_Path] = [package_dir]

    for parent in package_dir.parents:
        if (parent / "api-ms-win-core-winrt-error-l1-1-0.dll").exists():
            candidate_dirs.append(parent)
            break

    msys_root = _Path(_os.environ.get("MSYS2_ROOT", r"D:\msys64"))
    candidate_dirs.extend((msys_root / "mingw64" / "bin", msys_root / "ucrt64" / "bin"))

    for directory in candidate_dirs:
        if directory.is_dir():
            _DLL_DIRECTORY_HANDLES.append(_os.add_dll_directory(str(directory)))


_add_windows_dll_directories()

from ritk._ritk import (
    filter,  # noqa: F401
    io,  # noqa: F401
    metrics,  # noqa: F401
    registration,  # noqa: F401
    segmentation,  # noqa: F401
    statistics,  # noqa: F401
)
from ritk._ritk import image as _image_mod  # noqa: F401 (submodule reference)

# Surface the Image class at the top-level namespace so users can write
# ``ritk.Image(array, spacing=...)`` instead of ``ritk.image.Image(...)``.
image = _image_mod  # expose ritk.image as top-level attribute
Image = _image_mod.Image  # extract from already-imported submodule attribute
ColorImage = _image_mod.ColorImage  # RGB/vector multi-component image

import sys as _sys

# Register PyO3 submodule objects in sys.modules so that
# "import ritk.filter", "import ritk.io", etc. work as expected.
# PyO3 add_submodule() exposes submodules as attributes but does not
# automatically register them as importable paths; we do it here.
_sys.modules.setdefault("ritk.filter", filter)
_sys.modules.setdefault("ritk.io", io)
_sys.modules.setdefault("ritk.metrics", metrics)
_sys.modules.setdefault("ritk.registration", registration)
_sys.modules.setdefault("ritk.segmentation", segmentation)
_sys.modules.setdefault("ritk.statistics", statistics)
_sys.modules.setdefault("ritk.image", _image_mod)

__all__ = [
    "Image",
    "io",
    "filter",
    "metrics",
    "registration",
    "segmentation",
    "statistics",
]

__version__ = "0.12.4"

