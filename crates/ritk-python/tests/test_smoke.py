"""
Smoke tests for the ritk Python package.

Verify the ritk wheel can be imported and the public API surface is reachable.
These tests do NOT require medical image files on disk.

Run with: pytest crates/ritk-python/tests/test_smoke.py
"""

import importlib
import sys

# Import verification


def test_ritk_importable():
    """The top-level ritk package must be importable."""
    import ritk  # noqa: F401


def test_ritk_has_expected_submodules():
    """All public submodules declared in ritk/__init__.py must be importable."""
    import ritk

    expected = ["image", "io", "filter", "registration", "segmentation", "statistics"]
    for name in expected:
        assert (
            hasattr(ritk, name) or importlib.util.find_spec(f"ritk.{name}") is not None
        ), f"ritk.{name} submodule is missing from installed package"


def test_ritk_filter_submodule_importable():
    import ritk.filter  # noqa: F401


def test_ritk_segmentation_submodule_importable():
    import ritk.segmentation  # noqa: F401


def test_ritk_registration_submodule_importable():
    import ritk.registration  # noqa: F401


def test_ritk_statistics_submodule_importable():
    import ritk.statistics  # noqa: F401


def test_ritk_io_submodule_importable():
    import ritk.io  # noqa: F401


# API surface verification


def test_filter_public_functions_exist():
    """Key filter functions must be callable attributes."""
    import ritk.filter as rf

    required = [
        "gaussian_filter",
        "discrete_gaussian",
        "median_filter",
        "bilateral_filter",
        "n4_bias_correction",
        "anisotropic_diffusion",
        "gradient_magnitude",
        "laplacian",
        "frangi_vesselness",
        "canny_edge_detect",
        "recursive_gaussian",
        "sobel_gradient",
        "grayscale_erosion",
        "grayscale_dilation",
    ]
    missing = [fn for fn in required if not callable(getattr(rf, fn, None))]
    assert not missing, f"Missing callable functions in ritk.filter: {missing}"


def test_segmentation_public_functions_exist():
    import ritk.segmentation as rs

    required = [
        "otsu_threshold",
        "li_threshold",
        "yen_threshold",
        "connected_threshold_segment",
        "confidence_connected_segment",
        "kmeans_segment",
        "watershed_segment",
        "binary_erosion",
        "binary_dilation",
        "skeletonization",
        "connected_components",
        "chan_vese_segment",
        "geodesic_active_contour_segment",
    ]
    missing = [fn for fn in required if not callable(getattr(rs, fn, None))]
    assert not missing, f"Missing callable functions in ritk.segmentation: {missing}"


def test_registration_public_functions_exist():
    import ritk.registration as rr

    required = [
        "diffeomorphic_demons_register",
        "symmetric_demons_register",
        "inverse_consistent_demons_register",
        "syn_register",
        "bspline_ffd_register",
        "lddmm_register",
    ]
    missing = [fn for fn in required if not callable(getattr(rr, fn, None))]
    assert not missing, f"Missing callable functions in ritk.registration: {missing}"


def test_statistics_public_functions_exist():
    import ritk.statistics as rstat

    required = [
        "compute_statistics",
        "dice_coefficient",
        "psnr",
        "ssim",
        "zscore_normalize",
        "minmax_normalize",
    ]
    missing = [fn for fn in required if not callable(getattr(rstat, fn, None))]
    assert not missing, f"Missing callable functions in ritk.statistics: {missing}"


# Version / metadata verification


def test_ritk_has_version():
    """The installed package must expose __version__."""
    import ritk

    assert hasattr(ritk, "__version__"), "ritk.__version__ is missing"
    assert isinstance(ritk.__version__, str), (
        f"__version__ must be str, got {type(ritk.__version__)}"
    )
    assert ritk.__version__, "__version__ must not be empty"


def test_python_version_is_supported():
    """Ensure we are running on Python 3.9+."""
    assert sys.version_info >= (3, 9), f"RITK requires Python >= 3.9; got {sys.version}"
