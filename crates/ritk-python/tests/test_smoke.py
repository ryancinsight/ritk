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


def test_ritk_top_level_exports_match_public_contract():
    """Top-level package exports must match the documented public contract."""
    import ritk

    expected_exports = {
        "Image",
        "io",
        "filter",
        "registration",
        "segmentation",
        "statistics",
    }

    missing = [name for name in expected_exports if not hasattr(ritk, name)]
    assert not missing, f"ritk top-level exports are missing: {missing}"


def test_ritk___all___matches_public_contract():
    """__all__ must enumerate the supported top-level exports in stable order."""
    import ritk

    expected = [
        "Image",
        "io",
        "filter",
        "registration",
        "segmentation",
        "statistics",
    ]

    assert hasattr(ritk, "__all__"), "ritk.__all__ is missing"
    assert isinstance(ritk.__all__, list), (
        f"ritk.__all__ must be list, got {type(ritk.__all__)}"
    )
    assert ritk.__all__ == expected, (
        f"ritk.__all__ drifted from expected export order: {ritk.__all__}"
    )


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


def test_io_public_functions_exist():
    import ritk.io as rio

    required = [
        "read_image",
        "write_image",
        "read_transform",
        "write_transform",
    ]
    missing = [fn for fn in required if not callable(getattr(rio, fn, None))]
    assert not missing, f"Missing callable functions in ritk.io: {missing}"


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
        "laplacian_of_gaussian",
        "recursive_gaussian",
        "sobel_gradient",
        "grayscale_erosion",
        "grayscale_dilation",
        "curvature_anisotropic_diffusion",
        "sato_line_filter",
        "rescale_intensity",
        "intensity_windowing",
        "threshold_below",
        "threshold_above",
        "threshold_outside",
        "sigmoid_filter",
        "binary_threshold",
        "white_top_hat",
        "black_top_hat",
        "hit_or_miss",
        "label_dilation",
        "label_erosion",
        "label_opening",
        "label_closing",
        "morphological_reconstruction",
        "resample_image",
        "distance_transform",
    ]
    missing = [fn for fn in required if not callable(getattr(rf, fn, None))]
    assert not missing, f"Missing callable functions in ritk.filter: {missing}"


def test_segmentation_public_functions_exist():
    import ritk.segmentation as rs

    required = [
        "otsu_threshold",
        "li_threshold",
        "yen_threshold",
        "kapur_threshold",
        "triangle_threshold",
        "multi_otsu_threshold",
        "connected_components",
        "connected_threshold_segment",
        "kmeans_segment",
        "watershed_segment",
        "binary_erosion",
        "binary_dilation",
        "binary_opening",
        "binary_closing",
        "binary_fill_holes",
        "morphological_gradient",
        "chan_vese_segment",
        "geodesic_active_contour_segment",
        "shape_detection_segment",
        "threshold_level_set_segment",
        "laplacian_level_set_segment",
        "confidence_connected_segment",
        "neighborhood_connected_segment",
        "skeletonization",
        "label_shape_statistics",
    ]
    missing = [fn for fn in required if not callable(getattr(rs, fn, None))]
    assert not missing, f"Missing callable functions in ritk.segmentation: {missing}"


def test_registration_public_functions_exist():
    import ritk.registration as rr

    required = [
        "demons_register",
        "diffeomorphic_demons_register",
        "symmetric_demons_register",
        "inverse_consistent_demons_register",
        "multires_demons_register",
        "syn_register",
        "bspline_ffd_register",
        "multires_syn_register",
        "bspline_syn_register",
        "lddmm_register",
        "build_atlas",
        "majority_vote_fusion",
        "joint_label_fusion_py",
    ]
    missing = [fn for fn in required if not callable(getattr(rr, fn, None))]
    assert not missing, f"Missing callable functions in ritk.registration: {missing}"


def test_statistics_public_functions_exist():
    import ritk.statistics as rstat

    required = [
        "compute_statistics",
        "masked_statistics",
        "dice_coefficient",
        "hausdorff_distance",
        "mean_surface_distance",
        "psnr",
        "ssim",
        "estimate_noise",
        "minmax_normalize",
        "minmax_normalize_range",
        "zscore_normalize",
        "histogram_match",
        "white_stripe_normalize",
        "nyul_udupa_normalize",
        "compute_label_intensity_statistics",
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
