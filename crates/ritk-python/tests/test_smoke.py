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
        "metrics",
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
        "anonymize_dicom_dir",
        "read_mesh",
        "write_mesh",
    ]
    missing = [fn for fn in required if not callable(getattr(rio, fn, None))]
    assert not missing, f"Missing callable functions in ritk.io: {missing}"


def test_filter_public_functions_exist():
    """Key filter functions must be callable attributes."""
    import ritk.filter as rf

    required = [
        "gaussian_filter",
        "discrete_gaussian",
        "discrete_gaussian_derivative",
        "median_filter",
        "mean_filter",
        "bilateral_filter",
        "n4_bias_correction",
        "anisotropic_diffusion",
        "gradient_magnitude",
        "laplacian",
        "frangi_vesselness",
        "canny_edge_detect",
        "laplacian_of_gaussian",
        "recursive_gaussian",
        "recursive_gaussian_directional",
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
        "blend_images",
        "add_images",
        "nary_add",
        "nary_maximum",
        "subtract_images",
        "multiply_images",
        "divide_images",
        "binary_magnitude_images",
        "crop",
        "cyclic_shift",
        "divide_floor_images",
        "divide_real_images",
        "equal_images",
        "greater_equal_images",
        "greater_images",
        "less_equal_images",
        "less_images",
        "not_equal_images",
        "round_image",
        "signed_distance_map",
        "ternary_add_images",
        "ternary_magnitude_images",
        "ternary_magnitude_squared_images",
        "unary_minus_image",
        "minimum_images",
        "maximum_images",
        "squared_difference_images",
        "absolute_value_difference_images",
        "atan2_images",
        "pow_images",
        "abs_image",
        "sqrt_image",
        "square_image",
        "exp_image",
        "log_image",
        "log10_image",
        "exp_negative_image",
        "sin_image",
        "cos_image",
        "tan_image",
        "asin_image",
        "acos_image",
        "atan_image",
        "bounded_reciprocal_image",
        "clamp_image",
        "invert_intensity",
        "mask_image",
        "mask_negated_image",
        "white_top_hat",
        "black_top_hat",
        "hit_or_miss",
        "label_dilation",
        "label_erosion",
        "label_opening",
        "label_closing",
        "binary_contour",
        "label_contour",
        "voting_binary",
        "morphological_reconstruction",
        "h_maxima",
        "h_minima",
        "h_convex",
        "h_concave",
        "regional_maxima",
        "regional_minima",
        "valued_regional_maxima",
        "valued_regional_minima",
        "opening_by_reconstruction",
        "closing_by_reconstruction",
        "grayscale_closing",
        "grayscale_opening",
        "grayscale_fillhole",
        "grayscale_grind_peak",
        "flip",
        "constant_pad",
        "mirror_pad",
        "wrap_pad",
        "region_of_interest",
        "permute_axes",
        "paste",
        "resample_image",
        "rotate_image",
        "shift_image",
        "zoom_image",
        "distance_transform",
        "additive_gaussian_noise",
        "salt_and_pepper_noise",
        "shot_noise",
        "speckle_noise",
        "bin_shrink",
        "coherence_enhancing_diffusion",
        "forward_fft",
        "inverse_fft",
        "fft_shift",
        "fft_convolve",
        "fft_convolve_3d",
        "fft_normalized_correlate",
        "fft_normalized_correlate_3d",
        "fft_ideal_low_pass",
        "fft_ideal_high_pass",
        "fft_butterworth_low_pass",
        "fft_butterworth_high_pass",
        "richardson_lucy_deconvolution",
        "landweber_deconvolution",
        "wiener_deconvolution",
        "tikhonov_deconvolution",
        "max_intensity_projection",
        "min_intensity_projection",
        "mean_intensity_projection",
        "sum_intensity_projection",
        "stddev_intensity_projection",
        "median_intensity_projection",
        "normalize_image",
        "unsharp_mask",
        "zero_crossing_image",
        "and_images",
        "binary_not",
        "binary_projection",
        "binary_threshold_projection",
        "binomial_blur",
        "checker_board",
        "complex_to_imaginary",
        "complex_to_modulus",
        "complex_to_phase",
        "complex_to_real",
        "curvature_flow",
        "derivative",
        "double_threshold",
        "expand",
        "join_series",
        "magnitude_and_phase_to_complex",
        "masked_assign",
        "normalize_to_constant",
        "not_image",
        "or_images",
        "real_and_imaginary_to_complex",
        "shrink",
        "slice_image",
        "tile",
        "xor_images",
        "zero_flux_neumann_pad",
        "modulus",
        "box_mean",
        "box_sigma",
        "colliding_fronts",
        "fast_marching",
        "fft_pad",
        "gabor_image_source",
        "gaussian_image_source",
        "grid_image_source",
        "inverse_deconvolution",
        "projected_landweber_deconvolution",
        "rank",
        "voting_binary_hole_filling",
        "voting_binary_iterative_hole_filling",
        "warp",
        "bspline_decomposition",
        "binary_thinning",
        "binary_pruning",
        "erode_object_morphology",
        "real_to_half_hermitian_forward_fft",
        "half_hermitian_to_real_inverse_fft",
        "laplacian_sharpening",
        "zero_crossing_based_edge_detection",
        "iso_contour_distance",
        "local_noise",
        "stochastic_fractal_dimension",
        "transform_to_displacement_field",
        "transform_geometry",
        "invert_displacement_field",
        "iterative_inverse_displacement_field",
        "dicom_orient",
        "adaptive_histogram_equalization",
        "approximate_signed_distance_map",
        "normalized_correlation",
        "masked_fft_normalized_correlation",
        "reinitialize_level_set",
        "bitwise_not",
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
        "huang_threshold",
        "intermodes_threshold",
        "isodata_threshold",
        "kittler_illingworth_threshold",
        "moments_threshold",
        "renyi_entropy_threshold",
        "shanbhag_threshold",
        "multi_otsu_threshold",
        "connected_components",
        "relabel_components",
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
        "binary_threshold_segment",
        "marker_watershed_segment",
        "growcut_segment",
        "staple_ensemble",
        "change_label",
        "scalar_connected_component",
        "isolated_connected_segment",
        "morphological_watershed",
        "threshold_maximum_connected_components",
        "multi_label_staple",
        "label_set_dilate",
        "label_set_erode",
        "merge_label_map",
        "relabel_label_map",
        "toboggan",
        "vector_connected_component",
        "vector_confidence_connected_segment",
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
        "global_mi_register",
        "cma_mi_register",
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
        "jacobian_determinant",
        "analyze_jacobian",
        "extended_label_shape_statistics_py",
        "label_overlap_measures",
        "similarity_index",
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
