"""Type stubs for the ``_ritk.filter`` submodule (compiled by PyO3/maturin)."""

from __future__ import annotations

from ritk._ritk.image import Image

def gaussian_filter(image: Image, sigma: float) -> Image: ...
def discrete_gaussian(
    image: Image,
    variance: float,
    maximum_error: float = 0.01,
    use_image_spacing: bool = True,
) -> Image: ...
def median_filter(image: Image, radius: int = 1) -> Image: ...
def bilateral_filter(
    image: Image, spatial_sigma: float, range_sigma: float
) -> Image: ...
def n4_bias_correction(
    image: Image,
    num_fitting_levels: int = 4,
    num_iterations: int = 50,
    noise_estimate: float = 0.01,
) -> Image: ...
def anisotropic_diffusion(
    image: Image,
    iterations: int = 20,
    conductance: float = 3.0,
    time_step: float = 0.0625,
    exponential: bool = True,
) -> Image: ...
def gradient_magnitude(image: Image) -> Image: ...
def laplacian(image: Image) -> Image: ...
def frangi_vesselness(
    image: Image,
    scales: list[float] | None = None,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 15.0,
    bright_vessels: bool = True,
) -> Image: ...
def canny_edge_detect(
    image: Image,
    sigma: float = 1.0,
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
) -> Image: ...
def laplacian_of_gaussian(image: Image, sigma: float = 1.0) -> Image: ...
def recursive_gaussian(image: Image, sigma: float = 1.0, order: int = 0) -> Image: ...
def sobel_gradient(image: Image) -> Image: ...
def grayscale_erosion(image: Image, radius: int = 1) -> Image: ...
def grayscale_dilation(image: Image, radius: int = 1) -> Image: ...
def curvature_anisotropic_diffusion(
    image: Image,
    iterations: int = 20,
    time_step: float = 0.0625,
) -> Image: ...
def sato_line_filter(
    image: Image,
    scales: list[float] | None = None,
    alpha: float = 0.5,
    bright_tubes: bool = True,
) -> Image: ...
def rescale_intensity(
    image: Image,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> Image: ...
def intensity_windowing(
    image: Image,
    window_min: float,
    window_max: float,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> Image: ...
def threshold_below(
    image: Image,
    threshold: float,
    outside_value: float = 0.0,
) -> Image: ...
def threshold_above(
    image: Image,
    threshold: float,
    outside_value: float = 0.0,
) -> Image: ...
def threshold_outside(
    image: Image,
    lower: float,
    upper: float,
    outside_value: float = 0.0,
) -> Image: ...
def sigmoid_filter(
    image: Image,
    alpha: float,
    beta: float,
    min_output: float = 0.0,
    max_output: float = 1.0,
) -> Image: ...
def binary_threshold(
    image: Image,
    lower_threshold: float,
    upper_threshold: float,
    foreground: float = 1.0,
    background: float = 0.0,
) -> Image: ...
def white_top_hat(image: Image, radius: int) -> Image: ...
def black_top_hat(image: Image, radius: int) -> Image: ...
def hit_or_miss(image: Image, fg_radius: int, bg_radius: int) -> Image: ...
def label_dilation(image: Image, radius: int) -> Image: ...
def label_erosion(image: Image, radius: int = 1) -> Image: ...
def label_opening(image: Image, radius: int = 1) -> Image: ...
def label_closing(image: Image, radius: int = 1) -> Image: ...
def morphological_reconstruction(
    marker: Image,
    mask: Image,
    mode: str = "dilation",
) -> Image: ...
def resample_image(
    image: Image,
    spacing_z: float = 1.0,
    spacing_y: float = 1.0,
    spacing_x: float = 1.0,
    mode: str = "linear",
) -> Image: ...
def distance_transform(
    image: Image,
    foreground_threshold: float = 0.5,
    squared: bool = False,
) -> Image:
    """Euclidean distance transform (Meijster et al. 2000).

    For each background voxel, computes distance to nearest foreground voxel
    in physical units (respecting image spacing). Foreground voxels get 0.0.

    Args:
        image:                Binary input image.
        foreground_threshold: Voxels above this value are foreground (default 0.5).
        squared:              If True, return squared distances (default False).

    Returns:
        Distance image with identical shape and spatial metadata.
    """
    ...
