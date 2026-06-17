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
def normalize_image(image: Image) -> Image: ...
def bilateral_filter(
    image: Image, spatial_sigma: float, range_sigma: float
) -> Image: ...
def n4_bias_correction(
    image: Image,
    num_fitting_levels: int = 4,
    num_iterations: int = 50,
    noise_estimate: float = 0.01,
    shrink_factor: int = 4,
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
def unsharp_mask(
    image: Image,
    sigma: float = 1.0,
    amount: float = 0.5,
    threshold: float = 0.0,
    clamp: bool = False,
) -> Image: ...
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
def zero_crossing_image(
    image: Image,
    foreground_value: float = 1.0,
    background_value: float = 0.0,
) -> Image: ...
def blend_images(a: Image, b: Image, alpha: float = 0.5) -> Image:
    """Linearly blend two co-registered images.

    out(x) = (1 - alpha) * a(x) + alpha * b(x)

    alpha=0 returns a; alpha=1 returns b. Both images must have identical shapes.
    Spatial metadata is preserved from a. ITK Parity: BlendImageFilter.
    """
    ...

def add_images(a: Image, b: Image) -> Image:
    """Pixelwise addition: out(x) = a(x) + b(x). ITK Parity: AddImageFilter."""
    ...

def subtract_images(a: Image, b: Image) -> Image:
    """Pixelwise subtraction: out(x) = a(x) - b(x). ITK Parity: SubtractImageFilter."""
    ...

def multiply_images(a: Image, b: Image) -> Image:
    """Pixelwise multiplication: out(x) = a(x) * b(x). ITK Parity: MultiplyImageFilter."""
    ...

def divide_images(a: Image, b: Image) -> Image:
    """Pixelwise division: out(x) = a(x) / b(x); division by zero yields 0. ITK Parity: DivideImageFilter."""
    ...

def minimum_images(a: Image, b: Image) -> Image:
    """Pixelwise minimum: out(x) = min(a(x), b(x)). ITK Parity: MinimumImageFilter."""
    ...

def maximum_images(a: Image, b: Image) -> Image:
    """Pixelwise maximum: out(x) = max(a(x), b(x)). ITK Parity: MaximumImageFilter."""
    ...

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
def rotate_image(
    image: Image,
    angle_x: float = 0.0,
    angle_y: float = 0.0,
    angle_z: float = 0.0,
    mode: str = "linear",
    default_pixel_value: float = 0.0,
) -> Image: ...
def shift_image(
    image: Image,
    shift_z: float = 0.0,
    shift_y: float = 0.0,
    shift_x: float = 0.0,
    mode: str = "linear",
    default_pixel_value: float = 0.0,
) -> Image: ...
def zoom_image(
    image: Image,
    zoom_z: float = 1.0,
    zoom_y: float = 1.0,
    zoom_x: float = 1.0,
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

# -- Noise generators ---------------------------------------------------------

def additive_gaussian_noise(
    image: Image, std: float, mean: float = 0.0, seed: int = 42
) -> Image: ...
def salt_and_pepper_noise(
    image: Image, probability: float, seed: int = 42
) -> Image: ...
def shot_noise(image: Image, scale: float, seed: int = 42) -> Image: ...
def speckle_noise(image: Image, std: float, seed: int = 42) -> Image: ...

# -- Resampling / diffusion ---------------------------------------------------

def bin_shrink(
    image: Image, factor_z: int = 2, factor_y: int = 2, factor_x: int = 2
) -> Image: ...
def coherence_enhancing_diffusion(
    image: Image,
    sigma: float = 3.0,
    contrast: float = 1e-10,
    alpha: float = 0.001,
    time_step: float = 0.0625,
    iterations: int = 10,
) -> Image: ...

# -- FFT spectral filters -----------------------------------------------------

def forward_fft(image: Image) -> Image: ...
def inverse_fft(image: Image) -> Image: ...
def fft_shift(image: Image) -> Image: ...
def fft_convolve(image: Image, kernel: Image) -> Image: ...
def fft_convolve_3d(volume: Image, kernel: Image) -> Image: ...
def fft_normalized_correlate(image: Image, template: Image) -> Image: ...
def fft_normalized_correlate_3d(volume: Image, template: Image) -> Image: ...
def fft_ideal_low_pass(image: Image, cutoff: float = 0.3) -> Image: ...
def fft_ideal_high_pass(image: Image, cutoff: float = 0.3) -> Image: ...
def fft_butterworth_low_pass(
    image: Image, cutoff: float = 0.3, order: int = 2
) -> Image: ...
def fft_butterworth_high_pass(
    image: Image, cutoff: float = 0.3, order: int = 2
) -> Image: ...

# -- Deconvolution ------------------------------------------------------------

def richardson_lucy_deconvolution(
    image: Image, kernel: Image, max_iterations: int = 30, tolerance: float = 1e-06
) -> Image: ...
def landweber_deconvolution(
    image: Image,
    kernel: Image,
    step_size: float = 0.1,
    max_iterations: int = 100,
    tolerance: float = 1e-06,
) -> Image: ...
def wiener_deconvolution(
    image: Image, kernel: Image, noise_to_signal: float = 0.01
) -> Image: ...
def tikhonov_deconvolution(
    image: Image, kernel: Image, lambda_: float = 0.01
) -> Image: ...

# -- Intensity projections ----------------------------------------------------

def max_intensity_projection(image: Image, axis: int = 0) -> Image: ...
def min_intensity_projection(image: Image, axis: int = 0) -> Image: ...
def mean_intensity_projection(image: Image, axis: int = 0) -> Image: ...
def sum_intensity_projection(image: Image, axis: int = 0) -> Image: ...
def stddev_intensity_projection(image: Image, axis: int = 0) -> Image: ...
