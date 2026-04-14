"""Type stubs for the ``_ritk.filter`` submodule (compiled by PyO3/maturin)."""

from __future__ import annotations

from ritk._ritk.image import Image

def gaussian_filter(image: Image, sigma: float) -> Image: ...
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
