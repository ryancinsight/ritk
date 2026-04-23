"""Type stubs for the ``_ritk.statistics`` submodule (compiled by PyO3/maturin).

Descriptive statistics, image comparison metrics, noise estimation,
and intensity normalization functions.  All delegate to
``ritk_core::statistics`` implementations (SSOT).
"""

from __future__ import annotations

from ritk._ritk.image import Image

def compute_statistics(image: Image) -> dict[str, float]:
    """Compute descriptive statistics over all voxels.

    Returns dict with keys: min, max, mean, std, p25, p50, p75.
    """
    ...

def masked_statistics(image: Image, mask: Image) -> dict[str, float]:
    """Compute descriptive statistics over foreground voxels (mask > 0.5).

    Returns dict with keys: min, max, mean, std, p25, p50, p75.
    """
    ...

def dice_coefficient(image1: Image, image2: Image) -> float:
    """Sørensen–Dice coefficient between two binary masks. Returns value in [0, 1]."""
    ...

def hausdorff_distance(image1: Image, image2: Image) -> float:
    """Symmetric Hausdorff distance (mm) between two binary masks.

    Spacing is derived from ``image1``'s voxel spacing.
    """
    ...

def mean_surface_distance(image1: Image, image2: Image) -> float:
    """Symmetric mean surface distance (mm) between two binary masks.

    Spacing is derived from ``image1``'s voxel spacing.
    """
    ...

def psnr(image1: Image, image2: Image, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio (dB).

    Args:
        image1:  Test image.
        image2:  Reference image (same shape).
        max_val: Dynamic range (default 1.0 for normalized images).
    """
    ...

def ssim(image1: Image, image2: Image, max_val: float = 1.0) -> float:
    """Structural similarity index (Wang et al. 2004).

    Args:
        image1:  Test image.
        image2:  Reference image (same shape).
        max_val: Dynamic range (default 1.0 for normalized images).

    Returns value in [-1, 1].
    """
    ...

def estimate_noise(image: Image, mask: Image | None = None) -> float:
    """MAD-based Gaussian noise σ̂ estimate.

    If *mask* is provided, only foreground voxels (mask > 0.5) contribute.
    """
    ...

def minmax_normalize(image: Image) -> Image:
    """Min-max rescale to [0, 1]."""
    ...

def minmax_normalize_range(image: Image, target_min: float, target_max: float) -> Image:
    """Min-max rescale to [target_min, target_max]."""
    ...

def zscore_normalize(image: Image) -> Image:
    """Z-score standardization (zero mean, unit variance)."""
    ...

def histogram_match(source: Image, reference: Image) -> Image:
    """CDF-based histogram matching of *source* to *reference*."""
    ...

def white_stripe_normalize(
    image: Image,
    mask: Image | None = None,
    contrast: str | None = None,
    width: float | None = None,
) -> tuple[Image, float, float, float, int]:
    """Shinohara et al. (2014) white stripe MRI normalization.

    Args:
        image:    Input brain MRI.
        mask:     Optional brain mask (foreground > 0.5).
        contrast: ``"t1"`` or ``"t2"`` (default ``"t1"``).
        width:    Stripe half-width in quantile units (default 0.05).

    Returns:
        (normalized_image, mu, sigma, wm_peak, stripe_size).
    """
    ...

def nyul_udupa_normalize(
    image: Image,
    training_images: list[Image],
) -> Image:
    """Nyul-Udupa piecewise-linear histogram standardization.

    Args:
        image:           Image to normalize.
        training_images: Non-empty list of images used to learn standard landmarks.

    Returns:
        Normalized image with the same shape and spatial metadata as *image*.

    Raises:
        RuntimeError: if training_images is empty or normalization fails.
    """
    ...
