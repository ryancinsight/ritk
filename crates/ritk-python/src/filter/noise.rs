//! Python bindings for noise simulation filters (GAP-262-FLT-05).

use crate::errors::RitkResult;
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    AdditiveGaussianNoiseFilter, PatchBasedDenoisingImageFilter, SaltAndPepperNoiseFilter,
    ShotNoiseFilter, SpeckleNoiseFilter,
};
use std::sync::Arc;

/// Add additive Gaussian noise to a 3-D image.
///
/// ```text
/// I'(x) = I(x) + N(mean, std)
/// ```
///
/// Args:
///     image: Input 3-D PyImage.
///     std: Standard deviation of the Gaussian noise distribution.
///     mean: Mean of the Gaussian noise distribution (default 0.0).
///     seed: Random seed for deterministic output (default 42).
///
/// Returns:
///     PyImage with additive Gaussian noise applied.
#[pyfunction]
#[pyo3(signature = (image, std, mean=0.0, seed=42_u32))]
pub fn additive_gaussian_noise(
    py: Python<'_>,
    image: &PyImage,
    std: f64,
    mean: f64,
    seed: u32,
) -> RitkResult<PyImage> {
    let img = Arc::clone(&image.inner);
    let result = py
        .allow_threads(|| {
            AdditiveGaussianNoiseFilter::new(std)
                .with_mean(mean)
                .with_seed(seed)
                .apply(img.as_ref())
        })
        .map_err(|e| crate::errors::RitkPyError::runtime(e.to_string()))?;
    Ok(into_py_image(result))
}

/// Apply salt-and-pepper (impulse) noise to a 3-D image.
///
/// Independently replaces each voxel with min or max value at the given probability.
///
/// Args:
///     image: Input 3-D PyImage.
///     probability: Probability of each voxel being replaced (0.0–1.0).
///     seed: Random seed (default 42).
///
/// Returns:
///     PyImage with salt-and-pepper noise.
#[pyfunction]
#[pyo3(signature = (image, probability, seed=42_u32))]
pub fn salt_and_pepper_noise(
    py: Python<'_>,
    image: &PyImage,
    probability: f64,
    seed: u32,
) -> RitkResult<PyImage> {
    let img = Arc::clone(&image.inner);
    let result = py
        .allow_threads(|| {
            SaltAndPepperNoiseFilter::new(probability)
                .with_seed(seed)
                .apply(img.as_ref())
        })
        .map_err(|e| crate::errors::RitkPyError::runtime(e.to_string()))?;
    Ok(into_py_image(result))
}

/// Apply Poisson (shot) noise to a 3-D image.
///
/// ```text
/// I'(x) = Poisson(scale · max(I(x), 0)) / scale
/// ```
///
/// Args:
///     image: Input 3-D PyImage.
///     scale: Photon-count scale (higher = less noise). Typical: 0.1–100.0.
///     seed: Random seed (default 42).
///
/// Returns:
///     PyImage with Poisson noise.
#[pyfunction]
#[pyo3(signature = (image, scale, seed=42_u32))]
pub fn shot_noise(py: Python<'_>, image: &PyImage, scale: f64, seed: u32) -> RitkResult<PyImage> {
    let img = Arc::clone(&image.inner);
    let result = py
        .allow_threads(|| {
            ShotNoiseFilter::new(scale)
                .with_seed(seed)
                .apply(img.as_ref())
        })
        .map_err(|e| crate::errors::RitkPyError::runtime(e.to_string()))?;
    Ok(into_py_image(result))
}

/// Apply speckle (multiplicative) noise to a 3-D image.
///
/// ```text
/// I'(x) = I(x) · (1 + N(0, std))
/// ```
///
/// Characteristic of ultrasound / coherent imaging modalities.
///
/// Args:
///     image: Input 3-D PyImage.
///     std: Standard deviation of the multiplicative noise factor.
///     seed: Random seed (default 42).
///
/// Returns:
///     PyImage with speckle noise.
#[pyfunction]
#[pyo3(signature = (image, std, seed=42_u32))]
pub fn speckle_noise(py: Python<'_>, image: &PyImage, std: f64, seed: u32) -> RitkResult<PyImage> {
    let img = Arc::clone(&image.inner);
    let result = py
        .allow_threads(|| {
            SpeckleNoiseFilter::new(std)
                .with_seed(seed)
                .apply(img.as_ref())
        })
        .map_err(|e| crate::errors::RitkPyError::runtime(e.to_string()))?;
    Ok(into_py_image(result))
}

// ── PatchBasedDenoising ───────────────────────────────────────────────────────

/// Patch-based denoising, bit-exact to single-threaded
/// `SimpleITK.PatchBasedDenoising` (Gaussian noise model, fixed bandwidth).
///
/// Faithful ITK port: Gaussian-kernel joint-entropy gradient over patches drawn
/// by the GaussianRandomSpatialNeighborSubsampler (ITK MersenneTwister, seed 0),
/// visited in ImageBoundaryFacesCalculator order.
///
/// Args:
///     image:                    Input 3-D PyImage (nz==1 ⇒ 2-D).
///     number_of_iterations:     Denoising passes (default 1).
///     number_of_sample_patches: Patches sampled per pixel (default 200).
///     patch_radius:             Half-size of each patch per axis (default 4).
///     sample_variance:          Variance of the Gaussian sampling domain (default 400).
///     kernel_sigma:             Gaussian kernel bandwidth σ (default 400).
///
/// Returns:
///     Denoised PyImage (matches single-threaded sitk).
#[pyfunction]
#[pyo3(signature = (image, number_of_iterations=1, number_of_sample_patches=200,
                    patch_radius=4, sample_variance=400.0, kernel_sigma=400.0,
                    kernel_bandwidth_estimation=false))]
pub fn patch_based_denoising(
    py: Python<'_>,
    image: &PyImage,
    number_of_iterations: usize,
    number_of_sample_patches: usize,
    patch_radius: usize,
    sample_variance: f64,
    kernel_sigma: f64,
    kernel_bandwidth_estimation: bool,
) -> RitkResult<PyImage> {
    if kernel_bandwidth_estimation {
        return Err(crate::errors::RitkPyError::runtime(
            "Kernel bandwidth estimation is not implemented in ritk; set kernel_bandwidth_estimation=False".to_string()
        ));
    }
    let arc = Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        PatchBasedDenoisingImageFilter {
            number_of_iterations,
            number_of_sample_patches,
            patch_radius,
            sample_variance,
            kernel_sigma,
        }
        .apply(arc.as_ref())
    });
    result
        .map(into_py_image)
        .map_err(|e| crate::errors::RitkPyError::runtime(e.to_string()))
}
