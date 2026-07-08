use crate::image::{into_py_image, vec_to_image, PyImage};
use pyo3::prelude::*;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_filter::{
    gabor_image_source as core_gabor_image_source,
    gaussian_image_source as core_gaussian_image_source,
    grid_image_source as core_grid_image_source,
};
/// Generate a Gaussian blob image (`itk::GaussianImageSource` / `sitk.GaussianSource`).
///
/// `out(index) = scale · exp(−½ · Σ_d ((origin_d + index_d·spacing_d − mean_d)/sigma_d)²)`
/// (non-normalised; peak value = `scale`). All `(x, y, z)` tuples are in sitk
/// axis order; the produced image carries the given spacing/origin (identity
/// direction). ITK Parity: GaussianImageSource.
#[pyfunction]
#[pyo3(signature = (size, sigma, mean, scale=255.0, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)))]
pub fn gaussian_image_source(
    py: Python<'_>,
    size: (usize, usize, usize),
    sigma: (f64, f64, f64),
    mean: (f64, f64, f64),
    scale: f64,
    origin: (f64, f64, f64),
    spacing: (f64, f64, f64),
) -> PyImage {
    let (buf, dims) = py.allow_threads(|| {
        core_gaussian_image_source(
            [size.0, size.1, size.2],
            [sigma.0, sigma.1, sigma.2],
            [mean.0, mean.1, mean.2],
            scale,
            [origin.0, origin.1, origin.2],
            [spacing.0, spacing.1, spacing.2],
        )
    });
    into_py_image(vec_to_image(
        buf,
        dims,
        Point::new([origin.2, origin.1, origin.0]),
        Spacing::new([spacing.2, spacing.1, spacing.0]),
        Direction::identity(),
    ))
}

/// Generate a grid-pattern image (`itk::GridImageSource` / `sitk.GridSource`):
/// dark periodic Gaussian lines on a bright background,
/// `out = scale·Π_{selected d}(1 − Σ_lines exp(−(p_d−line)²/(2σ_d²)))`.
/// All `(x, y, z)` tuples are in sitk axis order. ITK Parity: GridImageSource.
#[pyfunction]
#[pyo3(signature = (size, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), sigma=(0.5, 0.5, 0.5), grid_spacing=(4.0, 4.0, 4.0), grid_offset=(0.0, 0.0, 0.0), scale=255.0, which_dimensions=(true, true, true)))]
#[allow(clippy::too_many_arguments)]
pub fn grid_image_source(
    py: Python<'_>,
    size: (usize, usize, usize),
    spacing: (f64, f64, f64),
    origin: (f64, f64, f64),
    sigma: (f64, f64, f64),
    grid_spacing: (f64, f64, f64),
    grid_offset: (f64, f64, f64),
    scale: f64,
    which_dimensions: (bool, bool, bool),
) -> PyImage {
    let (buf, dims) = py.allow_threads(|| {
        core_grid_image_source(
            [size.0, size.1, size.2],
            [spacing.0, spacing.1, spacing.2],
            [origin.0, origin.1, origin.2],
            [sigma.0, sigma.1, sigma.2],
            [grid_spacing.0, grid_spacing.1, grid_spacing.2],
            [grid_offset.0, grid_offset.1, grid_offset.2],
            scale,
            [which_dimensions.0, which_dimensions.1, which_dimensions.2],
        )
    });
    into_py_image(vec_to_image(
        buf,
        dims,
        Point::new([origin.2, origin.1, origin.0]),
        Spacing::new([spacing.2, spacing.1, spacing.0]),
        Direction::identity(),
    ))
}

/// Generate a Gabor-wavelet image (`itk::GaborImageSource` / `sitk.GaborSource`):
/// a Gaussian envelope modulated by a cosine along x (the real part),
/// `out = exp(−½·Σ((p_d−mean_d)/sigma_d)²)·cos(2π·frequency·(p_x−mean_x))`.
/// All `(x, y, z)` tuples are in sitk axis order. ITK Parity: GaborImageSource.
#[pyfunction]
#[pyo3(signature = (size, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), sigma=(16.0, 16.0, 16.0), mean=(32.0, 32.0, 32.0), frequency=0.4))]
#[allow(clippy::too_many_arguments)]
pub fn gabor_image_source(
    py: Python<'_>,
    size: (usize, usize, usize),
    spacing: (f64, f64, f64),
    origin: (f64, f64, f64),
    sigma: (f64, f64, f64),
    mean: (f64, f64, f64),
    frequency: f64,
) -> PyImage {
    let (buf, dims) = py.allow_threads(|| {
        core_gabor_image_source(
            [size.0, size.1, size.2],
            [spacing.0, spacing.1, spacing.2],
            [origin.0, origin.1, origin.2],
            [sigma.0, sigma.1, sigma.2],
            [mean.0, mean.1, mean.2],
            frequency,
        )
    });
    into_py_image(vec_to_image(
        buf,
        dims,
        Point::new([origin.2, origin.1, origin.0]),
        Spacing::new([spacing.2, spacing.1, spacing.0]),
        Direction::identity(),
    ))
}
