//! Sealed [`AutoThreshold`] trait: shared extract→histogram→mask scaffold for
//! single-threshold auto-selection algorithms.
//!
//! Implementors supply only [`AutoThreshold::compute_threshold`], which receives
//! a pre-built `&[u32]` histogram and returns the threshold intensity.  The
//! common extract→histogram and threshold→binary-mask pipelines are provided as
//! blanket default methods ([`AutoThreshold::compute`] /
//! [`AutoThreshold::apply`]).
//!
//! # Sealing
//! The trait is sealed via a private `sealed::Sealed` supertrait: only the five
//! types enumerated in this module (`OtsuThreshold`, `LiThreshold`,
//! `YenThreshold`, `KapurThreshold`, `TriangleThreshold`) may implement it.

use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

use super::kapur::KapurThreshold;
use super::li::LiThreshold;
use super::otsu::OtsuThreshold;
use super::triangle::TriangleThreshold;
use super::yen::YenThreshold;

// ── Sealed supertrait ──────────────────────────────────────────────────────────

mod sealed {
    /// Private marker that prevents external crates from implementing
    /// [`AutoThreshold`](super::AutoThreshold).
    pub trait Sealed {}
}

impl sealed::Sealed for OtsuThreshold {}
impl sealed::Sealed for LiThreshold {}
impl sealed::Sealed for YenThreshold {}
impl sealed::Sealed for KapurThreshold {}
impl sealed::Sealed for TriangleThreshold {}

// ── Trait ──────────────────────────────────────────────────────────────────────

/// Sealed trait for single-threshold auto-selection algorithms.
///
/// Implementors provide only the histogram-analysis kernel
/// ([`compute_threshold`](AutoThreshold::compute_threshold)); the shared
/// scaffold — intensity extraction, histogram construction, and binary-mask
/// application — is provided as blanket default methods.
///
/// # Blanket methods
/// - [`compute`](AutoThreshold::compute): extract → histogram → threshold.
/// - [`apply`](AutoThreshold::apply): compute threshold → binary mask image.
///
/// # Sealing
/// Only the five concrete threshold types defined in this crate implement this
/// trait.  External implementors are prevented by the private `sealed::Sealed`
/// supertrait.
pub trait AutoThreshold: sealed::Sealed {
    /// Number of equally-spaced histogram bins to use when building the
    /// intensity histogram.
    fn num_bins(&self) -> usize;

    /// Compute the threshold intensity from a pre-built histogram.
    ///
    /// This is the algorithm-specific kernel.  The blanket
    /// [`compute`](AutoThreshold::compute) calls this method after extracting
    /// pixel intensities and building the histogram.
    ///
    /// # Arguments
    /// - `hist`   : raw per-bin pixel counts; `hist.len() == n_bins`.
    /// - `n_bins` : number of bins (equals `hist.len()`; supplied for
    ///   convenience so implementations need not call `.len()`).
    /// - `x_min`  : minimum intensity value in the image.
    /// - `x_max`  : maximum intensity value in the image.
    ///
    /// # Preconditions (guaranteed by the blanket `compute` before this call)
    /// - `x_max > x_min` — constant images are handled before `compute_threshold`
    ///   is called.
    /// - `n_bins >= 2`.
    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32;

    /// Extract image intensities, build an intensity histogram, and compute the
    /// threshold intensity.
    ///
    /// Returns `0.0` for an empty image and `x_min` for a constant image
    /// (degenerate cases handled before delegating to
    /// [`compute_threshold`](AutoThreshold::compute_threshold)).
    fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        let (vals, _) = extract_vec_infallible(image);
        let slice: &[f32] = &vals;

        if slice.is_empty() {
            return 0.0;
        }

        let x_min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let x_max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Degenerate: constant image has no separable classes.
        if (x_max - x_min).abs() < f32::EPSILON {
            return x_min;
        }

        let n_bins = self.num_bins();
        let hist = build_histogram(slice, n_bins, x_min, x_max);
        self.compute_threshold(&hist, n_bins, x_min, x_max)
    }

    /// Apply the auto-threshold to produce a binary mask.
    ///
    /// - Pixels with intensity ≥ t\* → `1.0` (foreground).
    /// - Pixels with intensity < t\* → `0.0` (background).
    ///
    /// Spatial metadata (origin, spacing, direction) is preserved exactly.
    fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        // Use fully-qualified syntax to call the trait method, not any
        // same-named inherent method on the concrete type.
        let threshold = <Self as AutoThreshold>::compute(self, image);

        let device = image.data().device();
        let shape: [usize; D] = image.shape();
        let (vals, _) = extract_vec_infallible(image);

        let output: Vec<f32> = vals
            .iter()
            .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
            .collect();

        let tensor = Tensor::<B, D>::from_data(TensorData::new(output, Shape::new(shape)), &device);

        Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        )
    }
}

// ── Histogram construction helper ──────────────────────────────────────────────

/// Build an equally-spaced histogram with `num_bins` bins over `[x_min, x_max]`.
///
/// Bin mapping:
///   `bin(v) = ⌊(v − x_min) / (x_max − x_min) · (num_bins − 1)⌋`
/// clamped to `[0, num_bins − 1]`.
///
/// The formula uses `f32` arithmetic (`(v - x_min) / range * num_bins_m1`) to
/// match the histogram-building convention used by the existing
/// `compute_*_from_slice` public utilities.
///
/// # Preconditions
/// - `num_bins >= 2`
/// - `x_max > x_min`
pub(crate) fn build_histogram(slice: &[f32], num_bins: usize, x_min: f32, x_max: f32) -> Vec<u32> {
    debug_assert!(num_bins >= 2, "invariant: num_bins >= 2");
    debug_assert!(x_max > x_min, "invariant: x_max > x_min");

    let range = x_max - x_min;
    let num_bins_m1 = (num_bins - 1) as f32;
    let mut counts = vec![0u32; num_bins];

    for &v in slice {
        // Formula matches the existing compute_*_from_slice implementations.
        let bin = ((v - x_min) / range * num_bins_m1).floor() as usize;
        let bin = bin.min(num_bins - 1);
        counts[bin] += 1;
    }

    counts
}
