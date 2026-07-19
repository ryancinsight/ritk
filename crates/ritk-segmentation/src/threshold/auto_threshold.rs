//! Sealed [`AutoThreshold`] trait: shared extract→histogram→mask scaffold for
//! single-threshold auto-selection algorithms.
//!
//! Implementors supply only [`AutoThreshold::compute_threshold`], which receives
//! a pre-built `&[u32]` histogram and returns the threshold intensity.  The
//! common extract→histogram and threshold→binary-mask pipelines are provided as
//! blanket default methods ([`AutoThreshold::compute`] /
//! [`AutoThreshold::apply`] and their Coeus-native counterparts).
//!
//! # Non-finite intensities
//!
//! NaN and ±Inf samples are excluded from histogram statistics and always map
//! to background (`0.0`). An input with no finite samples has threshold `0.0`
//! and an all-background mask.
//!
//! # Sealing
//! The trait is sealed via a private `sealed::Sealed` supertrait. Its twelve
//! implementors are the threshold strategies explicitly registered below.

use ritk_image::tensor::{Backend, Tensor};
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
impl sealed::Sealed for super::isodata::IsoDataThreshold {}
impl sealed::Sealed for super::moments::MomentsThreshold {}
impl sealed::Sealed for super::huang::HuangThreshold {}
impl sealed::Sealed for super::intermodes::IntermodesThreshold {}
impl sealed::Sealed for super::shanbhag::ShanbhagThreshold {}
impl sealed::Sealed for super::kittler::KittlerIllingworthThreshold {}
impl sealed::Sealed for super::renyi::RenyiEntropyThreshold {}

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
/// - [`compute_native`](AutoThreshold::compute_native): native slice → threshold.
/// - [`apply_native_with_threshold`](AutoThreshold::apply_native_with_threshold):
///   native slice → threshold and binary mask without duplicate extraction.
///
/// # Sealing
/// Only the twelve concrete threshold types registered in this module implement
/// this trait. External implementors are prevented by the private
/// `sealed::Sealed` supertrait.
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
    fn compute<B: Backend, const D: usize>(&self, image: &Image<f32, B, D>) -> f32 {
        let (vals, _) = extract_vec_infallible(image);
        threshold_from_slice(self, &vals)
    }

    /// Apply the auto-threshold to produce a binary mask.
    ///
    /// - Pixels with intensity ≥ t\* → `1.0` (foreground).
    /// - Pixels with intensity < t\* → `0.0` (background).
    ///
    /// Spatial metadata (origin, spacing, direction) is preserved exactly.
    fn apply<B: Backend, const D: usize>(&self, image: &Image<f32, B, D>) -> Image<f32, B, D> {
        // Extract the volume once and reuse the slice for both threshold
        // selection and masking. The previous form called `compute` (which
        // extracts internally) and then extracted a second time, cloning and
        // copying the whole volume twice per `apply`.
        let (vals, shape) = extract_vec_infallible(image);
        let (_, output) = threshold_mask_from_slice(self, &vals);

        let device = B::default();
        let tensor = Tensor::<f32, B>::from_slice_on(shape, &output, &device);

        Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        )
        .expect("invariant: segmentation output tensor preserves the image rank")
    }

    /// Compute the threshold directly from a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend storage is not available as a
    /// contiguous host slice.
    fn compute_native<B, const D: usize>(
        &self,
        image: &ritk_image::Image<f32, B, D>,
    ) -> anyhow::Result<f32>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        Ok(threshold_from_slice(self, image.data_slice()?))
    }

    /// Apply the threshold to a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend storage is not available as a
    /// contiguous host slice or when the output image cannot be constructed.
    fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        self.apply_native_with_threshold(image, backend)
            .map(|(output, _)| output)
    }

    /// Compute the threshold and native mask from one host-slice extraction.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend storage is not available as a
    /// contiguous host slice or when the output image cannot be constructed.
    fn apply_native_with_threshold<B, const D: usize>(
        &self,
        image: &ritk_image::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<(ritk_image::Image<f32, B, D>, f32)>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (threshold, output) = threshold_mask_from_slice(self, image.data_slice()?);
        let mask = ritk_image::Image::from_flat_on(
            output,
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )?;
        Ok((mask, threshold))
    }
}

// ── ITK histogram model ─────────────────────────────────────────────────────────
//
// ITK's `HistogramThresholdImageFilter` builds its histogram with
// `itk::Statistics::ImageToHistogramFilter` under `AutoMinimumMaximum`.  For a
// real-valued image that filter places the lower bin edge at the image minimum
// and the upper bin edge a *margin* above the image maximum:
//
//   max_edge   = x_max + (x_max − x_min) / (n_bins · MARGINAL_SCALE)
//   bin_width  = (max_edge − x_min) / n_bins
//   bin(v)     = clamp(⌊(v − x_min) / bin_width⌋, 0, n_bins − 1)
//
// with `MarginalScale = 100` (the ITK default).  The threshold calculators then
// report either the **right edge** of the selected bin (Otsu / multi-Otsu, via
// `Histogram::GetBinMax`) or the **bin centre** (Li / Yen / Kapur / Triangle, via
// `Histogram::GetMeasurement`).  This module is the single source of truth for
// that geometry; every algorithm derives its reported intensity from it so the
// trait path and the `compute_*_from_slice` path stay bit-identical.

/// ITK `ImageToHistogramFilter` default marginal scale.
pub(crate) const HISTOGRAM_MARGINAL_SCALE: f64 = 100.0;

/// ITK histogram bin width over `[x_min, x_max]` with `n_bins` bins.
///
/// Computed in `f64` to match ITK's `MeasurementType` accumulation, then the
/// caller maps intensities through it.  `x_max > x_min` and `n_bins >= 2` are
/// preconditions guaranteed by the degenerate-case guards in
/// [`threshold_from_slice`].
pub(crate) fn itk_bin_width(x_min: f32, x_max: f32, n_bins: usize) -> f64 {
    let (lo, hi) = (x_min as f64, x_max as f64);
    let max_edge = hi + (hi - lo) / (n_bins as f64 * HISTOGRAM_MARGINAL_SCALE);
    (max_edge - lo) / n_bins as f64
}

/// Centre intensity of bin `k`: `x_min + (k + 0.5) · bin_width`.
///
/// Matches ITK `Histogram::GetMeasurement(k, 0)` (Li / Yen / Kapur / Triangle).
pub(crate) fn bin_center(x_min: f32, bin_width: f64, k: usize) -> f32 {
    (x_min as f64 + (k as f64 + 0.5) * bin_width) as f32
}

/// Right edge of bin `k`: `x_min + (k + 1) · bin_width`.
///
/// Matches ITK `Histogram::GetBinMax(0, k)` (Otsu / multi-Otsu).
pub(crate) fn bin_right_edge(x_min: f32, bin_width: f64, k: usize) -> f32 {
    (x_min as f64 + (k as f64 + 1.0) * bin_width) as f32
}

// ── Histogram construction helper ──────────────────────────────────────────────

/// Build the ITK-convention histogram with `num_bins` bins over `[x_min, x_max]`.
///
/// Bin mapping uses [`itk_bin_width`]:
///   `bin(v) = clamp(⌊(v − x_min) / bin_width⌋, 0, num_bins − 1)`
///
/// # Preconditions
/// - `num_bins >= 2`
/// - `x_max > x_min`
pub(crate) fn build_histogram(slice: &[f32], num_bins: usize, x_min: f32, x_max: f32) -> Vec<u32> {
    debug_assert!(num_bins >= 2, "invariant: num_bins >= 2");
    debug_assert!(x_max > x_min, "invariant: x_max > x_min");

    let bin_width = itk_bin_width(x_min, x_max, num_bins);
    let lo = x_min as f64;
    let mut counts = vec![0u32; num_bins];

    for &v in slice.iter().filter(|value| value.is_finite()) {
        let bin = ((v as f64 - lo) / bin_width).floor() as usize;
        let bin = bin.min(num_bins - 1);
        counts[bin] += 1;
    }

    counts
}

/// Shared extract→histogram→threshold pipeline over a flat intensity slice.
///
/// This is the single entry point behind both [`AutoThreshold::compute`] (which
/// extracts the slice from an [`Image`]) and the per-algorithm
/// `compute_*_from_slice` convenience functions, guaranteeing they agree
/// bit-for-bit.
///
/// Returns `0.0` for an empty or all-nonfinite slice and `x_min` for a constant
/// finite slice. Non-finite samples do not contribute to the histogram.
pub(crate) fn threshold_from_slice<A: AutoThreshold + ?Sized>(algo: &A, slice: &[f32]) -> f32 {
    let Some((x_min, x_max, _)) = finite_bounds(slice) else {
        return 0.0;
    };

    // Degenerate: constant image has no separable classes.
    if (x_max - x_min).abs() < f32::EPSILON {
        return x_min;
    }

    let n_bins = algo.num_bins();
    let hist = build_histogram(slice, n_bins, x_min, x_max);
    algo.compute_threshold(&hist, n_bins, x_min, x_max)
}

/// Return finite minimum, maximum, and sample count without allocating.
pub(crate) fn finite_bounds(slice: &[f32]) -> Option<(f32, f32, usize)> {
    let mut finite = slice.iter().copied().filter(|value| value.is_finite());
    let first = finite.next()?;
    let (minimum, maximum, additional) = finite.fold(
        (first, first, 0_usize),
        |(minimum, maximum, count), value| (minimum.min(value), maximum.max(value), count + 1),
    );
    Some((minimum, maximum, additional + 1))
}

fn threshold_mask_from_slice<A: AutoThreshold + ?Sized>(
    algorithm: &A,
    slice: &[f32],
) -> (f32, Vec<f32>) {
    let threshold = threshold_from_slice(algorithm, slice);
    let mask = slice
        .iter()
        .map(|&value| {
            if value.is_finite() && value >= threshold {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    (threshold, mask)
}

#[cfg(test)]
#[path = "tests_auto_threshold_native.rs"]
mod tests_auto_threshold_native;
