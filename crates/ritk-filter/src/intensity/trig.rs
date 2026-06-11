//! Pixelwise trigonometric and bounded-reciprocal intensity filters.
//!
//! Each filter is a pure pixelwise map `f : f32 → f32` applied independently to
//! every voxel in a 3-D image. Spatial metadata (origin, spacing, direction) is
//! preserved identically in every output image.
//!
//! All trigonometric arguments are interpreted as radians (matching ITK convention).
//!
//! # ITK / ImageJ / SimpleITK parity
//!
//! | Filter                          | ITK class                             | Note                    |
//! |---------------------------------|---------------------------------------|-------------------------|
//! | `AtanImageFilter`               | `AtanImageFilter`                     | atan(x) ∈ (−π/2, π/2)  |
//! | `SinImageFilter`                | `SinImageFilter`                      | sin(x), x in radians    |
//! | `CosImageFilter`                | `CosImageFilter`                      | cos(x), x in radians    |
//! | `TanImageFilter`                | `TanImageFilter`                      | tan(x), x in radians    |
//! | `AsinImageFilter`               | `AsinImageFilter`                     | asin(x), domain [−1,1]  |
//! | `AcosImageFilter`               | `AcosImageFilter`                     | acos(x), domain [−1,1]  |
//! | `BoundedReciprocalImageFilter`  | `BoundedReciprocalImageFilter`        | 1 / (1 + |x|)           |

use ritk_core::filter::ops::{extract_vec_infallible as extract_vec, rebuild};
use ritk_image::Image;
use burn::tensor::backend::Backend;

// ── AtanImageFilter ───────────────────────────────────────────────────────────

/// Pixelwise arctangent filter.
///
/// # Mathematical Specification
///
/// `out(x) = atan(in(x))`
///
/// Range: `(−π/2, π/2)` for all finite inputs.
///
/// # Properties
/// - Bijective on ℝ with range `(−π/2, π/2)`.
/// - Odd function: `atan(−x) = −atan(x)`.
/// - Preserves spatial metadata.
/// - O(N) time and output space.
///
/// # References
/// - ITK `itk::AtanImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AtanImageFilter;

impl AtanImageFilter {
    /// Construct a new `AtanImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the arctangent transform to every voxel.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::atan).collect();
        rebuild(out, dims, image)
    }
}

// ── SinImageFilter ────────────────────────────────────────────────────────────

/// Pixelwise sine filter.
///
/// # Mathematical Specification
///
/// `out(x) = sin(in(x))`
///
/// Input interpreted as radians. Range: `[−1, 1]` for real inputs.
///
/// # Properties
/// - `sin(0) = 0`, `sin(π/2) = 1`, `sin(π) ≈ 0`.
/// - Preserves spatial metadata.
/// - O(N) time and output space.
///
/// # References
/// - ITK `itk::SinImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SinImageFilter;

impl SinImageFilter {
    /// Construct a new `SinImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the sine transform to every voxel (input in radians).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::sin).collect();
        rebuild(out, dims, image)
    }
}

// ── CosImageFilter ────────────────────────────────────────────────────────────

/// Pixelwise cosine filter.
///
/// # Mathematical Specification
///
/// `out(x) = cos(in(x))`
///
/// Input interpreted as radians. Range: `[−1, 1]` for real inputs.
///
/// # Properties
/// - `cos(0) = 1`, `cos(π/2) ≈ 0`, `cos(π) = −1`.
/// - Even function: `cos(−x) = cos(x)`.
/// - Preserves spatial metadata.
/// - O(N) time and output space.
///
/// # References
/// - ITK `itk::CosImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CosImageFilter;

impl CosImageFilter {
    /// Construct a new `CosImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the cosine transform to every voxel (input in radians).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::cos).collect();
        rebuild(out, dims, image)
    }
}

// ── TanImageFilter ────────────────────────────────────────────────────────────

/// Pixelwise tangent filter.
///
/// # Mathematical Specification
///
/// `out(x) = tan(in(x))`
///
/// Input interpreted as radians. Undefined at `x = π/2 + nπ` (produces `±∞` or `NaN`).
///
/// # Properties
/// - `tan(0) = 0`, `tan(π/4) = 1`.
/// - Odd function: `tan(−x) = −tan(x)`.
/// - Preserves spatial metadata.
/// - O(N) time and output space.
///
/// # References
/// - ITK `itk::TanImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TanImageFilter;

impl TanImageFilter {
    /// Construct a new `TanImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the tangent transform to every voxel (input in radians).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::tan).collect();
        rebuild(out, dims, image)
    }
}

// ── AsinImageFilter ───────────────────────────────────────────────────────────

/// Pixelwise arcsine filter.
///
/// # Mathematical Specification
///
/// `out(x) = asin(in(x))`
///
/// Domain: `[−1, 1]`. Range: `[−π/2, π/2]`. Values outside `[−1, 1]` produce `NaN`.
///
/// # Properties
/// - `asin(0) = 0`, `asin(1) = π/2`, `asin(−1) = −π/2`.
/// - Odd function: `asin(−x) = −asin(x)`.
/// - Preserves spatial metadata.
/// - O(N) time and output space.
///
/// # References
/// - ITK `itk::AsinImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AsinImageFilter;

impl AsinImageFilter {
    /// Construct a new `AsinImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the arcsine transform to every voxel.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::asin).collect();
        rebuild(out, dims, image)
    }
}

// ── AcosImageFilter ───────────────────────────────────────────────────────────

/// Pixelwise arccosine filter.
///
/// # Mathematical Specification
///
/// `out(x) = acos(in(x))`
///
/// Domain: `[−1, 1]`. Range: `[0, π]`. Values outside `[−1, 1]` produce `NaN`.
///
/// # Properties
/// - `acos(1) = 0`, `acos(0) = π/2`, `acos(−1) = π`.
/// - Complement relation: `asin(x) + acos(x) = π/2` for `x ∈ [−1, 1]`.
/// - Preserves spatial metadata.
/// - O(N) time and output space.
///
/// # References
/// - ITK `itk::AcosImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AcosImageFilter;

impl AcosImageFilter {
    /// Construct a new `AcosImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the arccosine transform to every voxel.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::acos).collect();
        rebuild(out, dims, image)
    }
}

// ── BoundedReciprocalImageFilter ──────────────────────────────────────────────

/// Pixelwise bounded-reciprocal filter.
///
/// # Mathematical Specification
///
/// `out(x) = 1 / (1 + |in(x)|)`
///
/// Range: `(0, 1]` for all finite inputs. Achieves maximum `1.0` at `in(x) = 0`.
///
/// # Properties
/// - Monotone decreasing on `[0, ∞)`: larger `|x|` → smaller output.
/// - At zero: `out(0) = 1`.
/// - Asymptote: `lim_{|x|→∞} out(x) = 0`.
/// - Bounded: `out(x) ∈ (0, 1]` for all finite inputs.
/// - Preserves spatial metadata.
/// - O(N) time and output space.
///
/// # References
/// - ITK `itk::BoundedReciprocalImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoundedReciprocalImageFilter;

impl BoundedReciprocalImageFilter {
    /// Construct a new `BoundedReciprocalImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the bounded-reciprocal transform `1 / (1 + |x|)` to every voxel.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals
            .into_iter()
            .map(|x| 1.0_f32 / (1.0 + x.abs()))
            .collect();
        rebuild(out, dims, image)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_trig.rs"]
mod tests_trig;
