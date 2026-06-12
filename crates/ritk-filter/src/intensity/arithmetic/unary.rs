//! Generic unary pixelwise intensity filter.
//!
//! A single generic implementation `UnaryImageFilter<Op>` replaces the five
//! identical scaffolds (`abs`, `sqrt`, `exp`, `log`, `square`), which differed
//! only in one closure.  Variation is encoded through the sealed `UnaryPixelOp`
//! trait; each ZST marker struct implements one operation.  Type aliases
//! (`AbsImageFilter`, `SqrtImageFilter`, …) preserve the original public API.

use std::marker::PhantomData;

use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_tensor_ops::{extract_vec_infallible as extract_vec, rebuild};

// ── Sealed trait infrastructure ──────────────────────────────────────────────

mod sealed {
    pub trait Sealed {}
}

/// Elementwise `f32 → f32` operation applied by [`UnaryImageFilter`].
///
/// This trait is **sealed**: external crates cannot implement it.  Only the
/// five marker structs defined in this module (`Abs`, `Sqrt`, `Exp`, `Log`,
/// `Square`) are valid implementations.
pub trait UnaryPixelOp: sealed::Sealed {
    /// Apply the operation to a single voxel value.
    fn apply(v: f32) -> f32;
}

// ── Operation marker ZSTs ────────────────────────────────────────────────────

/// Operation marker for `out(x) = |in(x)|`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Abs;

/// Operation marker for `out(x) = √in(x)`.
///
/// IEEE 754: negative inputs yield `NaN` (matching ITK behaviour).
#[derive(Debug, Clone, Copy, Default)]
pub struct Sqrt;

/// Operation marker for `out(x) = e^{in(x)}`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Exp;

/// Operation marker for `out(x) = ln(in(x))`.
///
/// IEEE 754: `ln(0) = −∞`, `ln(negative) = NaN` (matching ITK behaviour).
#[derive(Debug, Clone, Copy, Default)]
pub struct Log;

/// Operation marker for `out(x) = in(x)²`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Square;

impl sealed::Sealed for Abs {}
impl sealed::Sealed for Sqrt {}
impl sealed::Sealed for Exp {}
impl sealed::Sealed for Log {}
impl sealed::Sealed for Square {}

impl UnaryPixelOp for Abs {
    #[inline]
    fn apply(v: f32) -> f32 {
        v.abs()
    }
}

impl UnaryPixelOp for Sqrt {
    #[inline]
    fn apply(v: f32) -> f32 {
        v.sqrt()
    }
}

impl UnaryPixelOp for Exp {
    #[inline]
    fn apply(v: f32) -> f32 {
        v.exp()
    }
}

impl UnaryPixelOp for Log {
    #[inline]
    fn apply(v: f32) -> f32 {
        v.ln()
    }
}

impl UnaryPixelOp for Square {
    #[inline]
    fn apply(v: f32) -> f32 {
        v * v
    }
}

// ── Generic filter ────────────────────────────────────────────────────────────

/// Generic pixelwise unary intensity filter parameterised by an [`UnaryPixelOp`].
///
/// Applies `Op::apply` independently to every voxel.  Spatial metadata
/// (origin, spacing, direction) is preserved identically in the output image.
///
/// # Type parameters
/// - `Op`: a sealed [`UnaryPixelOp`] marker ZST selecting the operation.
///
/// # Complexity
/// O(N) time, O(N) output space.
///
/// # Examples
/// ```ignore
/// let out = AbsImageFilter::new().apply(&img);
/// let out = SqrtImageFilter::new().apply(&img);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct UnaryImageFilter<Op: UnaryPixelOp> {
    _op: PhantomData<Op>,
}

impl<Op: UnaryPixelOp> UnaryImageFilter<Op> {
    /// Construct a new `UnaryImageFilter`.
    pub fn new() -> Self {
        Self { _op: PhantomData }
    }

    /// Apply the operation pixelwise to `image`.
    ///
    /// Works for any spatial dimensionality `D`; spatial metadata is preserved.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(|v| Op::apply(v)).collect();
        rebuild(out, dims, image)
    }
}

// ── Public type aliases ───────────────────────────────────────────────────────

/// Pixelwise absolute value filter.  `out(x) = |in(x)|`.
///
/// # References
/// - ITK `itk::AbsImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Abs.
pub type AbsImageFilter = UnaryImageFilter<Abs>;

/// Pixelwise square-root filter.  `out(x) = √in(x)`.
///
/// Negative inputs produce `NaN` (IEEE 754 / ITK behaviour).
///
/// # References
/// - ITK `itk::SqrtImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Square Root.
pub type SqrtImageFilter = UnaryImageFilter<Sqrt>;

/// Pixelwise natural exponential filter.  `out(x) = e^{in(x)}`.
///
/// # References
/// - ITK `itk::ExpImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Exp.
pub type ExpImageFilter = UnaryImageFilter<Exp>;

/// Pixelwise natural logarithm filter.  `out(x) = ln(in(x))`.
///
/// `ln(0) = −∞`, `ln(negative) = NaN` (IEEE 754 / ITK behaviour).
///
/// # References
/// - ITK `itk::LogImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Log.
pub type LogImageFilter = UnaryImageFilter<Log>;

/// Pixelwise squaring filter.  `out(x) = in(x)²`.
///
/// Non-negative output for all real inputs.
///
/// # References
/// - ITK `itk::SquareImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Square.
pub type SquareImageFilter = UnaryImageFilter<Square>;
