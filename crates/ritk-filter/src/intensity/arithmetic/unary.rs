//! Generic unary pixelwise intensity filter.
//!
//! A single generic implementation `UnaryImageFilter<Op>` replaces the
//! identical per-operation scaffolds (`abs`, `sqrt`, `exp`, `log`, `square`,
//! `log10`, `exp_negative`), which differed only in one closure.  Variation is
//! encoded through the sealed `UnaryPixelOp` trait; each ZST marker struct
//! implements one operation.  Type aliases (`AbsImageFilter`, `SqrtImageFilter`,
//! …) preserve the original public API.

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
/// marker structs defined in this module (`Abs`, `Sqrt`, `Exp`, `Log`,
/// `Square`, `Log10`, `ExpNegative`) are valid implementations.
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

/// Operation marker for `out(x) = log₁₀(in(x))`.
///
/// IEEE 754: `log₁₀(0) = −∞`, `log₁₀(negative) = NaN` (matching ITK
/// `Log10ImageFilter`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Log10;

/// Operation marker for `out(x) = e^{−in(x)}`.
///
/// Matches ITK `ExpNegativeImageFilter` (`std::exp(-x)`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ExpNegative;

/// Operation marker for `out(x) = −in(x)`.
///
/// Matches ITK `UnaryMinusImageFilter`.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnaryMinus;

/// Operation marker for `out(x) = round(in(x))` to the nearest integer.
///
/// Matches ITK `RoundImageFilter` / `itk::Math::Round` — **round half-integer
/// up** (toward +∞): `floor(x + 0.5)`. This differs from Rust `f32::round`
/// (round half away from zero) on negative half-integers (e.g. −2.5 → −2 here,
/// −3 for `f32::round`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Round;

impl sealed::Sealed for Abs {}
impl sealed::Sealed for Sqrt {}
impl sealed::Sealed for Exp {}
impl sealed::Sealed for Log {}
impl sealed::Sealed for Square {}
impl sealed::Sealed for Log10 {}
impl sealed::Sealed for ExpNegative {}
impl sealed::Sealed for UnaryMinus {}
impl sealed::Sealed for Round {}

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

impl UnaryPixelOp for Log10 {
    #[inline]
    fn apply(v: f32) -> f32 {
        v.log10()
    }
}

impl UnaryPixelOp for ExpNegative {
    #[inline]
    fn apply(v: f32) -> f32 {
        (-v).exp()
    }
}

impl UnaryPixelOp for UnaryMinus {
    #[inline]
    fn apply(v: f32) -> f32 {
        -v
    }
}

impl UnaryPixelOp for Round {
    #[inline]
    fn apply(v: f32) -> f32 {
        // ITK Math::Round = round half-integer up (floor(x + 0.5)).
        (v + 0.5).floor()
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

/// Pixelwise base-10 logarithm filter.  `out(x) = log₁₀(in(x))`.
///
/// `log₁₀(0) = −∞`, `log₁₀(negative) = NaN` (IEEE 754 / ITK behaviour).
///
/// # References
/// - ITK `itk::Log10ImageFilter<TInputImage, TOutputImage>`.
pub type Log10ImageFilter = UnaryImageFilter<Log10>;

/// Pixelwise negative-exponential filter.  `out(x) = e^{−in(x)}`.
///
/// # References
/// - ITK `itk::ExpNegativeImageFilter<TInputImage, TOutputImage>`.
pub type ExpNegativeImageFilter = UnaryImageFilter<ExpNegative>;

/// Pixelwise unary minus filter.  `out(x) = −in(x)`.
///
/// # References
/// - ITK `itk::UnaryMinusImageFilter<TInputImage, TOutputImage>`.
pub type UnaryMinusImageFilter = UnaryImageFilter<UnaryMinus>;

/// Pixelwise round-to-nearest-integer filter.  `out(x) = ⌊in(x) + ½⌋`.
///
/// Round half-integer up (toward +∞), matching ITK `itk::Math::Round`.
///
/// # References
/// - ITK `itk::RoundImageFilter<TInputImage, TOutputImage>`.
pub type RoundImageFilter = UnaryImageFilter<Round>;

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_image::test_support as ts;

    type B = burn_ndarray::NdArray<f32>;

    /// `log₁₀(10ⁿ) = n` for exact powers of ten, matching ITK `Log10ImageFilter`.
    #[test]
    fn log10_of_powers_of_ten() {
        let img = ts::make_image::<B, 3>(vec![1.0, 10.0, 100.0, 1000.0], [1, 1, 4]);
        let out = Log10ImageFilter::new().apply(&img);
        let v = out.data_slice().into_owned();
        for (got, exp) in v.iter().zip([0.0_f32, 1.0, 2.0, 3.0]) {
            assert!((got - exp).abs() < 1e-5, "log10: got {got}, expected {exp}");
        }
    }

    /// `e^{−0} = 1` and `e^{−x}` is the reciprocal of `e^{x}`, matching ITK
    /// `ExpNegativeImageFilter`.
    #[test]
    fn exp_negative_matches_reciprocal_exp() {
        let xs = vec![0.0_f32, 1.0, 2.5, -1.0];
        let img = ts::make_image::<B, 3>(xs.clone(), [1, 1, 4]);
        let out = ExpNegativeImageFilter::new().apply(&img);
        let v = out.data_slice().into_owned();
        for (got, &x) in v.iter().zip(xs.iter()) {
            let expected = (-x).exp();
            assert!(
                (got - expected).abs() <= 1e-6 * expected.abs().max(1.0),
                "exp(-{x}): got {got}, expected {expected}"
            );
        }
    }

    /// Unary minus negates every voxel.
    #[test]
    fn unary_minus_negates() {
        let img = ts::make_image::<B, 3>(vec![0.0, 3.0, -2.5, 7.0], [1, 1, 4]);
        let out = UnaryMinusImageFilter::new().apply(&img);
        assert_eq!(out.data_slice().into_owned(), vec![0.0, -3.0, 2.5, -7.0]);
    }

    /// Round half-integer up (ITK `Math::Round`): note −2.5 → −2, not −3.
    #[test]
    fn round_half_integer_up() {
        let img = ts::make_image::<B, 3>(vec![2.4, 2.5, 2.6, -2.5, -2.4, -0.5], [1, 1, 6]);
        let out = RoundImageFilter::new().apply(&img);
        assert_eq!(
            out.data_slice().into_owned(),
            vec![2.0, 3.0, 3.0, -2.0, -2.0, 0.0]
        );
    }
}
