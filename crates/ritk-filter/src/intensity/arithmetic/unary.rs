//! Generic unary pixelwise intensity filter.
//!
//! A single generic implementation `UnaryImageFilter<Op>` replaces the
//! identical per-operation scaffolds (`abs`, `sqrt`, `exp`, `log`, `square`,
//! `log10`, `exp_negative`), which differed only in one closure.  Variation is
//! encoded through the sealed `UnaryPixelOp` trait; each ZST marker struct
//! implements one operation.  Type aliases (`AbsImageFilter`, `SqrtImageFilter`,
//! â€¦) preserve the original public API.

use std::marker::PhantomData;

use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec_infallible as extract_vec, rebuild};

// â”€â”€ Sealed trait infrastructure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

mod sealed {
    pub trait Sealed {}
}

/// Elementwise `f32 â†’ f32` operation applied by [`UnaryImageFilter`].
///
/// This trait is **sealed**: external crates cannot implement it.  Only the
/// marker structs defined in this module (`Abs`, `Sqrt`, `Exp`, `Log`,
/// `Square`, `Log10`, `ExpNegative`) are valid implementations.
pub trait UnaryPixelOp: sealed::Sealed {
    /// Apply the operation to a single voxel value.
    fn apply(v: f32) -> f32;
}

// â”€â”€ Operation marker ZSTs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Operation marker for `out(x) = |in(x)|`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Abs;

/// Operation marker for `out(x) = âˆšin(x)`.
///
/// IEEE 754: negative inputs yield `NaN` (matching ITK behaviour).
#[derive(Debug, Clone, Copy, Default)]
pub struct Sqrt;

/// Operation marker for `out(x) = e^{in(x)}`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Exp;

/// Operation marker for `out(x) = ln(in(x))`.
///
/// IEEE 754: `ln(0) = âˆ’âˆž`, `ln(negative) = NaN` (matching ITK behaviour).
#[derive(Debug, Clone, Copy, Default)]
pub struct Log;

/// Operation marker for `out(x) = in(x)Â²`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Square;

/// Operation marker for `out(x) = logâ‚â‚€(in(x))`.
///
/// IEEE 754: `logâ‚â‚€(0) = âˆ’âˆž`, `logâ‚â‚€(negative) = NaN` (matching ITK
/// `Log10ImageFilter`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Log10;

/// Operation marker for `out(x) = e^{âˆ’in(x)}`.
///
/// Matches ITK `ExpNegativeImageFilter` (`std::exp(-x)`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ExpNegative;

/// Operation marker for `out(x) = âˆ’in(x)`.
///
/// Matches ITK `UnaryMinusImageFilter`.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnaryMinus;

/// Operation marker for `out(x) = round(in(x))` to the nearest integer.
///
/// Matches ITK `RoundImageFilter` / `itk::Math::Round` â€” **round half-integer
/// up** (toward +âˆž): `floor(x + 0.5)`. This differs from Rust `f32::round`
/// (round half away from zero) on negative half-integers (e.g. âˆ’2.5 â†’ âˆ’2 here,
/// âˆ’3 for `f32::round`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Round;

/// Operation marker for the logical NOT of a mask: `out(x) = 1` where
/// `in(x) == 0`, else `0`.
///
/// Matches ITK `NotImageFilter` (`!A` in NumericTraits): any nonzero input is
/// "true" â†’ `0`, only an exact zero is "false" â†’ `1`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Not;

/// Operation marker for `out(x) = atan(x)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Atan;
/// Operation marker for `out(x) = sin(x)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sin;
/// Operation marker for `out(x) = cos(x)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cos;
/// Operation marker for `out(x) = tan(x)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tan;
/// Operation marker for `out(x) = asin(x)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Asin;
/// Operation marker for `out(x) = acos(x)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Acos;
/// Operation marker for `out(x) = 1 / (1 + |x|)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoundedReciprocal;

impl sealed::Sealed for Abs {}
impl sealed::Sealed for Sqrt {}
impl sealed::Sealed for Exp {}
impl sealed::Sealed for Log {}
impl sealed::Sealed for Square {}
impl sealed::Sealed for Log10 {}
impl sealed::Sealed for ExpNegative {}
impl sealed::Sealed for UnaryMinus {}
impl sealed::Sealed for Round {}
impl sealed::Sealed for Not {}
impl sealed::Sealed for Atan {}
impl sealed::Sealed for Sin {}
impl sealed::Sealed for Cos {}
impl sealed::Sealed for Tan {}
impl sealed::Sealed for Asin {}
impl sealed::Sealed for Acos {}
impl sealed::Sealed for BoundedReciprocal {}

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

impl UnaryPixelOp for Not {
    #[inline]
    fn apply(v: f32) -> f32 {
        // ITK `!A`: nonzero â†’ false (0), exact zero â†’ true (1).
        (v == 0.0) as u8 as f32
    }
}

impl UnaryPixelOp for Atan {
    #[inline]
    fn apply(v: f32) -> f32 {
        v.atan()
    }
}
impl UnaryPixelOp for Sin {
    #[inline]
    fn apply(v: f32) -> f32 {
        v.sin()
    }
}
impl UnaryPixelOp for Cos {
    #[inline]
    fn apply(v: f32) -> f32 {
        v.cos()
    }
}
impl UnaryPixelOp for Tan {
    #[inline]
    fn apply(v: f32) -> f32 {
        v.tan()
    }
}
impl UnaryPixelOp for Asin {
    #[inline]
    fn apply(v: f32) -> f32 {
        v.asin()
    }
}
impl UnaryPixelOp for Acos {
    #[inline]
    fn apply(v: f32) -> f32 {
        v.acos()
    }
}
impl UnaryPixelOp for BoundedReciprocal {
    #[inline]
    fn apply(v: f32) -> f32 {
        1.0 / (1.0 + v.abs())
    }
}

// â”€â”€ Generic filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<f32, B, D>) -> Image<f32, B, D> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(|v| Op::apply(v)).collect();
        rebuild(out, dims, image)
    }

    /// Apply the operation pixelwise to a Coeus-native 3-D `image`.
    ///
    /// Mirrors [`Self::apply`] on the Atlas-native substrate for the common
    /// 3-D medical-image case. Spatial metadata (origin, spacing, direction) is
    /// preserved. The operation is identical to the Burn path â€” both route
    /// through the same flat-buffer core.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |vals, _dims| {
            vals.iter().copied().map(|v| Op::apply(v)).collect()
        })
    }
}

// â”€â”€ Public type aliases â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Pixelwise absolute value filter.  `out(x) = |in(x)|`.
///
/// # References
/// - ITK `itk::AbsImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Abs.
pub type AbsImageFilter = UnaryImageFilter<Abs>;

/// Pixelwise square-root filter.  `out(x) = âˆšin(x)`.
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
/// `ln(0) = âˆ’âˆž`, `ln(negative) = NaN` (IEEE 754 / ITK behaviour).
///
/// # References
/// - ITK `itk::LogImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Log.
pub type LogImageFilter = UnaryImageFilter<Log>;

/// Pixelwise squaring filter.  `out(x) = in(x)Â²`.
///
/// Non-negative output for all real inputs.
///
/// # References
/// - ITK `itk::SquareImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Square.
pub type SquareImageFilter = UnaryImageFilter<Square>;

/// Pixelwise base-10 logarithm filter.  `out(x) = logâ‚â‚€(in(x))`.
///
/// `logâ‚â‚€(0) = âˆ’âˆž`, `logâ‚â‚€(negative) = NaN` (IEEE 754 / ITK behaviour).
///
/// # References
/// - ITK `itk::Log10ImageFilter<TInputImage, TOutputImage>`.
pub type Log10ImageFilter = UnaryImageFilter<Log10>;

/// Pixelwise negative-exponential filter.  `out(x) = e^{âˆ’in(x)}`.
///
/// # References
/// - ITK `itk::ExpNegativeImageFilter<TInputImage, TOutputImage>`.
pub type ExpNegativeImageFilter = UnaryImageFilter<ExpNegative>;

/// Pixelwise unary minus filter.  `out(x) = âˆ’in(x)`.
///
/// # References
/// - ITK `itk::UnaryMinusImageFilter<TInputImage, TOutputImage>`.
pub type UnaryMinusImageFilter = UnaryImageFilter<UnaryMinus>;

/// Pixelwise round-to-nearest-integer filter.  `out(x) = âŒŠin(x) + Â½âŒ‹`.
///
/// Round half-integer up (toward +âˆž), matching ITK `itk::Math::Round`.
///
/// # References
/// - ITK `itk::RoundImageFilter<TInputImage, TOutputImage>`.
pub type RoundImageFilter = UnaryImageFilter<Round>;

/// Pixelwise logical NOT of a mask.  `out(x) = 1` where `in(x) == 0`, else `0`.
///
/// Any nonzero value is treated as "true" (â†’ `0`); only an exact zero is
/// "false" (â†’ `1`).
///
/// # References
/// - ITK `itk::NotImageFilter<TInputImage, TOutputImage>`.
pub type NotImageFilter = UnaryImageFilter<Not>;

/// Pixelwise arctangent filter.
pub type AtanImageFilter = UnaryImageFilter<Atan>;
/// Pixelwise sine filter.
pub type SinImageFilter = UnaryImageFilter<Sin>;
/// Pixelwise cosine filter.
pub type CosImageFilter = UnaryImageFilter<Cos>;
/// Pixelwise tangent filter.
pub type TanImageFilter = UnaryImageFilter<Tan>;
/// Pixelwise arcsine filter.
pub type AsinImageFilter = UnaryImageFilter<Asin>;
/// Pixelwise arccosine filter.
pub type AcosImageFilter = UnaryImageFilter<Acos>;
/// Pixelwise bounded reciprocal filter.
pub type BoundedReciprocalImageFilter = UnaryImageFilter<BoundedReciprocal>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_support::{make_native_image, native_vals};
    use coeus_core::SequentialBackend;

    /// `logâ‚â‚€(10â¿) = n` for exact powers of ten, matching ITK `Log10ImageFilter`.
    #[test]
    fn log10_of_powers_of_ten() {
        let img = make_native_image(vec![1.0, 10.0, 100.0, 1000.0], [1, 1, 4]);
        let out = Log10ImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        let v = native_vals(&out);
        for (got, exp) in v.iter().zip([0.0_f32, 1.0, 2.0, 3.0]) {
            assert!((got - exp).abs() < 1e-5, "log10: got {got}, expected {exp}");
        }
    }

    /// `e^{âˆ’0} = 1` and `e^{âˆ’x}` is the reciprocal of `e^{x}`, matching ITK
    /// `ExpNegativeImageFilter`.
    #[test]
    fn exp_negative_matches_reciprocal_exp() {
        let xs = vec![0.0_f32, 1.0, 2.5, -1.0];
        let img = make_native_image(xs.clone(), [1, 1, 4]);
        let out = ExpNegativeImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        let v = native_vals(&out);
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
        let img = make_native_image(vec![0.0, 3.0, -2.5, 7.0], [1, 1, 4]);
        let out = UnaryMinusImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        assert_eq!(native_vals(&out), vec![0.0, -3.0, 2.5, -7.0]);
    }

    /// Round half-integer up (ITK `Math::Round`): note âˆ’2.5 â†’ âˆ’2, not âˆ’3.
    #[test]
    fn round_half_integer_up() {
        let img = make_native_image(vec![2.4, 2.5, 2.6, -2.5, -2.4, -0.5], [1, 1, 6]);
        let out = RoundImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        assert_eq!(native_vals(&out), vec![2.0, 3.0, 3.0, -2.0, -2.0, 0.0]);
    }

    /// Logical NOT (ITK `NotImageFilter`): only exact zero maps to `1`; any
    /// nonzero value (incl. negatives and fractions) maps to `0`.
    #[test]
    fn logical_not_maps_zero_to_one() {
        let img = make_native_image(vec![0.0, 1.0, 2.0, -3.0, 0.5], [1, 1, 5]);
        let out = NotImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        assert_eq!(native_vals(&out), vec![1.0, 0.0, 0.0, 0.0, 0.0]);
    }
}
