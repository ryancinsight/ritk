//! The scalar arithmetic the band update needs, for the two filters' precisions.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// The arithmetic the band update needs, for the two filters' precisions.
///
/// `from_f64`/`to_f64` exist so the shared constants (`1`, `½`, the `1e-6` nest
/// floor) and the `f64` RMS accumulator can be written once; the comparisons and
/// the band arithmetic itself stay in `T`, so `f32` runs exactly as `f32`.
pub(crate) trait SparseScalar:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
}

impl SparseScalar for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn abs(self) -> Self {
        f32::abs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl SparseScalar for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}
