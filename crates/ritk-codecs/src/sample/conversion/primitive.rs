//! The single boundary for sample conversions between Rust numeric primitives.
//!
//! These casts follow [the Rust Reference's numeric cast rules]: integer
//! narrowing keeps the low target bits, integer-to-float rounds to nearest
//! with ties to even, float-to-integer truncates toward zero and saturates,
//! and float narrowing rounds to the target precision.
//!
//! [the Rust Reference's numeric cast rules]: https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast

#![expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "Rust primitive cast semantics are the explicit lossy-conversion policy"
)]

/// Primitive numeric conversions used by the sealed sample vocabulary.
pub(super) trait PrimitiveSample: Sized {
    /// Convert an unsigned integer sample using Rust's primitive cast rules.
    fn from_unsigned_sample(value: u64) -> Self;

    /// Convert a signed integer sample using Rust's primitive cast rules.
    fn from_signed_sample(value: i64) -> Self;

    /// Convert a real-valued sample using Rust's primitive cast rules.
    fn from_real_sample(value: f64) -> Self;
}

impl PrimitiveSample for u8 {
    fn from_unsigned_sample(value: u64) -> Self { value as u8 }
    fn from_signed_sample(value: i64) -> Self { value as u8 }
    fn from_real_sample(value: f64) -> Self { value as u8 }
}

impl PrimitiveSample for i8 {
    fn from_unsigned_sample(value: u64) -> Self { value as i8 }
    fn from_signed_sample(value: i64) -> Self { value as i8 }
    fn from_real_sample(value: f64) -> Self { value as i8 }
}

impl PrimitiveSample for u16 {
    fn from_unsigned_sample(value: u64) -> Self { value as u16 }
    fn from_signed_sample(value: i64) -> Self { value as u16 }
    fn from_real_sample(value: f64) -> Self { value as u16 }
}

impl PrimitiveSample for i16 {
    fn from_unsigned_sample(value: u64) -> Self { value as i16 }
    fn from_signed_sample(value: i64) -> Self { value as i16 }
    fn from_real_sample(value: f64) -> Self { value as i16 }
}

impl PrimitiveSample for u32 {
    fn from_unsigned_sample(value: u64) -> Self { value as u32 }
    fn from_signed_sample(value: i64) -> Self { value as u32 }
    fn from_real_sample(value: f64) -> Self { value as u32 }
}

impl PrimitiveSample for i32 {
    fn from_unsigned_sample(value: u64) -> Self { value as i32 }
    fn from_signed_sample(value: i64) -> Self { value as i32 }
    fn from_real_sample(value: f64) -> Self { value as i32 }
}

impl PrimitiveSample for u64 {
    fn from_unsigned_sample(value: u64) -> Self { value }
    fn from_signed_sample(value: i64) -> Self { value.cast_unsigned() }
    fn from_real_sample(value: f64) -> Self { value as u64 }
}

impl PrimitiveSample for i64 {
    fn from_unsigned_sample(value: u64) -> Self { value.cast_signed() }
    fn from_signed_sample(value: i64) -> Self { value }
    fn from_real_sample(value: f64) -> Self { value as i64 }
}

impl PrimitiveSample for f32 {
    fn from_unsigned_sample(value: u64) -> Self { value as f32 }
    fn from_signed_sample(value: i64) -> Self { value as f32 }
    fn from_real_sample(value: f64) -> Self { value as f32 }
}

impl PrimitiveSample for f64 {
    fn from_unsigned_sample(value: u64) -> Self { value as f64 }
    fn from_signed_sample(value: i64) -> Self { value as f64 }
    fn from_real_sample(value: f64) -> Self { value }
}

#[cfg(test)]
mod tests {
    use super::PrimitiveSample;

    #[test]
    fn float_narrowing_rounds_at_the_binary32_precision() {
        assert_eq!(
            <f32 as PrimitiveSample>::from_real_sample(1.0 + f64::EPSILON).to_bits(),
            1.0_f32.to_bits()
        );
        assert_eq!(
            <f32 as PrimitiveSample>::from_real_sample(1.0 + 3.0 * 2.0_f64.powi(-24)).to_bits(),
            (1.0_f32.to_bits() + 2)
        );

        let least_subnormal = f64::from(f32::from_bits(1));
        assert_eq!(
            <f32 as PrimitiveSample>::from_real_sample(least_subnormal).to_bits(),
            1
        );
        assert_eq!(
            <f32 as PrimitiveSample>::from_real_sample(least_subnormal / 2.0).to_bits(),
            0
        );
        assert!(<f32 as PrimitiveSample>::from_real_sample(-0.0).is_sign_negative());
    }

    #[test]
    fn float_to_integer_truncates_and_saturates() {
        assert_eq!(<i8 as PrimitiveSample>::from_real_sample(-1.9), -1);
        assert_eq!(<u16 as PrimitiveSample>::from_real_sample(2.9), 2);
        assert_eq!(<u8 as PrimitiveSample>::from_real_sample(-1.0), 0);
        assert_eq!(
            <i32 as PrimitiveSample>::from_real_sample(f64::INFINITY),
            i32::MAX
        );
        assert_eq!(
            <i32 as PrimitiveSample>::from_real_sample(f64::NEG_INFINITY),
            i32::MIN
        );
        assert_eq!(<u64 as PrimitiveSample>::from_real_sample(f64::NAN), 0);
    }

    #[test]
    fn integer_casts_keep_the_low_bits_and_round_when_targeting_float() {
        assert_eq!(<u8 as PrimitiveSample>::from_signed_sample(-1), u8::MAX);
        assert_eq!(<i8 as PrimitiveSample>::from_unsigned_sample(u64::MAX), -1);
        assert_eq!(
            <f32 as PrimitiveSample>::from_unsigned_sample(16_777_217),
            16_777_216.0
        );
        assert_eq!(
            <f64 as PrimitiveSample>::from_unsigned_sample(9_007_199_254_740_993),
            9_007_199_254_740_992.0
        );
    }
}
