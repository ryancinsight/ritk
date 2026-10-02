//! Explicit numeric conversion for the fixed-width sample set.
//!
//! Conversion behavior follows [the Rust Reference's numeric cast rules].
//!
//! [the Rust Reference's numeric cast rules]: https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast

/// A finite binary floating-point value represented as an integer significand
/// times a power of two.
#[derive(Clone, Copy)]
struct FiniteFloat {
    negative: bool,
    significand: u64,
    exponent: i32,
}

/// Narrow an unsigned integer to an unsigned sample width by retaining its
/// low `width` bits.
pub(super) fn unsigned_to_unsigned(value: u64, width: u32) -> u64 {
    if width == 64 {
        return value;
    }
    let modulus = 1_u128
        .checked_shl(width)
        .expect("invariant: sample width is below 128 bits");
    let low_bits = u128::from(value) % modulus;
    u64::try_from(low_bits).expect("invariant: narrowed unsigned sample fits u64")
}

/// Narrow a signed integer to an unsigned sample width by retaining its low
/// `width` bits.
pub(super) fn signed_to_unsigned(value: i64, width: u32) -> u64 {
    let modulus = 1_i128
        .checked_shl(width)
        .expect("invariant: sample width is below 128 bits");
    let low_bits = i128::from(value).rem_euclid(modulus);
    u64::try_from(low_bits).expect("invariant: narrowed unsigned sample fits u64")
}

/// Narrow an unsigned integer to a signed sample width using two's-complement
/// interpretation of its low `width` bits.
pub(super) fn unsigned_to_signed(value: u64, width: u32) -> i64 {
    let modulus = 1_i128
        .checked_shl(width)
        .expect("invariant: sample width is below 128 bits");
    let sign_boundary = modulus / 2;
    let low_bits = i128::from(value) % modulus;
    let signed = if low_bits >= sign_boundary {
        low_bits - modulus
    } else {
        low_bits
    };
    i64::try_from(signed).expect("invariant: narrowed signed sample fits i64")
}

/// Narrow a signed integer to a signed sample width using two's-complement
/// interpretation of its low `width` bits.
pub(super) fn signed_to_signed(value: i64, width: u32) -> i64 {
    let modulus = 1_i128
        .checked_shl(width)
        .expect("invariant: sample width is below 128 bits");
    let sign_boundary = modulus / 2;
    let low_bits = i128::from(value).rem_euclid(modulus);
    let signed = if low_bits >= sign_boundary {
        low_bits - modulus
    } else {
        low_bits
    };
    i64::try_from(signed).expect("invariant: narrowed signed sample fits i64")
}

/// Convert an unsigned integer to an IEEE binary format, rounding to nearest
/// with ties to even.
pub(super) fn integer_to_float(
    value: u64,
    negative: bool,
    fraction_bits: u32,
    exponent_bias: u32,
    sign_position: u32,
) -> u64 {
    if value == 0 {
        return 0;
    }

    let bit_count = u64::BITS - value.leading_zeros();
    let precision = fraction_bits + 1;
    let mut exponent = bit_count - 1;
    let mut significand = if bit_count > precision {
        round_right(value, bit_count - precision)
    } else {
        value
            .checked_shl(precision - bit_count)
            .expect("invariant: normalized sample significand fits u64")
    };
    let carry = 1_u64
        .checked_shl(precision)
        .expect("invariant: sample precision is below 64 bits");
    if significand == carry {
        significand >>= 1;
        exponent = exponent
            .checked_add(1)
            .expect("invariant: integer sample exponent fits u32");
    }

    let exponent_field = exponent
        .checked_add(exponent_bias)
        .expect("invariant: encoded exponent fits u32");
    let fraction_mask = 1_u64
        .checked_shl(fraction_bits)
        .expect("invariant: fraction width is below 64 bits")
        - 1;
    let sign = if negative {
        1_u64
            .checked_shl(sign_position)
            .expect("invariant: sign position fits u64")
    } else {
        0
    };
    let encoded_exponent = u64::from(exponent_field)
        .checked_shl(fraction_bits)
        .expect("invariant: encoded exponent fits u64");
    sign | encoded_exponent | (significand & fraction_mask)
}

/// Convert a finite `f64` representation to an IEEE binary32 value. Special
/// values retain their sign; NaNs become quiet NaNs with a nonzero payload.
pub(super) fn real_to_float(
    value: f64,
    fraction_bits: u32,
    exponent_bias: u32,
    sign_position: u32,
) -> f32 {
    let bits = value.to_bits();
    let sign = bits >> 63 != 0;
    let sign_word = if sign {
        1_u64
            .checked_shl(sign_position)
            .expect("invariant: sign position fits u64")
    } else {
        0
    };
    let source_exponent = (bits >> 52) & 0x7ff;
    let source_fraction = bits & ((1_u64 << 52) - 1);
    let exponent_limit = 0x7ff;

    if source_exponent == exponent_limit {
        let exponent_word = 0xff_u64
            .checked_shl(fraction_bits)
            .expect("invariant: binary32 exponent fits u64");
        let payload = if source_fraction == 0 {
            0
        } else {
            (source_fraction >> 29) | (1_u64 << (fraction_bits - 1))
        };
        return float_from_bits(sign_word | exponent_word | payload);
    }

    let Some(parts) = finite_parts(bits) else {
        unreachable!("invariant: a finite source exponent has finite parts")
    };
    if parts.significand == 0 {
        return float_from_bits(sign_word);
    }

    let source_bit_count = u64::BITS - parts.significand.leading_zeros();
    let leading_exponent = i32::try_from(source_bit_count - 1)
        .expect("invariant: source significand bit count fits i32")
        .checked_add(parts.exponent)
        .expect("invariant: source exponent sum fits i32");
    let max_exponent = i32::try_from(exponent_bias).expect("invariant: exponent bias fits i32");
    let min_exponent = 1 - max_exponent;

    if leading_exponent > max_exponent {
        return float_from_bits(sign_word | (0xff_u64 << fraction_bits));
    }

    let precision = fraction_bits + 1;
    if leading_exponent >= min_exponent {
        let mut significand = if source_bit_count > precision {
            round_right(parts.significand, source_bit_count - precision)
        } else {
            parts
                .significand
                .checked_shl(precision - source_bit_count)
                .expect("invariant: normalized significand fits u64")
        };
        let carry = 1_u64
            .checked_shl(precision)
            .expect("invariant: target precision is below 64 bits");
        let mut exponent = leading_exponent;
        if significand == carry {
            significand >>= 1;
            exponent = exponent
                .checked_add(1)
                .expect("invariant: target exponent fits i32");
        }
        if exponent > max_exponent {
            return float_from_bits(sign_word | (0xff_u64 << fraction_bits));
        }
        let exponent_field = u64::try_from(exponent + max_exponent)
            .expect("invariant: normalized exponent is nonnegative");
        let encoded_exponent = exponent_field
            .checked_shl(fraction_bits)
            .expect("invariant: encoded exponent fits u64");
        let fraction_mask = 1_u64
            .checked_shl(fraction_bits)
            .expect("invariant: fraction width is below 64 bits")
            - 1;
        return float_from_bits(sign_word | encoded_exponent | (significand & fraction_mask));
    }

    let subnormal_exponent = min_exponent
        .checked_sub(i32::try_from(fraction_bits).expect("invariant: fraction width fits i32"))
        .expect("invariant: subnormal exponent fits i32");
    let scale = parts
        .exponent
        .checked_sub(subnormal_exponent)
        .expect("invariant: subnormal scaling exponent fits i32");
    let subnormal = if scale >= 0 {
        parts
            .significand
            .checked_shl(u32::try_from(scale).expect("invariant: scale is nonnegative"))
            .expect("invariant: subnormal significand fits u64")
    } else {
        round_right(parts.significand, scale.unsigned_abs())
    };
    float_from_bits(sign_word | subnormal)
}

/// Truncate a real sample toward zero and saturate at the unsigned target
/// width. NaN and negative values map to zero.
pub(super) fn float_to_unsigned(value: f64, width: u32) -> u64 {
    let maximum = (1_u128
        .checked_shl(width)
        .expect("invariant: sample width is below 128 bits"))
        - 1;
    if value.is_nan() || value <= 0.0 {
        return 0;
    }
    if value == f64::INFINITY {
        return u64::try_from(maximum).expect("invariant: sample width is at most 64 bits");
    }

    let parts = finite_parts(value.to_bits())
        .expect("invariant: finite nonnegative sample has finite parts");
    let magnitude = truncated_magnitude(parts);
    let bounded = if magnitude > maximum {
        maximum
    } else {
        magnitude
    };
    u64::try_from(bounded).expect("invariant: saturated sample fits u64")
}

/// Truncate a real sample toward zero and saturate at the signed target
/// width. NaN maps to zero.
pub(super) fn float_to_signed(value: f64, width: u32) -> i64 {
    let limit = 1_i128
        .checked_shl(width - 1)
        .expect("invariant: signed sample width is below 128 bits");
    let minimum = -limit;
    let maximum = limit - 1;
    if value.is_nan() {
        return 0;
    }
    if value == f64::INFINITY {
        return i64::try_from(maximum).expect("invariant: saturated sample fits i64");
    }
    if value == f64::NEG_INFINITY {
        return i64::try_from(minimum).expect("invariant: saturated sample fits i64");
    }

    let parts = finite_parts(value.to_bits()).expect("invariant: finite sample has finite parts");
    let magnitude = i128::try_from(truncated_magnitude(parts)).unwrap_or(i128::MAX);
    let signed = if parts.negative {
        if magnitude >= limit {
            minimum
        } else {
            -magnitude
        }
    } else if magnitude > maximum {
        maximum
    } else {
        magnitude
    };
    i64::try_from(signed).expect("invariant: saturated sample fits i64")
}

/// Decode a finite IEEE binary64 value as `significand * 2^exponent`.
fn finite_parts(bits: u64) -> Option<FiniteFloat> {
    let negative = bits >> 63 != 0;
    let exponent_field = (bits >> 52) & 0x7ff;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent_field == 0x7ff {
        return None;
    }
    if exponent_field == 0 {
        return Some(FiniteFloat {
            negative,
            significand: fraction,
            exponent: -1074,
        });
    }
    let significand = (1_u64 << 52) | fraction;
    let exponent =
        i32::try_from(exponent_field).expect("invariant: exponent field fits i32") - 1023 - 52;
    Some(FiniteFloat {
        negative,
        significand,
        exponent,
    })
}

/// Truncate a nonnegative binary value to its integer magnitude. Values too
/// large for `u128` saturate so the target-width conversion can clamp them.
fn truncated_magnitude(parts: FiniteFloat) -> u128 {
    if parts.significand == 0 {
        return 0;
    }
    if parts.exponent >= 0 {
        let shift = u32::try_from(parts.exponent).expect("invariant: exponent is nonnegative");
        let significand = u128::from(parts.significand);
        if shift > significand.leading_zeros() {
            return u128::MAX;
        }
        return significand
            .checked_shl(shift)
            .expect("invariant: bounded significand shift fits u128");
    }
    let shift = parts.exponent.unsigned_abs();
    if shift >= u64::BITS {
        0
    } else {
        u128::from(parts.significand >> shift)
    }
}

/// Round a positive integer right by `shift`, using round-to-nearest, ties to
/// even.
fn round_right(value: u64, shift: u32) -> u64 {
    if shift == 0 {
        return value;
    }
    if shift > u64::BITS {
        return 0;
    }
    let retained = if shift == u64::BITS {
        0
    } else {
        value >> shift
    };
    let discarded = if shift == u64::BITS {
        value
    } else {
        value & ((1_u64 << shift) - 1)
    };
    let halfway = 1_u64
        .checked_shl(shift - 1)
        .expect("invariant: half-way bit fits u64");
    if discarded > halfway || (discarded == halfway && retained & 1 == 1) {
        retained
            .checked_add(1)
            .expect("invariant: rounded significand fits u64")
    } else {
        retained
    }
}

/// Construct a binary32 value from its encoded bits.
fn float_from_bits(bits: u64) -> f32 {
    f32::from_bits(u32::try_from(bits).expect("invariant: binary32 bits fit u32"))
}

#[cfg(test)]
mod tests {
    use super::{
        float_to_signed, float_to_unsigned, integer_to_float, real_to_float, signed_to_signed,
        signed_to_unsigned, unsigned_to_signed, unsigned_to_unsigned,
    };

    #[test]
    fn integer_narrowing_retains_the_low_target_bits() {
        assert_eq!(unsigned_to_unsigned(u64::MAX, 8), u64::from(u8::MAX));
        assert_eq!(signed_to_unsigned(-1, 16), u64::from(u16::MAX));
        assert_eq!(unsigned_to_signed(u64::MAX, 8), -1);
        assert_eq!(signed_to_signed(-129, 8), 127);
        assert_eq!(signed_to_signed(i64::MIN, 64), i64::MIN);
        assert_eq!(unsigned_to_signed(u64::MAX, 64), -1);
    }

    #[test]
    fn integer_to_float_rounds_directly_to_the_target_significand() {
        assert_eq!(
            integer_to_float(1, false, 23, 127, 31),
            u64::from(1.0_f32.to_bits())
        );
        assert_eq!(
            integer_to_float(2, false, 23, 127, 31),
            u64::from(2.0_f32.to_bits())
        );
        assert_eq!(integer_to_float(2, false, 52, 1023, 63), 2.0_f64.to_bits());
        let single_tie = integer_to_float(16_777_217, false, 23, 127, 31);
        assert_eq!(single_tie, u64::from(16_777_216_f32.to_bits()));
        let single_odd_tie = integer_to_float(16_777_219, false, 23, 127, 31);
        assert_eq!(single_odd_tie, u64::from(16_777_220_f32.to_bits()));

        let double_tie = integer_to_float(9_007_199_254_740_993, false, 52, 1023, 63);
        assert_eq!(double_tie, 9_007_199_254_740_992_f64.to_bits());
        let negative = integer_to_float(3, true, 23, 127, 31);
        assert_eq!(negative, u64::from((-3.0_f32).to_bits()));
    }

    #[test]
    fn real_to_float_rounds_normal_subnormal_and_special_values() {
        assert_eq!(real_to_float(2.0, 23, 127, 31), 2.0_f32);
        assert_eq!(real_to_float(0.1, 23, 127, 31).to_bits(), 0x3dcc_cccd);
        assert_eq!(real_to_float(1.0 + 2.0_f64.powi(-24), 23, 127, 31), 1.0);
        assert_eq!(
            real_to_float(1.0 + 3.0 * 2.0_f64.powi(-24), 23, 127, 31),
            f32::from_bits(1.0_f32.to_bits() + 2)
        );
        let least = f64::from(f32::from_bits(1));
        assert_eq!(real_to_float(least, 23, 127, 31).to_bits(), 1);
        assert_eq!(real_to_float(least / 2.0, 23, 127, 31).to_bits(), 0);
        assert_eq!(real_to_float(f64::MAX, 23, 127, 31), f32::INFINITY);
        assert_eq!(
            real_to_float(f64::NEG_INFINITY, 23, 127, 31),
            f32::NEG_INFINITY
        );
        let negative_nan = real_to_float(f64::from_bits(0xfff8_0000_0000_1234), 23, 127, 31);
        assert!(negative_nan.is_nan());
        assert!(negative_nan.is_sign_negative());
        assert_eq!(
            real_to_float(-0.0, 23, 127, 31).to_bits(),
            (-0.0_f32).to_bits()
        );
    }

    #[test]
    fn real_to_integer_truncates_toward_zero_and_saturates() {
        assert_eq!(float_to_signed(-1.5, 32), -1);
        assert_eq!(float_to_signed(2.9, 32), 2);
        assert_eq!(float_to_signed(f64::NAN, 32), 0);
        assert_eq!(float_to_signed(f64::INFINITY, 32), i64::from(i32::MAX));
        assert_eq!(float_to_signed(f64::NEG_INFINITY, 32), i64::from(i32::MIN));
        assert_eq!(float_to_signed(-1.0e300, 64), i64::MIN);
        assert_eq!(float_to_unsigned(-1.5, 16), 0);
        assert_eq!(float_to_unsigned(2.9, 16), 2);
        assert_eq!(float_to_unsigned(f64::NAN, 16), 0);
        assert_eq!(float_to_unsigned(f64::INFINITY, 16), u64::from(u16::MAX));
        assert_eq!(float_to_unsigned(f64::MAX, 64), u64::MAX);
        let large = 2.0_f64.powi(128);
        assert_eq!(float_to_unsigned(large, 64), u64::MAX);
        assert_eq!(float_to_signed(large, 64), i64::MAX);
        assert_eq!(float_to_signed(-large, 64), i64::MIN);
    }
}
