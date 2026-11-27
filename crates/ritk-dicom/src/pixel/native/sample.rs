//! Stored integer and IEEE floating-point values at their encoded precision.
use crate::PixelLayout;
use num_bigint::{BigInt, Sign, ToBigInt};
use std::fmt::{self, Display};

#[derive(Clone, Debug, PartialEq)]
pub enum StoredScalar {
    Unsigned(u64),
    Signed(i64),
    Wide(BigInt),
    Float(f32),
    Double(f64),
}

impl StoredScalar {
    pub(super) fn integer(word: &[u8], stored: u16, signed: bool) -> Self {
        let used = &word[..usize::from(stored).div_ceil(8)];
        if stored <= 64 {
            let mut bytes = [0; 8];
            bytes[..used.len()].copy_from_slice(used);
            let shift = 64 - stored;
            let aligned = u64::from_le_bytes(bytes) << shift;
            return if signed {
                Self::Signed(i64::from_le_bytes(aligned.to_le_bytes()) >> shift)
            } else {
                Self::Unsigned(aligned >> shift)
            };
        }
        let modulus = BigInt::from(1u8) << usize::from(stored);
        let value = BigInt::from_bytes_le(num_bigint::Sign::Plus, used) & (&modulus - 1u8);
        Self::Wide(if signed && value.bit(u64::from(stored - 1)) {
            value - modulus
        } else {
            value
        })
    }

    fn to_f64(&self) -> f64 {
        match self {
            Self::Unsigned(n) => *n as f64,
            Self::Signed(n) => *n as f64,
            Self::Wide(n) => {
                let magnitude = n.magnitude();
                let value = if let Ok(word) = u128::try_from(magnitude) {
                    word as f64
                } else if magnitude.bits() > 1024 {
                    f64::INFINITY
                } else {
                    // Keep the leading 53 bits and round ties to an even significand.
                    let shift = magnitude.bits() - 53;
                    let mut high = u64::try_from(magnitude >> shift).expect("53-bit significand");
                    if magnitude.bit(shift - 1)
                        && (high % 2 == 1
                            || magnitude
                                .trailing_zeros()
                                .is_some_and(|zeros| zeros < shift - 1))
                    {
                        high += 1;
                    }
                    high as f64 * 2f64.powi(i32::try_from(shift).expect("finite exponent"))
                };
                if n.sign() == Sign::Minus {
                    -value
                } else {
                    value
                }
            }
            Self::Float(n) => f64::from(*n),
            Self::Double(n) => *n,
        }
    }

    /// Difference from a modality window centre before conversion to display precision.
    pub fn window_offset(&self, slope: f64, intercept: f64, center: f64) -> f64 {
        // Where f64 can no longer distinguish adjacent integers, subtract the
        // integral window origin in the stored domain before narrowing.
        if slope.fract() == 0.0
            && matches!(self, Self::Unsigned(_) | Self::Signed(_) | Self::Wide(_))
            && center.abs().max(intercept.abs()) >= (1u64 << 52) as f64
            && let (Some(origin), Some(offset), Some(scale)) =
                (center.to_bigint(), intercept.to_bigint(), slope.to_bigint())
        {
            let delta = match self {
                Self::Unsigned(n) => BigInt::from(*n),
                Self::Signed(n) => BigInt::from(*n),
                Self::Wide(n) => n.clone(),
                _ => unreachable!(),
            } * scale
                + offset
                - origin;
            return Self::Wide(delta).to_f64() + (intercept.fract() - center.fract());
        }
        self.to_f64().mul_add(slope, intercept - center)
    }

    pub(super) fn display_value(&self, layout: PixelLayout) -> f32 {
        if layout.rescale_slope == 1. && layout.rescale_intercept == 0. {
            return match self {
                Self::Signed(n) => *n as f32,
                Self::Unsigned(n) => *n as f32,
                Self::Wide(n) => {
                    let value =
                        u128::try_from(n.magnitude()).map_or(f32::INFINITY, |word| word as f32);
                    if n.sign() == Sign::Minus {
                        -value
                    } else {
                        value
                    }
                }
                Self::Float(n) => *n,
                Self::Double(n) => *n as f32,
            };
        }
        let value = self.to_f64();
        (value * f64::from(layout.rescale_slope) + f64::from(layout.rescale_intercept)) as f32
    }
}

impl Display for StoredScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsigned(n) => Display::fmt(n, f),
            Self::Signed(n) => Display::fmt(n, f),
            Self::Wide(n) => Display::fmt(n, f),
            Self::Float(n) => Display::fmt(n, f),
            Self::Double(n) => Display::fmt(n, f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_byte_aligned_us_widths_admit_the_full_stored_sign_bit() {
        for bytes in 1u16..=u16::MAX / 8 {
            let allocated = bytes * 8;
            let raw = vec![0xff; usize::from(bytes)];
            let signed = StoredScalar::integer(&raw, allocated, true);
            assert!(
                matches!(signed, StoredScalar::Signed(-1))
                    || matches!(signed,StoredScalar::Wide(ref n) if n == &BigInt::from(-1))
            );
            assert_eq!(
                StoredScalar::integer(&raw, 1, false),
                StoredScalar::Unsigned(1)
            );
        }
    }
}
