//! Stored integer and IEEE floating-point values at their encoded precision.
use crate::PixelLayout;
use num_bigint::{BigInt, Sign};
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
            Self::Wide(n) => super::affine::scaled(n, 0),
            Self::Float(n) => f64::from(*n),
            Self::Double(n) => *n,
        }
    }

    /// Difference from a modality window centre before conversion to display precision.
    pub fn window_offset(&self, slope: f64, intercept: f64, center: f64) -> f64 {
        let approximate = self.to_f64();
        if slope.is_finite()
            && intercept.is_finite()
            && center.is_finite()
            && approximate.abs().max(center.abs()).max(intercept.abs()) >= (1u64 << 52) as f64
        {
            let (n, exponent) = match self {
                Self::Unsigned(n) => (BigInt::from(*n), 0),
                Self::Signed(n) => (BigInt::from(*n), 0),
                Self::Wide(n) => (n.clone(), 0),
                _ if approximate.is_finite() => {
                    let (mantissa, exponent) = super::affine::parts(approximate);
                    (BigInt::from(mantissa), exponent)
                }
                _ => return approximate.mul_add(slope, intercept - center),
            };
            return super::affine::centered(n, exponent, slope, intercept, center);
        }
        approximate.mul_add(slope, intercept - center)
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
        self.window_offset(
            f64::from(layout.rescale_slope),
            f64::from(layout.rescale_intercept),
            0.,
        ) as f32
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
