//! Exact centering of integer samples under a binary64 modality rescale.
use num_bigint::{BigInt, Sign};

/// Finite IEEE value as an integer significand times a power of two.
pub(super) fn parts(n: f64) -> (i64, i32) {
    let bits = n.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    let significand = fraction | if biased == 0 { 0 } else { 1u64 << 52 };
    if significand == 0 {
        return (0, 0);
    }
    let zeros = significand.trailing_zeros();
    let mantissa = (significand >> zeros) as i64;
    let signed = if n.is_sign_negative() {
        -mantissa
    } else {
        mantissa
    };
    (signed, biased.max(1) - 1023 - 52 + zeros as i32)
}

/// Round an integer times a power of two once, including at the subnormal boundary.
pub(super) fn scaled(n: &BigInt, exponent: i32) -> f64 {
    let magnitude = n.magnitude();
    if magnitude.bits() == 0 {
        return 0.;
    }
    // Pixel widths are bounded by the US attribute. Aligning three finite
    // binary64 terms can add at most the IEEE exponent range to that width.
    let leading = i32::try_from(magnitude.bits()).expect("DICOM scalar bit length") - 1 + exponent;
    let value = if leading > 1023 {
        f64::INFINITY
    } else if leading < -1075 {
        0.
    } else {
        let quantum = (leading - 52).max(-1074);
        let shift = quantum - exponent;
        let significand = if shift > 0 {
            let shift = shift as u32;
            let half = u64::from(shift - 1);
            let mut high = u64::try_from(magnitude >> shift).expect("53-bit significand");
            if magnitude.bit(half)
                && (high % 2 == 1 || magnitude.trailing_zeros().is_some_and(|z| z < half))
            {
                high += 1;
            }
            high
        } else {
            u64::try_from(magnitude).expect("53-bit significand") << (-shift)
        };
        let unit = if quantum < -1022 {
            1u64 << (quantum + 1074)
        } else {
            ((quantum + 1023) as u64) << 52
        };
        significand as f64 * f64::from_bits(unit)
    };
    if n.sign() == Sign::Minus {
        -value
    } else {
        value
    }
}

pub(super) fn centered(n: BigInt, exponent: i32, slope: f64, intercept: f64, center: f64) -> f64 {
    let (scale, se) = parts(slope);
    let (offset, ie) = parts(intercept);
    let (origin, ce) = parts(center);
    let se = se + exponent;
    let power = se.min(ie).min(ce);
    let result = ((n * scale) << (se - power)) + (BigInt::from(offset) << (ie - power))
        - (BigInt::from(origin) << (ce - power));
    scaled(&result, power)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_rescale_retains_adjacent_values_after_cancellation() {
        let n = (BigInt::from(1u8) << 60) + 1;
        assert_eq!(centered(n, 0, 0.5, 0.125, 2f64.powi(59)), 0.625);
        let enormous = BigInt::from(1u8) << 1100;
        assert_eq!(
            centered(enormous, 0, 2f64.powi(-100), 0., 0.),
            2f64.powi(1000)
        );
    }

    #[test]
    fn subnormal_halfway_cases_round_to_even() {
        assert_eq!(scaled(&BigInt::from(1), -1075).to_bits(), 0);
        assert_eq!(scaled(&BigInt::from(3), -1075).to_bits(), 2);
        assert_eq!(
            scaled(&BigInt::from(-1), -1075).to_bits(),
            (-0.0f64).to_bits()
        );
    }
}
