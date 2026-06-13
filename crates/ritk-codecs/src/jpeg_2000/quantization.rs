//! Scalar quantization for the irreversible 9/7 path (ISO 15444-1 §E.1).
//!
//! Lossy JPEG 2000 quantizes each wavelet coefficient `y` of subband `b` by a
//! dead-zone scalar quantizer with step size `Δ_b`:
//!
//! ```text
//! q = sign(y) · floor(|y| / Δ_b)
//! ```
//!
//! and reconstructs (no bit-plane truncation) at the bin mid-point
//! (reconstruction bias `r = 0.5`, §E.1.1.2):
//!
//! ```text
//! ŷ = 0                       if q = 0
//! ŷ = sign(q)·(|q| + 0.5)·Δ_b otherwise
//! ```
//!
//! The step size is transmitted in the QCD/QCC marker as an exponent ε_b and an
//! 11-bit mantissa μ_b relative to the subband dynamic-range exponent `R_b`
//! (§E.1.1, eq E-3):
//!
//! ```text
//! Δ_b = 2^(R_b − ε_b) · (1 + μ_b / 2^11)
//! ```

/// Mantissa precision: μ_b occupies the low 11 bits of the scalar SPqcd entry.
const MANTISSA_BITS: u32 = 11;
const MANTISSA_SCALE: f32 = (1u32 << MANTISSA_BITS) as f32; // 2048
const MANTISSA_MAX: u32 = (1u32 << MANTISSA_BITS) - 1; // 2047
const EXPONENT_MAX: u32 = (1u32 << 5) - 1; // ε_b is 5 bits

/// Reconstruct the step size `Δ_b` from the dynamic-range exponent `R_b` and the
/// transmitted (ε_b, μ_b) pair (ISO 15444-1 eq E-3).
#[inline]
pub fn step_size(r_b: u32, exponent: u32, mantissa: u32) -> f32 {
    let exp = i32::from(r_b as i16) - i32::from(exponent as i16);
    (1.0 + mantissa as f32 / MANTISSA_SCALE) * 2f32.powi(exp)
}

/// Pack an (ε_b, μ_b) pair into the 16-bit scalar SPqcd field: ε in bits 15–11,
/// μ in bits 10–0.
#[inline]
pub fn pack_spqcd(exponent: u32, mantissa: u32) -> u16 {
    (((exponent & EXPONENT_MAX) << MANTISSA_BITS) | (mantissa & MANTISSA_MAX)) as u16
}

/// Dead-zone quantize a coefficient with step `delta` (§E.1.1).
#[inline]
pub fn quantize(coeff: f32, delta: f32) -> i32 {
    let q = (coeff.abs() / delta).floor() as i64;
    let q = q.min(i64::from(i32::MAX)) as i32;
    if coeff < 0.0 {
        -q
    } else {
        q
    }
}

/// Mid-point dequantize a quantized index with step `delta` (§E.1.1.2, r = 0.5).
#[inline]
pub fn dequantize(q: i32, delta: f32) -> f32 {
    if q == 0 {
        0.0
    } else {
        let mag = (q.unsigned_abs() as f32 + 0.5) * delta;
        if q < 0 {
            -mag
        } else {
            mag
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_size_matches_eq_e3() {
        // Δ_b = 2^(R−ε)·(1 + μ/2048).  Unit step when ε = R, μ = 0.
        assert!((step_size(16, 16, 0) - 1.0).abs() < 1e-6);
        // ε one below R doubles the base; μ adds half an octave at μ = 1024.
        assert!((step_size(16, 15, 0) - 2.0).abs() < 1e-6);
        assert!((step_size(16, 15, 1024) - 3.0).abs() < 1e-5);
        // ε above R gives a sub-unit step.
        assert!((step_size(16, 18, 0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn spqcd_pack_places_exponent_and_mantissa() {
        for &(e, m) in &[(0u32, 0u32), (15, 2047), (31, 1024), (11, 7)] {
            let packed = pack_spqcd(e, m);
            assert_eq!(u32::from(packed) >> MANTISSA_BITS, e);
            assert_eq!(u32::from(packed) & MANTISSA_MAX, m);
        }
    }

    #[test]
    fn quantize_dequantize_error_is_within_half_step() {
        let delta = 2.5f32;
        for i in -100..=100 {
            let coeff = i as f32 * 0.37;
            let q = quantize(coeff, delta);
            let r = dequantize(q, delta);
            if q != 0 {
                // Reconstruction error never exceeds Δ (dead-zone is wider).
                assert!(
                    (r - coeff).abs() <= delta,
                    "coeff={coeff} q={q} r={r} Δ={delta}"
                );
            }
        }
    }

    #[test]
    fn quantize_is_sign_symmetric_and_dead_zone() {
        let delta = 4.0f32;
        assert_eq!(quantize(0.0, delta), 0);
        assert_eq!(quantize(3.9, delta), 0); // dead zone |y| < Δ → 0
        assert_eq!(quantize(-3.9, delta), 0);
        assert_eq!(quantize(4.1, delta), 1);
        assert_eq!(quantize(-4.1, delta), -1);
        assert_eq!(quantize(8.0, delta), 2);
    }
}
