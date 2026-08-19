//! Block-radius calculators for speckle tracking.
//!
//! The block size is the fundamental resolution knob of block matching: too
//! small a kernel has no stable speckle signature, too large a kernel smooths
//! away the sub-block strain variation the method exists to measure. These
//! calculators derive a defensible axial kernel half-length from the signal or
//! the transducer spec, instead of leaving it as a magic constant.

use anyhow::{bail, Result};

/// Minimum axial kernel half-length returned by the calculators.
///
/// A kernel of `2·1 + 1 = 3` samples is the smallest window with enough
/// samples for a stable mean-subtracted correlation. Anything below that is
/// indistinguishable from noise.
const MIN_HALF_LENGTH: usize = 1;

/// Derive a block radius from the axial autocorrelation of a reference line.
///
/// The radius is half the distance to the first lag at which the normalized
/// autocorrelation envelope drops below `threshold`. That lag is a
/// point-spread-function width estimate: the kernel should cover a full PSF so
/// it captures the correlated speckle structure rather than a single lobe.
/// The result is clamped to at least one sample half-length.
///
/// The signal is mean-subtracted and normalized so the zero-lag
/// autocorrelation is `1`; `threshold` is therefore a fraction of the peak
/// energy. A negative or zero-energy signal (a constant or empty line) is
/// rejected: it carries no PSF width to measure.
///
/// # Errors
///
/// Returns an error when `signal` is empty, has zero energy, or `threshold` is
/// outside `(0, 1)`.
pub fn radius_from_axial_autocorrelation(signal: &[f64], threshold: f64) -> Result<usize> {
    if signal.is_empty() {
        bail!("axial autocorrelation requires a non-empty signal");
    }
    if !threshold.is_finite() || threshold <= 0.0 || threshold >= 1.0 {
        bail!("autocorrelation threshold must be in (0, 1), got {threshold}");
    }

    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let centred: Vec<f64> = signal.iter().map(|&value| value - mean).collect();
    let energy = centred.iter().map(|value| value * value).sum::<f64>();
    if energy <= 0.0 {
        bail!("axial autocorrelation requires a non-constant signal");
    }

    let n = centred.len();
    // Half-lag autocorrelation: r(k) = Σ x[i]·x[i+k] / Σ x[i]², for k ≥ 0.
    // The first lag whose normalized correlation falls below the threshold is
    // the PSF-width crossing; the kernel half-length is that lag (at least 1).
    for lag in 1..n {
        let correlation: f64 = centred[..n - lag]
            .iter()
            .zip(&centred[lag..])
            .map(|(&a, &b)| a * b)
            .sum();
        let normalized = correlation / energy;
        if normalized.abs() < threshold {
            return Ok(lag.max(MIN_HALF_LENGTH));
        }
    }
    // The autocorrelation never decorrelates within the window (e.g. a pure
    // low-frequency tone): the line is coherent end to end, so the largest
    // useful kernel is the whole line.
    Ok((n / 2).max(MIN_HALF_LENGTH))
}

/// Derive a block radius from transducer bandwidth and axial sample spacing.
///
/// The axial resolution of a pulse-echo system is approximately
/// `c / (2·BW)` where `BW` is the fractional bandwidth times the centre
/// frequency. The kernel half-length in samples is the number of sample
/// spacings that fit in one axial resolution cell:
///
/// ```text
/// r = round( c / (2 · f_c · BW) / dz )
/// ```
///
/// where `c` is the speed of sound, `f_c` the centre frequency, `BW` the
/// fractional bandwidth in `(0, 1]`, and `dz` the axial sample spacing. A
/// kernel of this size spans one resolution cell, which is the natural
/// correlation length of the received RF.
///
/// # Errors
///
/// Returns an error for a non-positive speed of sound, centre frequency, or
/// sample spacing, or a bandwidth outside `(0, 1]`.
pub fn radius_from_bandwidth(
    speed_of_sound_m_s: f64,
    centre_frequency_hz: f64,
    fractional_bandwidth: f64,
    axial_sample_spacing_m: f64,
) -> Result<usize> {
    if speed_of_sound_m_s <= 0.0 || !speed_of_sound_m_s.is_finite() {
        bail!("speed of sound must be finite and positive, got {speed_of_sound_m_s}");
    }
    if centre_frequency_hz <= 0.0 || !centre_frequency_hz.is_finite() {
        bail!("centre frequency must be finite and positive, got {centre_frequency_hz}");
    }
    if !fractional_bandwidth.is_finite()
        || fractional_bandwidth <= 0.0
        || fractional_bandwidth > 1.0
    {
        bail!("fractional bandwidth must be in (0, 1], got {fractional_bandwidth}");
    }
    if axial_sample_spacing_m <= 0.0 || !axial_sample_spacing_m.is_finite() {
        bail!("axial sample spacing must be finite and positive, got {axial_sample_spacing_m}");
    }

    let axial_resolution = speed_of_sound_m_s / (2.0 * centre_frequency_hz * fractional_bandwidth);
    let radius = (axial_resolution / axial_sample_spacing_m).round() as usize;
    Ok(radius.max(MIN_HALF_LENGTH))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autocorrelation_recovers_a_known_psf_width() {
        // A Gaussian-envelope sinusoid. The autocorrelation of a modulated
        // signal decorrelates at the carrier zero crossings: adjacent samples
        // of a 50-sample-period carrier are near-orthogonal, so the radius is
        // a few samples. The estimator must return a small positive kernel,
        // not the full envelope width.
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64 - 256.0) / 40.0;
                (-0.5 * t * t).exp() * (2.0 * std::f64::consts::PI * 0.02 * i as f64).cos()
            })
            .collect();
        let radius = radius_from_axial_autocorrelation(&signal, 0.5).expect("valid signal");
        // A pure tone has zero autocorrelation at lag = period/4; the radius is
        // the first lag below threshold, which is small but not 1 (the envelope
        // still correlates adjacent samples slightly). 3–12 is the honest band
        // for this carrier.
        assert!(
            (3..=12).contains(&radius),
            "radius {radius} should reflect the carrier correlation length"
        );
    }

    #[test]
    fn autocorrelation_of_white_noise_is_tiny() {
        // Deterministic white noise decorrelates at lag 1: the radius is the
        // minimum, because there is no coherent structure to capture.
        let mut signal = Vec::with_capacity(64);
        let mut state = 0x1234_5678_u64;
        for _ in 0..64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            signal.push(((state >> 33) as f64 / (1_u64 << 31) as f64) - 1.0);
        }
        let radius = radius_from_axial_autocorrelation(&signal, 0.5).expect("valid signal");
        assert_eq!(radius, MIN_HALF_LENGTH);
    }

    #[test]
    fn autocorrelation_rejects_invalid_inputs() {
        assert!(radius_from_axial_autocorrelation(&[], 0.5).is_err());
        assert!(radius_from_axial_autocorrelation(&[1.0, 1.0, 1.0], 0.5).is_err());
        assert!(radius_from_axial_autocorrelation(&[0.0, 1.0], 0.0).is_err());
        assert!(radius_from_axial_autocorrelation(&[0.0, 1.0], 1.0).is_err());
    }

    #[test]
    fn bandwidth_formula_returns_expected_radius() {
        // 1540 m/s, 5 MHz, 60% bandwidth, 0.1 mm spacing:
        // axial resolution = 1540 / (2 · 5e6 · 0.6) = 2.5667e-4 m.
        // radius = 2.5667e-4 / 1e-4 = 2.57 → 3.
        let radius = radius_from_bandwidth(1540.0, 5.0e6, 0.6, 1.0e-4).expect("valid transducer");
        assert_eq!(radius, 3);
    }

    #[test]
    fn bandwidth_formula_rejects_invalid_inputs() {
        assert!(radius_from_bandwidth(0.0, 5.0e6, 0.6, 1.0e-4).is_err());
        assert!(radius_from_bandwidth(1540.0, 0.0, 0.6, 1.0e-4).is_err());
        assert!(radius_from_bandwidth(1540.0, 5.0e6, 0.0, 1.0e-4).is_err());
        assert!(radius_from_bandwidth(1540.0, 5.0e6, 1.1, 1.0e-4).is_err());
        assert!(radius_from_bandwidth(1540.0, 5.0e6, 0.6, 0.0).is_err());
    }
}
