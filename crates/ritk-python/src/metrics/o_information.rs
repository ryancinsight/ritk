use anyhow::Result;
use ritk_core::statistics::information::{
    dual_total_correlation as core_dtc, o_information as core_oi,
};

pub(super) fn dtc_slices(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    core_dtc(channels, num_bins)
}

pub(super) fn oi_slices(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    core_oi(channels, num_bins)
}

#[cfg(test)]
pub(super) fn oi_direct_slices(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    core_oi_direct(channels, num_bins)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize, modulus: usize) -> Vec<f32> {
        (0..n).map(|i| (i % modulus) as f32).collect()
    }

    #[test]
    fn dtc_two_identical_slices_is_non_negative() {
        let a = ramp(64, 8);
        let dtc = dtc_slices(&[a.as_slice(), a.as_slice()], 8).unwrap();
        assert!(dtc >= 0.0, "DTC must be ≥ 0, got {dtc}");
    }

    #[test]
    fn oi_two_slices_is_zero() {
        let a = ramp(64, 8);
        let b = ramp(64, 4);
        let oi = oi_slices(&[a.as_slice(), b.as_slice()], 8).unwrap();
        assert!(oi.abs() < 1e-9, "Ω(X,Y) must be 0 for n=2, got {oi}");
    }

    #[test]
    fn oi_direct_matches_oi_three_channels() {
        let a = ramp(128, 8);
        let b = ramp(128, 4);
        let c = ramp(128, 6);
        let channels: &[&[f32]] = &[a.as_slice(), b.as_slice(), c.as_slice()];
        let oi = oi_slices(channels, 8).unwrap();
        let oi_d = oi_direct_slices(channels, 8).unwrap();
        assert!((oi - oi_d).abs() < 1e-9, "oi={oi:.10} must equal oi_direct={oi_d:.10}");
    }
}
