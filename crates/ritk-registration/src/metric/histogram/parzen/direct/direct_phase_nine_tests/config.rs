//! ParzenConfig encapsulation (ARCH-322-03) and dead-code gating correctness (DEAD-322-02).

use super::super::types::ParzenConfig;
use super::super::*;

#[test]
fn parzen_config_private_fields_cannot_be_constructed_inconsistently() {
    for sigma_sq in [0.01, 0.1, 1.0, 4.0, 16.0, 100.0] {
        let cfg = ParzenConfig::new(sigma_sq);
        assert!(
            (cfg.sigma_sq() - sigma_sq).abs() < 1e-10,
            "sigma_sq() mismatch for {sigma_sq}: got {}",
            cfg.sigma_sq()
        );
        assert_eq!(
            cfg.half_width(),
            compute_half_width(sigma_sq),
            "half_width mismatch for sigma_sq={sigma_sq}"
        );
        assert!(
            (cfg.inv_2sigma_sq() - (-0.5 / sigma_sq)).abs() < 1e-10,
            "inv_2sigma_sq mismatch for sigma_sq={sigma_sq}: got {}",
            cfg.inv_2sigma_sq()
        );
    }
}

#[test]
fn parzen_config_from_intensity_sigma_encapsulation() {
    let cfg = ParzenConfig::from_intensity_sigma(8.0, 0.0, 255.0, 32);
    let sigma_sq = cfg.sigma_sq();
    assert_eq!(
        cfg.half_width(),
        compute_half_width(sigma_sq),
        "from_intensity_sigma half_width inconsistent"
    );
    assert!(
        (cfg.inv_2sigma_sq() - (-0.5 / sigma_sq)).abs() < 1e-10,
        "from_intensity_sigma inv_2sigma_sq inconsistent"
    );
}

#[test]
fn bin_range_iter_returns_correct_indices() {
    use super::super::types::BinRange;
    let range = BinRange::new(10, 3, 32);
    let indices: Vec<usize> = range.iter().collect();
    assert_eq!(indices, vec![7, 8, 9, 10, 11, 12, 13]);
}

#[test]
fn bin_range_len_matches_iter_count() {
    use super::super::types::BinRange;
    let range = BinRange::new(15, 3, 32);
    assert_eq!(range.len(), range.iter().count());
}

#[test]
fn bin_range_len_at_boundary() {
    use super::super::types::BinRange;
    let range = BinRange::new(1, 3, 32);
    assert_eq!(range.len(), 5);
    assert_eq!(range.iter().count(), 5);
}

#[test]
fn stack_weights_len_matches_iter() {
    let cfg = ParzenConfig::new(1.0);
    let (_, weights) = cfg.compute_weights(15.3, 32);
    assert_eq!(weights.len(), weights.iter().count());
}
