//! SampleWindow edge cases (TEST-322-05).

use super::super::sample::SampleWindow;
use super::super::types::ParzenConfig;

#[test]
fn sample_window_at_exact_bin_center() {
    let num_bins = 16;
    let sigma_sq = 1.0;
    let fix_cfg = ParzenConfig::new(sigma_sq);
    let mov_cfg = ParzenConfig::new(sigma_sq);
    let fixed = vec![5.0_f32; 1];
    let moving = vec![10.0; 1];

    let window = SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, None)
        .expect("exact center should be in-bounds");

    assert_eq!(window.f_range().lo, 2);
    assert_eq!(window.f_range().hi, 8);
    assert_eq!(window.m_range().lo, 7);
    assert_eq!(window.m_range().hi, 13);
}

#[test]
fn sample_window_at_zero() {
    let num_bins = 16;
    let sigma_sq = 1.0;
    let fix_cfg = ParzenConfig::new(sigma_sq);
    let mov_cfg = ParzenConfig::new(sigma_sq);
    let fixed = vec![0.0_f32; 1];
    let moving = vec![8.0; 1];

    let window = SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, None)
        .expect("value at 0 should be in-bounds");

    assert_eq!(window.f_range().lo, 0);
    assert_eq!(window.f_range().hi, 3);
}

#[test]
fn sample_window_at_upper_boundary() {
    let num_bins = 16;
    let sigma_sq = 1.0;
    let fix_cfg = ParzenConfig::new(sigma_sq);
    let mov_cfg = ParzenConfig::new(sigma_sq);
    let fixed = vec![15.0_f32; 1];
    let moving = vec![8.0; 1];

    let window = SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, None)
        .expect("value at upper boundary should be in-bounds");

    assert_eq!(window.f_range().lo, 12);
    assert_eq!(window.f_range().hi, 15);
}

#[test]
fn sample_window_oob_mask_excludes_sample() {
    let num_bins = 16;
    let sigma_sq = 1.0;
    let fix_cfg = ParzenConfig::new(sigma_sq);
    let mov_cfg = ParzenConfig::new(sigma_sq);
    let fixed = vec![8.0_f32; 1];
    let moving = vec![8.0; 1];
    let mask = vec![0.0_f32; 1];

    assert!(
        SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, Some(&mask)).is_none(),
        "OOB mask 0.0 should exclude sample"
    );
}

#[test]
fn sample_window_oob_mask_includes_sample() {
    let num_bins = 16;
    let sigma_sq = 1.0;
    let fix_cfg = ParzenConfig::new(sigma_sq);
    let mov_cfg = ParzenConfig::new(sigma_sq);
    let fixed = vec![8.0_f32; 1];
    let moving = vec![8.0; 1];
    let mask = vec![1.0_f32; 1];

    assert!(
        SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, Some(&mask)).is_some(),
        "OOB mask 1.0 should include sample"
    );
}

#[test]
fn sample_window_moving_only_at_boundary() {
    let num_bins = 16;
    let mov_cfg = ParzenConfig::new(1.0);
    let moving = vec![0.5_f32; 1];

    let (m_val, m_range, _weights, _inv_sum_m) =
        SampleWindow::new_moving_only(0, &moving, num_bins, &mov_cfg, None)
            .expect("value near 0 should be in-bounds");

    assert!((m_val - 0.5).abs() < 1e-10);
    assert_eq!(m_range.lo, 0);
}

#[test]
fn sample_window_different_sigmas() {
    let num_bins = 32;
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(4.0);
    let fixed = vec![15.0_f32; 1];
    let moving = vec![15.0; 1];

    let window = SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, None)
        .expect("should be in-bounds");

    assert_eq!(window.f_range().len(), 7);
    assert_eq!(window.m_range().len(), 13);
}
