//! SampleWindow tests (MEM-316-01, FIX-316-07, ARCH-317-01).

use super::super::sample::SampleWindow;
use super::super::types::ParzenConfig;

#[test]
fn sample_window_in_bounds() {
    let fixed = vec![15.3, 20.7];
    let moving = vec![12.0, 18.5];
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);
    let window = SampleWindow::new(0, &fixed, &moving, 32, &fix_cfg, &mov_cfg, None);
    assert!(window.is_some());
    let w = window.unwrap();
    assert_eq!(w.f_val, 15.3);
    assert_eq!(w.m_val, 12.0);
    assert_eq!(w.f_range().lo, 12);
    assert_eq!(w.f_range().hi, 18);
    assert_eq!(w.m_range().lo, 9);
    assert_eq!(w.m_range().hi, 15);
    // Verify pre-computed weights
    assert!(!w.f_weights.is_empty(), "fixed weights should be populated");
    assert!(
        !w.m_weights.is_empty(),
        "moving weights should be populated"
    );
    assert_eq!(w.f_weights.len as usize, w.f_range().len());
    assert_eq!(w.m_weights.len as usize, w.m_range().len());
}

#[test]
fn sample_window_oob_mask_excludes() {
    let fixed = vec![15.3];
    let moving = vec![12.0];
    let oob = vec![0.0f32]; // excluded
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);
    let window = SampleWindow::new(0, &fixed, &moving, 32, &fix_cfg, &mov_cfg, Some(&oob));
    assert!(window.is_none());
}

#[test]
fn sample_window_in_bounds_mask() {
    let fixed = vec![15.3];
    let moving = vec![12.0];
    let oob = vec![1.0f32]; // in-bounds
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);
    let window = SampleWindow::new(0, &fixed, &moving, 32, &fix_cfg, &mov_cfg, Some(&oob));
    assert!(window.is_some());
}

#[test]
fn sample_window_moving_only_in_bounds() {
    let moving = vec![12.0, 18.5];
    let mov_cfg = ParzenConfig::new(1.0);
    let result = SampleWindow::new_moving_only(1, &moving, 32, &mov_cfg, None);
    assert!(result.is_some());
    let (m_val, m_range, m_weights, _inv_sum_m) = result.unwrap();
    assert_eq!(m_val, 18.5);
    assert_eq!(m_range.lo, 15);
    assert_eq!(m_range.hi, 21);
    assert_eq!(m_weights.len(), m_range.len());
}

#[test]
fn sample_window_moving_only_oob() {
    let moving = vec![12.0];
    let oob = vec![0.0f32];
    let mov_cfg = ParzenConfig::new(1.0);
    let result = SampleWindow::new_moving_only(0, &moving, 32, &mov_cfg, Some(&oob));
    assert!(result.is_none());
}
