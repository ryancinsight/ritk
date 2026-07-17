use super::*;
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;

type B = LegacyBurnBackend;

/// Sample (Bessel) standard deviation over the clipped window, matching ITK:
/// `[10,20,30,40,50]` r=1 → `[7.071, 10, 10, 10, 7.071]` (interior `[20,30,40]`
/// gives sample std 10, NOT population 8.165; boundary `[10,20]` gives 7.071).
#[test]
fn box_sigma_is_sample_std_with_shrink_window() {
    let img = ts::burn_compat::make_image::<B, 3>(vec![10.0, 20.0, 30.0, 40.0, 50.0], [1, 1, 5]);
    let out = BoxSigmaImageFilter::new([0, 0, 1]).apply(&img);
    let v = out.data_slice().into_owned();
    // boundary windows have sample std sqrt(50) = 7.0710678…
    let edge = 50.0f32.sqrt();
    let expected = [edge, 10.0, 10.0, 10.0, edge];
    for (got, exp) in v.iter().zip(expected) {
        assert!((got - exp).abs() < 1e-4, "got {got}, expected {exp}");
    }
}

/// A constant image has zero variance everywhere.
#[test]
fn box_sigma_constant_is_zero() {
    let img = ts::burn_compat::make_image::<B, 3>(vec![7.0; 27], [3, 3, 3]);
    let out = BoxSigmaImageFilter::new([1, 1, 1]).apply(&img);
    for &x in out.data_slice().iter() {
        assert!(x.abs() < 1e-4, "got {x}");
    }
}

/// Radius 0 (single-voxel window, n=1) yields all-zero (sample std undefined → 0).
#[test]
fn box_sigma_radius_zero_is_zero() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let img = ts::burn_compat::make_image::<B, 3>(data, [2, 3, 4]);
    let out = BoxSigmaImageFilter::new([0, 0, 0]).apply(&img);
    assert_eq!(out.data_slice().into_owned(), vec![0.0; 24]);
}
