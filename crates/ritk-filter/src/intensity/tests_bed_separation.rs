use super::*;
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;
use ritk_image::Image;

type B = LegacyBurnBackend;

fn make_image(values: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::burn_compat::make_image::<B, 3>(values, dims)
}

#[test]
fn test_threshold_foreground() {
    let input = vec![-1000.0_f32, -200.0, 0.0, 120.0];
    let mask = threshold_foreground(&input, -350.0);
    assert_eq!(mask, vec![0, 1, 1, 1]);
}

#[test]
fn test_keep_largest_component_selects_body() {
    let dims = [1, 4, 4];
    let mask = vec![0u8, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1];
    let filtered = keep_largest_component(&mask, dims);
    assert_eq!(filtered.iter().filter(|&&v| v != 0).count(), 4);
    assert_eq!(filtered[5], 1);
    assert_eq!(filtered[6], 1);
    assert_eq!(filtered[9], 1);
    assert_eq!(filtered[10], 1);
    assert_eq!(filtered[15], 0);
}

#[test]
fn test_mask_preserves_foreground_and_removes_background() {
    let dims = [1, 2, 4];
    let values = vec![
        -1000.0, -1000.0, -1000.0, -1000.0, -200.0, -150.0, 20.0, 30.0,
    ];
    let img = make_image(values, dims);
    let filter = BedSeparationFilter::new(BedSeparationConfig::default());
    let out = filter.mask(&img).unwrap();
    let vals = out.data_slice();
    assert_eq!(vals.len(), 8);
    assert_eq!(vals.iter().filter(|&&v| v > 0.5).count(), 8);
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[1], 1.0);
    assert_eq!(vals[2], 1.0);
    assert_eq!(vals[3], 1.0);
    assert_eq!(vals[4], 1.0);
    assert_eq!(vals[5], 1.0);
    assert_eq!(vals[6], 1.0);
    assert_eq!(vals[7], 1.0);
}

#[test]
fn test_apply_uses_outside_value() {
    let dims = [1, 1, 4];
    let values = vec![-1000.0, -500.0, 50.0, 200.0];
    let img = make_image(values, dims);
    let config = BedSeparationConfig {
        body_threshold: -600.0,
        outside_value: -2048.0,
        component_policy: ComponentPolicy::All,
        closing_radius: 0,
        opening_radius: 0,
        ..Default::default()
    };

    let filter = BedSeparationFilter::new(config);
    let out = filter.apply(&img).unwrap();
    let vals = out.data_slice();

    assert_eq!(&*vals, &[-2048.0, -500.0, 50.0, 200.0]);
}

#[test]
fn test_binary_morphology_round_trip_identity_radius_zero() {
    let mask = vec![0u8, 1, 0, 1, 1, 0, 0, 1];
    let dims = [1, 2, 4];
    assert_eq!(binary_opening(&mask, dims, 0), mask);
    assert_eq!(binary_closing(&mask, dims, 0), mask);
}
