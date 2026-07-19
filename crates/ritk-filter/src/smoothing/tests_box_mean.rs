use super::*;
use ritk_image::test_support as ts;

type B = coeus_core::SequentialBackend;

/// Shrink-window boundary (ITK BoxMean): `[10,20,30,40,50]` r=1 →
/// `[15,20,30,40,45]` (out[0] = (10+20)/2, NOT the clamped Mean value 13.33).
#[test]
fn box_mean_shrinks_window_at_boundary() {
    let img = ts::make_image::<f32, B, 3>(vec![10.0, 20.0, 30.0, 40.0, 50.0], [1, 1, 5]);
    let out = BoxMeanImageFilter::new([0, 0, 1]).apply(&img);
    let v = out
        .data_slice()
        .expect("invariant: contiguous host storage")
        .to_vec();
    let expected = [15.0f32, 20.0, 30.0, 40.0, 45.0];
    for (got, exp) in v.iter().zip(expected) {
        assert!((got - exp).abs() < 1e-5, "got {got}, expected {exp}");
    }
}

/// A constant image is preserved exactly (mean of any window of c is c).
#[test]
fn box_mean_constant_preserved() {
    let img = ts::make_image::<f32, B, 3>(vec![7.0; 27], [3, 3, 3]);
    let out = BoxMeanImageFilter::new([1, 1, 1]).apply(&img);
    for &x in out
        .data_slice()
        .expect("invariant: contiguous host storage")
        .iter()
    {
        assert!((x - 7.0).abs() < 1e-5, "got {x}");
    }
}

/// Radius 0 is the identity.
#[test]
fn box_mean_radius_zero_is_identity() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let img = ts::make_image::<f32, B, 3>(data.clone(), [2, 3, 4]);
    let out = BoxMeanImageFilter::new([0, 0, 0]).apply(&img);
    assert_eq!(
        out.data_slice()
            .expect("invariant: contiguous host storage")
            .to_vec(),
        data
    );
}
