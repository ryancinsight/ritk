use super::*;
use burn_ndarray::NdArray;
use coeus_core::SequentialBackend;
use ritk_image::native::Image as NativeImage;
use ritk_image::test_support as ts;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

/// Hysteresis: outer-band voxels connected to an inner-band voxel become
/// foreground; isolated inner-band voxels stay; pure-outer islands drop.
/// Verified against `sitk.DoubleThreshold` (probe).
#[test]
fn double_threshold_hysteresis_matches_sitk_probe() {
    // [0,30,55,60,55,30,0,52,0]; t1=20,t2=50,t3=70,t4=80.
    let img = ts::make_image::<B, 3>(
        vec![0.0, 30.0, 55.0, 60.0, 55.0, 30.0, 0.0, 52.0, 0.0],
        [1, 1, 9],
    );
    let out = DoubleThresholdImageFilter::new(20.0, 50.0, 70.0, 80.0, 1.0, 0.0)
        .apply(&img)
        .unwrap();
    assert_eq!(
        out.data_slice().into_owned(),
        vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    );
}

/// A pure-outer voxel with no inner-band neighbour is dropped.
#[test]
fn double_threshold_isolated_outer_dropped() {
    // index1 (30) is in [20,80] but not [50,70] and has no inner neighbour → 0.
    let img = ts::make_image::<B, 3>(vec![0.0, 30.0, 0.0], [1, 1, 3]);
    let out = DoubleThresholdImageFilter::new(20.0, 50.0, 70.0, 80.0, 1.0, 0.0)
        .apply(&img)
        .unwrap();
    assert_eq!(out.data_slice().into_owned(), vec![0.0, 0.0, 0.0]);
}

/// Inside/outside values are honoured.
#[test]
fn double_threshold_custom_values() {
    let img = ts::make_image::<B, 3>(vec![60.0, 0.0], [1, 1, 2]);
    let out = DoubleThresholdImageFilter::new(20.0, 50.0, 70.0, 80.0, 7.0, -1.0)
        .apply(&img)
        .unwrap();
    assert_eq!(out.data_slice().into_owned(), vec![7.0, -1.0]);
}

#[test]
fn native_double_threshold_retains_connected_outer_band() {
    let image = NativeImage::from_flat_on(
        vec![0.0, 2.0, 5.0, 2.0, 0.0],
        [1, 1, 5],
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = DoubleThresholdImageFilter::new(1.0, 4.0, 6.0, 8.0, 1.0, 0.0)
        .apply_native(&image, &SequentialBackend)
        .expect("native double threshold succeeds");
    assert_eq!(
        output.data_slice().expect("invariant: contiguous storage"),
        &[0.0, 1.0, 1.0, 1.0, 0.0]
    );
    assert_eq!(output.origin(), image.origin());
    assert_eq!(output.spacing(), image.spacing());
    assert_eq!(output.direction(), image.direction());
}
