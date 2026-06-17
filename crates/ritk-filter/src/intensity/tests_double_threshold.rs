use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;

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
