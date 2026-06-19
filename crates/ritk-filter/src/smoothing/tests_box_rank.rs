use super::*;
use ritk_image::test_support as ts;

type B = burn_ndarray::NdArray<f32>;

/// Median (rank 0.5) over the clipped window: `[10,20,30,40,50]` r=1 →
/// `[10,20,30,40,40]` (boundary windows shrink to in-bounds voxels).
#[test]
fn rank_median_shrink_window() {
    let img = ts::make_image::<B, 3>(vec![10.0, 20.0, 30.0, 40.0, 50.0], [1, 1, 5]);
    let out = RankImageFilter::new([0, 0, 1], 0.5).apply(&img);
    assert_eq!(
        out.data_slice().into_owned(),
        vec![10.0, 20.0, 30.0, 40.0, 40.0]
    );
}

/// Floor index (not round): a 4-element window `[10,20,30,40]` at rank 0.5 picks
/// index `floor(0.5·3)=1` → 20, not 30.
#[test]
fn rank_uses_floor_index() {
    let img = ts::make_image::<B, 3>(vec![10.0, 20.0, 30.0, 40.0], [1, 1, 4]);
    let out = RankImageFilter::new([0, 0, 2], 0.5).apply(&img);
    // idx1 window = full [10,20,30,40] → floor(0.5·3)=1 → 20.
    assert_eq!(out.data_slice().into_owned()[1], 20.0);
}

/// rank 0.0 = min, rank 1.0 = max over the (clipped) window.
#[test]
fn rank_extremes_are_min_and_max() {
    let img = ts::make_image::<B, 3>(vec![10.0, 20.0, 30.0, 40.0, 50.0], [1, 1, 5]);
    let lo = RankImageFilter::new([0, 0, 1], 0.0).apply(&img);
    let hi = RankImageFilter::new([0, 0, 1], 1.0).apply(&img);
    assert_eq!(
        lo.data_slice().into_owned(),
        vec![10.0, 10.0, 20.0, 30.0, 40.0]
    );
    assert_eq!(
        hi.data_slice().into_owned(),
        vec![20.0, 30.0, 40.0, 50.0, 50.0]
    );
}
