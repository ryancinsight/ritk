//! Differential coverage: `distance_transform_coeus` must be value-identical
//! to the Burn-generic `DistanceTransformImageFilter::apply` it mirrors —
//! both call the same `euclidean_dt` core, so divergence would indicate a
//! boundary-wrapping bug, not an algorithmic one.

use super::distance_transform_coeus;
use crate::distance::euclidean::DistanceTransformImageFilter;
use crate::distance::types::BinarizationThreshold;
use burn_ndarray::NdArray;
use coeus_core::SequentialBackend;
use ritk_image::coeus::Image as CoeusImage;
use ritk_image::test_support as ts;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type BurnBackend = NdArray<f32>;

fn assert_matches_burn(vals: Vec<f32>, dims: [usize; 3]) {
    let burn_image = ts::make_image::<BurnBackend, 3>(vals.clone(), dims);
    let burn_result = DistanceTransformImageFilter::new()
        .apply(&burn_image)
        .expect("burn distance transform");
    let (burn_vals, _) = extract_vec_infallible(&burn_result);

    let coeus_image = CoeusImage::from_flat_on(
        vals,
        dims,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("coeus image construction");
    let coeus_result =
        distance_transform_coeus(&coeus_image, BinarizationThreshold::DEFAULT, &SequentialBackend)
            .expect("coeus distance transform");
    let coeus_vals = coeus_result.data_slice().expect("coeus result slice");

    assert_eq!(coeus_vals.len(), burn_vals.len());
    for (i, (&c, &b)) in coeus_vals.iter().zip(burn_vals.iter()).enumerate() {
        assert_eq!(
            c, b,
            "coeus/burn distance transform divergence at flat index {i}: coeus={c}, burn={b}"
        );
    }
}

#[test]
fn matches_burn_single_foreground_voxel() {
    let dims = [5usize, 5, 5];
    let mut fg = vec![0.0f32; 5 * 5 * 5];
    fg[0] = 1.0;
    assert_matches_burn(fg, dims);
}

#[test]
fn matches_burn_all_foreground() {
    assert_matches_burn(vec![1.0f32; 4 * 4 * 4], [4, 4, 4]);
}

#[test]
fn matches_burn_all_background() {
    assert_matches_burn(vec![0.0f32; 3 * 3 * 3], [3, 3, 3]);
}

#[test]
fn matches_burn_scattered_foreground() {
    let dims = [6usize, 5, 4];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n)
        .map(|i| if i % 7 == 0 { 1.0 } else { 0.0 })
        .collect();
    assert_matches_burn(vals, dims);
}
