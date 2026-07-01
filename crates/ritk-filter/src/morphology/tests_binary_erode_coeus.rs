//! Differential coverage: `binary_erode_coeus` must be value-identical to
//! the Burn-generic `BinaryErodeFilter::apply` it mirrors — both call the
//! identical `erode_binary_3d` core, so divergence would indicate a
//! boundary-wrapping bug, not an algorithmic one.

use super::binary_erode_coeus;
use crate::morphology::BinaryErodeFilter;
use burn_ndarray::NdArray;
use coeus_core::SequentialBackend;
use ritk_image::coeus::Image as CoeusImage;
use ritk_image::test_support as ts;
use ritk_spatial::{Direction, Point, Spacing};

type BurnBackend = NdArray<f32>;

fn assert_matches_burn(vals: Vec<f32>, dims: [usize; 3], radius: usize) {
    let burn_image = ts::make_image::<BurnBackend, 3>(vals.clone(), dims);
    let burn_result = BinaryErodeFilter::new(radius)
        .apply(&burn_image)
        .expect("burn binary erosion");
    let burn_vals = burn_result
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();

    let coeus_image = CoeusImage::from_flat_on(
        vals,
        dims,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("coeus image construction");
    let coeus_result = binary_erode_coeus(
        &coeus_image,
        radius,
        Default::default(),
        &SequentialBackend,
    )
    .expect("coeus binary erosion");
    let coeus_vals = coeus_result.data_slice().expect("coeus result slice");

    assert_eq!(coeus_vals.len(), burn_vals.len());
    for (i, (&c, &b)) in coeus_vals.iter().zip(burn_vals.iter()).enumerate() {
        assert_eq!(
            c, b,
            "coeus/burn binary erosion divergence at flat index {i}: coeus={c}, burn={b}"
        );
    }
}

#[test]
fn matches_burn_radius_zero_identity() {
    let vals = vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    assert_matches_burn(vals, [2, 2, 2], 0);
}

#[test]
fn matches_burn_all_foreground_radius_one() {
    assert_matches_burn(vec![1.0f32; 27], [3, 3, 3], 1);
}

#[test]
fn matches_burn_scattered_foreground_radius_one() {
    let dims = [6usize, 5, 4];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect();
    assert_matches_burn(vals, dims, 1);
}

#[test]
fn matches_burn_all_background() {
    assert_matches_burn(vec![0.0f32; 8], [2, 2, 2], 1);
}
