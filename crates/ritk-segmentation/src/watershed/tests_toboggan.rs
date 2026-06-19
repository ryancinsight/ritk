//! Differential tests for [`toboggan`] against SimpleITK reference output.
//!
//! Expected label images are captured verbatim from `sitk.Toboggan` on the same
//! 2-D reliefs — an external oracle, not a ritk self-comparison.

use super::toboggan;
use ritk_image::test_support as ts;
use ritk_tensor_ops::extract_vec_infallible;

type B = burn_ndarray::NdArray<f32>;

fn z1(flat: Vec<f32>, rows: usize, cols: usize) -> ritk_image::Image<B, 3> {
    ts::make_image::<B, 3>(flat, [1, rows, cols])
}

fn run(img: &ritk_image::Image<B, 3>) -> Vec<f32> {
    extract_vec_infallible(&toboggan(img)).0
}

fn f(v: &[i32]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

#[test]
fn single_basin_all_one_label() {
    // Bowl with a single minimum → every voxel slides to label 2.
    let img = z1(
        f(&[5, 4, 3, 4, 5, 4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4]),
        4,
        5,
    );
    assert_eq!(run(&img), f(&[2; 20]));
}

#[test]
fn four_corner_basins_match_sitk() {
    // Four corner minima separated by a central ridge.
    let img = z1(
        f(&[
            1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5, 4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 2, 1,
        ]),
        5,
        5,
    );
    let expect = f(&[
        2, 2, 3, 3, 3, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 4, 4, 5, 5, 5, 4, 4, 5, 5, 5,
    ]);
    assert_eq!(run(&img), expect);
}

#[test]
fn ridge_split_matches_sitk() {
    let img = z1(f(&[1, 5, 1, 5, 5, 5, 1, 5, 1]), 3, 3);
    assert_eq!(run(&img), f(&[2, 3, 3, 4, 3, 3, 4, 3, 3]));
}
