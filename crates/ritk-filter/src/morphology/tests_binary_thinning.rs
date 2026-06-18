use super::BinaryThinningFilter;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = NdArray<f32>;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, dims)
}

/// A 1-pixel-wide horizontal line is already a skeleton and is preserved:
/// interior pixels have exactly two on-neighbours (transitions = 2 ≠ 1, fails
/// test B) and the endpoints have a single on-neighbour (fails test A).
#[test]
fn binary_thinning_preserves_thin_line() {
    let (nz, ny, nx) = (1usize, 5, 9);
    let mut vals = vec![0.0f32; ny * nx];
    for x in 2..=6 {
        vals[2 * nx + x] = 1.0; // interior horizontal line, away from x-borders
    }
    let out = BinaryThinningFilter::new().apply(&make(vals.clone(), [nz, ny, nx]));
    let (ov, _) = extract_vec_infallible(&out);
    assert_eq!(ov, vals, "a 1-pixel line must survive thinning unchanged");
}

/// A solid block thins to a strictly smaller, still-binary skeleton.
#[test]
fn binary_thinning_reduces_block_and_is_binary() {
    let (nz, ny, nx) = (1usize, 7, 7);
    let mut vals = vec![0.0f32; ny * nx];
    for y in 1..6 {
        for x in 1..6 {
            vals[y * nx + x] = 1.0; // solid 5×5 block
        }
    }
    let before = vals.iter().filter(|&&v| v == 1.0).count();
    let out = BinaryThinningFilter::new().apply(&make(vals, [nz, ny, nx]));
    let (ov, _) = extract_vec_infallible(&out);
    assert!(
        ov.iter().all(|&v| v == 0.0 || v == 1.0),
        "output must be binary"
    );
    let after = ov.iter().filter(|&&v| v == 1.0).count();
    assert!(
        after < before && after > 0,
        "thinning must reduce a solid block"
    );
}

/// An all-background image is unchanged.
#[test]
fn binary_thinning_empty_is_identity() {
    let dims = [1usize, 4, 5];
    let n: usize = dims.iter().product();
    let out = BinaryThinningFilter::new().apply(&make(vec![0.0; n], dims));
    let (ov, _) = extract_vec_infallible(&out);
    assert!(ov.iter().all(|&v| v == 0.0));
    assert_eq!(out.shape(), dims);
    assert_eq!(out.spacing()[0], 1.0);
}
