use super::normalized_correlation;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn img(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, dims)
}

/// Where the image neighbourhood equals the template, the normalized correlation
/// is exactly 1.0 (numerator √(N−1)·std over denominator √(N−1)·std).
#[test]
fn ncc_self_match_is_one() {
    let tpl = vec![0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0];
    let dims = [1, 3, 3];
    let image = img(tpl.clone(), dims);
    let template = img(tpl, dims);
    let mask = img(vec![1.0; 9], dims);
    let out = normalized_correlation(&image, &mask, &template)
        .expect("infallible: validated precondition");
    let (v, _) = extract_vec_infallible(&out);
    let center = 3 + 1; // (z,y,x) = (0,1,1); flat index = 0*9 + 1*3 + 1
    assert!(
        (v[center] - 1.0).abs() < 1e-5,
        "self-match NCC = {}, want 1",
        v[center]
    );
}

/// NCC is invariant to a positive affine transform of the neighbourhood: an
/// image that is `a·T + b` still correlates to 1.0 at the aligned center.
#[test]
fn ncc_affine_invariance() {
    let tpl = vec![0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0];
    let scaled: Vec<f32> = tpl.iter().map(|&t| 3.0 * t + 5.0).collect();
    let dims = [1, 3, 3];
    let out = normalized_correlation(
        &img(scaled, dims),
        &img(vec![1.0; 9], dims),
        &img(tpl, dims),
    )
    .expect("infallible: validated precondition");
    let (v, _) = extract_vec_infallible(&out);
    assert!(
        (v[4] - 1.0).abs() < 1e-5,
        "affine-scaled NCC = {}, want 1",
        v[4]
    );
}

/// Masked-out voxels are zero.
#[test]
fn ncc_masked_out_is_zero() {
    let tpl = vec![0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0];
    let dims = [1, 3, 3];
    let mut mask = vec![1.0f32; 9];
    mask[0] = 0.0;
    let out = normalized_correlation(&img(tpl.clone(), dims), &img(mask, dims), &img(tpl, dims))
        .expect("infallible: validated precondition");
    let (v, _) = extract_vec_infallible(&out);
    assert_eq!(v[0], 0.0, "masked-out voxel is zero");
}

/// A shape mismatch is rejected.
#[test]
fn ncc_shape_mismatch_errors() {
    let image = img(vec![0.0; 9], [1, 3, 3]);
    let mask = img(vec![1.0; 6], [1, 2, 3]);
    let template = img(vec![1.0; 9], [1, 3, 3]);
    assert!(normalized_correlation(&image, &mask, &template).is_err());
}
