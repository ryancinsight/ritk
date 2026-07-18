use super::OrientImageFilter;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, dims)
}

/// With an identity direction, the image's own orientation code is "SPL"
/// (image axes x, y, z = tensor axes 2, 1, 0 point world z+, y+, x+). Orienting
/// to the self-code is a no-op.
#[test]
fn orient_to_self_code_is_identity() {
    let data: Vec<f32> = (0..6).map(|v| v as f32).collect();
    let img = make(data.clone(), [1, 2, 3]);
    let out = OrientImageFilter::from_code("SPL")
        .unwrap()
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 2, 3]);
    let (ov, _) = extract_vec_infallible(&out);
    assert_eq!(ov, data, "self-code orientation must not change data");
    assert_eq!(out.origin(), img.origin());
    assert_eq!(out.spacing(), img.spacing());
}

/// "IAR" reverses each image axis from "SPL" (S→I, P→A, L→R), a full reversal of
/// all three axes with no permutation: out[z][y][x] = in[z][Y-1-y][X-1-x].
#[test]
fn orient_full_reversal() {
    let data: Vec<f32> = (0..6).map(|v| v as f32).collect();
    let img = make(data, [1, 2, 3]); // y0=[0,1,2], y1=[3,4,5]
    let out = OrientImageFilter::from_code("IAR")
        .unwrap()
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 2, 3]);
    let (ov, _) = extract_vec_infallible(&out);
    assert_eq!(ov, vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
}

/// A permuting code reorders the axes: it preserves the voxel multiset and
/// permutes the shape, leaving the data a rearrangement of the input.
#[test]
fn orient_permutation_preserves_multiset() {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let img = make(data.clone(), [2, 3, 4]);
    // "LPS" maps image axes to world x+, y+, z+; for identity direction that
    // permutes the tensor axes (z↔x) relative to the "SPL" self-code.
    let out = OrientImageFilter::from_code("LPS")
        .unwrap()
        .apply(&img)
        .unwrap();
    let (mut ov, _) = extract_vec_infallible(&out);
    let mut sorted = data;
    ov.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(ov, sorted, "orientation must preserve the voxel multiset");
    assert_eq!(out.shape(), [4, 3, 2], "tensor axes z and x swap");
}

/// Malformed codes are rejected, not silently accepted.
#[test]
fn orient_invalid_codes_error() {
    let img = make(vec![0.0; 6], [1, 2, 3]);
    let f = OrientImageFilter::from_code("XYZ");
    assert!(f.is_err(), "unknown letters must error");
    assert!(
        OrientImageFilter::from_code("LL").is_err(),
        "wrong length must error"
    );
    // Repeated anatomical axis (two letters on the L/R axis).
    let dup = OrientImageFilter::from_code("LRP");
    assert!(dup.is_err(), "repeated anatomical axis must error");
    // A valid filter still applies cleanly.
    assert!(OrientImageFilter::from_code("SPL")
        .unwrap()
        .apply(&img)
        .is_ok());
}
