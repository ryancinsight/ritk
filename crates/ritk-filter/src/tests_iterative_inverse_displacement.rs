use super::IterativeInverseDisplacementField;
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = LegacyBurnBackend;

fn img(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, dims)
}

/// The inverse of the zero field is the zero field (exactly).
#[test]
fn iterative_invert_zero_field_is_zero() {
    let dims = [4, 5, 6];
    let n: usize = dims.iter().product();
    let z = || img(vec![0.0f32; n], dims);
    let (vx, vy, vz) = IterativeInverseDisplacementField::default().apply(&z(), &z(), &z());
    for c in [&vx, &vy, &vz] {
        let (v, _) = extract_vec_infallible(c);
        assert!(v.iter().all(|&x| x == 0.0), "zero field inverts to zero");
    }
}

/// For a spatially constant displacement `u = (a, 0, 0)`, the interior of the
/// inverse approaches `−a`.
#[test]
fn iterative_invert_constant_field_interior_is_negated() {
    let dims = [7, 7, 7];
    let n: usize = dims.iter().product();
    let a = 0.5f32;
    let dx = img(vec![a; n], dims);
    let zero = img(vec![0.0f32; n], dims);
    let (vx, _, _) = IterativeInverseDisplacementField::default().apply(&dx, &zero, &zero);
    let (vxv, _) = extract_vec_infallible(&vx);
    let center = (3 * 7 + 3) * 7 + 3;
    assert!(
        (vxv[center] + a).abs() < 0.2,
        "interior inverse x = {} should approach -{a}",
        vxv[center]
    );
}

/// Output geometry matches the input field.
#[test]
fn iterative_invert_preserves_geometry() {
    let dims = [3, 4, 5];
    let n: usize = dims.iter().product();
    let z = || img(vec![0.0f32; n], dims);
    let (vx, _, _) = IterativeInverseDisplacementField::default().apply(&z(), &z(), &z());
    assert_eq!(vx.shape(), dims);
    assert_eq!(vx.spacing()[0], 1.0);
}
