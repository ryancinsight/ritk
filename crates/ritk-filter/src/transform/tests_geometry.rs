use super::transform_geometry;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::Point;
use ritk_tensor_ops::extract_vec_infallible;

type B = NdArray<f32>;

const ID: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn img_with_origin(origin: [f64; 3]) -> Image<B, 3> {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    ts::make_image_with::<B, 3>(data, [2, 3, 4], Some(Point::new(origin)), None, None)
}

/// The identity transform leaves geometry and data unchanged.
#[test]
fn transform_geometry_identity_is_noop() {
    let img = img_with_origin([10.0, 20.0, 30.0]);
    let out = transform_geometry(&img, ID, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]).unwrap();
    assert_eq!(out.origin(), img.origin());
    assert_eq!(out.spacing(), img.spacing());
    assert_eq!(out.direction().0, img.direction().0);
    let (ov, _) = extract_vec_infallible(&out);
    let (iv, _) = extract_vec_infallible(&img);
    assert_eq!(ov, iv, "data must be unchanged");
}

/// Pure translation: `A = I` ⇒ `origin' = origin − t` (center cancels), direction
/// and data unchanged.
#[test]
fn transform_geometry_translation_shifts_origin() {
    let img = img_with_origin([10.0, 20.0, 30.0]);
    let out = transform_geometry(&img, ID, [2.0, -1.0, 0.5], [5.0, 5.0, 5.0]).unwrap();
    let o = out.origin();
    assert!((o[0] - 8.0).abs() < 1e-12, "origin x {} != 8", o[0]);
    assert!((o[1] - 21.0).abs() < 1e-12, "origin y {} != 21", o[1]);
    assert!((o[2] - 29.5).abs() < 1e-12, "origin z {} != 29.5", o[2]);
    assert_eq!(out.direction().0, img.direction().0);
}

/// A 90° rotation about z (proper rotation) sets `D' = A⁻¹·D = Aᵀ` for an
/// identity input direction.
#[test]
fn transform_geometry_rotation_inverts_direction() {
    let img = img_with_origin([0.0, 0.0, 0.0]);
    let rot = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let out = transform_geometry(&img, rot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]).unwrap();
    let d = out.direction().0;
    // A⁻¹ = Aᵀ = [[0,1,0],[-1,0,0],[0,0,1]]
    let expected = nalgebra::Matrix3::new(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (d[(i, j)] - expected[(i, j)]).abs() < 1e-12,
                "dir[{i}][{j}] mismatch"
            );
        }
    }
}

/// A singular matrix has no inverse and is rejected.
#[test]
fn transform_geometry_singular_matrix_errors() {
    let img = img_with_origin([0.0, 0.0, 0.0]);
    let singular = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]];
    assert!(transform_geometry(&img, singular, [0.0; 3], [0.0; 3]).is_err());
}
