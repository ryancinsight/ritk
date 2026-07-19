use super::transform_to_displacement_field;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make(dims: [usize; 3]) -> Image<f32, B, 3> {
    let n: usize = dims.iter().product();
    ts::make_image::<f32, B, 3>(vec![0.0f32; n], dims)
}

const ID: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// The identity transform produces a zero displacement field.
#[test]
fn transform_to_displacement_identity_is_zero() {
    let dims = [4usize, 5, 6];
    let (dz, dy, dx) =
        transform_to_displacement_field(&make(dims), ID, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]).unwrap();
    for img in [&dz, &dy, &dx] {
        let (v, _) = extract_vec_infallible(img);
        assert!(
            v.iter().all(|&c| c.abs() < 1e-6),
            "identity field must be 0"
        );
    }
}

/// A pure translation `M = I` gives a constant field equal to the translation —
/// convention-agnostic, since `D(p) = (p − c) + c + t − p = t` for every `p`,
/// independent of how physical points are laid out.
#[test]
fn transform_to_displacement_translation_is_constant() {
    let dims = [3usize, 4, 5];
    let (tx, ty, tz) = (1.5f64, -2.0, 0.75);
    let (dz, dy, dx) =
        transform_to_displacement_field(&make(dims), ID, [tx, ty, tz], [1.0, 1.0, 1.0]).unwrap();
    let check = |img: &Image<f32, B, 3>, want: f64| {
        let (v, _) = extract_vec_infallible(img);
        assert!(
            v.iter().all(|&c| (c as f64 - want).abs() < 1e-5),
            "component must be the constant {want}"
        );
    };
    check(&dx, tx);
    check(&dy, ty);
    check(&dz, tz);
}

/// Output components share the reference geometry and shape.
#[test]
fn transform_to_displacement_preserves_geometry() {
    let dims = [2usize, 3, 4];
    let img = make(dims);
    let (dz, dy, dx) =
        transform_to_displacement_field(&img, ID, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]).unwrap();
    for comp in [&dz, &dy, &dx] {
        assert_eq!(comp.shape(), dims);
        assert_eq!(comp.spacing()[0], img.spacing()[0]);
    }
}
