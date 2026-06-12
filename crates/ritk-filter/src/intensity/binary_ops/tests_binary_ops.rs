use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0_f64, 0.0, 0.0]),
        Spacing::new([1.0_f64, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

// --- AddImageFilter ------------------------------------------------------

#[test]
fn add_filter_computes_elementwise_sum() {
    let a = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
    let b = make_image(vec![10.0, 20.0, 30.0, 40.0], [1, 2, 2]);
    let out = AddImageFilter::new().apply(&a, &b).unwrap();
    let v = voxels(&out);
    let expected = [11.0f32, 22.0, 33.0, 44.0];
    for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "[{}] expected {}, got {}",
            i,
            exp,
            got
        );
    }
}

#[test]
fn add_filter_preserves_spatial_metadata() {
    let a = make_image(vec![1.0; 8], [2, 2, 2]);
    let b = make_image(vec![2.0; 8], [2, 2, 2]);
    let out = AddImageFilter::new().apply(&a, &b).unwrap();
    assert_eq!(out.shape(), a.shape());
    assert_eq!(out.spacing(), a.spacing());
}

#[test]
fn add_filter_shape_mismatch_returns_error() {
    let a = make_image(vec![1.0; 4], [1, 2, 2]);
    let b = make_image(vec![1.0; 8], [2, 2, 2]);
    assert!(AddImageFilter::new().apply(&a, &b).is_err());
}

// --- SubtractImageFilter -------------------------------------------------

#[test]
fn subtract_filter_computes_elementwise_difference() {
    let a = make_image(vec![10.0, 20.0, 30.0, 40.0], [1, 2, 2]);
    let b = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
    let out = SubtractImageFilter::new().apply(&a, &b).unwrap();
    let v = voxels(&out);
    let expected = [9.0f32, 18.0, 27.0, 36.0];
    for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "[{}] expected {}, got {}",
            i,
            exp,
            got
        );
    }
}

#[test]
fn subtract_filter_self_minus_self_is_zero() {
    let a = make_image(vec![5.0, 3.0, 7.0, 2.0], [1, 2, 2]);
    let out = SubtractImageFilter::new().apply(&a, &a).unwrap();
    let v = voxels(&out);
    for (i, &val) in v.iter().enumerate() {
        assert!((val - 0.0).abs() < 1e-5, "[{}] expected 0, got {}", i, val);
    }
}

// --- MultiplyImageFilter -------------------------------------------------

#[test]
fn multiply_filter_computes_elementwise_product() {
    let a = make_image(vec![2.0, 3.0, 4.0, 5.0], [1, 2, 2]);
    let b = make_image(vec![3.0, 4.0, 5.0, 6.0], [1, 2, 2]);
    let out = MultiplyImageFilter::new().apply(&a, &b).unwrap();
    let v = voxels(&out);
    let expected = [6.0f32, 12.0, 20.0, 30.0];
    for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "[{}] expected {}, got {}",
            i,
            exp,
            got
        );
    }
}

#[test]
fn multiply_filter_by_zero_image_yields_zeros() {
    let a = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
    let z = make_image(vec![0.0; 4], [1, 2, 2]);
    let out = MultiplyImageFilter::new().apply(&a, &z).unwrap();
    let v = voxels(&out);
    for &val in &v {
        assert!((val - 0.0).abs() < 1e-5, "expected 0, got {}", val);
    }
}

// --- DivideImageFilter ---------------------------------------------------

#[test]
fn divide_filter_computes_elementwise_quotient() {
    let a = make_image(vec![10.0, 20.0, 30.0, 40.0], [1, 2, 2]);
    let b = make_image(vec![2.0, 4.0, 5.0, 8.0], [1, 2, 2]);
    let out = DivideImageFilter::new().apply(&a, &b).unwrap();
    let v = voxels(&out);
    let expected = [5.0f32, 5.0, 6.0, 5.0];
    for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "[{}] expected {}, got {}",
            i,
            exp,
            got
        );
    }
}

#[test]
fn divide_filter_division_by_zero_yields_zero() {
    let a = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
    let b = make_image(vec![0.0, 1.0, 0.0, 2.0], [1, 2, 2]);
    let out = DivideImageFilter::new().apply(&a, &b).unwrap();
    let v = voxels(&out);
    assert!(
        (v[0] - 0.0).abs() < 1e-5,
        "div-by-zero at [0]: got {}",
        v[0]
    );
    assert!((v[1] - 2.0).abs() < 1e-5, "[1]: expected 2, got {}", v[1]);
    assert!(
        (v[2] - 0.0).abs() < 1e-5,
        "div-by-zero at [2]: got {}",
        v[2]
    );
    assert!((v[3] - 2.0).abs() < 1e-5, "[3]: expected 2, got {}", v[3]);
}

// --- ImageMinFilter / ImageMaxFilter -------------------------------------

#[test]
fn min_filter_returns_elementwise_minimum() {
    let a = make_image(vec![1.0, 5.0, 3.0, 7.0], [1, 2, 2]);
    let b = make_image(vec![4.0, 2.0, 6.0, 1.0], [1, 2, 2]);
    let out = ImageMinFilter::new().apply(&a, &b).unwrap();
    let v = voxels(&out);
    let expected = [1.0f32, 2.0, 3.0, 1.0];
    for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "[{}] expected {}, got {}",
            i,
            exp,
            got
        );
    }
}

#[test]
fn max_filter_returns_elementwise_maximum() {
    let a = make_image(vec![1.0, 5.0, 3.0, 7.0], [1, 2, 2]);
    let b = make_image(vec![4.0, 2.0, 6.0, 1.0], [1, 2, 2]);
    let out = ImageMaxFilter::new().apply(&a, &b).unwrap();
    let v = voxels(&out);
    let expected = [4.0f32, 5.0, 6.0, 7.0];
    for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "[{}] expected {}, got {}",
            i,
            exp,
            got
        );
    }
}

// --- Generic BinaryOpFilter directly ------------------------------------

#[test]
fn generic_binary_op_filter_matches_specialized() {
    let a = make_image(vec![3.0, 7.0, 2.0, 9.0], [1, 2, 2]);
    let b = make_image(vec![1.0, 4.0, 6.0, 3.0], [1, 2, 2]);

    // Verify the generic path produces the same results as the type aliases
    let add_out = BinaryOpFilter::<AddOp>::new().apply(&a, &b).unwrap();
    let sub_out = BinaryOpFilter::<SubtractOp>::new().apply(&a, &b).unwrap();
    let mul_out = BinaryOpFilter::<MultiplyOp>::new().apply(&a, &b).unwrap();
    let div_out = BinaryOpFilter::<DivideOp>::new().apply(&a, &b).unwrap();
    let min_out = BinaryOpFilter::<MinOp>::new().apply(&a, &b).unwrap();
    let max_out = BinaryOpFilter::<MaxOp>::new().apply(&a, &b).unwrap();

    let add_v = voxels(&add_out);
    assert!((add_v[0] - 4.0).abs() < 1e-5);
    assert!((add_v[1] - 11.0).abs() < 1e-5);

    let sub_v = voxels(&sub_out);
    assert!((sub_v[0] - 2.0).abs() < 1e-5);
    assert!((sub_v[1] - 3.0).abs() < 1e-5);

    let mul_v = voxels(&mul_out);
    assert!((mul_v[0] - 3.0).abs() < 1e-5);
    assert!((mul_v[1] - 28.0).abs() < 1e-5);

    let div_v = voxels(&div_out);
    assert!((div_v[0] - 3.0).abs() < 1e-5);
    assert!((div_v[1] - 1.75).abs() < 1e-4);

    let min_v = voxels(&min_out);
    assert!((min_v[0] - 1.0).abs() < 1e-5);
    assert!((min_v[1] - 4.0).abs() < 1e-5);

    let max_v = voxels(&max_out);
    assert!((max_v[0] - 3.0).abs() < 1e-5);
    assert!((max_v[1] - 7.0).abs() < 1e-5);
}
