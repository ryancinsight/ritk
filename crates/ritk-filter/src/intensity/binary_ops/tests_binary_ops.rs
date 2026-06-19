use super::*;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_image::test_support as ts;

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, dims)
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

// --- SquaredDifferenceImageFilter / AbsoluteValueDifferenceImageFilter ----

#[test]
fn squared_difference_computes_squared_residual() {
    let a = make_image(vec![3.0, 10.0, -2.0], [1, 1, 3]);
    let b = make_image(vec![1.0, 4.0, 1.0], [1, 1, 3]);
    let out = SquaredDifferenceImageFilter::new().apply(&a, &b).unwrap();
    let v = voxels(&out);
    // (3-1)²=4, (10-4)²=36, (-2-1)²=9
    for (got, exp) in v.iter().zip([4.0f32, 36.0, 9.0]) {
        assert!(
            (got - exp).abs() < 1e-4,
            "squared diff: got {got}, expected {exp}"
        );
    }
}

#[test]
fn absolute_value_difference_is_symmetric_and_nonnegative() {
    let a = make_image(vec![3.0, 4.0], [1, 1, 2]);
    let b = make_image(vec![1.0, 7.0], [1, 1, 2]);
    let ab = voxels(
        &AbsoluteValueDifferenceImageFilter::new()
            .apply(&a, &b)
            .unwrap(),
    );
    let ba = voxels(
        &AbsoluteValueDifferenceImageFilter::new()
            .apply(&b, &a)
            .unwrap(),
    );
    for (x, y) in ab.iter().zip(ba.iter()) {
        assert_eq!(x, y, "|a-b| must equal |b-a|");
        assert!(*x >= 0.0, "absolute difference must be non-negative");
    }
    // |3-1|=2, |4-7|=3
    for (got, exp) in ab.iter().zip([2.0f32, 3.0]) {
        assert!(
            (got - exp).abs() < 1e-5,
            "abs diff: got {got}, expected {exp}"
        );
    }
}

// --- Atan2ImageFilter / PowImageFilter ------------------------------------

#[test]
fn atan2_matches_std_atan2() {
    let a = make_image(vec![1.0, 1.0, 0.0, -1.0], [1, 1, 4]);
    let b = make_image(vec![1.0, 0.0, 1.0, -1.0], [1, 1, 4]);
    let out = voxels(&Atan2ImageFilter::new().apply(&a, &b).unwrap());
    let ya = [1.0f32, 1.0, 0.0, -1.0];
    let yb = [1.0f32, 0.0, 1.0, -1.0];
    for (i, got) in out.iter().enumerate() {
        let exp = ya[i].atan2(yb[i]);
        assert!(
            (got - exp).abs() < 1e-6,
            "atan2[{i}]: got {got}, expected {exp}"
        );
    }
}

#[test]
fn pow_matches_std_powf() {
    let a = make_image(vec![2.0, 3.0, 9.0, 5.0], [1, 1, 4]);
    let b = make_image(vec![3.0, 2.0, 0.5, 0.0], [1, 1, 4]);
    let out = voxels(&PowImageFilter::new().apply(&a, &b).unwrap());
    // 2³=8, 3²=9, 9^0.5=3, 5⁰=1
    for (got, exp) in out.iter().zip([8.0f32, 9.0, 3.0, 1.0]) {
        assert!((got - exp).abs() < 1e-5, "pow: got {got}, expected {exp}");
    }
}

// --- BinaryMagnitudeImageFilter ------------------------------------------

#[test]
fn binary_magnitude_computes_hypotenuse() {
    let a = make_image(vec![3.0, 5.0, 0.0], [1, 1, 3]);
    let b = make_image(vec![4.0, 12.0, 0.0], [1, 1, 3]);
    let out = voxels(&BinaryMagnitudeImageFilter::new().apply(&a, &b).unwrap());
    // 3-4-5, 5-12-13, 0-0-0
    for (got, exp) in out.iter().zip([5.0f32, 13.0, 0.0]) {
        assert!(
            (got - exp).abs() < 1e-5,
            "binary magnitude: got {got}, expected {exp}"
        );
    }
}

// --- Comparison filters (Equal/NotEqual/Greater/GreaterEqual/Less/LessEqual) --

#[test]
fn comparison_filters_produce_binary_masks() {
    let a = make_image(vec![1.0, 2.0, 3.0, 2.0], [1, 1, 4]);
    let b = make_image(vec![2.0, 2.0, 1.0, 5.0], [1, 1, 4]);
    let eq = voxels(&EqualImageFilter::new().apply(&a, &b).unwrap());
    let ne = voxels(&NotEqualImageFilter::new().apply(&a, &b).unwrap());
    let gt = voxels(&GreaterImageFilter::new().apply(&a, &b).unwrap());
    let ge = voxels(&GreaterEqualImageFilter::new().apply(&a, &b).unwrap());
    let lt = voxels(&LessImageFilter::new().apply(&a, &b).unwrap());
    let le = voxels(&LessEqualImageFilter::new().apply(&a, &b).unwrap());
    // a=[1,2,3,2], b=[2,2,1,5]
    assert_eq!(eq, vec![0.0, 1.0, 0.0, 0.0]);
    assert_eq!(ne, vec![1.0, 0.0, 1.0, 1.0]);
    assert_eq!(gt, vec![0.0, 0.0, 1.0, 0.0]);
    assert_eq!(ge, vec![0.0, 1.0, 1.0, 0.0]);
    assert_eq!(lt, vec![1.0, 0.0, 0.0, 1.0]);
    assert_eq!(le, vec![1.0, 1.0, 0.0, 1.0]);
    // eq + ne == 1 everywhere; gt + le == 1; lt + ge == 1
    for i in 0..4 {
        assert_eq!(eq[i] + ne[i], 1.0);
        assert_eq!(gt[i] + le[i], 1.0);
        assert_eq!(lt[i] + ge[i], 1.0);
    }
}

// --- Logical mask filters (And/Or/Xor) -----------------------------------

#[test]
fn logical_filters_match_itk_truth_tables() {
    // Binary masks: a=[0,0,1,1], b=[0,1,0,1] (ITK treats >0 as true)
    let a = make_image(vec![0.0, 0.0, 1.0, 1.0], [1, 1, 4]);
    let b = make_image(vec![0.0, 1.0, 0.0, 1.0], [1, 1, 4]);
    let and = voxels(&AndImageFilter::new().apply(&a, &b).unwrap());
    let or = voxels(&OrImageFilter::new().apply(&a, &b).unwrap());
    let xor = voxels(&XorImageFilter::new().apply(&a, &b).unwrap());
    assert_eq!(and, vec![0.0, 0.0, 0.0, 1.0]);
    assert_eq!(or, vec![0.0, 1.0, 1.0, 1.0]);
    assert_eq!(xor, vec![0.0, 1.0, 1.0, 0.0]);
    // De Morgan cross-check: a XOR b == (a OR b) AND NOT(a AND b)
    for i in 0..4 {
        let de_morgan = if or[i] > 0.5 && and[i] < 0.5 {
            1.0
        } else {
            0.0
        };
        assert_eq!(xor[i], de_morgan, "xor[{i}] vs (a|b)&!(a&b)");
    }
}
