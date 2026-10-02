use crate::sample::{Rescale, SampleError, SampleType};

#[test]
fn rescale_rejects_non_finite_coefficients() {
    for (slope, intercept) in [
        (f64::NAN, 0.0),
        (1.0, f64::INFINITY),
        (f64::NEG_INFINITY, 1.0),
    ] {
        let err = Rescale::new(slope, intercept).expect_err("non-finite coefficient");
        assert!(matches!(err, SampleError::NonFiniteRescale { .. }), "{err}");
    }
    let rescale = Rescale::new(-2.5, 1024.0).expect("finite coefficients");
    assert_eq!((rescale.slope(), rescale.intercept()), (-2.5, 1024.0));
}

#[test]
fn rescale_maps_floats_in_their_own_arithmetic() {
    let rescale = Rescale::new(2.0, -1024.0).expect("finite coefficients");
    let mut ct = [0.0_f32, 1.5, 1000.0];
    rescale.apply(&mut ct).expect("float target");
    assert_eq!(ct, [-1024.0, -1021.0, 976.0]);

    // 2^53 + 2 is exact in f64 and lost in f32: the map runs in T.
    let wide = 9_007_199_254_740_994.0_f64;
    let mut values = [wide];
    Rescale::new(1.0, 2.0)
        .expect("finite coefficients")
        .apply(&mut values)
        .expect("float target");
    assert_eq!(values, [wide + 2.0]);
}

#[test]
fn rescale_identity_accepts_every_type_unchanged() {
    let identity = Rescale::new(1.0, 0.0).expect("finite coefficients");
    assert!(identity.is_identity());
    assert_eq!(identity, Rescale::IDENTITY);
    let mut labels = [0_u32, 7, u32::MAX];
    identity
        .apply(&mut labels)
        .expect("the identity applies to integers");
    assert_eq!(labels, [0, 7, u32::MAX]);
}

#[test]
fn rescale_into_an_integer_type_is_rejected_unchanged() {
    let rescale = Rescale::new(0.5, 0.0).expect("finite coefficients");
    assert!(!rescale.is_identity());
    let mut stored = [3_i16, -4];
    let err = rescale.apply(&mut stored).expect_err("integer target");
    assert!(matches!(
        err,
        SampleError::IntegerRescale {
            sample_type: SampleType::I16,
            slope: 0.5,
            intercept: 0.0
        }
    ));
    assert_eq!(stored, [3, -4]);
}

/// One rounding after the product and one after the sum, both in `f32`:
/// `(1 + 2^-23)(1 + 2^-22) - 1` rounds its `2^-45` term away at the product,
/// where `f64` arithmetic, or a fused multiply-add, keeps it.
#[test]
fn rescale_rounds_in_the_requested_float_type() {
    let mut values = [1.0 + f32::EPSILON];
    Rescale::new(1.0 + 2.0 * f64::from(f32::EPSILON), -1.0)
        .expect("finite coefficients")
        .apply(&mut values)
        .expect("float target");
    assert_eq!(values, [3.0 * f32::EPSILON]);
}

#[test]
fn rescale_coefficient_past_the_float_range_is_rejected_unchanged() {
    let rescale = Rescale::new(1e300, 0.0).expect("finite in f64");
    let mut narrow = [2.0_f32];
    let err = rescale.apply(&mut narrow).expect_err("1e300 overflows f32");
    assert!(matches!(
        err,
        SampleError::RescaleOutOfRange {
            sample_type: SampleType::F32,
            slope: 1e300,
            intercept: 0.0
        }
    ));
    assert_eq!(narrow, [2.0]);
    let mut wide = [2.0_f64];
    rescale.apply(&mut wide).expect("1e300 is an f64");
    assert_eq!(wide, [2e300]);
}

/// 1e-50 is below the smallest f32 subnormal (about 1.4e-45), so it rounds to
/// zero in f32 and would map every sample to the intercept.
#[test]
fn rescale_slope_that_vanishes_in_the_float_type_is_rejected_unchanged() {
    let rescale = Rescale::new(1e-50, 7.0).expect("finite in f64");
    let mut narrow = [2.0_f32];
    let err = rescale
        .apply(&mut narrow)
        .expect_err("1e-50 underflows f32");
    assert!(matches!(
        err,
        SampleError::RescaleOutOfRange {
            sample_type: SampleType::F32,
            slope: 1e-50,
            intercept: 7.0
        }
    ));
    assert_eq!(narrow, [2.0]);
    // In f64 the slope survives and still moves a large enough sample.
    let mut wide = [1e40_f64];
    rescale.apply(&mut wide).expect("1e-50 is an f64");
    assert_eq!(wide, [1e40_f64 * 1e-50 + 7.0]);
    assert!(wide[0] > 7.0);
}

/// An intercept past the f32 range is refused even with a representable
/// slope.
#[test]
fn rescale_intercept_past_the_float_range_is_rejected_unchanged() {
    let rescale = Rescale::new(1.0, 1e300).expect("finite in f64");
    let mut narrow = [2.0_f32];
    let err = rescale.apply(&mut narrow).expect_err("1e300 overflows f32");
    assert!(matches!(
        err,
        SampleError::RescaleOutOfRange {
            sample_type: SampleType::F32,
            slope: 1.0,
            intercept: 1e300
        }
    ));
    assert_eq!(narrow, [2.0]);
}

/// A slope of exactly zero is not an underflow: it maps every sample to the
/// intercept, as the header declares.
#[test]
fn rescale_with_a_zero_slope_maps_to_the_intercept() {
    let rescale = Rescale::new(0.0, -7.5).expect("finite");
    let mut values = [3.0_f32, -1.0, 1e30];
    rescale.apply(&mut values).expect("zero is representable");
    assert_eq!(values, [-7.5; 3]);
}

#[test]
fn rescale_errors_name_the_coefficients() {
    let integer = SampleError::IntegerRescale {
        sample_type: SampleType::I16,
        slope: 0.5,
        intercept: -3.0,
    };
    assert_eq!(
        integer.to_string(),
        "rescale y = 0.5 * x + -3 has no faithful i16 result; \
         read the stored samples and the rescale separately"
    );
    let range = SampleError::RescaleOutOfRange {
        sample_type: SampleType::F32,
        slope: 1e300,
        intercept: 0.0,
    };
    assert_eq!(
        range.to_string(),
        "rescale slope 1e300 and intercept 0.0 fall outside the range of f32"
    );
}
