use super::*;
use crate::scaling::IntegerScaling;
use eunomia::NumericElement;
use ritk_codecs::sample::Rescale;
use std::fmt::Debug;

/// One stored datatype range, `valid_range`, and real range.
struct Case {
    storage: [f64; 2],
    valid: [f64; 2],
    real: [f64; 2],
}

const CASES: [Case; 5] = [
    // The offset-dominated case: stored 60001 is one step above `valid_min`.
    Case {
        storage: [0.0, 65_535.0],
        valid: [60_000.0, 65_535.0],
        real: [0.0, 1.0],
    },
    // Signed storage, the default valid range, a real range not centred on zero.
    Case {
        storage: [-32_768.0, 32_767.0],
        valid: [-32_768.0, 32_767.0],
        real: [-1_000.0, 3_000.0],
    },
    // A negative `valid_min` and a real range far from the stored scale.
    Case {
        storage: [-32_768.0, 32_767.0],
        valid: [-500.0, 20_000.0],
        real: [-5.0, 12_345.678],
    },
    // A fractional multiplier and a nonzero real minimum.
    Case {
        storage: [0.0, 255.0],
        valid: [100.0, 200.0],
        real: [0.5, 2.5],
    },
    // An entirely negative real range.
    Case {
        storage: [0.0, 65_535.0],
        valid: [1_000.0, 2_000.0],
        real: [-3.0, -1.0],
    },
];

/// libminc `miconvert_voxel_to_real` (`libsrc2/convert.c`).
fn libminc(stored: f64, valid: [f64; 2], real: [f64; 2]) -> f64 {
    (stored - valid[0]) / (valid[1] - valid[0]) * (real[1] - real[0]) + real[0]
}

/// The unit roundoff `u = epsilon / 2` of `T`.
fn unit_roundoff<T: Sample>() -> f64 {
    if T::TYPE.byte_width() == 4 {
        f64::from(f32::EPSILON) / 2.0
    } else {
        f64::EPSILON / 2.0
    }
}

fn map_of(case: &Case) -> RealValueMap {
    IntegerScaling::new(
        case.valid,
        case.storage,
        &[case.real[0]],
        &[case.real[1]],
        1,
        1,
    )
    .expect("valid case")
    .slice_maps()[0]
}

/// Assert `|map(v) - libminc(v)| <= 10u (P + R)` for every stored integer in
/// the valid range.
///
/// Derivation, with `u` the unit roundoff of `T` and `d = v - valid_min`,
/// `a = (real_max - real_min) / (valid_max - valid_min)`, `P = |d a|`,
/// `R = |real_min|`:
///
/// - `d` is exact: `v`, `valid_min`, and their difference are integers below
///   2^24, which `T` represents exactly.
/// - `a` carries two f64 roundings (`real_max - real_min`, the quotient) and
///   one conversion to `T`: relative error at most `u + 2·2^-53 <= 3u`.
/// - the multiply rounds once (`u`), the conversion of `real_min` once (`u`),
///   the add once (`u`): `|map(v) - exact| <= 3uP + uP + uR + u(P + R) + O(u²)
///   <= 6u(P + R)`.
/// - libminc's formula in f64 has four roundings, so
///   `|libminc - exact| <= 4·2^-53 (P + R)`, and rounding it to `T` adds at
///   most `u |y| <= u (P + R)`.
///
/// The implementation-minus-reference difference is therefore at most
/// `6u + u + 4·2^-53 <= 7u + 4·2^-53` for f32 (`u = 2^-24`, so `<= 8u`), and
/// for f64, whose reference needs no rounding to `T`, `6u + 4u = 10u`: both
/// are within `10u (P + R)`.
fn agrees_with_libminc<T: Sample + Debug>(case: &Case) {
    let map = map_of(case);
    let slope = (case.real[1] - case.real[0]) / (case.valid[1] - case.valid[0]);
    let u = unit_roundoff::<T>();
    let (low, high) = (
        i64::from_real_sample(case.valid[0]),
        i64::from_real_sample(case.valid[1]),
    );
    for stored in low..=high {
        let stored = f64::from_signed_sample(stored);
        let mut values = [T::from_real_sample(stored)];
        map.apply(&mut values).expect("floating-point map");
        let got = NumericElement::to_f64(values[0]);
        let want =
            NumericElement::to_f64(T::from_real_sample(libminc(stored, case.valid, case.real)));
        let bound =
            10.0 * u * ((stored - case.valid[0]) * slope).abs() + 10.0 * u * case.real[0].abs();
        let error = (got - want).abs();
        assert!(
            error <= bound,
            "{} v={stored} map {got} libminc {want} bound {bound}",
            T::TYPE
        );
    }
}

#[test]
fn maps_agree_with_libminc_within_the_derived_bound() {
    for case in &CASES {
        agrees_with_libminc::<f32>(case);
        agrees_with_libminc::<f64>(case);
    }
}

#[test]
fn the_folded_intercept_form_violates_the_bound_where_the_offset_is_applied_first() {
    // v = 60001 with valid_range [60000, 65535] and real range [0, 1]:
    // libminc gives 1/5535 = 1.8067e-4. Folding `valid_min` into the
    // intercept (`x * a + (real_min - valid_min * a)`) cancels 10.84 against
    // 10.84 in f32 and returns 1.8024e-4.
    let case = &CASES[0];
    let slope = (case.real[1] - case.real[0]) / (case.valid[1] - case.valid[0]);
    let folded = Rescale::new(slope, case.real[0] - case.valid[0] * slope).expect("finite");
    let mut folded_value = [60_001.0_f32];
    folded.apply(&mut folded_value).expect("f32 map");
    let want = f64::from(f32::from_real_sample(libminc(
        60_001.0, case.valid, case.real,
    )));
    let bound = 10.0 * unit_roundoff::<f32>() * slope;

    assert!(
        (f64::from(folded_value[0]) - want).abs() > bound,
        "the folded form was expected to miss the bound: {}",
        folded_value[0]
    );
    let mut offset_first = [60_001.0_f32];
    map_of(case).apply(&mut offset_first).expect("f32 map");
    assert!((f64::from(offset_first[0]) - want).abs() <= bound);
}

#[test]
fn the_offset_is_exact_for_stored_integers() {
    // real range [0, 5535] over valid_range [60000, 65535] has slope 1, so the
    // real value is exactly `stored - 60000`.
    let map = RealValueMap::new(60_000.0, 1.0, 0.0).expect("finite");
    let mut values: Vec<f32> = (60_000_u32..=65_535)
        .map(|value| f32::from_unsigned_sample(u64::from(value)))
        .collect();
    map.apply(&mut values).expect("f32 map");
    let expected: Vec<f32> = (0_u32..=5_535)
        .map(|value| f32::from_unsigned_sample(u64::from(value)))
        .collect();
    assert_eq!(values, expected);
}

#[test]
fn a_map_whose_range_equals_its_valid_range_is_the_identity() {
    let map = RealValueMap::new(-32_768.0, 1.0, -32_768.0).expect("finite");
    assert!(map.is_identity());
    assert_eq!(map, RealValueMap::IDENTITY);
    let mut values = [-32_768_i16, 0, 32_767];
    map.apply(&mut values).expect("the identity suits integers");
    assert_eq!(values, [-32_768, 0, 32_767]);
}

#[test]
fn an_offset_or_scaled_map_refuses_an_integer_type_and_changes_nothing() {
    for map in [
        RealValueMap::new(5.0, 1.0, 0.0).expect("finite"),
        RealValueMap::new(0.0, 0.5, 0.0).expect("finite"),
    ] {
        let mut values = [5_i16, 6, 7];
        let error = map
            .apply(&mut values)
            .expect_err("no faithful integer form");
        assert!(error.to_string().contains("i16"), "{error}");
        assert_eq!(values, [5, 6, 7]);
    }
}

#[test]
fn a_refused_scale_leaves_the_values_unshifted() {
    // 1e-50 underflows to zero in f32, so the scale is refused after the offset
    // would have run.
    let map = RealValueMap::new(10.0, 1e-50, 0.0).expect("finite in f64");
    let mut values = [10.0_f32, 11.0];
    map.apply(&mut values).expect_err("slope underflows in f32");
    assert_eq!(values, [10.0, 11.0]);
}

#[test]
fn non_finite_coefficients_are_rejected() {
    for (minimum, slope, intercept) in [
        (f64::NAN, 1.0, 0.0),
        (0.0, f64::INFINITY, 0.0),
        (0.0, 1.0, f64::NAN),
    ] {
        assert!(RealValueMap::new(minimum, slope, intercept).is_err());
    }
}
