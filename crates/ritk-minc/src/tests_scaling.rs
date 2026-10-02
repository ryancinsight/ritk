use super::*;
use std::fmt::Debug;

const STORAGE_RANGE: [f64; 2] = [-32_768.0, 32_767.0];

fn scaling(valid_range: [f64; 2], minima: &[f64], maxima: &[f64]) -> IntegerScaling {
    IntegerScaling::new(valid_range, STORAGE_RANGE, minima, maxima, 4, 8)
        .expect("valid scaling fixture")
}

/// `stored` integers mapped by slice `slice` of `scaling`, in `T` arithmetic.
fn mapped<T: Sample>(scaling: &IntegerScaling, slice: usize, stored: &[i16]) -> Vec<T> {
    let mut values: Vec<T> = stored
        .iter()
        .copied()
        .map(|value| T::from_signed_sample(i64::from(value)))
        .collect();
    scaling.slice_maps()[slice]
        .apply(&mut values)
        .expect("floating-point maps apply");
    values
}

fn expected<T: Sample>(values: &[f64]) -> Vec<T> {
    values.iter().copied().map(T::from_real_sample).collect()
}

fn assert_mapped<T: Sample + Debug>(
    scaling: &IntegerScaling,
    slice: usize,
    stored: &[i16],
    real: &[f64],
) {
    assert_eq!(
        mapped::<T>(scaling, slice, stored),
        expected::<T>(real),
        "{}",
        T::TYPE
    );
}

fn global_range_maps_endpoints_and_midpoint<T: Sample + Debug>() {
    let scaling = scaling([0.0, 100.0], &[-100.0], &[300.0]);

    assert_eq!(scaling.slice_maps().len(), 2, "one map per slice");
    for slice in 0..2 {
        assert_mapped::<T>(&scaling, slice, &[0, 50, 100], &[-100.0, 100.0, 300.0]);
    }
}

fn per_slice_ranges_select_first_spatial_axis<T: Sample + Debug>() {
    let scaling = scaling([0.0, 100.0], &[-1_000.0, 0.0], &[1_000.0, 200.0]);

    assert_mapped::<T>(
        &scaling,
        0,
        &[0, 25, 50, 100],
        &[-1_000.0, -500.0, 0.0, 1_000.0],
    );
    assert_mapped::<T>(&scaling, 1, &[0, 25, 50, 100], &[0.0, 50.0, 100.0, 200.0]);
}

fn default_real_range_maps_to_the_unit_interval<T: Sample + Debug>() {
    let scaling = scaling([0.0, 100.0], &[0.0], &[1.0]);

    assert_mapped::<T>(&scaling, 0, &[0, 25, 50, 100], &[0.0, 0.25, 0.5, 1.0]);
}

fn uniform_real_slice_maps_every_valid_value_to_constant<T: Sample + Debug>() {
    let scaling = scaling([0.0, 100.0], &[7.5], &[7.5]);

    assert_mapped::<T>(&scaling, 0, &[0, 83, 100], &[7.5, 7.5, 7.5]);
}

#[test]
fn global_range_maps_in_every_float_type() {
    global_range_maps_endpoints_and_midpoint::<f32>();
    global_range_maps_endpoints_and_midpoint::<f64>();
}

#[test]
fn per_slice_ranges_map_in_every_float_type() {
    per_slice_ranges_select_first_spatial_axis::<f32>();
    per_slice_ranges_select_first_spatial_axis::<f64>();
}

#[test]
fn default_real_range_maps_in_every_float_type() {
    default_real_range_maps_to_the_unit_interval::<f32>();
    default_real_range_maps_to_the_unit_interval::<f64>();
}

#[test]
fn uniform_real_slice_maps_in_every_float_type() {
    uniform_real_slice_maps_every_valid_value_to_constant::<f32>();
    uniform_real_slice_maps_every_valid_value_to_constant::<f64>();
}

#[test]
fn reversed_valid_range_has_identical_semantics() {
    let forward = scaling([0.0, 100.0], &[-100.0], &[300.0]);
    let reverse = scaling([100.0, 0.0], &[-100.0], &[300.0]);

    assert_eq!(forward.slice_maps(), reverse.slice_maps());
}

#[test]
fn a_real_range_equal_to_the_valid_range_is_the_identity() {
    let signed = scaling(STORAGE_RANGE, &[STORAGE_RANGE[0]], &[STORAGE_RANGE[1]]);
    let unsigned = IntegerScaling::new([0.0, 255.0], [0.0, 255.0], &[0.0], &[255.0], 4, 8)
        .expect("valid scaling fixture");

    assert_eq!(signed.slice_maps(), [RealValueMap::IDENTITY; 2]);
    assert_eq!(unsigned.slice_maps(), [RealValueMap::IDENTITY; 2]);
}

#[test]
fn a_nonidentity_map_refuses_an_integer_type() {
    let scaling = scaling([0.0, 100.0], &[0.0], &[1.0]);
    let mut values = [0_i16, 50, 100, 25];

    let error = scaling.slice_maps()[0]
        .apply(&mut values)
        .expect_err("a fractional map has no integer result");

    assert!(
        error.to_string().contains("i16"),
        "unexpected error: {error}"
    );
    assert_eq!(values, [0, 50, 100, 25], "a refused map changes nothing");
}

#[test]
fn malformed_ranges_are_rejected() {
    let degenerate = IntegerScaling::new([1.0, 1.0], STORAGE_RANGE, &[0.0], &[1.0], 4, 8)
        .expect_err("degenerate valid range must fail");
    assert!(
        degenerate.to_string().contains("endpoints must differ"),
        "unexpected error: {degenerate:#}"
    );

    let inverted_real = IntegerScaling::new([0.0, 1.0], STORAGE_RANGE, &[2.0], &[1.0], 4, 8)
        .expect_err("inverted real range must fail");
    assert!(
        inverted_real.to_string().contains("image-min 2 greater"),
        "unexpected error: {inverted_real:#}"
    );

    let wrong_slice_count =
        IntegerScaling::new([0.0, 1.0], STORAGE_RANGE, &[0.0; 3], &[1.0; 3], 4, 8)
            .expect_err("wrong per-slice range count must fail");
    assert!(
        wrong_slice_count
            .to_string()
            .contains("one entry per slice (2), got 3"),
        "unexpected error: {wrong_slice_count:#}"
    );

    let outside_storage = IntegerScaling::new(
        [STORAGE_RANGE[0] - 1.0, 100.0],
        STORAGE_RANGE,
        &[0.0],
        &[1.0],
        4,
        8,
    )
    .expect_err("valid range outside the stored datatype must fail");
    assert!(
        outside_storage
            .to_string()
            .contains("exceeds the stored datatype range"),
        "unexpected error: {outside_storage:#}"
    );

    let length_mismatch = IntegerScaling::new([0.0, 1.0], STORAGE_RANGE, &[0.0, 0.0], &[1.0], 4, 8)
        .expect_err("image-min/image-max length mismatch must fail");
    assert!(
        length_mismatch.to_string().contains("length mismatch"),
        "unexpected error: {length_mismatch:#}"
    );

    let inconsistent_geometry =
        IntegerScaling::new([0.0, 1.0], STORAGE_RANGE, &[0.0], &[1.0], 3, 8)
            .expect_err("non-divisible scaling geometry must fail");
    assert!(
        inconsistent_geometry
            .to_string()
            .contains("geometry is inconsistent"),
        "unexpected error: {inconsistent_geometry:#}"
    );

    let non_finite = IntegerScaling::new([0.0, 1.0], STORAGE_RANGE, &[0.0], &[f64::INFINITY], 4, 8)
        .expect_err("non-finite real range must fail");
    assert!(
        non_finite.to_string().contains("must be finite"),
        "unexpected error: {non_finite:#}"
    );
}

#[test]
fn a_non_finite_valid_range_endpoint_is_rejected() {
    for (valid_range, shown) in [
        ([f64::NAN, 1.0], "[NaN, 1]"),
        ([0.0, f64::NAN], "[0, NaN]"),
        ([0.0, f64::INFINITY], "[0, inf]"),
        ([f64::NEG_INFINITY, 1.0], "[-inf, 1]"),
    ] {
        let error = IntegerScaling::new(valid_range, STORAGE_RANGE, &[0.0], &[1.0], 4, 8)
            .expect_err("a non-finite valid_range endpoint must fail");
        assert_eq!(
            error.to_string(),
            format!("MINC2 valid_range must contain finite endpoints, got {shown}")
        );
    }
}

#[test]
fn stored_samples_outside_valid_range_are_rejected_with_their_index() {
    let scaling = scaling([0.0, 100.0], &[0.0], &[1.0]);

    scaling
        .check_stored(&SampleBuffer::I16(vec![0, 25, 50, 100, 0, 25, 50, 100]))
        .expect("every sample is inside valid_range");
    let high = scaling
        .check_stored(&SampleBuffer::I16(vec![0, 25, 101, 100]))
        .expect_err("101 is above valid_range");
    assert!(
        high.to_string().contains("stored voxel 2 value 101"),
        "unexpected error: {high:#}"
    );
    let low = scaling
        .check_stored(&SampleBuffer::F64(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -0.5]))
        .expect_err("-0.5 is below valid_range");
    assert!(
        low.to_string().contains("stored voxel 6 value -0.5"),
        "unexpected error: {low:#}"
    );
}

#[test]
fn default_integer_ranges_match_storage_types() {
    assert_eq!(
        integer_storage_range(SampleType::I16),
        Some([f64::from(i16::MIN), f64::from(i16::MAX)])
    );
    assert_eq!(
        integer_storage_range(SampleType::U8),
        Some([0.0, f64::from(u8::MAX)])
    );
    assert_eq!(integer_storage_range(SampleType::F32), None);
    assert_eq!(integer_storage_range(SampleType::F64), None);
}

#[test]
fn a_scalar_range_allocates_nothing_for_a_header_claimed_slice_count() {
    // 2^40 slices of one voxel: expanding a map per slice would reserve 32 TiB.
    let slice_count = 1_usize << 40;
    let scaling = IntegerScaling::new([0.0, 100.0], STORAGE_RANGE, &[0.0], &[1.0], 1, slice_count)
        .expect("a scalar range is valid for any slice count");

    assert_eq!(scaling.maps.len(), 1, "one held map, not one per slice");
    assert_eq!(scaling.slice_count, slice_count);
}

#[test]
fn a_per_slice_range_must_match_the_claimed_slice_count() {
    let error = IntegerScaling::new(
        [0.0, 100.0],
        STORAGE_RANGE,
        &[0.0, 0.0],
        &[1.0, 1.0],
        1,
        1_usize << 40,
    )
    .expect_err("two ranges do not cover 2^40 slices");

    assert!(
        error.to_string().contains("one entry per slice"),
        "unexpected error: {error:#}"
    );
}
