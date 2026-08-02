use super::*;
use consus_core::ByteOrder;
use core::num::NonZeroUsize;

fn scaling(valid_range: [f64; 2], minima: Vec<f64>, maxima: Vec<f64>) -> IntegerScaling {
    IntegerScaling::new(
        valid_range,
        [f64::from(i16::MIN), f64::from(i16::MAX)],
        minima,
        maxima,
        4,
        8,
    )
    .expect("valid scaling fixture")
}

#[test]
fn global_range_maps_endpoints_and_midpoint() {
    let scaling = scaling([0.0, 100.0], vec![-100.0], vec![300.0]);

    assert_eq!(scaling.scale(0.0, 0).expect("lower endpoint"), -100.0);
    assert_eq!(scaling.scale(50.0, 1).expect("midpoint"), 100.0);
    assert_eq!(scaling.scale(100.0, 7).expect("upper endpoint"), 300.0);
}

#[test]
fn per_slice_ranges_select_first_spatial_axis() {
    let scaling = scaling([0.0, 100.0], vec![-1_000.0, 0.0], vec![1_000.0, 200.0]);

    assert_eq!(scaling.scale(25.0, 1).expect("slice zero"), -500.0);
    assert_eq!(scaling.scale(25.0, 5).expect("slice one"), 50.0);
}

#[test]
fn reversed_valid_range_has_identical_semantics() {
    let forward = scaling([0.0, 100.0], vec![-100.0], vec![300.0]);
    let reverse = scaling([100.0, 0.0], vec![-100.0], vec![300.0]);

    assert_eq!(
        forward.scale(25.0, 2).expect("forward range"),
        reverse.scale(25.0, 2).expect("reversed range")
    );
}

#[test]
fn uniform_real_slice_maps_every_valid_value_to_constant() {
    let scaling = scaling([0.0, 100.0], vec![7.5], vec![7.5]);

    assert_eq!(scaling.scale(0.0, 0).expect("lower"), 7.5);
    assert_eq!(scaling.scale(83.0, 3).expect("interior"), 7.5);
    assert_eq!(scaling.scale(100.0, 7).expect("upper"), 7.5);
}

#[test]
fn malformed_ranges_and_out_of_range_samples_are_rejected() {
    let storage_range = [f64::from(i16::MIN), f64::from(i16::MAX)];
    let degenerate = IntegerScaling::new([1.0, 1.0], storage_range, vec![0.0], vec![1.0], 4, 8)
        .expect_err("degenerate valid range must fail");
    assert!(
        degenerate.to_string().contains("endpoints must differ"),
        "unexpected error: {degenerate:#}"
    );

    let inverted_real = IntegerScaling::new([0.0, 1.0], storage_range, vec![2.0], vec![1.0], 4, 8)
        .expect_err("inverted real range must fail");
    assert!(
        inverted_real.to_string().contains("image-min 2 greater"),
        "unexpected error: {inverted_real:#}"
    );

    let wrong_slice_count =
        IntegerScaling::new([0.0, 1.0], storage_range, vec![0.0; 3], vec![1.0; 3], 4, 8)
            .expect_err("wrong per-slice range count must fail");
    assert!(
        wrong_slice_count
            .to_string()
            .contains("one entry per slice (2), got 3"),
        "unexpected error: {wrong_slice_count:#}"
    );

    let outside_storage = IntegerScaling::new(
        [f64::from(i16::MIN) - 1.0, 100.0],
        storage_range,
        vec![0.0],
        vec![1.0],
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

    let scaling = scaling([0.0, 100.0], vec![0.0], vec![1.0]);
    let error = scaling
        .scale(101.0, 6)
        .expect_err("out-of-range sample must not be silently converted");
    assert!(error.to_string().contains("voxel 6"));
}

#[test]
fn default_integer_ranges_match_storage_types() {
    let signed = Datatype::Integer {
        bits: NonZeroUsize::new(16).expect("nonzero"),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let unsigned = Datatype::Integer {
        bits: NonZeroUsize::new(8).expect("nonzero"),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let float = Datatype::Float {
        bits: NonZeroUsize::new(32).expect("nonzero"),
        byte_order: ByteOrder::LittleEndian,
    };

    assert_eq!(
        default_integer_valid_range(&signed).expect("signed range"),
        Some([f64::from(i16::MIN), f64::from(i16::MAX)])
    );
    assert_eq!(
        default_integer_valid_range(&unsigned).expect("unsigned range"),
        Some([0.0, f64::from(u8::MAX)])
    );
    assert_eq!(default_integer_valid_range(&float).expect("float"), None);
}
