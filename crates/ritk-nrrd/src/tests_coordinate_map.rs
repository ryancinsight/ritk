//! Coordinate-map persistence oracles.
#![expect(
    clippy::unwrap_used,
    reason = "fixture unwraps on well-formed in-memory headers; ratchet RITK-UNWRAP-1"
)]

use super::*;
use ritk_spatial::{
    CoordinateMap, CurvilinearArray, Direction, PhasedArray3D, SliceSeries, SliceTransform,
};

fn curvilinear() -> CoordinateMap {
    CoordinateMap::CurvilinearArray(
        CurvilinearArray::try_new(1.0e-4, 0.06, 0.5_f64.to_radians(), 5.0_f64.to_radians())
            .expect("geometry"),
    )
}

fn phased() -> CoordinateMap {
    CoordinateMap::PhasedArray3D(
        PhasedArray3D::try_new(
            1.0e-4,
            0.01,
            0.75_f64.to_radians(),
            1.5_f64.to_radians(),
            -0.4,
            -0.2,
        )
        .expect("geometry"),
    )
}

fn slice_series() -> CoordinateMap {
    let first = SliceTransform::new(
        Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        [-12.5, 6.0, 3.0],
    );
    let second = SliceTransform::new(
        Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        [-12.5, 6.0, 5.5],
    );
    CoordinateMap::SliceSeries(SliceSeries::try_new(vec![first, second]).expect("valid sweep"))
}

/// Every non-Cartesian map must survive encode/decode exactly.
///
/// The parameters are `f64` written in Rust's shortest round-trip form, so
/// equality here is exact rather than epsilon-bounded: a lossy encoding would
/// silently shift every physical point the map produces.
#[test]
fn non_cartesian_maps_round_trip_exactly() {
    for original in [curvilinear(), phased(), slice_series()] {
        let encoded = encode(&original).expect("non-Cartesian maps encode");
        let decoded = decode(&encoded).expect("decode");
        assert_eq!(decoded, original, "payload was: {encoded}");
    }
}

#[test]
fn slice_series_rejects_wrong_component_counts_and_nonfinite_values() {
    let short_transform = "slice_series count=1 transforms=1,0,0,0,1,0,0,0,1,0,0";
    assert!(decode(short_transform)
        .expect_err("slice transforms require twelve components")
        .to_string()
        .contains("fewer than 12 components"));
    let nonfinite = "slice_series count=1 transforms=NaN,0,0,0,1,0,0,0,1,0,0,0";
    assert!(decode(nonfinite)
        .expect_err("slice transforms must be finite")
        .to_string()
        .contains("non-finite"));
}

/// Cartesian is written by omission, so absence and Cartesian must be the same
/// statement — otherwise every pre-existing NRRD would fail to load.
#[test]
fn cartesian_is_encoded_by_omission() {
    assert!(encode(&CoordinateMap::Cartesian).is_none());
    let empty = std::collections::HashMap::new();
    assert_eq!(
        from_header(&empty).expect("absent map is Cartesian"),
        CoordinateMap::Cartesian
    );
    // An explicit tag is still accepted, so a file written by a tool that
    // spells it out loads identically.
    assert_eq!(
        decode("cartesian").expect("explicit cartesian"),
        CoordinateMap::Cartesian
    );
}

/// Named parameters must be order-independent: a reader that depended on
/// position would misread a file whose writer reordered them.
#[test]
fn parameters_are_order_independent() {
    let forward = "curvilinear radius_sample_size=0.0001 first_sample_distance=0.06 \
                   lateral_angular_separation=0.008726646259971648 first_lateral_angle=-0.5";
    let shuffled = "curvilinear first_lateral_angle=-0.5 \
                    lateral_angular_separation=0.008726646259971648 \
                    first_sample_distance=0.06 radius_sample_size=0.0001";
    assert_eq!(decode(forward).unwrap(), decode(shuffled).unwrap());
}

/// A malformed map is an error, never a silent fall back to Cartesian.
///
/// Falling back would hand the caller beam data labelled as a raster — exactly
/// the failure this field exists to prevent, and undetectable downstream.
#[test]
fn malformed_maps_are_rejected_rather_than_defaulted() {
    // Unknown tag: a file from a newer ritk.
    assert!(decode("slice_series foo=1").is_err());
    // Missing a required parameter.
    assert!(decode("curvilinear radius_sample_size=0.0001").is_err());
    // Unparseable number.
    assert!(decode(
        "curvilinear radius_sample_size=abc first_sample_distance=0.06 \
         lateral_angular_separation=0.0087 first_lateral_angle=-0.5"
    )
    .is_err());
    // Parameter without '='.
    assert!(decode("curvilinear radius_sample_size").is_err());
    // Empty value.
    assert!(decode("").is_err());
    // Values the geometry itself rejects must not be resurrected from a file.
    assert!(decode(
        "curvilinear radius_sample_size=0 first_sample_distance=0.06 \
         lateral_angular_separation=0.0087 first_lateral_angle=-0.5"
    )
    .is_err());
}

#[test]
fn duplicate_named_parameters_are_rejected() {
    let duplicate_scalar = "curvilinear radius_sample_size=0.0001 radius_sample_size=0.0002 \
        first_sample_distance=0.06 lateral_angular_separation=0.0087 first_lateral_angle=-0.5";
    let error = decode(duplicate_scalar).expect_err("the first duplicate must not win silently");
    assert!(error
        .to_string()
        .contains("duplicate ritk_coordinate_map parameter"));

    let duplicate_count = "slice_series count=1 count=2 transforms=1,0,0,0,1,0,0,0,1,0,0,0";
    let error = decode(duplicate_count).expect_err("the first count must not win silently");
    assert!(error
        .to_string()
        .contains("duplicate ritk_coordinate_map parameter 'count'"));

    let duplicate_transforms = "slice_series count=1 transforms=1,0,0,0,1,0,0,0,1,0,0,0 \
        transforms=1,0,0,0,1,0,0,0,1,0,0,0";
    let error =
        decode(duplicate_transforms).expect_err("the first transform list must not win silently");
    assert!(error
        .to_string()
        .contains("duplicate ritk_coordinate_map parameter 'transforms'"));
}

/// The header helper must find the key and reject a bad payload under it.
#[test]
fn header_lookup_uses_the_documented_key() {
    let mut headers = std::collections::HashMap::new();
    headers.insert(
        COORDINATE_MAP_KEY.to_string(),
        encode(&curvilinear()).unwrap(),
    );
    assert_eq!(from_header(&headers).unwrap(), curvilinear());

    headers.insert(COORDINATE_MAP_KEY.to_string(), "nonsense".to_string());
    assert!(from_header(&headers).is_err());
}
