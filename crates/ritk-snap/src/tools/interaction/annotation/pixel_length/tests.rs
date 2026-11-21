use super::{MeasurementError, PixelLength};

#[test]
fn pixel_norm_preserves_geometric_symmetries() -> Result<(), MeasurementError> {
    // The integer 3-4-5 triangle and powers of two have exactly representable
    // coordinates and distances throughout this finite exponent range.
    let mut scale = 1.0;
    for _ in 0..64 {
        for translation in [0.0, 8.0 * scale, -8.0 * scale] {
            let start = [translation, translation];
            let end = [translation + 3.0 * scale, translation + 4.0 * scale];
            for endpoints in [
                [start, end],
                [end, start],
                [[start[1], start[0]], [end[1], end[0]]],
            ] {
                let length = PixelLength::try_from(endpoints)?;
                assert_eq!(length.pixels(), 5.0 * scale);
                let stored: [[f32; 2]; 2] = length.into();
                assert_eq!(stored, endpoints);
            }
        }
        scale *= 2.0;
    }
    Ok(())
}

#[test]
fn zero_tiny_and_large_finite_lengths_remain_distinct() -> Result<(), MeasurementError> {
    for (endpoints, expected) in [
        ([[7.0, -3.0], [7.0, -3.0]], 0.0),
        ([[0.0, 0.0], [f32::from_bits(1), 0.0]], f32::from_bits(1)),
        ([[0.0, 0.0], [f32::MAX, 0.0]], f32::MAX),
        ([[f32::MAX, f32::MAX], [f32::MAX, f32::MAX]], 0.0),
    ] {
        assert_eq!(PixelLength::try_from(endpoints)?.pixels(), expected);
    }
    Ok(())
}

#[test]
fn nonfinite_coordinates_report_their_exact_location() {
    for point in 0..2 {
        for component in 0..2 {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut endpoints = [[0.0; 2]; 2];
                endpoints[point][component] = invalid;
                assert!(matches!(
                    PixelLength::try_from(endpoints),
                    Err(MeasurementError::NonFiniteImageCoordinate {
                        point: rejected_point,
                        component: rejected_component,
                        value,
                    }) if rejected_point == point
                        && rejected_component == component
                        && value.to_bits() == invalid.to_bits()
                ));
            }
        }
    }
}

#[test]
fn finite_endpoints_can_still_overflow_the_distance() {
    for endpoints in [
        [[-f32::MAX, 0.0], [f32::MAX, 0.0]],
        [[0.0, 0.0], [f32::MAX, f32::MAX]],
    ] {
        assert_eq!(
            PixelLength::try_from(endpoints),
            Err(MeasurementError::NonFiniteResult {
                kind: "pixel length",
            })
        );
    }
}

#[test]
fn serde_retains_endpoint_bits_and_rejects_invalid_distances() {
    for endpoints in [
        [[-0.0, 0.0], [3.0, 4.0]],
        [[0.0, 0.0], [f32::from_bits(1), -0.0]],
    ] {
        let length = PixelLength::try_from(endpoints)
            .expect("invariant: these reference endpoints have finite distances");
        let encoded = serde_json::to_string(&length)
            .expect("invariant: validated endpoints contain only finite coordinates");
        let decoded: PixelLength = serde_json::from_str(&encoded)
            .expect("invariant: serialized finite endpoints retain the validated shape");
        let actual: [[f32; 2]; 2] = decoded.into();
        assert_eq!(
            actual.map(|point| point.map(f32::to_bits)),
            endpoints.map(|point| point.map(f32::to_bits))
        );
        assert_eq!(decoded.pixels(), length.pixels());
    }
    for encoded in [
        "[[-3.4028235e38,0.0],[3.4028235e38,0.0]]",
        "[[0.0,0.0],[3.4028235e38,3.4028235e38]]",
    ] {
        let error = serde_json::from_str::<PixelLength>(encoded)
            .expect_err("overflowing distance cannot form a finite pixel length");
        assert_eq!(error.classify(), serde_json::error::Category::Data);
        assert!(error
            .to_string()
            .contains("pixel length measurement result is not finite"));
    }
}

#[test]
fn serde_rejects_incomplete_extra_or_nonnumeric_coordinates() {
    use serde_json::error::Category::{Data, Syntax};

    // After the fixed-array visitor consumes two pairs, serde_json's end_seq
    // reports extra elements as TrailingCharacters, classified as Syntax.
    for (encoded, expected) in [
        ("[]", Data),
        ("[[0.0,0.0]]", Data),
        ("[[0.0],[1.0,1.0]]", Data),
        ("[[0.0,0.0],[1.0,1.0],[2.0,2.0]]", Syntax),
        ("[[null,0.0],[1.0,1.0]]", Data),
    ] {
        let error = serde_json::from_str::<PixelLength>(encoded)
            .expect_err("pixel endpoints require exactly two pairs of finite numbers");
        assert_eq!(error.classify(), expected, "input: {encoded}");
    }
}
