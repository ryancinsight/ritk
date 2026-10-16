//! An inverse over physical coordinates must not admit non-finite arithmetic.

use crate::{CartesianGridGeometry, CoordinateMap, Direction, NonCartesianGrid, Point, Spacing};
use std::sync::Once;

fn trace_tests() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ACTIVE)
            .with_test_writer()
            .init();
    });
}

fn reject_non_finite_entries<const D: usize>() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for row in 0..D {
            for column in 0..D {
                let _case = tracing::info_span!("case", rank = D, row, column, invalid).entered();
                let direction = tracing::info_span!("arrange").in_scope(|| {
                    let mut direction = Direction::<D>::identity();
                    direction[(row, column)] = invalid;
                    direction
                });
                let inverse = tracing::info_span!("act").in_scope(|| direction.try_inverse());
                tracing::info_span!("assert").in_scope(|| {
                    assert_eq!(inverse, None, "non-finite direction must have no inverse");
                });
            }
        }
    }
}

#[test]
fn non_finite_direction_entries_are_rejected_in_every_position() {
    trace_tests();
    reject_non_finite_entries::<1>();
    reject_non_finite_entries::<2>();
    reject_non_finite_entries::<3>();
}

#[test]
fn finite_arithmetic_overflow_does_not_return_a_corrupt_inverse() {
    trace_tests();
    // Working-matrix normalization overflows MAX/0.5. Working-matrix
    // elimination overflows MAX+MAX; its real inverse has finite entries,
    // but the unscaled algorithm must report failure instead of returning
    // the incorrect [[1,0],[0,0]]. Inverse-only elimination overflows -MAX*2
    // while every working-matrix entry stays finite.
    for (stage, rows) in [
        ("working normalization", [[0.5, f64::MAX], [0.0, 1.0]]),
        ("working elimination", [[1.0, f64::MAX], [-1.0, f64::MAX]]),
        ("inverse elimination", [[1.0, f64::MAX], [0.0, 0.5]]),
    ] {
        let _case = tracing::info_span!("case", stage).entered();
        let direction = tracing::info_span!("arrange").in_scope(|| Direction::from_rows(rows));
        let inverse = tracing::info_span!("act").in_scope(|| direction.try_inverse());
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(inverse, None, "unrepresentable {stage} must be reported");
        });
    }
}

#[test]
fn invalid_direction_is_propagated_through_cartesian_geometry() {
    trace_tests();
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let _case = tracing::info_span!("case", invalid).entered();
        let direction = tracing::info_span!("arrange")
            .in_scope(|| Direction::from_rows([[1.0, invalid], [0.0, 1.0]]));
        let error = tracing::info_span!("act").in_scope(|| {
            CartesianGridGeometry::new(
                &Point::new([3.0, 5.0]),
                &Spacing::try_new([2.0, 4.0]).expect("positive finite fixture spacing"),
                &direction,
                &CoordinateMap::Cartesian,
            )
            .expect_err("a non-finite transform pair must not be admitted")
        });
        tracing::info_span!("assert").in_scope(|| {
            assert!(matches!(error, NonCartesianGrid::SingularDirection));
            assert!(error.to_string().contains("finite"), "{error}");
        });
    }
}

#[test]
fn finite_shear_and_pivot_exchange_keep_the_exact_inverse() {
    trace_tests();
    let direction =
        tracing::info_span!("arrange").in_scope(|| Direction::from_rows([[0.0, 2.0], [1.0, 0.5]]));
    let inverse = tracing::info_span!("act")
        .in_scope(|| direction.try_inverse().expect("finite nonsingular fixture"));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(inverse, Direction::from_rows([[-0.25, 1.0], [0.5, 0.0]]));
        assert_eq!(direction * inverse, Direction::identity());
        assert_eq!(inverse * direction, Direction::identity());
    });
}

#[test]
fn extreme_finite_diagonal_keeps_its_exact_subnormal_inverse() {
    trace_tests();
    // 2^1023 and 2^-1023 are exact binary values. The former is normal and
    // the latter subnormal, so this also guards against rejecting a finite
    // representable inverse merely because an input or result is extreme.
    let (direction, expected) = tracing::info_span!("arrange").in_scope(|| {
        let large = f64::from_bits(0x7fe0_0000_0000_0000);
        let reciprocal = f64::from_bits(0x0008_0000_0000_0000);
        (
            Direction::from_rows([[large, 0.0], [0.0, large]]),
            Direction::from_rows([[reciprocal, 0.0], [0.0, reciprocal]]),
        )
    });
    let inverse = tracing::info_span!("act").in_scope(|| {
        direction
            .try_inverse()
            .expect("finite representable inverse")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(inverse, expected);
        assert_eq!(direction * inverse, Direction::identity());
    });
}
