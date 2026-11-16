//! Direction inversion must reject every non-finite arithmetic result.

use crate::Direction;
use std::sync::Once;

fn init_tracing() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ACTIVE)
            .with_test_writer()
            .try_init()
            .expect("invariant: the test binary initializes one tracing subscriber");
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
    init_tracing();
    reject_non_finite_entries::<1>();
    reject_non_finite_entries::<2>();
    reject_non_finite_entries::<3>();
}

#[test]
fn normalization_overflow_is_rejected() {
    init_tracing();
    let _case = tracing::info_span!("case", stage = "normalization").entered();
    let direction = tracing::info_span!("arrange")
        .in_scope(|| Direction::from_rows([[0.5, f64::MAX], [0.0, 1.0]]));
    let inverse = tracing::info_span!("act").in_scope(|| direction.try_inverse());
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(inverse, None, "normalization overflow must be reported");
    });
}

#[test]
fn elimination_overflow_is_rejected() {
    init_tracing();
    let _case = tracing::info_span!("case", stage = "elimination").entered();
    let direction = tracing::info_span!("arrange")
        .in_scope(|| Direction::from_rows([[1.0, f64::MAX], [-1.0, f64::MAX]]));
    let inverse = tracing::info_span!("act").in_scope(|| direction.try_inverse());
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(inverse, None, "elimination overflow must be reported");
    });
}

#[test]
fn inverse_only_elimination_overflow_is_rejected() {
    init_tracing();
    let _case = tracing::info_span!("case", stage = "inverse elimination").entered();
    let direction = tracing::info_span!("arrange")
        .in_scope(|| Direction::from_rows([[1.0, f64::MAX], [0.0, 0.5]]));
    let inverse = tracing::info_span!("act").in_scope(|| direction.try_inverse());
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(inverse, None, "inverse overflow must be reported");
    });
}

#[test]
fn exact_pivoted_shear_keeps_its_exact_inverse() {
    init_tracing();
    let direction =
        tracing::info_span!("arrange").in_scope(|| Direction::from_rows([[0.0, 2.0], [1.0, 0.5]]));
    let inverse = tracing::info_span!("act").in_scope(|| {
        direction
            .try_inverse()
            .expect("invariant: the fixture is nonsingular")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(inverse, Direction::from_rows([[-0.25, 1.0], [0.5, 0.0]]));
        assert_eq!(direction * inverse, Direction::identity());
        assert_eq!(inverse * direction, Direction::identity());
    });
}

#[test]
fn extreme_finite_diagonal_keeps_its_exact_subnormal_inverse() {
    init_tracing();
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
            .expect("invariant: the inverse is representable")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(inverse, expected);
        assert_eq!(direction * inverse, Direction::identity());
    });
}

#[test]
fn small_diagonal_above_threshold_keeps_its_exact_reciprocal() {
    init_tracing();
    let (direction, expected) = tracing::info_span!("arrange").in_scope(|| {
        let small = f64::from_bits(0x3f60_0000_0000_0000);
        let reciprocal = f64::from_bits(0x4080_0000_0000_0000);
        (
            Direction::from_rows([[small, 0.0], [0.0, small]]),
            Direction::from_rows([[reciprocal, 0.0], [0.0, reciprocal]]),
        )
    });
    let inverse = tracing::info_span!("act").in_scope(|| {
        direction
            .try_inverse()
            .expect("invariant: the pivot clears the threshold")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(inverse, expected);
        assert_eq!(direction * inverse, Direction::identity());
    });
}

#[test]
fn small_diagonal_below_threshold_stays_singular() {
    init_tracing();
    let direction = tracing::info_span!("arrange").in_scope(|| {
        let tiny = f64::from_bits(0x0008_0000_0000_0000);
        Direction::from_rows([[tiny, 0.0], [0.0, 1.0]])
    });
    let inverse = tracing::info_span!("act").in_scope(|| direction.try_inverse());
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(inverse, None, "a sub-threshold pivot remains singular");
    });
}
