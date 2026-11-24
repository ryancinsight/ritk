use super::usable_pixel_spacing;

#[test]
fn spacing_rejects_values_that_round_to_zero_in_metres() {
    let least_positive_millimetre = f64::from_bits(1);
    assert_eq!(usable_pixel_spacing([least_positive_millimetre, 1.0]), None);
    assert_eq!(usable_pixel_spacing([1.0, least_positive_millimetre]), None);
}

/// A conformant pair passes through byte-for-byte; the guard must not
/// perturb legitimate geometry.
#[test]
fn conformant_spacing_is_preserved_exactly() {
    assert_eq!(usable_pixel_spacing([0.7, 0.65]), Some([0.7, 0.65]));
}

/// `PixelSpacing = "0\0"` occurs in derived and secondary-capture objects.
/// It parses as a valid pair of f64 and previously reached `Spacing::new`,
/// whose assert aborts the process.
#[test]
fn zero_spacing_is_rejected_rather_than_reaching_the_constructor() {
    assert_eq!(usable_pixel_spacing([0.0, 0.0]), None);
    assert_eq!(usable_pixel_spacing([0.0, 1.0]), None);
    assert_eq!(usable_pixel_spacing([1.0, 0.0]), None);
}

#[test]
fn negative_spacing_is_rejected() {
    assert_eq!(usable_pixel_spacing([-1.0, 1.0]), None);
    assert_eq!(usable_pixel_spacing([1.0, -0.5]), None);
}

#[test]
fn non_finite_spacing_is_rejected() {
    assert_eq!(usable_pixel_spacing([f64::NAN, 1.0]), None);
    assert_eq!(usable_pixel_spacing([1.0, f64::INFINITY]), None);
    assert_eq!(usable_pixel_spacing([f64::NEG_INFINITY, 1.0]), None);
}

/// Every value this guard admits must satisfy `Spacing::try_new`, which is
/// the invariant the panicking `Spacing::new` asserts. Without this the two
/// validity definitions could drift apart silently.
#[test]
fn admitted_values_satisfy_the_spacing_invariant() {
    for candidate in [
        [0.7, 0.65],
        [1.0, 1.0],
        [f64::MIN_POSITIVE, 1e6],
        [0.0, 1.0],
        [-1.0, 1.0],
        [f64::NAN, 1.0],
        [f64::INFINITY, 1.0],
    ] {
        if let Some(accepted) = usable_pixel_spacing(candidate) {
            assert!(
                ritk_spatial::Spacing::<2>::try_new(accepted).is_ok(),
                "admitted {accepted:?} that Spacing rejects"
            );
        }
    }
}
