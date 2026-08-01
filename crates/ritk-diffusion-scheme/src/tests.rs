use ritk_spatial::Vector;

use crate::{
    DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme, GradientSchemeError,
    parse_fsl_bval, read_fsl_scheme,
};

fn weighting(value: f64) -> DiffusionWeighting {
    DiffusionWeighting::from_seconds_per_square_millimeter(value)
        .expect("finite nonnegative weighting")
}

#[test]
fn weighting_converts_to_canonical_si() {
    let b = weighting(1_000.0);
    assert_eq!(b.seconds_per_square_millimeter(), 1_000.0);
    assert_eq!(b.seconds_per_square_meter(), 1_000_000_000.0);
}

#[test]
fn weighting_rejects_negative_and_non_finite_values() {
    for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            DiffusionWeighting::from_seconds_per_square_millimeter(value),
            Err(GradientSchemeError::InvalidWeighting { value: actual, .. })
                if actual.to_bits() == value.to_bits()
        ));
    }
}

#[test]
fn gradient_contract_distinguishes_b0_and_weighted_entries() {
    assert!(GradientDirection::new(weighting(0.0), Vector::new([0.0, 0.0, 0.0])).is_ok());
    assert!(GradientDirection::new(weighting(1_000.0), Vector::new([1.0, 0.0, 0.0])).is_ok());
    assert!(GradientDirection::new(weighting(0.0), Vector::new([1.0, 0.0, 0.0])).is_err());
    assert!(GradientDirection::new(weighting(1_000.0), Vector::new([2.0, 0.0, 0.0])).is_err());
    assert!(GradientDirection::new(weighting(1_000.0), Vector::new([f64::NAN, 0.0, 0.0])).is_err());
}

#[test]
fn fsl_round_trip_preserves_order_shells_and_frame() {
    let scheme = read_fsl_scheme("0 1000 2000", "0 1 0\n0 0 1\n0 0 0")
        .expect("valid multi-shell FSL scheme");
    assert_eq!(scheme.frame(), GradientFrame::ImageAxis);
    assert_eq!(scheme.b0_indices(weighting(50.0)), vec![0]);
    assert_eq!(scheme.dwi_indices(weighting(50.0)), vec![1, 2]);
    assert_eq!(
        scheme
            .shells()
            .into_iter()
            .map(DiffusionWeighting::seconds_per_square_millimeter)
            .collect::<Vec<_>>(),
        vec![1_000.0, 2_000.0]
    );
    assert_eq!(
        scheme.directions()[1].direction(),
        Vector::new([1.0, 0.0, 0.0])
    );
}

#[test]
fn fsl_parser_rejects_non_finite_weighting() {
    let error = parse_fsl_bval("0 NaN").expect_err("NaN weighting must fail");
    assert!(matches!(
        error,
        GradientSchemeError::InvalidWeighting { index: 1, .. }
    ));
}

#[test]
fn proper_rotation_preserves_weightings_and_rotates_direction() {
    let scheme = GradientScheme::new(
        vec![
            GradientDirection::new(weighting(0.0), Vector::new([0.0, 0.0, 0.0])).expect("valid b0"),
            GradientDirection::new(weighting(1_000.0), Vector::new([1.0, 0.0, 0.0]))
                .expect("valid DWI"),
        ],
        GradientFrame::Lps,
    )
    .expect("valid scheme");
    let rotated = scheme
        .reorient([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        .expect("proper rotation");
    assert_eq!(
        rotated.directions()[0].direction(),
        Vector::new([0.0, 0.0, 0.0])
    );
    assert_eq!(
        rotated.directions()[1].direction(),
        Vector::new([0.0, 1.0, 0.0])
    );
    assert_eq!(rotated.directions()[1].weighting(), weighting(1_000.0));
}

#[test]
fn reflection_and_non_orthonormal_rotation_are_rejected() {
    let scheme = read_fsl_scheme("1000", "1\n0\n0").expect("valid scheme");
    let reflection = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert!(matches!(
        scheme.reorient(reflection),
        Err(GradientSchemeError::InvalidRotation(_))
    ));
    let scaled = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert!(matches!(
        scheme.reorient(scaled),
        Err(GradientSchemeError::InvalidRotation(_))
    ));
}
