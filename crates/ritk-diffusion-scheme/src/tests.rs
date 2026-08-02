mod reorient_per_volume;

use ritk_spatial::Vector;

use crate::{
    DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme, GradientSchemeError,
    parse_fsl_bval, read_fsl_scheme, read_mrtrix_scheme, write_fsl_scheme, write_mrtrix_scheme,
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

// ── ADR 0036 verification condition 8: FSL round-trip ───────────────────

#[test]
fn fsl_write_read_round_trip_recovers_identical_scheme() {
    let scheme = GradientScheme::new(
        vec![
            GradientDirection::new(weighting(0.0), Vector::new([0.0, 0.0, 0.0])).unwrap(),
            GradientDirection::new(weighting(500.0), Vector::new([0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0])).unwrap(),
            GradientDirection::new(weighting(1_000.0), Vector::new([0.0, 1.0, 0.0])).unwrap(),
            GradientDirection::new(weighting(2_000.0), Vector::new([0.0, 0.0, 1.0])).unwrap(),
        ],
        GradientFrame::ImageAxis,
    )
    .expect("valid multi-shell scheme");

    let (bval, bvec) = write_fsl_scheme(&scheme);
    let recovered = read_fsl_scheme(&bval, &bvec).expect("round-trip parse");

    assert_eq!(recovered.frame(), scheme.frame());
    assert_eq!(recovered.len(), scheme.len());
    for (original, recovered) in scheme.directions().iter().zip(recovered.directions().iter()) {
        assert_eq!(
            original.weighting(),
            recovered.weighting(),
            "weightings differ"
        );
        assert_eq!(
            original.direction(),
            recovered.direction(),
            "directions differ"
        );
    }
}

// ── ADR 0036 verification condition 8: MRtrix round-trip ────────────────

#[test]
fn mrtrix_write_read_round_trip_recovers_identical_scheme() {
    let scheme = GradientScheme::new(
        vec![
            GradientDirection::new(weighting(0.0), Vector::new([0.0, 0.0, 0.0])).unwrap(),
            GradientDirection::new(weighting(500.0), Vector::new([0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0])).unwrap(),
            GradientDirection::new(weighting(1_000.0), Vector::new([0.0, 1.0, 0.0])).unwrap(),
            GradientDirection::new(weighting(2_000.0), Vector::new([0.0, 0.0, 1.0])).unwrap(),
        ],
        GradientFrame::ImageAxis,
    )
    .expect("valid multi-shell scheme");

    let header = write_mrtrix_scheme(&scheme);
    let recovered = read_mrtrix_scheme(&header).expect("MRtrix round-trip parse");

    assert_eq!(recovered.frame(), GradientFrame::ImageAxis);
    assert_eq!(recovered.len(), scheme.len());
    for (original, recovered) in scheme.directions().iter().zip(recovered.directions().iter()) {
        assert_eq!(original.weighting(), recovered.weighting());
        assert_eq!(original.direction(), recovered.direction());
    }
}

#[test]
fn mrtrix_parser_rejects_malformed_headers() {
    // Missing DW_scheme key.
    assert!(read_mrtrix_scheme("NDim: 3\nEND\n").is_err());

    // Wrong column count.
    assert!(read_mrtrix_scheme("DW_scheme: 2,3\n0,0,0,0\n1,0,0,1000\n").is_err());

    // Dimension mismatch.
    assert!(read_mrtrix_scheme("DW_scheme: 4,4\n0,0,0,0\n1,0,0,1000\n0,1,0,1000\n").is_err());

    // No data rows.
    assert!(read_mrtrix_scheme("DW_scheme: 0,4\n").is_err());
}
