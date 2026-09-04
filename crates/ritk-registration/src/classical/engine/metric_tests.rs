use super::*;

fn metric(
    normalization: NmiNormalization,
    estimator: HistogramEstimator,
) -> MutualInformationMetric {
    MutualInformationMetric::with_ranges(
        8,
        IntensityRange::try_new(0.0, 1.0).expect("valid fixed range"),
        IntensityRange::try_new(-1.0, 1.0).expect("valid moving range"),
        normalization,
        estimator,
    )
    .expect("valid metric")
}

#[test]
fn partial_volume_weights_conserve_sample_mass() {
    for coordinate in [0.0, 0.125, 2.5, 6.999, 7.0] {
        let weights = linear_bin_weights(coordinate, 8);
        let mass: f64 = weights.into_iter().map(|(_, weight)| weight).sum();
        assert_eq!(mass, 1.0);
    }
}

#[test]
fn partial_volume_weights_are_continuous_inside_a_bin_interval() {
    let delta = 1.0e-6;
    let left = linear_bin_weights(2.5 - delta, 8);
    let right = linear_bin_weights(2.5 + delta, 8);
    assert_eq!(left[0].0, right[0].0);
    assert_eq!(left[1].0, right[1].0);
    // Two subtractions form the observed weight difference; eight ULPs
    // bound their rounding plus the literals' representation error.
    let rounding_bound = 8.0 * f64::EPSILON;
    for ((_, left_weight), (_, right_weight)) in left.into_iter().zip(right) {
        assert!(
            (left_weight - right_weight).abs() <= 2.0 * delta + rounding_bound,
            "linear kernel changed by more than its unit slope permits"
        );
    }
}

#[test]
fn joint_entropy_nmi_is_symmetric_with_ranges_exchanged() {
    let fixed = [0.0, 0.15, 0.4, 0.7, 1.0];
    let moving = [-1.0, -0.4, 0.2, 0.6, 1.0];
    let forward = metric(NmiNormalization::JointEntropy, HistogramEstimator::Discrete)
        .compute_masked_samples(&fixed, &moving, None)
        .expect("finite samples");
    let reverse = MutualInformationMetric::with_ranges(
        8,
        IntensityRange::try_new(-1.0, 1.0).expect("valid moving range"),
        IntensityRange::try_new(0.0, 1.0).expect("valid fixed range"),
        NmiNormalization::JointEntropy,
        HistogramEstimator::Discrete,
    )
    .expect("valid metric")
    .compute_masked_samples(&moving, &fixed, None)
    .expect("finite samples");
    assert_eq!(forward, reverse);
}

#[test]
fn mask_changes_the_selected_value_semantics() {
    let fixed = [0.0, 0.25, 0.5, 0.75, 1.0];
    let moving = [-1.0, -0.5, 0.0, 0.5, -1.0];
    let metric = metric(
        NmiNormalization::MeanEntropy,
        HistogramEstimator::MovingLinearPartialVolume,
    );
    let all = metric
        .compute_masked_samples(&fixed, &moving, None)
        .expect("finite samples");
    let selected = metric
        .compute_masked_samples(&fixed, &moving, Some(&[true, true, true, true, false]))
        .expect("non-empty mask");
    assert!(
        selected > all,
        "excluding the deliberately conflicting final pair must increase NMI: {all} -> {selected}"
    );
}

#[test]
fn invalid_inputs_are_typed_errors() {
    let error =
        MutualInformationMetric::new(1, 0.0, 1.0).expect_err("one histogram bin must be rejected");
    assert!(matches!(
        error,
        RegistrationError::InvalidInput(message)
            if message == "mutual-information histogram requires at least two bins, got 1"
    ));
    let error = IntensityRange::try_new(1.0, 1.0)
        .expect_err("a degenerate intensity range must be rejected");
    assert!(matches!(
        error,
        RegistrationError::InvalidInput(message)
            if message == "intensity range must be finite and increasing, got [1, 1]"
    ));
    let metric = metric(
        NmiNormalization::MeanEntropy,
        HistogramEstimator::MovingLinearPartialVolume,
    );
    let error = metric
        .compute_masked_samples(&[0.0], &[], None)
        .expect_err("mismatched sample lengths must be rejected");
    assert!(matches!(
        error,
        RegistrationError::InvalidInput(message)
            if message == "mutual-information sample lengths differ: 1 versus 0"
    ));
    let error = metric
        .compute_masked_samples(&[0.0], &[-1.0], Some(&[]))
        .expect_err("mismatched mask length must be rejected");
    assert!(matches!(
        error,
        RegistrationError::InvalidInput(message)
            if message == "mutual-information mask length 0 differs from sample length 1"
    ));
    let error = metric
        .compute_masked_samples(&[f64::NAN], &[-1.0], None)
        .expect_err("non-finite samples must be rejected");
    assert!(matches!(
        error,
        RegistrationError::InvalidInput(message)
            if message == "mutual-information sample 0 is not finite: fixed=NaN, moving=-1"
    ));
}
