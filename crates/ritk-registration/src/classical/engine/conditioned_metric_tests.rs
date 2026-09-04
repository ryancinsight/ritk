use std::num::NonZeroU8;

use super::*;

fn metric() -> MutualInformationMetric {
    MutualInformationMetric::with_ranges(
        2,
        IntensityRange::try_new(0.0, 1.0).expect("valid fixed range"),
        IntensityRange::try_new(0.0, 1.0).expect("valid moving range"),
        NmiNormalization::MeanEntropy,
        HistogramEstimator::Discrete,
    )
    .expect("valid metric")
}

#[test]
fn spatial_conditioning_resolves_a_global_histogram_ambiguity() {
    let fixed = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let aligned = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let mixed = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let regions = [0, 0, 0, 0, 1, 1, 1, 1];
    let base = metric();
    let global_aligned = base
        .compute_masked_samples(&fixed, &aligned, None)
        .expect("finite samples");
    let global_mixed = base
        .compute_masked_samples(&fixed, &mixed, None)
        .expect("finite samples");
    assert_eq!(global_aligned, global_mixed);

    let mut conditioned = SpatiallyConditionedMutualInformationMetric::try_new(
        base,
        NonZeroU8::new(2).expect("two is nonzero"),
    )
    .expect("valid conditioned metric");
    let conditioned_aligned = conditioned
        .compute_masked_samples(&fixed, &aligned, &regions, None)
        .expect("finite samples");
    let conditioned_mixed = conditioned
        .compute_masked_samples(&fixed, &mixed, &regions, None)
        .expect("finite samples");
    assert_eq!(conditioned_aligned, 1.0);
    assert_eq!(conditioned_mixed, 0.0);
}

#[test]
fn workspace_clears_histograms_between_evaluations() {
    let fixed = [0.0, 0.0, 1.0, 1.0];
    let aligned = [0.0, 0.0, 1.0, 1.0];
    let mixed = [0.0, 1.0, 0.0, 1.0];
    let regions = [0, 0, 0, 0];
    let mut conditioned = SpatiallyConditionedMutualInformationMetric::try_new(
        metric(),
        NonZeroU8::new(1).expect("one is nonzero"),
    )
    .expect("valid conditioned metric");
    let aligned_score = conditioned
        .compute_masked_samples(&fixed, &aligned, &regions, None)
        .expect("finite samples");
    let mixed_score = conditioned
        .compute_masked_samples(&fixed, &mixed, &regions, None)
        .expect("finite samples");
    assert_eq!(aligned_score, 1.0);
    assert_eq!(mixed_score, 0.0);
}

#[test]
fn invalid_region_inputs_are_typed_errors() {
    let mut conditioned = SpatiallyConditionedMutualInformationMetric::try_new(
        metric(),
        NonZeroU8::new(2).expect("two is nonzero"),
    )
    .expect("valid conditioned metric");
    assert!(conditioned
        .compute_masked_samples(&[0.0], &[0.0], &[], None)
        .is_err());
    assert!(conditioned
        .compute_masked_samples(&[0.0], &[0.0], &[2], None)
        .is_err());
    assert!(conditioned
        .compute_masked_samples(&[0.0], &[0.0], &[0], Some(&[]))
        .is_err());
}
