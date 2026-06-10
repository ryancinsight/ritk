use super::*;
use burn_ndarray::NdArray;

type B = NdArray<f32>;

#[test]
fn test_mutual_information_consolidation() {
    let max_intensity = 255.0;
    let num_bins = 50;

    // Mattes specific configuration check
    let m_mattes =
        MutualInformation::<B>::new_mattes(num_bins, 0.0, max_intensity, &Default::default());
    let expected_sigma = max_intensity / 50.0;

    assert_eq!(MutualInformationVariant::Mattes, m_mattes.variant);
    assert!((m_mattes.histogram_calculator.parzen_sigma - expected_sigma).abs() < 1e-6);

    // Check correct name routing
    assert_eq!(
        <MutualInformation<B> as Metric<B, 3>>::name(&m_mattes),
        "Mattes Mutual Information"
    );

    let m_std = MutualInformation::<B>::standard_default(&Default::default());
    assert_eq!(
        <MutualInformation<B> as Metric<B, 3>>::name(&m_std),
        "Mutual Information"
    );

    let m_nmi = MutualInformation::<B>::normalized_default(&Default::default());
    assert_eq!(
        <MutualInformation<B> as Metric<B, 3>>::name(&m_nmi),
        "Normalized Mutual Information"
    );
}

#[test]
fn test_sampling_clamp() {
    let m = MutualInformation::<B>::standard_default(&Default::default()).with_sampling(1.5);
    assert_eq!(m.sampling_percentage, None); // Clamps to 1.0, which disables stochastic branch

    let m = MutualInformation::<B>::standard_default(&Default::default()).with_sampling(-0.5);
    assert_eq!(m.sampling_percentage, Some(1e-4)); // Clamps to 1e-4
}
