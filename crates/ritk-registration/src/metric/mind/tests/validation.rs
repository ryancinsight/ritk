use anyhow::Result;

use super::{identity_image, synthetic_values};
use crate::metric::mind::sampling::linear_index;
use crate::metric::mind::{MindSscConfig, MindSscError, MindSscFixedPrep, MindSscSampling};

#[test]
fn malformed_configuration_and_domains_return_typed_errors() -> Result<()> {
    assert!(matches!(
        MindSscConfig::try_new([0, 1, 1], [2; 3], MindSscSampling::dense()),
        Err(MindSscError::InvalidPatchRadius { axis: 0, .. })
    ));
    assert!(matches!(
        MindSscSampling::try_stratified(0),
        Err(MindSscError::EmptySampleBudget)
    ));
    assert!(matches!(
        MindSscSampling::try_indices([1, 1]),
        Err(MindSscError::DuplicateSampleIndex { index: 1 })
    ));
    let small = identity_image(vec![0.0; 6 * 7 * 7], [6, 7, 7])?;
    assert!(matches!(
        MindSscFixedPrep::try_new(&small, MindSscConfig::default(), None, None),
        Err(MindSscError::ImageTooSmall { .. })
    ));
    let valid = identity_image(synthetic_values([9; 3]), [9; 3])?;
    assert!(matches!(
        MindSscFixedPrep::try_new(&valid, MindSscConfig::default(), Some(&[true]), None),
        Err(MindSscError::MaskLength { .. })
    ));
    assert!(matches!(
        MindSscFixedPrep::try_new(&valid, MindSscConfig::default(), None, Some(&[1.0])),
        Err(MindSscError::WeightLength { .. })
    ));
    let center = linear_index([4; 3], [9; 3])?;
    let config = MindSscConfig::try_new([1; 3], [2; 3], MindSscSampling::try_indices([center])?)?;
    let mut weights = vec![0.0; 9 * 9 * 9];
    weights[center] = -1.0;
    assert!(matches!(
        MindSscFixedPrep::try_new(&valid, config, None, Some(&weights)),
        Err(MindSscError::InvalidWeight { index, .. }) if index == center
    ));
    Ok(())
}

#[test]
fn selected_weight_arithmetic_has_distinct_typed_failures() -> Result<()> {
    let shape = [9; 3];
    let valid = identity_image(synthetic_values(shape), shape)?;
    let first = linear_index([4, 4, 4], shape)?;
    let second = linear_index([4, 4, 5], shape)?;

    let one_config =
        MindSscConfig::try_new([1; 3], [2; 3], MindSscSampling::try_indices([first])?)?;
    let mut denominator_weights = vec![0.0; shape.into_iter().product()];
    denominator_weights[first] = f32::MAX;
    assert!(matches!(
        MindSscFixedPrep::try_new(&valid, one_config, None, Some(&denominator_weights)),
        Err(MindSscError::NonFiniteDenominator { weight_sum }) if weight_sum == f32::MAX
    ));

    let two_config = MindSscConfig::try_new(
        [1; 3],
        [2; 3],
        MindSscSampling::try_indices([first, second])?,
    )?;
    let mut sum_weights = vec![0.0; shape.into_iter().product()];
    sum_weights[first] = f32::MAX;
    sum_weights[second] = f32::MAX;
    assert!(matches!(
        MindSscFixedPrep::try_new(&valid, two_config, None, Some(&sum_weights)),
        Err(MindSscError::NonFiniteSelectedWeightSum { value }) if value.is_infinite()
    ));
    Ok(())
}
