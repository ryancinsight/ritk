use super::descriptor::linear_field_values;
use super::{identity_image, synthetic_values, translation};
use crate::metric::mind::sampling::linear_index;
use crate::metric::mind::{mind_ssc_value, MindSscConfig, MindSscFixedPrep, MindSscSampling};
use crate::types::AffineTransform;
use anyhow::Result;

#[test]
fn constant_image_has_maximum_self_similarity() -> Result<()> {
    let image = identity_image(vec![4.0; 9 * 9 * 9], [9; 3])?;
    let prepared = MindSscFixedPrep::try_new(&image, MindSscConfig::default(), None, None)?;
    assert_eq!(prepared.eval(&image, &AffineTransform::IDENTITY)?, 1.0);
    Ok(())
}

#[test]
fn one_shot_and_prepared_paths_are_identical() -> Result<()> {
    let shape = [10; 3];
    let image = identity_image(synthetic_values(shape), shape)?;
    let config = MindSscConfig::try_new([1; 3], [2; 3], MindSscSampling::dense())?;
    let prepared = MindSscFixedPrep::try_new(&image, config.clone(), None, None)?;
    let transform = translation(0.25, -0.5, 0.75);
    assert_eq!(
        prepared.eval(&image, &transform)?,
        mind_ssc_value(&image, &image, &transform, config, None, None)?
    );
    Ok(())
}

#[test]
fn out_of_field_support_uses_background_and_fixed_denominator() -> Result<()> {
    let shape = [11; 3];
    let image = identity_image(linear_field_values(shape), shape)?;
    let center = [5; 3];
    let center_index = linear_index(center, shape)?;
    let config = MindSscConfig::try_new(
        [1; 3],
        [1; 3],
        MindSscSampling::try_indices([center_index])?,
    )?;
    // The fixed linear-field descriptor is the hand-derived word asserted in
    // descriptor::linear_field_matches_hand_derived_ssc_word. Zero background
    // produces all-one unary components; their Hamming distance is 28.
    let expected = 1.0 - 28.0_f32 / 60.0;
    let prepared = MindSscFixedPrep::try_new(&image, config, None, None)?;
    let actual = prepared.eval(&image, &translation(1000.0, 0.0, 0.0))?;
    assert_eq!(actual, expected);
    assert_eq!(prepared.selected_indices(), &[center_index]);
    Ok(())
}

#[test]
fn manufactured_translation_is_recovered() -> Result<()> {
    let shape = [15; 3];
    let fixed_values = synthetic_values(shape);
    let mut moving_values = vec![0.0; fixed_values.len()];
    for z in 0..shape[0] - 1 {
        for y in 0..shape[1] {
            for x in 0..shape[2] {
                moving_values[linear_index([z + 1, y, x], shape)?] =
                    fixed_values[linear_index([z, y, x], shape)?];
            }
        }
    }
    let fixed = identity_image(fixed_values, shape)?;
    let moving = identity_image(moving_values, shape)?;
    let centers = (4..=10)
        .flat_map(|z| (4..=10).flat_map(move |y| (4..=10).map(move |x| [z, y, x])))
        .map(|index| linear_index(index, shape))
        .collect::<Result<Vec<_>, _>>()?;
    let config = MindSscConfig::try_new([1; 3], [2; 3], MindSscSampling::try_indices(centers)?)?;
    let prepared = MindSscFixedPrep::try_new(&fixed, config, None, None)?;
    let candidates = [-1.0, 0.0, 1.0, 2.0]
        .map(|offset| prepared.eval(&moving, &translation(offset, 0.0, 0.0)))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let recovered = candidates
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| [-1.0, 0.0, 1.0, 2.0][index])
        .expect("four candidates");
    assert_eq!(recovered, 1.0);
    assert_eq!(candidates[2], 1.0);
    Ok(())
}

#[test]
fn weighted_scores_respect_the_similarity_range() -> Result<()> {
    let shape = [11; 3];
    let image = identity_image(synthetic_values(shape), shape)?;
    let centers = [[4, 4, 4], [5, 5, 5], [6, 6, 6]]
        .map(|index| linear_index(index, shape))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let config = MindSscConfig::try_new(
        [1; 3],
        [2; 3],
        MindSscSampling::try_indices(centers.clone())?,
    )?;
    let mut weights = vec![0.0; shape.into_iter().product()];
    for (index, weight) in centers.into_iter().zip([0.25, 1.0, 32.0]) {
        weights[index] = weight;
    }
    let prepared = MindSscFixedPrep::try_new(&image, config, None, Some(&weights))?;
    for transform in [
        AffineTransform::IDENTITY,
        translation(0.25, -0.5, 0.75),
        translation(1000.0, 0.0, 0.0),
    ] {
        let score = prepared.eval(&image, &transform)?;
        assert!((0.0..=1.0).contains(&score), "score={score}");
    }
    Ok(())
}
