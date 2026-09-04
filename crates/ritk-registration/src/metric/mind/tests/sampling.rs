use std::collections::BTreeSet;

use anyhow::Result;
use ritk_spatial::{Direction, Point, Spacing};

use super::{identity_image, image, synthetic_values, translation};
use crate::metric::mind::sampling::{select_indices, stratum_for_test};
use crate::metric::mind::{MindSscConfig, MindSscFixedPrep, MindSscSampling};

#[test]
fn default_sampling_is_repeatable_and_covers_each_stratum() -> Result<()> {
    let shape = [35; 3];
    let halo = [3; 3];
    let spacing = [1.0, 1.5, 2.0];
    let policy = MindSscSampling::try_stratified(64)?;
    let first = select_indices(&policy, shape, halo, spacing, None)?;
    let second = select_indices(&policy, shape, halo, spacing, None)?;
    assert_eq!(first, second);
    assert_eq!(first.len(), 64);
    let strata = first
        .iter()
        .map(|index| stratum_for_test(*index, shape, halo, spacing, 64))
        .collect::<Result<BTreeSet<_>, _>>()?;
    assert_eq!(strata.len(), 64);
    Ok(())
}

#[test]
fn default_uses_dense_domain_when_all_centers_fit() -> Result<()> {
    let shape = [10; 3];
    let values = synthetic_values(shape);
    let image = identity_image(values, shape)?;
    let default = MindSscFixedPrep::try_new(&image, MindSscConfig::default(), None, None)?;
    let dense_config = MindSscConfig::try_new([1; 3], [2; 3], MindSscSampling::dense())?;
    let dense = MindSscFixedPrep::try_new(&image, dense_config, None, None)?;
    assert_eq!(default.selected_indices(), dense.selected_indices());
    let transform = translation(0.5, 0.25, -0.25);
    assert_eq!(
        default.eval(&image, &transform)?,
        dense.eval(&image, &transform)?
    );
    Ok(())
}

#[test]
fn preparation_maps_metadata_spacing_to_c_order_strata() -> Result<()> {
    let shape = [35; 3];
    let metadata_spacing = [9.0, 2.0, 1.0];
    let c_order_spacing = [1.0, 2.0, 9.0];
    let policy = MindSscSampling::try_stratified(64)?;
    let config = MindSscConfig::try_new([1; 3], [2; 3], policy.clone())?;
    let fixed = image(
        synthetic_values(shape),
        shape,
        Point::origin(),
        Spacing::new(metadata_spacing),
        Direction::identity(),
    )?;
    let prepared = MindSscFixedPrep::try_new(&fixed, config, None, None)?;
    let expected = select_indices(&policy, shape, [3; 3], c_order_spacing, None)?;
    let unbridged = select_indices(&policy, shape, [3; 3], metadata_spacing, None)?;
    assert_eq!(prepared.selected_indices(), expected);
    assert_ne!(prepared.selected_indices(), unbridged);
    Ok(())
}
