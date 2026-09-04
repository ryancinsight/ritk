use anyhow::Result;
use proptest::prelude::*;

use super::synthetic_values;
use crate::metric::mind::descriptor::{descriptor_at, pack_distances, quantized_levels};
use crate::metric::mind::sampling::linear_index;
use crate::metric::mind::{MindSscConfig, MindSscSampling};

pub(super) fn linear_field_values(shape: [usize; 3]) -> Vec<f32> {
    let mut values = Vec::with_capacity(shape.into_iter().product());
    for z in 0..shape[0] {
        for y in 0..shape[1] {
            for x in 0..shape[2] {
                let value = u16::try_from(z + 2 * y + 4 * x).expect("test linear field fits u16");
                values.push(f32::from(value));
            }
        }
    }
    values
}

#[test]
fn linear_field_matches_hand_derived_ssc_word() -> Result<()> {
    let shape = [9; 3];
    let values = linear_field_values(shape);
    let config = MindSscConfig::try_new([1; 3], [1; 3], MindSscSampling::dense())?;
    let actual = descriptor_at([4; 3], config.geometry(), |index| {
        Ok(values[linear_index(index, shape)?])
    })?;
    // For I=z+2y+4x, the 3^3 patch distances are
    // [27,243,243,675,243,27,675,243,108,972,972,108]. Their mean is 378,
    // giving unary levels [5,3,3,1,3,5,1,3,4,0,0,4].
    assert_eq!(actual, 0x0780_0f38_7e70_9cff);
    assert_eq!(
        quantized_levels(actual),
        [5, 3, 3, 1, 3, 5, 1, 3, 4, 0, 0, 4]
    );
    Ok(())
}

#[test]
fn normalization_and_quantization_match_hand_derived_word() -> Result<()> {
    let packed = pack_distances([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])?;
    // Mean distance is 11/12. The minimum response maps to level 5; every
    // other response is exp(-12/11), which rounds to level 2.
    assert_eq!(packed, 0x018c_6318_c631_8c7f);
    assert_eq!(
        quantized_levels(packed),
        [5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    );
    Ok(())
}

#[test]
fn unary_packing_popcount_equals_quantized_l1() -> Result<()> {
    let left = pack_distances([0.0, 1.0, 3.0, 7.0, 2.0, 9.0, 4.0, 6.0, 5.0, 8.0, 12.0, 10.0])?;
    let right = pack_distances([9.0, 2.0, 0.0, 5.0, 1.0, 11.0, 7.0, 3.0, 8.0, 4.0, 6.0, 10.0])?;
    let direct = quantized_levels(left)
        .into_iter()
        .zip(quantized_levels(right))
        .map(|(a, b)| a.abs_diff(b))
        .sum::<u32>();
    assert_eq!((left ^ right).count_ones(), direct);
    assert_eq!(left >> 60, 0);
    assert_eq!(right >> 60, 0);
    Ok(())
}

#[test]
fn descriptor_is_affine_intensity_invariant() -> Result<()> {
    let shape = [11; 3];
    let values = synthetic_values(shape);
    let transformed = values
        .iter()
        .map(|value| value.mul_add(4.0, 32.0))
        .collect::<Vec<_>>();
    let config = MindSscConfig::default();
    let center = [5; 3];
    let original = descriptor_at(center, config.geometry(), |index| {
        Ok(values[linear_index(index, shape)?])
    })?;
    let affine = descriptor_at(center, config.geometry(), |index| {
        Ok(transformed[linear_index(index, shape)?])
    })?;
    assert_eq!(original, affine);
    Ok(())
}

proptest! {
    #[test]
    fn positive_power_of_two_scale_and_offset_preserve_descriptor(
        scale_power in 0_u32..5,
        offset in -32_i16..32,
    ) {
        let shape = [9; 3];
        let values = synthetic_values(shape);
        let scale = f32::from(1_u16 << scale_power);
        let offset = f32::from(offset);
        let transformed = values.iter().map(|value| value.mul_add(scale, offset)).collect::<Vec<_>>();
        let config = MindSscConfig::default();
        let center = [4; 3];
        let original = descriptor_at(center, config.geometry(), |index| {
            Ok(values[linear_index(index, shape)?])
        })?;
        let affine = descriptor_at(center, config.geometry(), |index| {
            Ok(transformed[linear_index(index, shape)?])
        })?;
        prop_assert_eq!(original, affine);
    }
}
