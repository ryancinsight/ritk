use anyhow::Result;
use proptest::prelude::*;

use super::synthetic_values;
use crate::metric::mind::descriptor::{
    descriptor_at, pack_distances, quantized_level, quantized_levels,
};
use crate::metric::mind::sampling::linear_index;
use crate::metric::mind::{MindSscConfig, MindSscSampling};

#[test]
fn quantization_rounds_half_ties_away_from_zero() {
    // All five scaled responses are exactly k + 0.5 in binary32. This pins
    // the quantizer independently of platform-specific exp precision.
    assert_eq!(
        [0.1_f32, 0.3, 0.5, 0.7, 0.9].map(|response| response * 5.0),
        [0.5, 1.5, 2.5, 3.5, 4.5]
    );
    assert_eq!(
        [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0].map(quantized_level),
        [0, 1, 2, 3, 4, 5, 5]
    );
    assert_eq!(
        [0.5_f32.next_down(), 0.5_f32.next_up()].map(quantized_level),
        [2, 3]
    );
}

pub(super) fn half_response_patch_values() -> Result<Vec<f32>> {
    let shape = [9; 3];
    let mut values = vec![0.0; shape.into_iter().product()];
    // Six radius-one patches with dilation three are disjoint and in bounds.
    // The first four are zero. In z/y/x traversal order, these dyadic samples
    // accumulate (via mul_add) squared sums LN_2 and 3-LN_2 exactly in binary32:
    // 11_629_080 / 2^24 and 38_702_568 / 2^24 respectively.
    let patches: [([usize; 3], &[u16]); 2] = [
        (
            [4, 4, 1],
            &[
                2048, 2048, 1024, 1024, 1024, 256, 128, 64, 64, 64, 16, 16, 4, 2, 2,
            ],
        ),
        (
            [4, 4, 7],
            &[
                4096, 4096, 2048, 512, 512, 512, 256, 256, 128, 128, 32, 32, 32, 16, 8, 8, 8, 4, 4,
                2, 2,
            ],
        ),
    ];
    for (center, samples) in patches {
        let mut samples = samples.iter().copied().chain(std::iter::repeat(0));
        let mut distance = 0.0_f32;
        for z in center[0] - 1..=center[0] + 1 {
            for y in center[1] - 1..=center[1] + 1 {
                for x in center[2] - 1..=center[2] + 1 {
                    let sample = f32::from(samples.next().expect("zero-padded samples")) / 4096.0;
                    values[linear_index([z, y, x], shape)?] = sample;
                    distance = sample.mul_add(sample, distance);
                }
            }
        }
        let expected = if center[2] == 1 {
            std::f32::consts::LN_2
        } else {
            3.0 - std::f32::consts::LN_2
        };
        assert_eq!(distance.to_bits(), expected.to_bits());
    }
    Ok(values)
}

#[test]
fn physical_patches_preserve_half_response_descriptor() -> Result<()> {
    let shape = [9; 3];
    let values = half_response_patch_values()?;
    let low = std::f32::consts::LN_2;
    let high = 3.0 - low;
    let distances = [
        0.0, 0.0, low, high, 0.0, 0.0, low, high, low, high, low, high,
    ];
    assert_eq!(distances.iter().sum::<f32>() / 12.0, 1.0);
    // Rust exp precision is platform-specific. For the scaled response near
    // 2.5, the half-away contract selects 2 below the tie and 3 at/above it.
    let level = if (-low).exp() * 5.0 < 2.5 { 2 } else { 3 };
    let expected = [5, 5, level, 0, 5, 5, level, 0, level, 0, level, 0];
    let config = MindSscConfig::try_new([1; 3], [3; 3], MindSscSampling::dense())?;
    let packed = descriptor_at([4; 3], config.geometry(), |index| {
        Ok(values[linear_index(index, shape)?])
    })?;
    assert_eq!(quantized_levels(packed), expected);
    assert_eq!(packed, pack_distances(distances)?);
    let constant = pack_distances([0.0; 12])?;
    assert_eq!((packed ^ constant).count_ones(), 40 - 4 * level);
    Ok(())
}

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
