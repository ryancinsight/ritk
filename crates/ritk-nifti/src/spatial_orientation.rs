//! Exact orientation tests for NIfTI spatial transforms.

use std::cmp::Ordering;

use anyhow::{bail, Result};

const FRACTION_BITS: u32 = 52;
const EXPONENT_BIAS: i32 = 1023;
const SUBNORMAL_EXPONENT: i32 = -1074;
const ACCUMULATOR_LIMBS: usize = 100;

#[derive(Clone, Copy)]
struct BinaryTerm {
    magnitude: [u64; 3],
    exponent: i32,
    negative: bool,
}

/// Returns the exact sign of the spatial 3×3 determinant.
///
/// Each finite binary64 coefficient is an integer significand times a power
/// of two. The six determinant products are accumulated as fixed-width
/// integers at a shared exponent, so classification stays exact even when the
/// determinant magnitude falls outside binary64's representable range. The
/// 100 limbs cover the full binary64 triple-product exponent span (-3222 to
/// 2913), 159 significand bits, and the carry from summing six terms.
pub(crate) fn sform_orientation(rows: [[f64; 4]; 3]) -> Result<Ordering> {
    for (row_index, row) in rows.iter().enumerate() {
        for (column_index, value) in row.iter().enumerate() {
            if !value.is_finite() {
                bail!("NIfTI sform entry [{row_index},{column_index}] must be finite");
            }
        }
    }

    let products = [
        ([(0, 0), (1, 1), (2, 2)], false),
        ([(0, 1), (1, 2), (2, 0)], false),
        ([(0, 2), (1, 0), (2, 1)], false),
        ([(0, 2), (1, 1), (2, 0)], true),
        ([(0, 1), (1, 0), (2, 2)], true),
        ([(0, 0), (1, 2), (2, 1)], true),
    ];
    let products = determinant_products(rows, &products)?;
    let mut minimum_exponent = None;
    for product in products.iter().flatten() {
        minimum_exponent = Some(minimum_exponent.map_or(product.exponent, |current: i32| {
            current.min(product.exponent)
        }));
    }
    let Some(minimum_exponent) = minimum_exponent else {
        return Ok(Ordering::Equal);
    };

    let mut positive = [0_u64; ACCUMULATOR_LIMBS];
    let mut negative = [0_u64; ACCUMULATOR_LIMBS];
    for product in products.iter().flatten() {
        let shift = usize::try_from(product.exponent - minimum_exponent)
            .map_err(|_| anyhow::anyhow!("NIfTI determinant exponent span is invalid"))?;
        let accumulator = if product.negative {
            &mut negative
        } else {
            &mut positive
        };
        add_shifted(accumulator, product.magnitude, shift)?;
    }

    Ok(compare_unsigned(&positive, &negative))
}

fn determinant_products(
    rows: [[f64; 4]; 3],
    products: &[([(usize, usize); 3], bool); 6],
) -> Result<[Option<BinaryTerm>; 6]> {
    let mut result = [None; 6];
    for (index, (indices, subtract)) in products.iter().enumerate() {
        let [(row_a, column_a), (row_b, column_b), (row_c, column_c)] = *indices;
        let first = get_coefficient(rows, row_a, column_a)?;
        let second = get_coefficient(rows, row_b, column_b)?;
        let third = get_coefficient(rows, row_c, column_c)?;
        let term = triple_product([first, second, third], *subtract)?;
        let slot = result
            .get_mut(index)
            .ok_or_else(|| anyhow::anyhow!("NIfTI determinant term index is invalid"))?;
        *slot = term;
    }
    Ok(result)
}

fn get_coefficient(rows: [[f64; 4]; 3], row: usize, column: usize) -> Result<f64> {
    rows.get(row)
        .and_then(|values| values.get(column))
        .copied()
        .ok_or_else(|| anyhow::anyhow!("NIfTI sform coefficient index is invalid"))
}

fn triple_product(values: [f64; 3], subtract: bool) -> Result<Option<BinaryTerm>> {
    let [value_a, value_b, value_c] = values;
    let first = decompose(value_a)?;
    let second = decompose(value_b)?;
    let third = decompose(value_c)?;
    if first.significand == 0 || second.significand == 0 || third.significand == 0 {
        return Ok(None);
    }

    let exponent = first
        .exponent
        .checked_add(second.exponent)
        .and_then(|value| value.checked_add(third.exponent))
        .ok_or_else(|| anyhow::anyhow!("NIfTI determinant exponent overflowed"))?;
    Ok(Some(BinaryTerm {
        magnitude: multiply_significands(first.significand, second.significand, third.significand)?,
        exponent,
        negative: first.negative ^ second.negative ^ third.negative ^ subtract,
    }))
}

struct Decomposed {
    significand: u64,
    exponent: i32,
    negative: bool,
}

fn decompose(value: f64) -> Result<Decomposed> {
    let bits = value.to_bits();
    let fraction_mask = (1_u64 << FRACTION_BITS) - 1;
    let fraction = bits & fraction_mask;
    let encoded_exponent = (bits >> FRACTION_BITS) & 0x7ff;
    let negative = bits >> 63 != 0;
    if encoded_exponent == 0x7ff {
        bail!("NIfTI sform coefficient must be finite");
    }
    if encoded_exponent == 0 {
        return Ok(Decomposed {
            significand: fraction,
            exponent: SUBNORMAL_EXPONENT,
            negative,
        });
    }

    let encoded_exponent = i32::try_from(encoded_exponent)
        .map_err(|_| anyhow::anyhow!("NIfTI binary64 exponent is out of range"))?;
    Ok(Decomposed {
        significand: fraction | (1_u64 << FRACTION_BITS),
        exponent: encoded_exponent
            - EXPONENT_BIAS
            - i32::try_from(FRACTION_BITS)
                .map_err(|_| anyhow::anyhow!("NIfTI binary64 fraction width is out of range"))?,
        negative,
    })
}

fn multiply_significands(first: u64, second: u64, third: u64) -> Result<[u64; 3]> {
    let first_product = u128::from(first) * u128::from(second);
    let low_mask = u128::from(u64::MAX);
    let low = u64::try_from(first_product & low_mask)
        .map_err(|_| anyhow::anyhow!("NIfTI determinant low product limb exceeds u64"))?;
    let high = u64::try_from(first_product >> u64::BITS)
        .map_err(|_| anyhow::anyhow!("NIfTI determinant high product limb exceeds u64"))?;
    let low_times_third = u128::from(low) * u128::from(third);
    let high_times_third = u128::from(high) * u128::from(third) + (low_times_third >> u64::BITS);
    Ok([
        u64::try_from(low_times_third & low_mask)
            .map_err(|_| anyhow::anyhow!("NIfTI determinant low product limb exceeds u64"))?,
        u64::try_from(high_times_third & low_mask)
            .map_err(|_| anyhow::anyhow!("NIfTI determinant middle product limb exceeds u64"))?,
        u64::try_from(high_times_third >> u64::BITS)
            .map_err(|_| anyhow::anyhow!("NIfTI determinant high product limb exceeds u64"))?,
    ])
}

fn add_shifted(
    accumulator: &mut [u64; ACCUMULATOR_LIMBS],
    value: [u64; 3],
    shift: usize,
) -> Result<()> {
    let limb_shift = shift
        / usize::try_from(u64::BITS)
            .map_err(|_| anyhow::anyhow!("NIfTI determinant limb width is out of range"))?;
    let bit_shift = u32::try_from(
        shift
            % usize::try_from(u64::BITS)
                .map_err(|_| anyhow::anyhow!("NIfTI determinant limb width is out of range"))?,
    )
    .map_err(|_| anyhow::anyhow!("NIfTI determinant bit shift is out of range"))?;

    for (source_index, word) in value.into_iter().enumerate() {
        if word == 0 {
            continue;
        }
        let target_index = limb_shift
            .checked_add(source_index)
            .ok_or_else(|| anyhow::anyhow!("NIfTI determinant limb index overflowed"))?;
        if bit_shift == 0 {
            add_word(accumulator, target_index, word)?;
        } else {
            add_word(accumulator, target_index, word << bit_shift)?;
            let carry_index = target_index
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("NIfTI determinant limb index overflowed"))?;
            add_word(accumulator, carry_index, word >> (u64::BITS - bit_shift))?;
        }
    }
    Ok(())
}

fn add_word(
    accumulator: &mut [u64; ACCUMULATOR_LIMBS],
    mut index: usize,
    mut word: u64,
) -> Result<()> {
    while word != 0 {
        let slot = accumulator
            .get_mut(index)
            .ok_or_else(|| anyhow::anyhow!("NIfTI determinant accumulator capacity exceeded"))?;
        let (sum, carry) = slot.overflowing_add(word);
        *slot = sum;
        word = u64::from(carry);
        index = index
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("NIfTI determinant limb index overflowed"))?;
    }
    Ok(())
}

fn compare_unsigned(
    first: &[u64; ACCUMULATOR_LIMBS],
    second: &[u64; ACCUMULATOR_LIMBS],
) -> Ordering {
    first
        .iter()
        .rev()
        .zip(second.iter().rev())
        .find_map(|(left, right)| (left != right).then(|| left.cmp(right)))
        .unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::{sform_orientation, FRACTION_BITS};
    use std::cmp::Ordering;

    fn power_of_two(exponent: u32) -> f64 {
        f64::from_bits(u64::from(exponent) << FRACTION_BITS)
    }

    #[test]
    fn determinant_sign_is_exact_when_magnitude_exceeds_binary64() {
        let large = power_of_two(1023 + 1000);
        let rows = [
            [large, 0.0, 0.0, 0.0],
            [0.0, large, 0.0, 0.0],
            [0.0, 0.0, large, 0.0],
        ];

        assert_eq!(
            sform_orientation(rows).expect("finite rows"),
            Ordering::Greater
        );
    }

    #[test]
    fn determinant_sign_is_exact_when_magnitude_underflows_binary64() {
        let small = power_of_two(1023 - 1000);
        let rows = [
            [small, 0.0, 0.0, 0.0],
            [0.0, small, 0.0, 0.0],
            [0.0, 0.0, small, 0.0],
        ];

        assert_eq!(
            sform_orientation(rows).expect("finite rows"),
            Ordering::Greater
        );
    }

    #[test]
    fn determinant_sign_is_exact_for_mixed_scale_rows_and_transpose() {
        let large = power_of_two(1023 + 1000);
        let small = power_of_two(1023 - 500);
        let rows = [
            [large, small, 0.0, 0.0],
            [large, 0.0, 0.0, 0.0],
            [0.0, 0.0, large, 0.0],
        ];
        let transpose = [
            [large, large, 0.0, 0.0],
            [small, 0.0, 0.0, 0.0],
            [0.0, 0.0, large, 0.0],
        ];

        assert_eq!(
            sform_orientation(rows).expect("finite rows"),
            Ordering::Less
        );
        assert_eq!(
            sform_orientation(transpose).expect("finite rows"),
            Ordering::Less
        );
    }

    #[test]
    fn determinant_accumulator_covers_the_full_binary64_exponent_span() {
        let large = f64::MAX;
        let small = f64::from_bits(1);
        let rows = [
            [large, small, 0.0, 0.0],
            [0.0, large, small, 0.0],
            [small, 0.0, large, 0.0],
        ];

        assert_eq!(
            sform_orientation(rows).expect("finite rows"),
            Ordering::Greater
        );
    }

    #[test]
    fn determinant_orientation_matches_independent_integer_cases() {
        let positive = [
            [1.0, 2.0, 3.0, 0.0],
            [0.0, 1.0, 4.0, 0.0],
            [5.0, 6.0, 0.0, 0.0],
        ];
        let negative = [
            [0.0, 1.0, 4.0, 0.0],
            [1.0, 2.0, 3.0, 0.0],
            [5.0, 6.0, 0.0, 0.0],
        ];
        let near_singular = [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, f64::from_bits(0x3ff0_0000_0000_0001), 1.0, 0.0],
            [1.0, 1.0, f64::from_bits(0x3ff0_0000_0000_0001), 0.0],
        ];

        assert_eq!(
            sform_orientation(positive).expect("finite rows"),
            Ordering::Greater
        );
        assert_eq!(
            sform_orientation(negative).expect("finite rows"),
            Ordering::Less
        );
        assert_eq!(
            sform_orientation(near_singular).expect("finite rows"),
            Ordering::Greater
        );
    }

    #[test]
    fn singular_spatial_rows_have_equal_orientation() {
        let rows = [
            [1.0, 2.0, 3.0, 0.0],
            [2.0, 4.0, 6.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ];

        assert_eq!(
            sform_orientation(rows).expect("finite rows"),
            Ordering::Equal
        );
    }

    #[test]
    fn non_finite_translation_is_rejected() {
        let rows = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, f64::INFINITY],
            [0.0, 0.0, 1.0, 0.0],
        ];

        let error = sform_orientation(rows).expect_err("non-finite translation is invalid");
        assert!(error.to_string().contains("must be finite"));
    }
}
