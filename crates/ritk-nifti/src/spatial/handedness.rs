use std::cmp::Ordering;

/// Orientation of an invertible affine's linear part.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpatialHandedness {
    /// The linear transform reverses orientation.
    Left,
    /// The linear transform preserves orientation.
    Right,
}

/// Return the handedness of the linear part of three NIfTI sform rows.
///
/// The determinant is evaluated exactly for the supplied binary64 values.
/// Each finite value is represented as an integer significand times a power
/// of two. Binary64 exponents span `-1074..=971`, and a product of three
/// 53-bit significands uses at most 159 bits. Aligning the six determinant
/// terms spans at most 6,135 bits; summing up to six signed terms adds at most
/// three bits, for a 6,297-bit bound. The 99-limb accumulator holds 6,336
/// bits, so it covers the full result without heap allocation. Exact
/// accumulation avoids scale overflow and cancellation residuals that can
/// misclassify an exactly singular sform.
pub(crate) fn sform_handedness(rows: [[f64; 4]; 3]) -> Option<SpatialHandedness> {
    if rows
        .iter()
        .take(3)
        .flat_map(|row| row.iter().take(3))
        .any(|coefficient| !coefficient.is_finite())
    {
        return None;
    }

    let terms = [
        determinant_term([rows[0][0], rows[1][1], rows[2][2]], false),
        determinant_term([rows[0][1], rows[1][2], rows[2][0]], false),
        determinant_term([rows[0][2], rows[1][0], rows[2][1]], false),
        determinant_term([rows[0][2], rows[1][1], rows[2][0]], true),
        determinant_term([rows[0][1], rows[1][0], rows[2][2]], true),
        determinant_term([rows[0][0], rows[1][2], rows[2][1]], true),
    ];
    let base_exponent = terms.iter().flatten().map(|term| term.exponent).min()?;
    let mut positive = [0_u64; EXACT_DETERMINANT_LIMBS];
    let mut negative = [0_u64; EXACT_DETERMINANT_LIMBS];
    for term in terms.into_iter().flatten() {
        let accumulator = if term.negative {
            &mut negative
        } else {
            &mut positive
        };
        add_scaled_product(accumulator, term, base_exponent);
    }

    match positive.iter().rev().cmp(negative.iter().rev()) {
        Ordering::Less => Some(SpatialHandedness::Left),
        Ordering::Equal => None,
        Ordering::Greater => Some(SpatialHandedness::Right),
    }
}

const EXACT_DETERMINANT_LIMBS: usize = 99;
const BINARY64_FRACTION_MASK: u64 = (1_u64 << 52) - 1;
const BINARY64_EXPONENT_MASK: u64 = 0x7ff;
const BINARY64_SIGN_MASK: u64 = 1_u64 << 63;

#[derive(Clone, Copy)]
struct BinaryCoefficient {
    negative: bool,
    significand: u64,
    exponent: i32,
}

#[derive(Clone, Copy)]
struct DeterminantTerm {
    negative: bool,
    significand: [u64; 3],
    exponent: i32,
}

fn binary_coefficient(value: f64) -> Option<BinaryCoefficient> {
    let bits = value.to_bits();
    let exponent_field = (bits >> 52) & BINARY64_EXPONENT_MASK;
    let fraction = bits & BINARY64_FRACTION_MASK;
    let negative = bits & BINARY64_SIGN_MASK != 0;

    if exponent_field == 0 {
        if fraction == 0 {
            None
        } else {
            Some(BinaryCoefficient {
                negative,
                significand: fraction,
                exponent: -1074,
            })
        }
    } else {
        Some(BinaryCoefficient {
            negative,
            significand: (1_u64 << 52) | fraction,
            exponent: i32::try_from(exponent_field).expect("invariant: binary64 exponent fits i32")
                - 1075,
        })
    }
}

fn determinant_term(values: [f64; 3], odd_permutation: bool) -> Option<DeterminantTerm> {
    let [first, second, third] = values;
    let first = binary_coefficient(first)?;
    let second = binary_coefficient(second)?;
    let third = binary_coefficient(third)?;
    let exponent = first
        .exponent
        .checked_add(second.exponent)?
        .checked_add(third.exponent)?;

    Some(DeterminantTerm {
        negative: first.negative ^ second.negative ^ third.negative ^ odd_permutation,
        significand: multiply_significands(
            first.significand,
            second.significand,
            third.significand,
        ),
        exponent,
    })
}

fn multiply_significands(first: u64, second: u64, third: u64) -> [u64; 3] {
    let [low, high] = multiply_words(first, second);
    let [low_low, low_high] = multiply_words(low, third);
    let [high_low, high_high] = multiply_words(high, third);
    let (middle, carry) = low_high.overflowing_add(high_low);
    let high = high_high
        .checked_add(u64::from(carry))
        .expect("invariant: three binary64 significands fit 159 bits");
    [low_low, middle, high]
}

fn multiply_words(first: u64, second: u64) -> [u64; 2] {
    let product = u128::from(first) * u128::from(second);
    [
        u64::try_from(product & u128::from(u64::MAX))
            .expect("invariant: low product limb fits u64"),
        u64::try_from(product >> 64).expect("invariant: high product limb fits u64"),
    ]
}

fn add_scaled_product(
    accumulator: &mut [u64; EXACT_DETERMINANT_LIMBS],
    term: DeterminantTerm,
    base_exponent: i32,
) {
    let shift = usize::try_from(
        term.exponent
            .checked_sub(base_exponent)
            .expect("invariant: determinant exponents are ordered"),
    )
    .expect("invariant: determinant exponent span is nonnegative");
    let word_offset = shift / 64;
    let bit_offset = u32::try_from(shift % 64).expect("invariant: bit offset is below 64");

    for (limb_offset, limb) in term.significand.into_iter().enumerate() {
        if limb == 0 {
            continue;
        }
        let destination = word_offset
            .checked_add(limb_offset)
            .expect("invariant: determinant accumulator index fits usize");
        add_word(accumulator, destination, limb << bit_offset);
        if bit_offset != 0 {
            let upper = limb >> (64 - bit_offset);
            if upper != 0 {
                add_word(
                    accumulator,
                    destination
                        .checked_add(1)
                        .expect("invariant: determinant accumulator index fits usize"),
                    upper,
                );
            }
        }
    }
}

fn add_word(accumulator: &mut [u64; EXACT_DETERMINANT_LIMBS], mut index: usize, mut value: u64) {
    while value != 0 {
        let slot = accumulator
            .get_mut(index)
            .expect("invariant: 99 limbs cover every binary64 determinant");
        let (sum, carry) = slot.overflowing_add(value);
        *slot = sum;
        if !carry {
            return;
        }
        index = index
            .checked_add(1)
            .expect("invariant: determinant accumulator index fits usize");
        value = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{sform_handedness, SpatialHandedness};
    use std::cmp::Ordering;

    #[test]
    fn exact_handedness_detects_singularity_after_header_encoding() {
        let rows = [
            [-3.0, -2.0, -1.0, 0.0],
            [-6.0, -5.0, -4.0, 0.0],
            [9.0, 7.0, 5.0, 0.0],
        ];

        assert_eq!(sform_handedness(rows), None);
    }

    #[test]
    fn exact_handedness_preserves_nearly_singular_orientation() {
        let separation = 2.0_f64.powi(-30);
        let rows = [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0 + separation, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];

        assert_eq!(sform_handedness(rows), Some(SpatialHandedness::Right));
    }

    #[test]
    fn exact_handedness_accumulates_the_binary64_exponent_span() {
        let smallest = f64::from_bits(1);
        let largest = f64::MAX;
        let rows = [
            [smallest, largest, 0.0, 0.0],
            [0.0, smallest, largest, 0.0],
            [largest, 0.0, smallest, 0.0],
        ];

        assert_eq!(sform_handedness(rows), Some(SpatialHandedness::Right));
    }

    #[test]
    fn exact_handedness_matches_integer_oracle_for_binary_matrices() {
        for mask in 0_u16..(1_u16 << 9) {
            let value = |bit: usize| u8::from(mask & (1_u16 << bit) != 0);
            let coefficients = [
                [value(0), value(1), value(2)],
                [value(3), value(4), value(5)],
                [value(6), value(7), value(8)],
            ];
            let rows = coefficients
                .map(|row| [f64::from(row[0]), f64::from(row[1]), f64::from(row[2]), 0.0]);
            let determinant = i128::from(coefficients[0][0])
                * i128::from(coefficients[1][1])
                * i128::from(coefficients[2][2])
                + i128::from(coefficients[0][1])
                    * i128::from(coefficients[1][2])
                    * i128::from(coefficients[2][0])
                + i128::from(coefficients[0][2])
                    * i128::from(coefficients[1][0])
                    * i128::from(coefficients[2][1])
                - i128::from(coefficients[0][2])
                    * i128::from(coefficients[1][1])
                    * i128::from(coefficients[2][0])
                - i128::from(coefficients[0][1])
                    * i128::from(coefficients[1][0])
                    * i128::from(coefficients[2][2])
                - i128::from(coefficients[0][0])
                    * i128::from(coefficients[1][2])
                    * i128::from(coefficients[2][1]);
            let expected = match determinant.cmp(&0) {
                Ordering::Less => Some(SpatialHandedness::Left),
                Ordering::Equal => None,
                Ordering::Greater => Some(SpatialHandedness::Right),
            };

            assert_eq!(sform_handedness(rows), expected, "mask {mask}");
        }
    }
}
