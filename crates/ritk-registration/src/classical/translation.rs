//! Exhaustive integer-voxel translation registration over borrowed volumes.

use eunomia::{FloatElement, RealField};
use thiserror::Error;

mod private {
    pub trait Sealed {}
}

/// Failure modes for integer-voxel translation registration.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TranslationRegistrationError {
    /// The declared dimensions overflowed or did not match both buffers.
    #[error("translation volume shape {dimensions:?} requires {expected:?} elements, but fixed={fixed_len} and moving={moving_len}")]
    ShapeMismatch {
        /// Volume dimensions in `[z, y, x]` order.
        dimensions: [usize; 3],
        /// Checked element count, or `None` when multiplication overflowed.
        expected: Option<usize>,
        /// Fixed-buffer element count.
        fixed_len: usize,
        /// Moving-buffer element count.
        moving_len: usize,
    },
    /// A search radius could not be represented as a signed offset.
    #[error("translation search radius {radius} on axis {axis} exceeds isize::MAX")]
    SearchRadiusOverflow {
        /// Axis in `[z, y, x]` order.
        axis: usize,
        /// Rejected radius.
        radius: usize,
    },
    /// An input voxel was NaN or infinite.
    #[error("translation {buffer} buffer contains a non-finite value at index {index}")]
    NonFiniteInput {
        /// Buffer containing the invalid value.
        buffer: &'static str,
        /// Flat voxel index.
        index: usize,
    },
    /// No candidate had a non-empty, numerically defined overlap.
    #[error("translation search found no candidate with a defined metric")]
    UndefinedMetric,
}

/// A zero-sized, statically dispatched translation similarity metric.
pub trait TranslationMetric<T: RealField>: private::Sealed {
    /// Per-candidate reduction state.
    type State;

    /// Creates an empty reduction state.
    fn start() -> Self::State;
    /// Accumulates one aligned voxel pair.
    fn accumulate(state: &mut Self::State, fixed: T, moving: T);
    /// Produces the candidate score, or `None` when the metric is undefined.
    fn finish(state: Self::State, count: usize) -> Option<T>;
    /// Returns whether `candidate` is preferable to `best`.
    fn better(candidate: T, best: T) -> bool;
}

/// Mean squared difference metric; lower scores are preferred.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeanSquaredDifference;

impl private::Sealed for MeanSquaredDifference {}

impl<T: RealField> TranslationMetric<T> for MeanSquaredDifference {
    type State = T;

    fn start() -> Self::State {
        T::ZERO
    }

    fn accumulate(state: &mut Self::State, fixed: T, moving: T) {
        let difference = fixed - moving;
        *state = difference.scalar_fmadd(difference, *state);
    }

    fn finish(state: Self::State, count: usize) -> Option<T> {
        (count != 0).then(|| state / T::from_f64(count as f64))
    }

    fn better(candidate: T, best: T) -> bool {
        candidate < best
    }
}

/// Normalized cross-correlation metric; higher scores are preferred.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NormalizedCrossCorrelation;

impl private::Sealed for NormalizedCrossCorrelation {}

impl<T: RealField> TranslationMetric<T> for NormalizedCrossCorrelation {
    type State = [T; 5];

    fn start() -> Self::State {
        [T::ZERO; 5]
    }

    fn accumulate(state: &mut Self::State, fixed: T, moving: T) {
        state[0] += fixed;
        state[1] += moving;
        state[2] = fixed.scalar_fmadd(fixed, state[2]);
        state[3] = moving.scalar_fmadd(moving, state[3]);
        state[4] = fixed.scalar_fmadd(moving, state[4]);
    }

    fn finish(state: Self::State, count: usize) -> Option<T> {
        if count == 0 {
            return None;
        }
        let count = T::from_f64(count as f64);
        let fixed_variance = state[2] - state[0] * state[0] / count;
        let moving_variance = state[3] - state[1] * state[1] / count;
        let variance_product = fixed_variance * moving_variance;
        if variance_product <= T::ZERO {
            return None;
        }
        let covariance = state[4] - state[0] * state[1] / count;
        Some(covariance / variance_product.sqrt())
    }

    fn better(candidate: T, best: T) -> bool {
        candidate > best
    }
}

/// Finds the integer translation that best aligns two same-shaped 3-D volumes.
///
/// Buffers use contiguous row-major `[z, y, x]` order. `max_shift` is the
/// inclusive search radius on each axis. The returned shift maps a fixed-image
/// coordinate to the corresponding moving-image coordinate.
///
/// The kernel borrows both buffers, performs no allocation, and computes in
/// `T`'s native precision. Metric selection is resolved statically through the
/// zero-sized `M` policy.
///
/// # Errors
///
/// Returns a typed error for invalid shapes, unrepresentable search radii,
/// non-finite voxels, or a metric that is undefined for every candidate.
pub fn register_translation<T, M>(
    fixed: &[T],
    moving: &[T],
    dimensions: [usize; 3],
    max_shift: [usize; 3],
) -> Result<[isize; 3], TranslationRegistrationError>
where
    T: RealField,
    M: TranslationMetric<T>,
{
    validate_inputs(fixed, moving, dimensions, max_shift)?;
    let radius = max_shift.map(|value| {
        isize::try_from(value).expect("invariant: search radii were validated as isize")
    });
    let mut best: Option<(T, [isize; 3])> = None;

    for dz in -radius[0]..=radius[0] {
        for dy in -radius[1]..=radius[1] {
            for dx in -radius[2]..=radius[2] {
                let mut state = M::start();
                let mut count = 0usize;
                for z in 0..dimensions[0] {
                    let Some(mz) = z.checked_add_signed(dz).filter(|&v| v < dimensions[0]) else {
                        continue;
                    };
                    for y in 0..dimensions[1] {
                        let Some(my) = y.checked_add_signed(dy).filter(|&v| v < dimensions[1])
                        else {
                            continue;
                        };
                        for x in 0..dimensions[2] {
                            let Some(mx) = x.checked_add_signed(dx).filter(|&v| v < dimensions[2])
                            else {
                                continue;
                            };
                            let fixed_index = (z * dimensions[1] + y) * dimensions[2] + x;
                            let moving_index = (mz * dimensions[1] + my) * dimensions[2] + mx;
                            M::accumulate(&mut state, fixed[fixed_index], moving[moving_index]);
                            count += 1;
                        }
                    }
                }
                let Some(score) = M::finish(state, count) else {
                    continue;
                };
                if best
                    .as_ref()
                    .is_none_or(|(current, _)| M::better(score, *current))
                {
                    best = Some((score, [dz, dy, dx]));
                }
            }
        }
    }

    best.map(|(_, shift)| shift)
        .ok_or(TranslationRegistrationError::UndefinedMetric)
}

fn validate_inputs<T: FloatElement>(
    fixed: &[T],
    moving: &[T],
    dimensions: [usize; 3],
    max_shift: [usize; 3],
) -> Result<(), TranslationRegistrationError> {
    let expected = dimensions[0]
        .checked_mul(dimensions[1])
        .and_then(|value| value.checked_mul(dimensions[2]));
    if expected != Some(fixed.len()) || expected != Some(moving.len()) {
        return Err(TranslationRegistrationError::ShapeMismatch {
            dimensions,
            expected,
            fixed_len: fixed.len(),
            moving_len: moving.len(),
        });
    }
    for (axis, radius) in max_shift.into_iter().enumerate() {
        if isize::try_from(radius).is_err() {
            return Err(TranslationRegistrationError::SearchRadiusOverflow { axis, radius });
        }
    }
    for (buffer, values) in [("fixed", fixed), ("moving", moving)] {
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(TranslationRegistrationError::NonFiniteInput { buffer, index });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translated<T: RealField>(source: &[T], dimensions: [usize; 3], shift: [isize; 3]) -> Vec<T> {
        let mut output = vec![T::ZERO; source.len()];
        for z in 0..dimensions[0] {
            for y in 0..dimensions[1] {
                for x in 0..dimensions[2] {
                    let Some(mz) = z
                        .checked_add_signed(shift[0])
                        .filter(|&v| v < dimensions[0])
                    else {
                        continue;
                    };
                    let Some(my) = y
                        .checked_add_signed(shift[1])
                        .filter(|&v| v < dimensions[1])
                    else {
                        continue;
                    };
                    let Some(mx) = x
                        .checked_add_signed(shift[2])
                        .filter(|&v| v < dimensions[2])
                    else {
                        continue;
                    };
                    let source_index = (z * dimensions[1] + y) * dimensions[2] + x;
                    let output_index = (mz * dimensions[1] + my) * dimensions[2] + mx;
                    output[output_index] = source[source_index];
                }
            }
        }
        output
    }

    fn impulse<T: RealField>(dimensions: [usize; 3], coordinate: [usize; 3]) -> Vec<T> {
        let mut values = vec![T::ZERO; dimensions.into_iter().product()];
        values[(coordinate[0] * dimensions[1] + coordinate[1]) * dimensions[2] + coordinate[2]] =
            T::ONE;
        values
    }

    fn recovers_shift<T: RealField, M: TranslationMetric<T>>() {
        let dimensions = [5, 6, 7];
        let fixed = impulse::<T>(dimensions, [2, 2, 3]);
        let expected = [1, -1, 2];
        let moving = translated(&fixed, dimensions, expected);
        assert_eq!(
            register_translation::<T, M>(&fixed, &moving, dimensions, [2, 2, 2]),
            Ok(expected)
        );
    }

    #[test]
    fn squared_difference_recovers_shift_for_supported_precisions() {
        recovers_shift::<f32, MeanSquaredDifference>();
        recovers_shift::<f64, MeanSquaredDifference>();
    }

    #[test]
    fn correlation_recovers_shift_for_supported_precisions() {
        recovers_shift::<f32, NormalizedCrossCorrelation>();
        recovers_shift::<f64, NormalizedCrossCorrelation>();
    }

    #[test]
    fn identical_volumes_select_zero_shift() {
        let dimensions = [3, 3, 3];
        let fixed: Vec<f64> = (0..27).map(|value| value as f64).collect();
        assert_eq!(
            register_translation::<_, MeanSquaredDifference>(&fixed, &fixed, dimensions, [1; 3]),
            Ok([0, 0, 0])
        );
    }

    #[test]
    fn invalid_shape_and_non_finite_values_are_rejected() {
        assert!(matches!(
            register_translation::<f64, MeanSquaredDifference>(&[0.0], &[0.0], [2, 1, 1], [0; 3]),
            Err(TranslationRegistrationError::ShapeMismatch { .. })
        ));
        assert_eq!(
            register_translation::<f64, MeanSquaredDifference>(&[f64::NAN], &[0.0], [1; 3], [0; 3]),
            Err(TranslationRegistrationError::NonFiniteInput {
                buffer: "fixed",
                index: 0
            })
        );
    }

    #[test]
    fn correlation_rejects_constant_volumes() {
        assert_eq!(
            register_translation::<f64, NormalizedCrossCorrelation>(
                &[1.0; 8], &[1.0; 8], [2; 3], [1; 3]
            ),
            Err(TranslationRegistrationError::UndefinedMetric)
        );
    }
}
