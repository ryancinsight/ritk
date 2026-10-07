//! Geometry predicates shared by DICOM import and export.

/// Whether the provided direction-cosine vectors are unit length and mutually
/// orthogonal within the input and arithmetic error bound.
///
/// PS3.3 C.7.6.2.1.1 requires each axis to have unit norm and each pair to be
/// orthogonal. The bound admits directions rounded once to `f32` then widened:
/// for component error at most `u`, a three-term squared norm or dot changes by
/// at most `6u + 3u²`; five `f64` operations add at most
/// `3 * gamma(5) * (1+u)²`. The same predicate validates the six-value row and
/// column vectors on input and the six- or nine-value direction arrays emitted
/// by RITK writers.
pub(crate) fn direction_cosines_are_orthonormal(direction: &[f64]) -> bool {
    if direction.is_empty() {
        return true;
    }
    if direction.len() != 6 && direction.len() != 9 {
        return false;
    }
    let axes = direction.chunks_exact(3);
    for (index, axis) in axes.clone().enumerate() {
        let norm_squared = axis.iter().map(|value| value * value).sum::<f64>();
        if (norm_squared - 1.0).abs() > DIRECTION_DOT_BOUND {
            return false;
        }
        for other in axes.clone().skip(index + 1) {
            let dot = axis.iter().zip(other).map(|(a, b)| a * b).sum::<f64>();
            if dot.abs() > DIRECTION_DOT_BOUND {
                return false;
            }
        }
    }
    true
}

const DIRECTION_UNIT_ROUNDOFF: f64 = 1.0 / 16_777_216.0;
const DOT_UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;
const DOT_GAMMA: f64 = 5.0 * DOT_UNIT_ROUNDOFF / (1.0 - 5.0 * DOT_UNIT_ROUNDOFF);
const DIRECTION_DOT_BOUND: f64 = 6.0 * DIRECTION_UNIT_ROUNDOFF
    + 3.0 * DIRECTION_UNIT_ROUNDOFF * DIRECTION_UNIT_ROUNDOFF
    + 3.0 * DOT_GAMMA * (1.0 + DIRECTION_UNIT_ROUNDOFF) * (1.0 + DIRECTION_UNIT_ROUNDOFF);
