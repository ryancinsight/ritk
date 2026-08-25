//! Echo-planar geometric distortion along the phase-encode axis.
//!
//! # The distortion
//!
//! An echo-planar readout traverses k-space slowly along one axis, so an
//! off-resonance field displaces signal along that phase-encode axis alone. The
//! displacement is proportional to the field and to the total readout time, and
//! it reverses with the phase-encode polarity — which is why acquiring a
//! reversed pair lets the field be recovered at all.
//!
//! Displacement is not the whole effect. Where neighbouring voxels are pushed
//! together their signal piles up, and where they are pulled apart it thins
//! out. Signal is conserved, so intensity scales by the Jacobian of the
//! mapping. A correction that moves voxels without rescaling them produces an
//! image whose geometry is right and whose intensities are wrong — invisible on
//! inspection, and a bias in every quantitative fit downstream.
//!
//! # Convention
//!
//! Conventions differ between toolchains, so this module pins one and states
//! it. For a field `f` in **voxels** and polarity sign `s = ±1`:
//!
//! ```text
//! observed(y) = true(y + s·f(y)) · |1 + s·∂f/∂y|
//! ```
//!
//! `f` maps an observed coordinate to the true coordinate it came from, at
//! positive polarity. [`distort`] applies that model; [`unwarp`] inverts it.
//!
//! # Validity
//!
//! The mapping is invertible only while `1 + s·∂f/∂y > 0`. At zero the voxel
//! grid folds — distinct true positions map to one observed position, and the
//! signal there is a sum that no unwarping can separate. That is a property of
//! the acquisition, not of the arithmetic, so a folding field is rejected
//! rather than clamped.
//!
//! # Reference
//!
//! Andersson, Skare, and Ashburner, "How to correct susceptibility distortions
//! in spin-echo echo-planar images: application to diffusion tensor imaging",
//! *NeuroImage* 20(2), 2003 — the reversed-polarity formulation, including the
//! Jacobian intensity term.

use leto::Array3;

use crate::classical::{RegistrationError, Result};

/// Image axis the phase encoding runs along.
///
/// Indices are into a `[z, y, x]` volume, matching the RITK tensor order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhaseEncodeAxis {
    /// Slowest axis (`z`, index 0).
    Depth,
    /// Middle axis (`y`, index 1) — the usual phase-encode axis for an axial
    /// EPI acquisition.
    Row,
    /// Fastest axis (`x`, index 2).
    Column,
}

impl PhaseEncodeAxis {
    /// Position of this axis in a `[z, y, x]` shape.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Depth => 0,
            Self::Row => 1,
            Self::Column => 2,
        }
    }
}

/// Traversal direction along the phase-encode axis.
///
/// The two polarities of a reversed pair distort oppositely, which is what
/// makes the field recoverable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhaseEncodePolarity {
    /// Increasing index — the "blip-up" acquisition.
    Positive,
    /// Decreasing index — the "blip-down" acquisition.
    Negative,
}

impl PhaseEncodePolarity {
    /// Sign this polarity applies to the displacement field.
    #[must_use]
    pub const fn sign(self) -> f64 {
        match self {
            Self::Positive => 1.0,
            Self::Negative => -1.0,
        }
    }

    /// The opposite polarity.
    #[must_use]
    pub const fn reversed(self) -> Self {
        match self {
            Self::Positive => Self::Negative,
            Self::Negative => Self::Positive,
        }
    }
}

/// Phase-encode axis and polarity of one acquisition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhaseEncoding {
    /// Axis the encoding runs along.
    pub axis: PhaseEncodeAxis,
    /// Traversal direction.
    pub polarity: PhaseEncodePolarity,
}

impl PhaseEncoding {
    /// Construct an encoding.
    #[must_use]
    pub const fn new(axis: PhaseEncodeAxis, polarity: PhaseEncodePolarity) -> Self {
        Self { axis, polarity }
    }

    /// The same axis with the opposite polarity — the other half of a reversed
    /// pair.
    #[must_use]
    pub const fn reversed(self) -> Self {
        Self {
            axis: self.axis,
            polarity: self.polarity.reversed(),
        }
    }
}

/// Smallest Jacobian treated as non-folding.
///
/// The mapping is singular at zero. A positive floor keeps a field that merely
/// approaches folding from producing an unbounded intensity scale, and it is
/// the value the error reports so a caller can see how close the field came.
const MINIMUM_JACOBIAN: f64 = 1.0e-3;

/// Apply the distortion a field produces under `encoding`.
///
/// The forward model: what the scanner would observe given the true image and
/// the off-resonance field. This is the model field estimation fits against.
///
/// `field` holds displacements in **voxels** along the encoding axis, on the
/// same grid as `image`.
///
/// # Errors
///
/// [`RegistrationError::InvalidInput`] when `field` and `image` differ in
/// shape, when either contains a non-finite value, or when the field folds the
/// voxel grid.
pub fn distort(
    image: &Array3<f64>,
    field: &Array3<f64>,
    encoding: PhaseEncoding,
) -> Result<Array3<f64>> {
    let shape = validated_shape(image, field)?;
    let axis = encoding.axis.index();
    let sign = encoding.polarity.sign();
    let length = shape[axis];

    let mut output = vec![0.0; shape[0] * shape[1] * shape[2]];
    for_each_line(shape, axis, |line| {
        let displacements: Vec<f64> = (0..length)
            .map(|position| sign * field[index_of(line, axis, position)])
            .collect();
        let source: Vec<f64> = (0..length)
            .map(|position| image[index_of(line, axis, position)])
            .collect();

        for position in 0..length {
            let jacobian = 1.0 + derivative(&displacements, position);
            if jacobian < MINIMUM_JACOBIAN {
                return Err(folding_error(position, jacobian));
            }
            let sampled = sample_linear(&source, position as f64 + displacements[position]);
            output[flat_index(index_of(line, axis, position), shape)] = sampled * jacobian;
        }
        Ok(())
    })?;

    Array3::from_shape_vec(shape, output).map_err(|error| {
        RegistrationError::InvalidInput(format!("distorted volume shape is invalid: {error}"))
    })
}

/// Recover the true image from an observed one, inverting [`distort`].
///
/// The observed-to-true mapping `y ↦ y + s·f(y)` is strictly increasing while
/// the field does not fold, so it has a unique inverse. This solves that
/// inverse per line rather than assuming small displacements, which keeps the
/// round trip exact instead of approximate.
///
/// # Errors
///
/// As [`distort`].
pub fn unwarp(
    observed: &Array3<f64>,
    field: &Array3<f64>,
    encoding: PhaseEncoding,
) -> Result<Array3<f64>> {
    let shape = validated_shape(observed, field)?;
    let axis = encoding.axis.index();
    let sign = encoding.polarity.sign();
    let length = shape[axis];

    let mut output = vec![0.0; shape[0] * shape[1] * shape[2]];
    for_each_line(shape, axis, |line| {
        let displacements: Vec<f64> = (0..length)
            .map(|position| sign * field[index_of(line, axis, position)])
            .collect();

        // Forward map of each observed sample, and the Jacobian that scaled it.
        let mut mapped = Vec::with_capacity(length);
        let mut jacobians = Vec::with_capacity(length);
        for position in 0..length {
            let jacobian = 1.0 + derivative(&displacements, position);
            if jacobian < MINIMUM_JACOBIAN {
                return Err(folding_error(position, jacobian));
            }
            mapped.push(position as f64 + displacements[position]);
            jacobians.push(jacobian);
        }

        // Undo the intensity scaling before resampling, so interpolation mixes
        // signal densities rather than already-scaled values.
        let unscaled: Vec<f64> = (0..length)
            .map(|position| observed[index_of(line, axis, position)] / jacobians[position])
            .collect();

        for target in 0..length {
            let source = invert_monotone(&mapped, target as f64);
            output[flat_index(index_of(line, axis, target), shape)] =
                sample_linear(&unscaled, source);
        }
        Ok(())
    })?;

    Array3::from_shape_vec(shape, output).map_err(|error| {
        RegistrationError::InvalidInput(format!("unwarped volume shape is invalid: {error}"))
    })
}

fn folding_error(position: usize, jacobian: f64) -> RegistrationError {
    RegistrationError::InvalidInput(format!(
        "displacement field folds the voxel grid at phase-encode position {position}: \
         Jacobian {jacobian} is below {MINIMUM_JACOBIAN}. Signal from distinct true \
         positions is summed there and cannot be separated by unwarping."
    ))
}

fn validated_shape(image: &Array3<f64>, field: &Array3<f64>) -> Result<[usize; 3]> {
    let shape = image.shape();
    if field.shape() != shape {
        return Err(RegistrationError::InvalidInput(format!(
            "field shape {:?} does not match image shape {shape:?}",
            field.shape()
        )));
    }
    for (name, array) in [("image", image), ("field", field)] {
        if array
            .as_slice()
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(RegistrationError::InvalidInput(format!(
                "{name} contains a non-finite value"
            )));
        }
    }
    Ok(shape)
}

/// Visit every line parallel to `axis`, passing the fixed coordinates of the
/// other two axes.
fn for_each_line<F>(shape: [usize; 3], axis: usize, mut body: F) -> Result<()>
where
    F: FnMut([usize; 3]) -> Result<()>,
{
    let others: Vec<usize> = (0..3).filter(|index| *index != axis).collect();
    for first in 0..shape[others[0]] {
        for second in 0..shape[others[1]] {
            let mut line = [0usize; 3];
            line[others[0]] = first;
            line[others[1]] = second;
            body(line)?;
        }
    }
    Ok(())
}

/// Row-major flat offset of `coordinate` in a `[z, y, x]` volume.
const fn flat_index(coordinate: [usize; 3], shape: [usize; 3]) -> usize {
    coordinate[0] * shape[1] * shape[2] + coordinate[1] * shape[2] + coordinate[2]
}

/// The coordinate of `position` along `axis` within `line`.
const fn index_of(line: [usize; 3], axis: usize, position: usize) -> [usize; 3] {
    let mut coordinate = line;
    coordinate[axis] = position;
    coordinate
}

/// Central difference of `values` at `position`, one-sided at the ends.
///
/// The field is sampled on the voxel grid, so the derivative that scales
/// intensity is the discrete one — using an analytic derivative of some fitted
/// field would not match the resampling actually performed.
fn derivative(values: &[f64], position: usize) -> f64 {
    let last = values.len() - 1;
    if values.len() == 1 {
        return 0.0;
    }
    if position == 0 {
        return values[1] - values[0];
    }
    if position == last {
        return values[last] - values[last - 1];
    }
    (values[position + 1] - values[position - 1]) / 2.0
}

/// Linear interpolation of `values` at a fractional index, clamped at the ends.
///
/// Clamping rather than zero-filling is the right edge behaviour here: signal
/// displaced from outside the field of view is unknown, and substituting zero
/// would introduce an artificial dark band that a later fit would try to
/// explain.
fn sample_linear(values: &[f64], position: f64) -> f64 {
    let last = values.len() - 1;
    if position <= 0.0 {
        return values[0];
    }
    if position >= last as f64 {
        return values[last];
    }
    let lower = position.floor();
    let weight = position - lower;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "ratchet RITK-LINT-1"
    )]
    let lower_index = lower as usize;
    values[lower_index] * (1.0 - weight) + values[lower_index + 1] * weight
}

/// Invert a strictly increasing sampled map at `target`.
///
/// `mapped[i]` is the image of grid position `i`. Returns the fractional
/// position whose image is `target`, by locating the bracketing pair and
/// interpolating within it.
fn invert_monotone(mapped: &[f64], target: f64) -> f64 {
    let last = mapped.len() - 1;
    if target <= mapped[0] {
        return 0.0;
    }
    if target >= mapped[last] {
        return last as f64;
    }
    // Monotone, so a binary search locates the bracket in log time and stays
    // correct for large displacements.
    let mut low = 0usize;
    let mut high = last;
    while high - low > 1 {
        let middle = (low + high) / 2;
        if mapped[middle] <= target {
            low = middle;
        } else {
            high = middle;
        }
    }
    let span = mapped[high] - mapped[low];
    if span <= 0.0 {
        return low as f64;
    }
    low as f64 + (target - mapped[low]) / span
}

#[cfg(test)]
mod tests;
