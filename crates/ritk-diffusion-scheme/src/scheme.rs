//! Acquisition-order scheme storage and reorientation.

use ritk_spatial::Vector;

use crate::{DiffusionWeighting, GradientDirection, GradientFrame, GradientSchemeError};

const ROTATION_TOLERANCE: f64 = 1.0e-9;

/// Validated acquisition scheme for one diffusion-weighted series.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientScheme {
    directions: Box<[GradientDirection]>,
    frame: GradientFrame,
}

impl GradientScheme {
    /// Construct an acquisition scheme in volume order.
    ///
    /// # Errors
    ///
    /// Returns [`GradientSchemeError::Empty`] for an empty scheme. Entries
    /// are reconstructed at their acquisition indices so an error identifies
    /// the exact invalid volume even when callers assembled them separately.
    pub fn new(
        directions: Vec<GradientDirection>,
        frame: GradientFrame,
    ) -> Result<Self, GradientSchemeError> {
        if directions.is_empty() {
            return Err(GradientSchemeError::Empty);
        }
        let directions = directions
            .into_iter()
            .enumerate()
            .map(|(index, entry)| {
                GradientDirection::at_index(entry.weighting(), entry.direction(), index)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        Ok(Self { directions, frame })
    }

    /// Construct from external s/mm² values and vectors.
    ///
    /// # Errors
    ///
    /// Returns the first weighting or direction error with its acquisition
    /// index, or [`GradientSchemeError::Empty`] when `pairs` is empty.
    pub fn from_seconds_per_square_millimeter(
        pairs: Vec<(f64, Vector<3>)>,
        frame: GradientFrame,
    ) -> Result<Self, GradientSchemeError> {
        if pairs.is_empty() {
            return Err(GradientSchemeError::Empty);
        }
        let directions = pairs
            .into_iter()
            .enumerate()
            .map(|(index, (value, direction))| {
                let weighting = DiffusionWeighting::at_index(value, index)?;
                GradientDirection::at_index(weighting, direction, index)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(directions, frame)
    }

    /// Entries in acquisition order.
    #[must_use]
    pub fn directions(&self) -> &[GradientDirection] {
        &self.directions
    }

    /// Declared coordinate frame for every direction.
    #[must_use]
    pub const fn frame(&self) -> GradientFrame {
        self.frame
    }

    /// Number of acquisition volumes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.directions.len()
    }

    /// Return whether the scheme contains no volumes.
    ///
    /// Validated schemes are never empty; this method supports generic
    /// collection-style code without weakening construction.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.directions.is_empty()
    }

    /// Indices whose weighting is at or below `threshold`.
    #[must_use]
    pub fn b0_indices(&self, threshold: DiffusionWeighting) -> Vec<usize> {
        self.directions
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| (entry.weighting() <= threshold).then_some(index))
            .collect()
    }

    /// Indices whose weighting is above `threshold`.
    #[must_use]
    pub fn dwi_indices(&self, threshold: DiffusionWeighting) -> Vec<usize> {
        self.directions
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| (entry.weighting() > threshold).then_some(index))
            .collect()
    }

    /// Unique nonzero shell weightings sorted in ascending order.
    #[must_use]
    pub fn shells(&self) -> Vec<DiffusionWeighting> {
        let mut values = self
            .directions
            .iter()
            .map(GradientDirection::weighting)
            .filter(|weighting| !weighting.is_unweighted())
            .collect::<Vec<_>>();
        values.sort_by(|left, right| {
            left.seconds_per_square_meter()
                .total_cmp(&right.seconds_per_square_meter())
        });
        values.dedup();
        values
    }

    /// Rotate all weighted gradients with a proper orthonormal matrix.
    ///
    /// # Errors
    ///
    /// Returns [`GradientSchemeError::InvalidRotation`] when `rotation`
    /// contains non-finite values, is not orthonormal within `1e-9`, or has
    /// determinant other than positive one within that tolerance.
    pub fn reorient(&self, rotation: [[f64; 3]; 3]) -> Result<Self, GradientSchemeError> {
        validate_rotation(rotation)?;
        let directions = self
            .directions
            .iter()
            .map(|entry| rotate_entry(entry, rotation))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(directions, self.frame)
    }

    /// Rotate each volume's gradient by that volume's own rotation.
    ///
    /// [`Self::reorient`] applies one rotation to the whole scheme, which is
    /// what a fixed frame change needs. Motion and eddy-current correction is
    /// the other case: each volume is registered independently, so volume `i`
    /// acquires its own rotation `R_i` and its gradient must move with it. A
    /// correction that leaves the scheme alone — or that applies one averaged
    /// rotation — yields a tensor field whose principal directions are wrong by
    /// the per-volume residual, while every intermediate result still looks
    /// entirely plausible. That silence is why this is a separate entry point
    /// rather than a flag on the single-rotation path.
    ///
    /// `rotations` is in acquisition order, one entry per volume. Unweighted
    /// (b = 0) volumes have no direction to rotate and are passed through, but
    /// they still consume their slot so the indices stay aligned with the
    /// series.
    ///
    /// # Errors
    ///
    /// Returns [`GradientSchemeError::RotationCountMismatch`] when `rotations`
    /// does not have exactly one entry per volume, or
    /// [`GradientSchemeError::InvalidRotation`] when any entry contains
    /// non-finite values, is not orthonormal within `1e-9`, or has determinant
    /// other than positive one within that tolerance.
    pub fn reorient_per_volume(
        &self,
        rotations: &[[[f64; 3]; 3]],
    ) -> Result<Self, GradientSchemeError> {
        if rotations.len() != self.directions.len() {
            return Err(GradientSchemeError::RotationCountMismatch {
                expected: self.directions.len(),
                actual: rotations.len(),
            });
        }
        for rotation in rotations {
            validate_rotation(*rotation)?;
        }

        let directions = self
            .directions
            .iter()
            .zip(rotations)
            .map(|(entry, rotation)| rotate_entry(entry, *rotation))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(directions, self.frame)
    }
}

/// Apply one rotation to one entry, passing unweighted volumes through.
///
/// A b = 0 volume carries no meaningful direction, so rotating its placeholder
/// would manufacture an orientation the acquisition never had.
fn rotate_entry(
    entry: &GradientDirection,
    rotation: [[f64; 3]; 3],
) -> Result<GradientDirection, GradientSchemeError> {
    if entry.weighting().is_unweighted() {
        return Ok(*entry);
    }
    let [x, y, z] = entry.direction().to_array();
    let rotated = Vector::new([
        rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z,
        rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z,
        rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z,
    ]);
    GradientDirection::new(entry.weighting(), rotated)
}

fn validate_rotation(rotation: [[f64; 3]; 3]) -> Result<(), GradientSchemeError> {
    if rotation.iter().flatten().any(|value| !value.is_finite()) {
        return Err(GradientSchemeError::InvalidRotation(
            "matrix contains a non-finite value".to_owned(),
        ));
    }
    for row in 0..3 {
        for column in 0..3 {
            let dot = (0..3)
                .map(|axis| rotation[axis][row] * rotation[axis][column])
                .sum::<f64>();
            let expected = if row == column { 1.0 } else { 0.0 };
            if (dot - expected).abs() > ROTATION_TOLERANCE {
                return Err(GradientSchemeError::InvalidRotation(format!(
                    "R^T R[{row},{column}] is {dot}, expected {expected}"
                )));
            }
        }
    }
    let determinant = rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]);
    if (determinant - 1.0).abs() > ROTATION_TOLERANCE {
        return Err(GradientSchemeError::InvalidRotation(format!(
            "determinant is {determinant}, expected +1"
        )));
    }
    Ok(())
}
