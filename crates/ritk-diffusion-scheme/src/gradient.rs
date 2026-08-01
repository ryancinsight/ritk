//! Validated gradient directions and coordinate frames.

use ritk_spatial::Vector;

use crate::{DiffusionWeighting, GradientSchemeError};

const UNIT_NORM_TOLERANCE: f64 = 1.0e-6;

/// Coordinate frame of diffusion gradient directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GradientFrame {
    /// Image index-axis coordinates, as used by FSL companion b-vectors.
    ImageAxis,
    /// Physical Left-Posterior-Superior patient coordinates.
    Lps,
}

/// One validated diffusion weighting and gradient direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientDirection {
    weighting: DiffusionWeighting,
    direction: Vector<3>,
}

impl GradientDirection {
    /// Construct a validated acquisition entry.
    ///
    /// An unweighted entry requires an exact zero vector. A weighted entry
    /// requires a finite unit vector within `1e-6` Euclidean norm.
    ///
    /// # Errors
    ///
    /// Returns [`GradientSchemeError::InvalidDirection`] when the direction
    /// does not satisfy the applicable zero/unit-vector contract.
    pub fn new(
        weighting: DiffusionWeighting,
        direction: Vector<3>,
    ) -> Result<Self, GradientSchemeError> {
        Self::at_index(weighting, direction, 0)
    }

    pub(crate) fn at_index(
        weighting: DiffusionWeighting,
        direction: Vector<3>,
        index: usize,
    ) -> Result<Self, GradientSchemeError> {
        let components = direction.to_array();
        if components.iter().any(|value| !value.is_finite()) {
            return Err(GradientSchemeError::InvalidDirection {
                index,
                reason: format!("components are not finite: {components:?}"),
            });
        }

        let norm = direction.norm();
        if weighting.is_unweighted() {
            if norm != 0.0 {
                return Err(GradientSchemeError::InvalidDirection {
                    index,
                    reason: format!("unweighted volume requires a zero vector, norm is {norm}"),
                });
            }
        } else if (norm - 1.0).abs() > UNIT_NORM_TOLERANCE {
            return Err(GradientSchemeError::InvalidDirection {
                index,
                reason: format!("weighted volume requires a unit vector, norm is {norm}"),
            });
        }

        Ok(Self {
            weighting,
            direction,
        })
    }

    /// Diffusion weighting for this volume.
    #[must_use]
    pub const fn weighting(&self) -> DiffusionWeighting {
        self.weighting
    }

    /// Gradient direction in the scheme's declared frame.
    #[must_use]
    pub const fn direction(&self) -> Vector<3> {
        self.direction
    }
}
