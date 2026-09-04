use super::super::error::{RegistrationError, Result};
use super::PARAMETER_COUNT;
use crate::types::AffineTransform;
use std::num::NonZeroU8;

/// Bounds and terminal resolution for anchored rigid registration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidSearchConfig {
    rotation_half_range_radians: f64,
    translation_half_range_mm: f64,
    final_rotation_resolution_radians: f64,
    final_translation_resolution_mm: f64,
    pub(super) structural_half_range_cells: NonZeroU8,
    pub(super) simplex_iteration_limit: usize,
}

impl RigidSearchConfig {
    /// Validate a bounded rigid-search configuration.
    ///
    /// Rotation values are degrees and translations are millimetres. The final
    /// resolution must not exceed its corresponding search half-range.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] when a value is non-finite,
    /// non-positive, the terminal resolution exceeds its half-range, or the
    /// simplex iteration limit is zero.
    pub fn try_new(
        rotation_half_range_deg: f64,
        translation_half_range_mm: f64,
        final_rotation_resolution_deg: f64,
        final_translation_resolution_mm: f64,
        simplex_iteration_limit: usize,
    ) -> Result<Self> {
        let values = [
            rotation_half_range_deg,
            translation_half_range_mm,
            final_rotation_resolution_deg,
            final_translation_resolution_mm,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(RegistrationError::InvalidInput(format!(
                "rigid-search ranges and resolutions must be finite and positive, got {values:?}"
            )));
        }
        if final_rotation_resolution_deg > rotation_half_range_deg
            || final_translation_resolution_mm > translation_half_range_mm
        {
            return Err(RegistrationError::InvalidInput(format!(
                "rigid-search terminal resolution [{final_rotation_resolution_deg} deg, \
                 {final_translation_resolution_mm} mm] exceeds half-range \
                 [{rotation_half_range_deg} deg, {translation_half_range_mm} mm]"
            )));
        }
        if simplex_iteration_limit == 0 {
            return Err(RegistrationError::InvalidInput(
                "rigid-search simplex iteration limit must be positive".to_owned(),
            ));
        }
        Ok(Self {
            rotation_half_range_radians: rotation_half_range_deg.to_radians(),
            translation_half_range_mm,
            final_rotation_resolution_radians: final_rotation_resolution_deg.to_radians(),
            final_translation_resolution_mm,
            structural_half_range_cells: NonZeroU8::MIN,
            simplex_iteration_limit,
        })
    }

    /// Set the structural-refinement half-range in terminal capture cells.
    ///
    /// The nonzero `u8` bound keeps the refinement finite. Global rigid-search
    /// bounds remain authoritative and may clip the requested local range.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroU8;
    /// use ritk_registration::RigidSearchConfig;
    ///
    /// let config = RigidSearchConfig::try_new(12.0, 8.0, 0.5, 0.75, 256)
    ///     .expect("valid bounded search")
    ///     .with_structural_half_range_cells(
    ///         NonZeroU8::new(3).expect("invariant: three is nonzero"),
    ///     );
    /// assert_eq!(config.structural_half_range_cells().get(), 3);
    /// ```
    #[must_use]
    pub const fn with_structural_half_range_cells(mut self, cells: NonZeroU8) -> Self {
        self.structural_half_range_cells = cells;
        self
    }

    /// Return the structural-refinement half-range in terminal capture cells.
    #[must_use]
    pub const fn structural_half_range_cells(self) -> NonZeroU8 {
        self.structural_half_range_cells
    }

    pub(super) fn global_bounds(self) -> [f64; PARAMETER_COUNT] {
        [
            self.rotation_half_range_radians,
            self.rotation_half_range_radians,
            self.rotation_half_range_radians,
            self.translation_half_range_mm,
            self.translation_half_range_mm,
            self.translation_half_range_mm,
        ]
    }

    pub(super) fn terminal_resolution(self) -> [f64; PARAMETER_COUNT] {
        [
            self.final_rotation_resolution_radians,
            self.final_rotation_resolution_radians,
            self.final_rotation_resolution_radians,
            self.final_translation_resolution_mm,
            self.final_translation_resolution_mm,
            self.final_translation_resolution_mm,
        ]
    }
}

/// NMI-capture and local structural-refinement candidates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct RigidSearchResult {
    /// Transform at the capture objective's optimum.
    pub capture_transform: AffineTransform,
    /// Transform after structural refinement inside the configured local range.
    pub structural_transform: AffineTransform,
    /// Capture-objective value at [`Self::capture_transform`].
    pub capture_score: f64,
    /// Structural-objective value at [`Self::structural_transform`].
    pub structural_score: f64,
    /// Whether capture terminated within one resolution step of a global bound.
    pub capture_saturated: bool,
    /// Whether structural refinement terminated within one convergence width
    /// of its effective local or global bound.
    pub structural_saturated: bool,
}
