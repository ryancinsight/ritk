//! Orientation-controlled oblique reslice construction.

use super::sampling::{
    add_scaled, patient_step_to_voxel, validate_volume, vector_norm, voxel_step,
};
use super::{ResliceError, ResliceInterpolation, ReslicePlane};
use crate::LoadedVolume;

const DEGREES_TO_RADIANS: f64 = std::f64::consts::PI / 180.0;

/// Yaw and pitch rotations for an oblique multiplanar plane.
///
/// Yaw rotates around the source row direction; pitch rotates around the
/// resulting horizontal direction. Angles are in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResliceOrientation {
    yaw_degrees: f64,
    pitch_degrees: f64,
}

impl ResliceOrientation {
    /// Construct a bounded orientation.
    ///
    /// Yaw accepts `[-180, 180]`; pitch accepts `[-90, 90]` so the plane has a
    /// unique pitch representation.
    ///
    /// # Errors
    /// Returns [`ResliceError::InvalidOrientation`] for non-finite or
    /// out-of-range angles.
    pub fn try_new(yaw_degrees: f64, pitch_degrees: f64) -> Result<Self, ResliceError> {
        if !yaw_degrees.is_finite()
            || !pitch_degrees.is_finite()
            || !(-180.0..=180.0).contains(&yaw_degrees)
            || !(-90.0..=90.0).contains(&pitch_degrees)
        {
            return Err(ResliceError::InvalidOrientation {
                yaw_degrees,
                pitch_degrees,
            });
        }
        Ok(Self {
            yaw_degrees,
            pitch_degrees,
        })
    }

    /// Return yaw in degrees.
    #[must_use]
    pub const fn yaw_degrees(self) -> f64 {
        self.yaw_degrees
    }

    /// Return pitch in degrees.
    #[must_use]
    pub const fn pitch_degrees(self) -> f64 {
        self.pitch_degrees
    }

    /// Apply a user-controlled rotation, wrapping yaw and clamping pitch.
    ///
    /// # Errors
    /// Returns [`ResliceError::InvalidOrientation`] when either delta is
    /// non-finite.
    pub fn rotated_by(
        self,
        yaw_delta_degrees: f64,
        pitch_delta_degrees: f64,
    ) -> Result<Self, ResliceError> {
        if !yaw_delta_degrees.is_finite() || !pitch_delta_degrees.is_finite() {
            return Err(ResliceError::InvalidOrientation {
                yaw_degrees: yaw_delta_degrees,
                pitch_degrees: pitch_delta_degrees,
            });
        }
        let yaw = (self.yaw_degrees + yaw_delta_degrees + 180.0).rem_euclid(360.0) - 180.0;
        let pitch = (self.pitch_degrees + pitch_delta_degrees).clamp(-90.0, 90.0);
        Self::try_new(yaw, pitch)
    }
}

impl Default for ResliceOrientation {
    fn default() -> Self {
        Self {
            yaw_degrees: 0.0,
            pitch_degrees: 0.0,
        }
    }
}

impl ReslicePlane {
    /// Build a centered one-sample oblique plane that remains inside the
    /// source volume.
    ///
    /// The output keeps source column and row spacing and their physical
    /// aspect ratio. Its rectangular field of view is uniformly reduced only
    /// as needed for every corner to remain within the source voxel bounds.
    ///
    /// # Errors
    /// Returns a source validation error, [`ResliceError::InvalidCenter`] for
    /// a center outside the source volume, or a typed plane-construction error.
    pub fn centered_oblique(
        volume: &LoadedVolume,
        center_voxel: [f64; 3],
        orientation: ResliceOrientation,
        interpolation: ResliceInterpolation,
    ) -> Result<Self, ResliceError> {
        let transform = validate_volume(volume)?;
        validate_center(center_voxel, volume.shape)?;

        let center_patient = transform.voxel_to_patient(center_voxel);
        let column_step = voxel_step(&transform, center_voxel, 2);
        let row_step = voxel_step(&transform, center_voxel, 1);
        let depth_step = voxel_step(&transform, center_voxel, 0);
        let column_spacing = vector_norm(column_step);
        let row_spacing = vector_norm(row_step);
        let depth_spacing = vector_norm(depth_step);
        let horizontal = normalize(column_step)?;
        let row_direction = normalize(row_step)?;
        let vertical = normalize(subtract(
            row_direction,
            scale(horizontal, dot(row_direction, horizontal)),
        ))?;

        let horizontal = rotate(horizontal, vertical, orientation.yaw_degrees);
        let vertical = rotate(vertical, horizontal, orientation.pitch_degrees);
        let normal = normalize(cross(horizontal, vertical))?;

        let half_width = (volume.shape[2].saturating_sub(1) as f64) * column_spacing * 0.5;
        let half_height = (volume.shape[1].saturating_sub(1) as f64) * row_spacing * 0.5;
        let horizontal_voxel = patient_step_to_voxel(&transform, center_patient, horizontal);
        let vertical_voxel = patient_step_to_voxel(&transform, center_patient, vertical);
        let corner_extent = std::array::from_fn::<_, 3, _>(|axis| {
            horizontal_voxel[axis].abs() * half_width + vertical_voxel[axis].abs() * half_height
        });
        if !half_width.is_finite()
            || !half_height.is_finite()
            || !corner_extent.into_iter().all(f64::is_finite)
        {
            return Err(ResliceError::InvalidFieldOfView);
        }

        let scale_factor = corner_extent
            .into_iter()
            .zip(center_voxel.into_iter().zip(volume.shape))
            .filter_map(|(extent, (center, size))| {
                (extent > 0.0)
                    .then_some(center.min(size.saturating_sub(1) as f64 - center) / extent)
            })
            .fold(1.0_f64, f64::min)
            .clamp(0.0, 1.0);
        let half_width = half_width * scale_factor;
        let half_height = half_height * scale_factor;
        let dimensions = [
            output_extent(half_width, column_spacing, volume.shape[2])?,
            output_extent(half_height, row_spacing, volume.shape[1])?,
        ];
        let actual_half_width = dimensions[0].saturating_sub(1) as f64 * column_spacing * 0.5;
        let actual_half_height = dimensions[1].saturating_sub(1) as f64 * row_spacing * 0.5;
        let origin = add_scaled(
            add_scaled(center_patient, horizontal, -actual_half_width),
            vertical,
            -actual_half_height,
        );
        let depth_spacing = column_spacing.min(row_spacing).min(depth_spacing);

        Self::try_new(
            volume,
            origin,
            scale(horizontal, column_spacing),
            scale(vertical, row_spacing),
            scale(normal, depth_spacing),
            dimensions,
            1,
            interpolation,
        )
    }

    /// Translate the plane by whole through-plane steps and validate the new
    /// corners against `volume`.
    ///
    /// # Errors
    /// Returns a typed geometry or bounds error when the translated plane is
    /// not valid for the source volume.
    pub fn shifted_along_depth(
        self,
        volume: &LoadedVolume,
        steps: f64,
    ) -> Result<Self, ResliceError> {
        if !steps.is_finite() {
            return Err(ResliceError::InvalidDepthOffset { steps });
        }
        let origin = add_scaled(self.origin, self.depth_step, steps);
        Self::try_new(
            volume,
            origin,
            self.horizontal_step,
            self.vertical_step,
            self.depth_step,
            self.dimensions,
            self.depth_samples,
            self.interpolation,
        )
    }
}

fn validate_center(center: [f64; 3], shape: [usize; 3]) -> Result<(), ResliceError> {
    if center.into_iter().zip(shape).any(|(value, size)| {
        !value.is_finite() || value < 0.0 || value > size.saturating_sub(1) as f64
    }) {
        return Err(ResliceError::InvalidCenter {
            center_voxel: center,
        });
    }
    Ok(())
}

fn output_extent(half_extent: f64, spacing: f64, maximum: usize) -> Result<usize, ResliceError> {
    let steps = (2.0 * half_extent / spacing).floor();
    if !steps.is_finite() || steps < 0.0 {
        return Err(ResliceError::InvalidFieldOfView);
    }
    let steps = steps.min(maximum.saturating_sub(1) as f64);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the extent is finite, nonnegative, and bounded by the source dimension"
    )]
    let steps = steps as usize;
    steps.checked_add(1).ok_or(ResliceError::InvalidFieldOfView)
}

fn normalize(vector: [f64; 3]) -> Result<[f64; 3], ResliceError> {
    let norm = vector_norm(vector);
    if !norm.is_finite() || norm == 0.0 {
        return Err(ResliceError::InvalidPlaneBasis);
    }
    Ok(scale(vector, norm.recip()))
}

fn rotate(vector: [f64; 3], axis: [f64; 3], degrees: f64) -> [f64; 3] {
    let angle = degrees * DEGREES_TO_RADIANS;
    let (sine, cosine) = angle.sin_cos();
    add_scaled(
        add_scaled(scale(vector, cosine), cross(axis, vector), sine),
        axis,
        dot(axis, vector) * (1.0 - cosine),
    )
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    vector.map(|component| component * factor)
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}
