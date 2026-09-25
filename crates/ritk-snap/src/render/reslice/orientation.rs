//! Orientation-controlled construction for physical reslice planes.

use super::sampling::{
    add_scaled, cross_product, patient_step_to_voxel, validate_volume, vector_norm, voxel_step,
};
use super::{ResliceError, ResliceInterpolation, ReslicePlane};
use crate::LoadedVolume;
use thiserror::Error;

const DEGREES_TO_RADIANS: f64 = std::f64::consts::PI / 180.0;

/// Failure while constructing or moving an orientation-controlled plane.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum ResliceOrientationError {
    /// The source volume or resulting plane failed reslice validation.
    #[error("reslice plane is invalid: {0}")]
    Source(#[from] ResliceError),
    /// The angles are not finite or exceed their documented ranges.
    #[error("orientation yaw {yaw_degrees} or pitch {pitch_degrees} is out of range")]
    InvalidOrientation {
        /// Requested yaw in degrees.
        yaw_degrees: f64,
        /// Requested pitch in degrees.
        pitch_degrees: f64,
    },
    /// The center lies outside the source's continuous voxel bounds.
    #[error("reslice center {center_voxel:?} lies outside the source volume")]
    InvalidCenter {
        /// Requested center in `[depth, row, column]` voxel coordinates.
        center_voxel: [f64; 3],
    },
    /// The physical field of view cannot be represented safely.
    #[error("reslice field of view is not representable")]
    InvalidFieldOfView,
    /// The plane offset is not finite.
    #[error("through-plane offset {steps} is not finite")]
    InvalidDepthOffset {
        /// Requested offset in plane-normal steps.
        steps: f64,
    },
    /// The source directions do not produce a finite, independent basis.
    #[error("source directions do not produce an independent plane basis")]
    InvalidPlaneBasis,
}

/// Yaw and pitch rotations for an oblique multiplanar plane.
///
/// Yaw rotates around the source row direction after it is projected
/// perpendicular to the source column direction; pitch rotates around the
/// resulting horizontal direction. Angles are in degrees.
///
/// # Examples
///
/// ```
/// use ritk_snap::render::ResliceOrientation;
///
/// let orientation = ResliceOrientation::try_new(30.0, -15.0)
///     .expect("angles are within the documented ranges");
/// assert_eq!(orientation.yaw_degrees(), 30.0);
/// assert_eq!(orientation.pitch_degrees(), -15.0);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq)]
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
    /// Returns [`ResliceOrientationError::InvalidOrientation`] for non-finite
    /// or out-of-range angles.
    pub fn try_new(yaw_degrees: f64, pitch_degrees: f64) -> Result<Self, ResliceOrientationError> {
        if !yaw_degrees.is_finite()
            || !pitch_degrees.is_finite()
            || !(-180.0..=180.0).contains(&yaw_degrees)
            || !(-90.0..=90.0).contains(&pitch_degrees)
        {
            return Err(ResliceOrientationError::InvalidOrientation {
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
    /// Returns [`ResliceOrientationError::InvalidOrientation`] when either
    /// delta is non-finite.
    pub fn rotated_by(
        self,
        yaw_delta_degrees: f64,
        pitch_delta_degrees: f64,
    ) -> Result<Self, ResliceOrientationError> {
        if !yaw_delta_degrees.is_finite() || !pitch_delta_degrees.is_finite() {
            return Err(ResliceOrientationError::InvalidOrientation {
                yaw_degrees: yaw_delta_degrees,
                pitch_degrees: pitch_delta_degrees,
            });
        }
        let yaw_delta = yaw_delta_degrees.rem_euclid(360.0);
        let yaw = (self.yaw_degrees + yaw_delta + 180.0).rem_euclid(360.0) - 180.0;
        let pitch = (self.pitch_degrees + pitch_delta_degrees).clamp(-90.0, 90.0);
        Self::try_new(yaw, pitch)
    }
}

impl ReslicePlane {
    /// Build a centered one-sample oblique plane within the source volume.
    ///
    /// The plane uses source column and row spacing, preserving their physical
    /// aspect ratio. Its rectangular field of view shrinks uniformly only as
    /// needed to keep every corner inside the source voxel bounds. Nearly
    /// parallel source axes use modified Gram-Schmidt with one
    /// reorthogonalization pass; this keeps the computed basis orthogonal to a
    /// small multiple of machine precision when the source affine is
    /// numerically nonsingular. See Giraud, Langou, and Rozložník, “The Loss
    /// of Orthogonality in the Gram-Schmidt Orthogonalization Process,”
    /// §2, eq. (2.2), pp. 1071–1072,
    /// <https://doi.org/10.1016/j.camwa.2005.08.009>.
    ///
    /// # Errors
    /// Returns [`ResliceOrientationError::Source`] for an invalid source or
    /// plane, [`ResliceOrientationError::InvalidCenter`] for a center outside
    /// the source, or [`ResliceOrientationError::InvalidFieldOfView`] when
    /// extents cannot be represented. It returns
    /// [`ResliceOrientationError::InvalidPlaneBasis`] when the affine cannot
    /// produce a finite, independent basis.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use ritk_snap::render::{
    /// #     ResliceInterpolation, ResliceOrientation, ResliceOrientationError,
    /// #     ReslicePlane,
    /// # };
    /// # use ritk_snap::LoadedVolume;
    /// # fn build(volume: &LoadedVolume) -> Result<ReslicePlane, ResliceOrientationError> {
    /// ReslicePlane::centered_oblique(
    ///     volume,
    ///     [12.0, 24.0, 32.0],
    ///     ResliceOrientation::default(),
    ///     ResliceInterpolation::Linear,
    /// )
    /// # }
    /// ```
    pub fn centered_oblique(
        volume: &LoadedVolume,
        center_voxel: [f64; 3],
        orientation: ResliceOrientation,
        interpolation: ResliceInterpolation,
    ) -> Result<Self, ResliceOrientationError> {
        let transform = validate_volume(volume)?;
        validate_center(center_voxel, volume.shape)?;

        let center_patient = transform.voxel_to_patient(center_voxel);
        let column_step = voxel_step(&transform, center_voxel, 2);
        let row_step = voxel_step(&transform, center_voxel, 1);
        let depth_step = voxel_step(&transform, center_voxel, 0);
        let column_spacing = vector_norm(column_step);
        let row_spacing = vector_norm(row_step);
        let depth_spacing = vector_norm(depth_step);
        let mut horizontal = normalize(column_step)?;
        let mut vertical = orthogonal_component(normalize(row_step)?, horizontal)?;

        horizontal = normalize(rotate(horizontal, vertical, orientation.yaw_degrees))?;
        vertical = orthogonal_component(vertical, horizontal)?;
        vertical = normalize(rotate(vertical, horizontal, orientation.pitch_degrees))?;
        vertical = orthogonal_component(vertical, horizontal)?;
        let normal = normalize(cross_product(horizontal, vertical))?;
        vertical = normalize(cross_product(normal, horizontal))?;

        let half_width =
            exact_voxel_index(volume.shape[2].saturating_sub(1))? * column_spacing * 0.5;
        let half_height = exact_voxel_index(volume.shape[1].saturating_sub(1))? * row_spacing * 0.5;
        let horizontal_voxel = patient_step_to_voxel(&transform, center_patient, horizontal);
        let vertical_voxel = patient_step_to_voxel(&transform, center_patient, vertical);
        let corner_extent = std::array::from_fn::<_, 3, _>(|axis| {
            horizontal_voxel[axis].abs() * half_width + vertical_voxel[axis].abs() * half_height
        });
        if !half_width.is_finite()
            || !half_height.is_finite()
            || !corner_extent.into_iter().all(f64::is_finite)
        {
            return Err(ResliceOrientationError::InvalidFieldOfView);
        }

        let source_maximum = [
            exact_voxel_index(volume.shape[0].saturating_sub(1))?,
            exact_voxel_index(volume.shape[1].saturating_sub(1))?,
            exact_voxel_index(volume.shape[2].saturating_sub(1))?,
        ];
        let scale_factor = corner_extent
            .into_iter()
            .zip(center_voxel.into_iter().zip(source_maximum))
            .filter_map(|(extent, (center, maximum))| {
                if extent > 0.0 {
                    Some(center.min(maximum - center) / extent)
                } else {
                    None
                }
            })
            .fold(1.0_f64, f64::min)
            .clamp(0.0, 1.0);
        let dimensions = [
            output_extent(half_width * scale_factor, column_spacing, volume.shape[2])?,
            output_extent(half_height * scale_factor, row_spacing, volume.shape[1])?,
        ];
        let actual_half_width =
            exact_voxel_index(dimensions[0].saturating_sub(1))? * column_spacing * 0.5;
        let actual_half_height =
            exact_voxel_index(dimensions[1].saturating_sub(1))? * row_spacing * 0.5;
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
        .map_err(Into::into)
    }

    /// Translate a plane along its normal and validate it against `volume`.
    ///
    /// This consumes and returns plane values rather than modifying one in
    /// place, so a failed translation leaves the original plane usable.
    ///
    /// # Errors
    /// Returns [`ResliceOrientationError::InvalidDepthOffset`] for a
    /// non-finite offset and [`ResliceOrientationError::Source`] when the
    /// translated plane fails source validation.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use ritk_snap::render::{ResliceOrientationError, ReslicePlane};
    /// # use ritk_snap::LoadedVolume;
    /// # fn shift(plane: ReslicePlane, volume: &LoadedVolume) -> Result<ReslicePlane, ResliceOrientationError> {
    /// plane.shifted_along_depth(volume, 1.0)
    /// # }
    /// ```
    pub fn shifted_along_depth(
        self,
        volume: &LoadedVolume,
        steps: f64,
    ) -> Result<Self, ResliceOrientationError> {
        if !steps.is_finite() {
            return Err(ResliceOrientationError::InvalidDepthOffset { steps });
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
        .map_err(Into::into)
    }
}

fn validate_center(center: [f64; 3], shape: [usize; 3]) -> Result<(), ResliceOrientationError> {
    let maximum = [
        exact_voxel_index(shape[0].saturating_sub(1))?,
        exact_voxel_index(shape[1].saturating_sub(1))?,
        exact_voxel_index(shape[2].saturating_sub(1))?,
    ];
    if center
        .into_iter()
        .zip(maximum)
        .any(|(value, maximum)| !value.is_finite() || value < 0.0 || value > maximum)
    {
        return Err(ResliceOrientationError::InvalidCenter {
            center_voxel: center,
        });
    }
    Ok(())
}

fn exact_voxel_index(index: usize) -> Result<f64, ResliceOrientationError> {
    const MAX_EXACT_INTEGER: u64 = 1_u64 << f64::MANTISSA_DIGITS;
    let index = u64::try_from(index).map_err(|_| ResliceOrientationError::InvalidFieldOfView)?;
    if index > MAX_EXACT_INTEGER {
        return Err(ResliceOrientationError::InvalidFieldOfView);
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "the bound above limits this conversion to exactly representable integer coordinates"
    )]
    let coordinate = index as f64;
    Ok(coordinate)
}

fn output_extent(
    half_extent: f64,
    spacing: f64,
    maximum: usize,
) -> Result<usize, ResliceOrientationError> {
    let steps = (2.0 * half_extent / spacing).floor();
    if !steps.is_finite() || steps < 0.0 {
        return Err(ResliceOrientationError::InvalidFieldOfView);
    }
    let steps = steps.min(exact_voxel_index(maximum.saturating_sub(1))?);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the finite nonnegative step count is bounded by a validated source dimension"
    )]
    let steps = steps as usize;
    steps
        .checked_add(1)
        .ok_or(ResliceOrientationError::InvalidFieldOfView)
}

fn normalize(vector: [f64; 3]) -> Result<[f64; 3], ResliceOrientationError> {
    let norm = vector_norm(vector);
    if !norm.is_finite() || norm == 0.0 {
        return Err(ResliceOrientationError::InvalidPlaneBasis);
    }
    Ok(scale(vector, norm.recip()))
}

fn orthogonal_component(
    candidate: [f64; 3],
    unit_axis: [f64; 3],
) -> Result<[f64; 3], ResliceOrientationError> {
    let first_residual = subtract(candidate, scale(unit_axis, dot(candidate, unit_axis)));
    let second_residual = subtract(
        first_residual,
        scale(unit_axis, dot(first_residual, unit_axis)),
    );
    normalize(second_residual)
}

fn rotate(vector: [f64; 3], axis: [f64; 3], degrees: f64) -> [f64; 3] {
    let angle = degrees * DEGREES_TO_RADIANS;
    let (sine, cosine) = angle.sin_cos();
    add_scaled(
        add_scaled(scale(vector, cosine), cross_product(axis, vector), sine),
        axis,
        dot(axis, vector) * (1.0 - cosine),
    )
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    vector.map(|component| component * factor)
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}
