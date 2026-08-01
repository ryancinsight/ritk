//! Aligning every volume of an acquisition series to one reference.
//!
//! # Scope
//!
//! This module is deliberately free of diffusion vocabulary. Motion correction
//! of a repeated acquisition is the same operation whether the volumes vary by
//! diffusion gradient, functional timepoint, or inversion time: register each
//! volume to a reference and report what moved.
//!
//! Keeping it that way also keeps the dependency direction right. A diffusion
//! consumer depends on registration; registration must not depend back on the
//! diffusion scheme. So [`register_series`] returns per-volume transforms and
//! rotations, and the caller applies them to whatever orientation-bearing data
//! it owns — for diffusion, through
//! `GradientScheme::reorient_per_volume`.
//!
//! # Why the rotation is reported separately
//!
//! Correcting the voxels is only half of a correction. Any direction-valued
//! data attached to a volume — a diffusion gradient, a tensor, an ODF — was
//! measured in that volume's original orientation, so it must move with the
//! transform. Omitting that step yields a result whose every intermediate looks
//! right and whose orientations are wrong by the per-volume residual.
//!
//! The rotation is *not* the transform's upper-left 3×3, which carries the
//! scale and shear an eddy-current correction also fits. It is the orthogonal
//! polar factor, extracted through [`rotation_from_linear`]. This module
//! reports it explicitly so a caller cannot accidentally take the raw linear
//! part.

use leto::Array3;
use ritk_spatial::rotation::{rotation_from_linear, RotationExtractionError};

use crate::classical::engine::ImageRegistration;
use crate::classical::{RegistrationError, Result};
use crate::types::AffineTransform;
use crate::validation::RegistrationQualityMetrics;

/// Which volume the rest of the series is aligned to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReferenceVolume {
    /// The first volume in acquisition order.
    ///
    /// For a diffusion series this is conventionally a b = 0 volume, which has
    /// the highest signal and no directional contrast — the reason acquisitions
    /// place one first.
    First,
    /// An explicit acquisition index.
    ///
    /// Use this when the first volume is not the best target: a series whose
    /// leading volume is corrupted, or one whose b = 0 sits elsewhere.
    Index(usize),
}

impl ReferenceVolume {
    /// Resolve to an index within a series of `count` volumes.
    fn resolve(self, count: usize) -> Result<usize> {
        let index = match self {
            Self::First => 0,
            Self::Index(index) => index,
        };
        if index >= count {
            return Err(RegistrationError::InvalidInput(format!(
                "reference volume index {index} is outside a series of {count} volumes"
            )));
        }
        Ok(index)
    }
}

/// How each volume is aligned to the reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeriesTransformModel {
    /// Six degrees of freedom: rotation and translation only.
    ///
    /// The correct model for subject motion, which cannot change the anatomy's
    /// size or shape. Fitting more than this lets noise deform the volume.
    Rigid,
    /// Nine degrees of freedom: rigid plus anisotropic scale.
    ///
    /// Eddy-current distortion is a scale and shear along the phase-encode
    /// direction that varies with the diffusion gradient, so a diffusion series
    /// needs the extra freedom this admits.
    Affine,
}

/// Series alignment configuration.
#[derive(Clone, Copy, Debug)]
pub struct SeriesRegistrationConfig {
    /// Which volume the rest align to.
    pub reference: ReferenceVolume,
    /// Degrees of freedom fitted per volume.
    pub model: SeriesTransformModel,
}

impl Default for SeriesRegistrationConfig {
    fn default() -> Self {
        Self {
            reference: ReferenceVolume::First,
            // Rigid is the safe default: it cannot deform anatomy, so a caller
            // that has not considered eddy currents does not silently receive a
            // shape-changing fit.
            model: SeriesTransformModel::Rigid,
        }
    }
}

/// Alignment of one volume to the reference.
#[derive(Clone, Debug)]
pub struct VolumeAlignment {
    /// Acquisition index of this volume.
    pub index: usize,
    /// Fitted transform mapping this volume onto the reference.
    pub transform: AffineTransform,
    /// Proper rotation carried by [`Self::transform`], scale and shear removed.
    ///
    /// Row-major. Apply this to any direction-valued data attached to the
    /// volume; do not use the transform's linear part directly.
    pub rotation: [[f64; 3]; 3],
    /// Similarity metrics for this volume's fit.
    ///
    /// `None` for the reference volume, which is assigned the identity rather
    /// than fitted. A zeroed metrics struct would claim a mutual information
    /// and correlation of zero for a registration that never ran.
    pub quality: Option<RegistrationQualityMetrics>,
}

impl VolumeAlignment {
    /// Whether this volume is the reference, which is aligned to itself.
    #[must_use]
    pub fn is_reference(&self, alignment: &SeriesAlignment) -> bool {
        self.index == alignment.reference_index
    }
}

/// Alignment of a whole series.
#[derive(Clone, Debug)]
pub struct SeriesAlignment {
    /// Index of the volume the others were aligned to.
    pub reference_index: usize,
    /// One entry per volume, in acquisition order.
    pub volumes: Vec<VolumeAlignment>,
}

impl SeriesAlignment {
    /// Per-volume rotations in acquisition order.
    ///
    /// Shaped for direct hand-off to a per-volume reorientation routine, so a
    /// caller never has to rebuild the list and risk reordering it.
    #[must_use]
    pub fn rotations(&self) -> Vec<[[f64; 3]; 3]> {
        self.volumes.iter().map(|volume| volume.rotation).collect()
    }
}

/// Identity rotation, used for the reference volume.
const IDENTITY_ROTATION: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Align every volume of `volumes` to the configured reference.
///
/// The reference is assigned the identity transform rather than being
/// registered to itself. Self-registration would return a near-identity fit
/// perturbed by optimizer noise, injecting a spurious rotation into the one
/// volume that is known to need none.
///
/// # Errors
///
/// [`RegistrationError::InvalidInput`] for an empty series or an out-of-range
/// reference index, the underlying registration error when a volume's fit
/// fails, and [`RegistrationError::InvalidInput`] when a fitted transform has
/// no extractable rotation — a reflected or rank-deficient fit, which means
/// that volume's registration failed rather than merely fitting poorly.
pub fn register_series(
    volumes: &[Array3<f64>],
    engine: &ImageRegistration,
    config: &SeriesRegistrationConfig,
) -> Result<SeriesAlignment> {
    if volumes.is_empty() {
        return Err(RegistrationError::InvalidInput(
            "cannot align an empty series".to_owned(),
        ));
    }
    let reference_index = config.reference.resolve(volumes.len())?;
    let reference = &volumes[reference_index];

    let aligned = volumes
        .iter()
        .enumerate()
        .map(|(index, volume)| {
            if index == reference_index {
                return Ok(VolumeAlignment {
                    index,
                    transform: AffineTransform::IDENTITY,
                    rotation: IDENTITY_ROTATION,
                    quality: None,
                });
            }

            let result = match config.model {
                SeriesTransformModel::Rigid => engine.rigid_registration_mutual_info(
                    volume,
                    reference,
                    &AffineTransform::IDENTITY,
                )?,
                SeriesTransformModel::Affine => engine.affine_registration_mutual_info(
                    volume,
                    reference,
                    &AffineTransform::IDENTITY,
                )?,
            };

            let rotation = rotation_of(&result.transform).map_err(|error| {
                RegistrationError::InvalidInput(format!(
                    "volume {index} produced a transform with no extractable rotation: {error}"
                ))
            })?;

            Ok(VolumeAlignment {
                index,
                transform: result.transform,
                rotation,
                quality: Some(result.quality),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(SeriesAlignment {
        reference_index,
        volumes: aligned,
    })
}

/// The proper rotation carried by a 4×4 affine's linear part.
///
/// Public so a caller holding a transform from elsewhere — a stored motion
/// estimate, a transform fitted by another routine — can reorient with the same
/// definition this module uses, rather than reaching for the raw upper-left
/// block.
///
/// # Errors
///
/// Propagates [`RotationExtractionError`] when the linear part is non-finite,
/// rank deficient, or orientation reversing.
pub fn rotation_of(
    transform: &AffineTransform,
) -> std::result::Result<[[f64; 3]; 3], RotationExtractionError> {
    let matrix = transform.as_array();
    // Row-major 4×4: the linear part is the leading 3 entries of the first
    // three rows.
    let linear = [
        [matrix[0], matrix[1], matrix[2]],
        [matrix[4], matrix[5], matrix[6]],
        [matrix[8], matrix[9], matrix[10]],
    ];
    rotation_from_linear(linear)
}

#[cfg(test)]
mod tests;
