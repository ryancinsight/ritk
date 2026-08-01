//! Standard DICOM diffusion-gradient extraction.
//!
//! DICOM PS3.3 defines Diffusion b-value `(0018,9087)` in s/mm² and
//! Diffusion Gradient Orientation `(0018,9089)` as direction cosines in the
//! patient frame. RITK patient coordinates are physical LPS, so successful
//! extraction produces [`GradientFrame::Lps`].
//!
//! Standard diffusion metadata normally appears in per-frame functional
//! groups for enhanced multi-frame MR. Top-level diffusion elements in classic
//! single-frame instances are less common and can reflect vendor-specific
//! conventions. This module supports only those standard top-level elements;
//! it rejects enhanced functional groups and private conventions rather than
//! guessing sequence or manufacturer semantics.

use std::path::Path;

use anyhow::{Context, Result};
use ritk_diffusion_scheme::{
    GradientFrame, GradientScheme, DEFAULT_B0_THRESHOLD_SECONDS_PER_SQUARE_MILLIMETER,
};
use ritk_spatial::Vector;

use crate::attribute::{tags, DicomAttributeRead};

/// Extract one external `(b-value, gradient)` pair from a DICOM object.
///
/// Returns `None` when neither standard top-level element is present.
///
/// # Errors
///
/// Returns an error when orientation is present without a b-value, orientation
/// is absent above the baseline threshold, the b-value cannot be decoded, the
/// direction does not contain exactly three components, or any component is
/// non-finite. A b-value at or below RITK's default 50 s/mm² baseline
/// threshold may omit orientation and is canonicalized by [`GradientScheme`].
/// Full unit and zero/unit-vector validation occurs when the pair enters the
/// scheme.
pub fn extract_diffusion_pair(
    object: &impl DicomAttributeRead,
) -> Result<Option<(f64, Vector<3>)>> {
    let weighting = object
        .optional_decimal(tags::DIFFUSION_B_VALUE, "DiffusionBValue")
        .context("failed to read Diffusion b-value")?;
    let direction = object
        .optional_decimal_vec(
            tags::DIFFUSION_GRADIENT_DIRECTION,
            "DiffusionGradientDirection",
        )
        .context("failed to read Diffusion Gradient Orientation")?;

    match (weighting, direction) {
        (None, None) => Ok(None),
        (Some(value), Some(components)) => {
            let components: [f64; 3] = components.try_into().map_err(|values: Vec<f64>| {
                anyhow::anyhow!(
                    "Diffusion Gradient Orientation must contain 3 components, got {}",
                    values.len()
                )
            })?;
            if !value.is_finite() || components.iter().any(|component| !component.is_finite()) {
                return Err(anyhow::anyhow!(
                    "Diffusion metadata must be finite: b={value}, direction={components:?}"
                ));
            }
            Ok(Some((value, Vector::new(components))))
        }
        (Some(value), None)
            if value.is_finite()
                && (0.0..=DEFAULT_B0_THRESHOLD_SECONDS_PER_SQUARE_MILLIMETER).contains(&value) =>
        {
            Ok(Some((value, Vector::new([0.0, 0.0, 0.0]))))
        }
        (Some(value), None) => Err(anyhow::anyhow!(
            "Diffusion b-value {value} exceeds the baseline threshold but Diffusion Gradient Orientation is missing"
        )),
        (None, Some(_)) => Err(anyhow::anyhow!(
            "Diffusion Gradient Orientation is present but Diffusion b-value is missing"
        )),
    }
}

/// Read one classic single-frame DICOM instance as a one-volume scheme.
///
/// # Errors
///
/// Returns an error when the file cannot be parsed, lacks a standard top-level
/// b-value, lacks orientation above the baseline threshold, or violates the
/// validated scheme contract. A baseline instance at or below 50 s/mm² may omit
/// orientation.
pub fn read_dicom_gradient_scheme_from_file<P: AsRef<Path>>(path: P) -> Result<GradientScheme> {
    read_dicom_gradient_scheme_from_files([path])
}

/// Read one representative classic DICOM file per acquisition volume.
///
/// `paths` order is acquisition order. This explicit surface avoids guessing
/// volume grouping from directories containing multiple slices per volume.
///
/// # Errors
///
/// Returns an error for an empty path sequence, an unreadable instance,
/// absent or malformed standard diffusion metadata, or a scheme invariant
/// violation.
pub fn read_dicom_gradient_scheme_from_files<I, P>(paths: I) -> Result<GradientScheme>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    let mut pairs = Vec::new();
    for (index, path) in paths.into_iter().enumerate() {
        let path = path.as_ref();
        let object = dicom::object::open_file(path)
            .with_context(|| format!("cannot open diffusion DICOM instance {path:?}"))?;
        let pair = extract_diffusion_pair(&object)?.ok_or_else(|| {
            anyhow::anyhow!(
                "diffusion DICOM instance {path:?} at acquisition index {index} lacks standard top-level diffusion metadata"
            )
        })?;
        pairs.push(pair);
    }
    GradientScheme::from_seconds_per_square_millimeter(pairs, GradientFrame::Lps)
        .map_err(anyhow::Error::from)
}

#[cfg(test)]
mod tests;
