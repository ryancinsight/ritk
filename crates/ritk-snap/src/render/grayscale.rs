//! DICOM grayscale presentation semantics.
//!
//! This module is the single implementation of window/level mapping and the
//! presentation metadata that surrounds it. Pixel decoding and modality
//! rescale happen at the RITK IO boundary; this layer maps the resulting
//! modality values to display samples.

use thiserror::Error;

use crate::LoadedVolume;
use ritk_io::{DicomObjectModel, DicomReadMetadata, DicomSliceMetadata, DicomTag};

const VOI_LUT_FUNCTION_TAG: DicomTag = DicomTag::new(0x0028, 0x1056);
const VOI_LUT_SEQUENCE_TAG: DicomTag = DicomTag::new(0x0028, 0x3010);

/// DICOM VOI LUT Function values admitted by the viewer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiLutFunction {
    /// DICOM default windowing function.
    Linear,
    /// Windowing function with exact centre and width boundaries.
    LinearExact,
    /// Logistic sigmoid windowing function.
    Sigmoid,
}

impl VoiLutFunction {
    /// Return the compact discriminant used by the GPU presentation uniform.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) const fn gpu_code(self) -> u32 {
        match self {
            Self::Linear => 0,
            Self::LinearExact => 1,
            Self::Sigmoid => 2,
        }
    }

    fn parse(value: &str) -> Result<Self, GrayscalePresentationError> {
        let normalized = value.trim().to_ascii_uppercase();
        match normalized.as_str() {
            "LINEAR" => Ok(Self::Linear),
            "LINEAR_EXACT" => Ok(Self::LinearExact),
            "SIGMOID" => Ok(Self::Sigmoid),
            _ => Err(GrayscalePresentationError::UnsupportedVoiFunction {
                value: value.trim().to_owned(),
            }),
        }
    }
}

/// Failure while resolving DICOM grayscale presentation metadata.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GrayscalePresentationError {
    /// A scalar image declares a photometric interpretation outside the two
    /// monochrome forms handled by the viewer.
    #[error(
        "unsupported scalar DICOM photometric interpretation {value:?}; expected MONOCHROME1 or MONOCHROME2"
    )]
    UnsupportedPhotometric {
        /// Declared DICOM Photometric Interpretation value.
        value: String,
    },
    /// The object declares a VOI LUT Function outside the admitted set.
    #[error(
        "unsupported DICOM VOI LUT Function {value:?}; accepted values are LINEAR, LINEAR_EXACT, and SIGMOID"
    )]
    UnsupportedVoiFunction {
        /// Declared DICOM VOI LUT Function value.
        value: String,
    },
    /// The object declares the table-based VOI LUT Sequence, which has no
    /// implementation in the current viewer presentation path.
    #[error("DICOM VOI LUT Sequence is not supported by the scalar viewer")]
    UnsupportedVoiLutSequence,
    /// Explicit VOI function values disagree across the loaded slices.
    #[error("DICOM VOI LUT Function differs across slices")]
    InconsistentVoiFunction,
}

/// Window centre and width used for display mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowLevel {
    /// Display window centre — the midpoint of the visible intensity range.
    pub center: f64,
    /// Display window width — the span of the visible intensity range.
    pub width: f64,
}

impl WindowLevel {
    /// Construct a window/level pair without changing the supplied values.
    pub const fn new(center: f64, width: f64) -> Self {
        Self { center, width }
    }

    /// Apply the DICOM default `LINEAR` function to one modality value.
    ///
    /// The default function follows DICOM PS3.3 C.11.2.1.2: the lower and
    /// upper thresholds are `c - 0.5 - (w - 1) / 2` and
    /// `c - 0.5 + (w - 1) / 2`; the upper threshold is exclusive. Widths
    /// below one are clamped to one because DICOM requires `w >= 1`.
    #[inline]
    pub fn apply(&self, value: f64) -> u8 {
        self.apply_with_function(value, VoiLutFunction::Linear)
    }

    /// Apply DICOM `LINEAR_EXACT` to one modality value.
    #[inline]
    pub fn apply_linear_exact(&self, value: f64) -> u8 {
        self.apply_with_function(value, VoiLutFunction::LinearExact)
    }

    /// Apply one admitted DICOM VOI LUT Function to one modality value.
    #[inline]
    pub fn apply_with_function(&self, value: f64, function: VoiLutFunction) -> u8 {
        if value.is_nan() {
            return 0;
        }
        let width = self.width.max(1.0);
        let normalized = match function {
            VoiLutFunction::Linear => linear(value, self.center, width),
            VoiLutFunction::LinearExact => linear_exact(value, self.center, width),
            VoiLutFunction::Sigmoid => sigmoid(value, self.center, width),
        };
        // All branches return a finite value in [0, 1]. Keeping the clamp at
        // this boundary also protects the narrowing conversion from arithmetic
        // round-off at a threshold.
        let scaled = (normalized.clamp(0.0, 1.0) * 255.0).round();
        #[expect(clippy::cast_possible_truncation, reason = "normalized display byte")]
        let byte = scaled as u8;
        byte
    }

    /// Apply the DICOM default function to every scalar sample.
    pub fn apply_slice(&self, pixels: &[f32]) -> Vec<u8> {
        pixels
            .iter()
            .map(|&pixel| self.apply(f64::from(pixel)))
            .collect()
    }
}

fn linear(value: f64, center: f64, width: f64) -> f64 {
    let lower = center - 0.5 - (width - 1.0) * 0.5;
    let upper = center - 0.5 + (width - 1.0) * 0.5;
    if value <= lower {
        0.0
    } else if value > upper {
        1.0
    } else {
        ((value - (center - 0.5)) / (width - 1.0) + 0.5).clamp(0.0, 1.0)
    }
}

fn linear_exact(value: f64, center: f64, width: f64) -> f64 {
    let lower = center - width * 0.5;
    let upper = center + width * 0.5;
    if value <= lower {
        0.0
    } else if value > upper {
        1.0
    } else {
        ((value - center) / width + 0.5).clamp(0.0, 1.0)
    }
}

fn sigmoid(value: f64, center: f64, width: f64) -> f64 {
    // This is DICOM's 4× logistic slope. `exp` maps infinities to the correct
    // limiting display value, so no input-dependent branch or panic is needed.
    1.0 / (1.0 + (4.0 * (center - value) / width).exp())
}

/// Resolved presentation metadata for one scalar volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrayscalePresentation {
    /// Function used to map modality values into display samples.
    pub voi_function: VoiLutFunction,
    /// Whether the mapped display sample is inverted for MONOCHROME1.
    pub invert: bool,
}

impl GrayscalePresentation {
    /// Resolve monochrome and VOI metadata from a loaded volume.
    ///
    /// A non-DICOM volume has no metadata and therefore uses the DICOM
    /// defaults `MONOCHROME2` and `LINEAR`. DICOM metadata is validated across
    /// every admitted slice so a later slice cannot silently change the
    /// display function selected for the volume.
    pub fn for_volume(volume: &LoadedVolume) -> Result<Self, GrayscalePresentationError> {
        let Some(metadata) = volume.metadata.as_deref() else {
            return Ok(Self::default());
        };
        Self::from_metadata(metadata)
    }

    /// Resolve presentation metadata from a DICOM metadata record.
    pub fn from_metadata(metadata: &DicomReadMetadata) -> Result<Self, GrayscalePresentationError> {
        let photometric = metadata
            .photometric_interpretation
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("MONOCHROME2");
        let invert = match photometric.to_ascii_uppercase().as_str() {
            "MONOCHROME1" => true,
            "MONOCHROME2" => false,
            value => {
                return Err(GrayscalePresentationError::UnsupportedPhotometric {
                    value: value.to_owned(),
                });
            }
        };

        let mut resolved = None;
        for slice in &metadata.slices {
            let function = resolve_slice_function(slice)?;
            let function = function.unwrap_or(VoiLutFunction::Linear);
            if let Some(previous) = resolved {
                if previous != function {
                    return Err(GrayscalePresentationError::InconsistentVoiFunction);
                }
            } else {
                resolved = Some(function);
            }
        }

        // A manually assembled or multi-frame metadata record may carry the
        // function on the series preservation object rather than a slice.
        let series_function = resolve_object_function(&metadata.preservation.object)?;
        if let Some(function) = series_function {
            if let Some(previous) = resolved {
                if previous != function {
                    return Err(GrayscalePresentationError::InconsistentVoiFunction);
                }
            } else {
                resolved = Some(function);
            }
        }
        Ok(Self {
            voi_function: resolved.unwrap_or(VoiLutFunction::Linear),
            invert,
        })
    }

    /// Map a modality value through the selected window and presentation.
    #[inline]
    pub fn apply(self, window: WindowLevel, value: f64) -> u8 {
        let mapped = window.apply_with_function(value, self.voi_function);
        if self.invert {
            255_u8.saturating_sub(mapped)
        } else {
            mapped
        }
    }

    /// Return the compact presentation bitfield used by the GPU uniform.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) const fn gpu_code(self) -> u32 {
        self.voi_function.gpu_code() | if self.invert { 4 } else { 0 }
    }
}

impl Default for GrayscalePresentation {
    fn default() -> Self {
        Self {
            voi_function: VoiLutFunction::Linear,
            invert: false,
        }
    }
}

fn resolve_slice_function(
    slice: &DicomSliceMetadata,
) -> Result<Option<VoiLutFunction>, GrayscalePresentationError> {
    resolve_object_function(&slice.preservation.object)
}

fn resolve_object_function(
    object: &DicomObjectModel,
) -> Result<Option<VoiLutFunction>, GrayscalePresentationError> {
    if object.get(VOI_LUT_SEQUENCE_TAG).is_some() {
        return Err(GrayscalePresentationError::UnsupportedVoiLutSequence);
    }
    let Some(node) = object.get(VOI_LUT_FUNCTION_TAG) else {
        return Ok(None);
    };
    let Some(value) = node.value.as_text() else {
        return Err(GrayscalePresentationError::UnsupportedVoiFunction {
            value: "non-text value".to_owned(),
        });
    };
    if value.contains('\\') {
        return Err(GrayscalePresentationError::UnsupportedVoiFunction {
            value: value.trim().to_owned(),
        });
    }
    VoiLutFunction::parse(value).map(Some)
}

#[cfg(test)]
#[path = "tests_grayscale.rs"]
mod tests;
