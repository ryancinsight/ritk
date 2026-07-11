//! Native DICOM RGB colour-volume loading.
//!
//! Exposes [`load_atlas_color_series`] and [`load_atlas_color_from_series`],
//! returning `Image<f32, MoiraiBackend, 4>` with shape `[depth, rows, cols, 3]`
//! and interleaved RGB samples in the channel axis.
//!
//! Both entry points route through the substrate-agnostic
//! [`load_color_volume_flat`](super::load_color_volume_flat) core (pixel decode
//! + RGB interleave into a flat `f32` buffer + shape/metadata) and construct
//! the Coeus-backed carrier via [`Image::from_flat`]. No tensor substrate is
//! used on this path.

use coeus_core::MoiraiBackend;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};

use super::super::color_common::RGB_CHANNELS;
use super::super::reader::{DicomReadMetadata, DicomSeriesInfo};
use super::{load_color_volume_flat, load_color_volume_flat_from_path};

// ── Error semantics ───────────────────────────────────────────────────────

/// Error variants for native colour-volume loading.
///
/// The [`load_color_volume_flat`](super::load_color_volume_flat) core uses
/// `anyhow` for both recoverable and unrecoverable conditions; this surface
/// maps them to two recoverable variants (decode-passthrough + carrier
/// construction). Display text preserves the core's diagnostic prefixes so
/// callers keep their diagnostic-text contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtlasColorError {
    /// The flat-decode core signalled a recoverable colour-series error.
    Legacy(String),
    /// Carrier construction failed (shape mismatch / channel axis conflict /
    /// non-finite origin/spacing).
    Conversion(String),
}

impl std::fmt::Display for AtlasColorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Legacy(s) => write!(f, "{s}"),
            Self::Conversion(s) => write!(f, "atlas color volume carrier construction failed: {s}"),
        }
    }
}

impl std::error::Error for AtlasColorError {}

// ── Load APIs ─────────────────────────────────────────────────────────────

/// Read a DICOM RGB series into a native 4-D colour volume.
///
/// Returns `Image<f32, MoiraiBackend, 4>` with shape `[depth, rows, cols, 3]`,
/// interleaved RGB samples in the channel axis, and the series spatial
/// metadata preserved verbatim.
pub fn load_atlas_color_series<P>(
    path: P,
) -> Result<(Image<f32, MoiraiBackend, 4>, DicomReadMetadata), AtlasColorError>
where
    P: AsRef<std::path::Path>,
{
    let (values, dims, metadata) =
        load_color_volume_flat_from_path(path).map_err(|e| AtlasColorError::Legacy(format!("{e:#}")))?;
    build_carrier(values, dims, metadata)
}

/// Load a DICOM RGB colour series from a pre-scanned series descriptor.
///
/// Zero-disk counterpart of [`load_atlas_color_series`].
pub fn load_atlas_color_from_series(
    series: DicomSeriesInfo,
) -> Result<(Image<f32, MoiraiBackend, 4>, DicomReadMetadata), AtlasColorError> {
    let (values, dims, metadata) =
        load_color_volume_flat(series.metadata).map_err(|e| AtlasColorError::Legacy(format!("{e:#}")))?;
    build_carrier(values, dims, metadata)
}

/// Build the native rank-4 carrier from the flat interleaved-RGB buffer.
///
/// The three physical axes carry the series origin/spacing/direction; the
/// channel axis is a non-spatial identity axis (origin offset 0, spacing 1,
/// identity direction).
fn build_carrier(
    values: Vec<f32>,
    dims: [usize; 4],
    metadata: DicomReadMetadata,
) -> Result<(Image<f32, MoiraiBackend, 4>, DicomReadMetadata), AtlasColorError> {
    if dims[3] != RGB_CHANNELS {
        return Err(AtlasColorError::Conversion(format!(
            "RGB volume channel axis must be {RGB_CHANNELS}, got {}",
            dims[3]
        )));
    }
    let origin = Point::<4>::new([
        metadata.origin[0],
        metadata.origin[1],
        metadata.origin[2],
        0.0,
    ]);
    let spacing = Spacing::<4>::new([
        metadata.spacing[0],
        metadata.spacing[1],
        metadata.spacing[2],
        1.0,
    ]);
    let direction = Direction::<4>::identity();
    Image::<f32, MoiraiBackend, 4>::from_flat(values, dims, origin, spacing, direction)
        .map(|carrier| (carrier, metadata))
        .map_err(|e| AtlasColorError::Conversion(e.to_string()))
}
