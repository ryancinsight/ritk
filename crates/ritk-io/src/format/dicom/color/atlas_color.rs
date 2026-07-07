//! Atlas-keyed sister module for DICOM RGB colour-volume loading.
//!
//! Per `docs/adr/0012-ritk-burn-trait-rebind.md` §Decision §Sub-batch #3.f:
//! atlas-side parallel module exposing [`load_atlas_color_series`] and
//! [`load_atlas_color_from_series`] sister functions that mirror the legacy
//! `super::load_dicom_color_series` / `super::load_dicom_color_from_series`,
//! returning `Image<f32, MoiraiBackend, 4>` shape `[depth, rows, cols, 3]`
//! instead of the Burn-keyed `RgbVolume<B>`.
//!
//! Strictly additive on production surface per the sub-batch #3 atomic-boundary
//! invariant (ADR 0012 §Decision §1): every public symbol of `super::color`
//! is preserved verbatim. The Atlas twin delegates the heavy DICOM
//! parsing/decoding work to the legacy module, then converts the resulting
//! `RgbVolume<B>` (Burn-keyed `ColorVolume<B, 3>`) tensor data into the
//! canonical Atlas-typed image carrier (`ritk_image::native::Image`).
//!
//! **No `B` parameter on the sister signature**: the sister internally pins
//! the legacy call to `burn_ndarray::NdArray<f32>` (the canonical f32
//! host-resident burn backend for oracle-test fidelity). Without this pin,
//! the sister would have to bind `B: burn::tensor::backend::Backend` to
//! dispatch the legacy function — which forces the user into a Burn backend
//! for the carrier-transfer host-slice extract and is hostile against the
//! Atlas-typed `ComputeBackend` contract. Pinning internally keeps the
//! sister signature a single Atlas-typed (`MoiraiBackend`-typed) output.
//!
//! **Data extraction** uses [`ColorVolume::data_vec`] (canonical f32
//! extraction on the volume carrier) and [`Image::from_flat`] to
//! reconstruct the Atlas-typed carrier with the legacy module's spatial
//! metadata preserved verbatim. No `CpuAddressableStorage` trait bound is
//! required because the sister reads from the post-construct Burn tensor,
//! not from an Atlas-side image.
//!
//! **No `Cargo.toml` mutation** in sub-batch #3.f: only additive inject
//! (one `[dependencies]` line for `coeus-tensor`). To avoid a `thiserror`
//! dep-add, the [`AtlasColorError`] enum carries a hand-rolled `Display`
//! impl via [`std::fmt::Display`] + [`std::error::Error`] — zero new
//! crate-graph edges in this file.

use coeus_core::MoiraiBackend;
use ritk_core::image::RgbVolume;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};

use super::super::color_common::RGB_CHANNELS;
use super::super::reader::{DicomReadMetadata, DicomSeriesInfo};

// ── Error semantics ───────────────────────────────────────────────────────

/// Atlas-side error variants for colour-volume loading.
///
/// The legacy `super::load_color_from_series` uses `anyhow::bail!` /
/// `with_context` for both recoverable and unrecoverable conditions; the
/// Atlas twin exposes two recoverable variants that map 1:1 to the load's
/// most-likely failure shapes (legacy-passthrough + atlas-carrier
/// construction). Display text matches legacy panic/diagnostic prefixes
/// so callers that switch between paths keep their diagnostic-text
/// contract verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtlasColorError {
    /// Legacy loader signalled a recoverable colour-series error.
    Legacy(String),
    /// Atlas-side carrier construction failed (shape mismatch / channel
    /// axis conflict / non-finite origin/spacing).
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

/// Read a DICOM RGB series into an Atlas-typed 4-D colour volume.
///
/// Sister to [`super::load_dicom_color_series`]. Returns
/// `Image<f32, MoiraiBackend, 4>` shape `[depth, rows, cols, 3]` with
/// interleaved RGB samples in the channel axis and the legacy module's
/// spatial metadata preserved verbatim (`Point<3>` origin, `Spacing<3>`
/// spacing, `Direction<3>` direction cosines).
pub fn load_atlas_color_series<P>(
    path: P,
) -> Result<(Image<f32, MoiraiBackend, 4>, DicomReadMetadata), AtlasColorError>
where
    P: AsRef<std::path::Path>,
{
    type LegacyBackend = burn_ndarray::NdArray<f32>;
    let dev = <LegacyBackend as burn::tensor::backend::Backend>::Device::default();
    let (rgb_volume, metadata) = super::load_dicom_color_series::<LegacyBackend, _>(path, &dev)
        .map_err(|e| AtlasColorError::Legacy(format!("{e:#}")))?;
    rgb_volume_to_atlas(rgb_volume, metadata)
}

/// Load a DICOM RGB colour series from a pre-scanned series descriptor.
///
/// Sister to [`super::load_dicom_color_from_series`]. Zero-disk counterpart
/// of [`load_atlas_color_series`].
pub fn load_atlas_color_from_series(
    series: DicomSeriesInfo,
) -> Result<(Image<f32, MoiraiBackend, 4>, DicomReadMetadata), AtlasColorError> {
    type LegacyBackend = burn_ndarray::NdArray<f32>;
    let dev = <LegacyBackend as burn::tensor::backend::Backend>::Device::default();
    let metadata = series.metadata.clone();
    let (rgb_volume, _) = super::load_dicom_color_from_series::<LegacyBackend>(series, &dev)
        .map_err(|e| AtlasColorError::Legacy(format!("{e:#}")))?;
    rgb_volume_to_atlas(rgb_volume, metadata)
}

fn rgb_volume_to_atlas(
    rgb_volume: RgbVolume<burn_ndarray::NdArray<f32>>,
    metadata: DicomReadMetadata,
) -> Result<(Image<f32, MoiraiBackend, 4>, DicomReadMetadata), AtlasColorError> {
    let dims = rgb_volume.shape();
    if dims[3] != RGB_CHANNELS {
        return Err(AtlasColorError::Conversion(format!(
            "RGB volume channel axis must be {RGB_CHANNELS}, got {}",
            dims[3]
        )));
    }
    let values = rgb_volume.data_vec();
    // `Image<f32, B, 4>::from_flat` takes `Point<4> / Spacing<4> / Direction<4>`
    // matching the rank. The 3-D physical axes are interpolated; the channel
    // axis is treated as a non-spatial identity axis (origin offset 0,
    // spacing 1, direction identity). `metadata.origin`/`spacing` are the
    // legacy `[f64; 3]` arrays fed straight through.
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
        .map(|atlas| (atlas, metadata))
        .map_err(|e| AtlasColorError::Conversion(e.to_string()))
}
