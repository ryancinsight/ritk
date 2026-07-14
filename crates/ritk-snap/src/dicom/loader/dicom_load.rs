//! DICOM series loading into LoadedVolume.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use coeus_core::SequentialBackend;
use ritk_io::{
    is_rgb_dicom_series, load_color_volume_flat, load_color_volume_flat_from_path,
    load_dicom_from_series, load_dicom_series_with_metadata,
};
use tracing::info;

use crate::LoadedVolume;

/// Load a DICOM series from a pre-scanned series descriptor into a [`LoadedVolume`].
///
/// This is the zero-disk counterpart of [`load_dicom_volume`]: callers that
/// have already obtained a scanned series descriptor (e.g. via
/// [`ritk_io::scan_dicom_instances`] or [`ritk_io::scan_dicom_part10_bytes`])
/// pass it directly instead of re-scanning a directory. Pixel decode uses
/// `part10_bytes` from the slice metadata when present, falling back to
/// file-path I/O otherwise.
///
/// Both scalar and RGB color series are supported.
pub fn load_volume_from_scanned_series(
    series: ritk_io::ScannedDicomSeries,
) -> Result<LoadedVolume> {
    let is_rgb = series
        .metadata
        .photometric_interpretation
        .as_deref()
        .is_some_and(|pi| pi.trim().eq_ignore_ascii_case("RGB"));
    if is_rgb {
        load_dicom_color_volume_from_scanned_series(series)
    } else {
        load_dicom_scalar_volume_from_scanned_series(series)
    }
}

/// Convert a scalar `(Image, DicomReadMetadata)` pair into a [`LoadedVolume`].
///
/// This deduplicates the image-to-LoadedVolume conversion logic shared by
/// [`load_dicom_volume`] and [`load_volume_from_scanned_series`].
///
/// # Parameters
/// - `image` — the reconstructed 3-D scalar image.
/// - `meta` — per-series DICOM metadata.
/// - `source` — optional filesystem source path (absent for SCP-received instances).
fn loaded_volume_from_scalar_image(
    image: ritk_image::native::Image<f32, SequentialBackend, 3>,
    meta: ritk_io::DicomReadMetadata,
    source: Option<std::path::PathBuf>,
    backend: &SequentialBackend,
) -> Result<LoadedVolume> {
    let shape = image.shape();
    let spacing = image.spacing().to_array();
    let origin = image.origin().to_array();
    let direction = image.direction().to_row_major();
    let pixels = image.data_cow_on(backend).into_owned();
    let modality = meta.modality;
    let patient_name = meta.patient_name.clone();
    let patient_id = meta.patient_id.clone();
    let study_date = meta.study_date;
    let series_description = meta.series_description.clone();
    Ok(LoadedVolume {
        data: Arc::new(pixels),
        shape,
        channels: 1,
        spacing,
        origin,
        direction,
        metadata: Some(Box::new(meta)),
        source,
        modality,
        patient_name,
        patient_id,
        study_date,
        series_description,
        series_time: None,
        patient_weight_kg: None,
        injected_dose_bq: None,
        radionuclide_half_life_s: None,
        radiopharmaceutical_start_time: None,
        decay_correction: None,
    })
}

/// Load a scalar DICOM series from a pre-scanned series descriptor.
fn load_dicom_scalar_volume_from_scanned_series(
    series: ritk_io::ScannedDicomSeries,
) -> Result<LoadedVolume> {
    let backend = SequentialBackend;
    let (image, meta) = load_native_dicom_from_series(series, &backend)
        .with_context(|| "failed to load DICOM series from scanned instances")?;
    loaded_volume_from_scalar_image(image, meta, None, &backend)
}

/// Build a colour [`LoadedVolume`] from the substrate-free flat RGB core.
///
/// `flat` is the interleaved-RGB `f32` buffer produced by
/// [`ritk_io::load_color_volume_flat`], `dims` its `[depth, rows, cols, 3]`
/// shape. Spatial metadata is taken verbatim from `meta`; the direction array
/// is the row-major readout of the column-major direction cosines, matching
/// the physical-axis convention used by the scalar loader.
fn loaded_volume_from_color_flat(
    flat: Vec<f32>,
    dims: [usize; 4],
    meta: ritk_io::DicomReadMetadata,
    source: Option<std::path::PathBuf>,
) -> LoadedVolume {
    let [depth, rows, cols, _channels] = dims;
    let shape = [depth, rows, cols];
    let spacing = meta.spacing;
    let origin = meta.origin;
    // `meta.direction` is column-major; the scalar path and the former
    // `RgbVolume` carrier both surface direction cosines in row-major order,
    // i.e. the transpose of the stored column-major array.
    let d = meta.direction;
    let direction = [d[0], d[3], d[6], d[1], d[4], d[7], d[2], d[5], d[8]];
    let modality = meta.modality;
    let patient_name = meta.patient_name.clone();
    let patient_id = meta.patient_id.clone();
    let study_date = meta.study_date;
    let series_description = meta.series_description.clone();
    LoadedVolume {
        data: Arc::new(flat),
        shape,
        channels: 3,
        spacing,
        origin,
        direction,
        metadata: Some(Box::new(meta)),
        source,
        modality,
        patient_name,
        patient_id,
        study_date,
        series_description,
        series_time: None,
        patient_weight_kg: None,
        injected_dose_bq: None,
        radionuclide_half_life_s: None,
        radiopharmaceutical_start_time: None,
        decay_correction: None,
    }
}

/// Load an RGB DICOM colour series from a pre-scanned series descriptor.
fn load_dicom_color_volume_from_scanned_series(
    series: ritk_io::ScannedDicomSeries,
) -> Result<LoadedVolume> {
    let (flat, dims, meta) = load_color_volume_flat(series.metadata)
        .with_context(|| "failed to load DICOM RGB series from scanned instances")?;
    Ok(loaded_volume_from_color_flat(flat, dims, meta, None))
}

/// Load a DICOM series from `folder` into a [`LoadedVolume`].
///
/// # Algorithm
/// 1. Detects whether the series is RGB via [`is_rgb_dicom_series`].
/// 2. For RGB series: calls [`ritk_io::load_color_volume_flat_from_path`] and
///    builds a [`LoadedVolume`] with `channels: 3` from the flat RGB buffer.
/// 3. For scalar series: calls [`load_dicom_series_with_metadata`].
/// 4. Extracts spatial metadata (spacing, origin, direction) from the image.
/// 5. Populates optional DICOM-specific fields from `DicomReadMetadata`.
///
/// # Errors
/// Propagates any error returned by `ritk_io`.
pub fn load_dicom_volume<P: AsRef<Path>>(folder: P) -> Result<LoadedVolume> {
    let folder = folder.as_ref();
    info!(path = %folder.display(), "loading DICOM volume");

    // Detect RGB colour series and route to the colour-volume loader.
    if is_rgb_dicom_series(folder).unwrap_or(false) {
        return load_dicom_color_volume(folder);
    }

    let backend = SequentialBackend;
    let (image, meta) = load_native_dicom_series_with_metadata(folder, &backend)
        .with_context(|| format!("failed to load DICOM series from '{}'", folder.display()))?;
    loaded_volume_from_scalar_image(image, meta, Some(folder.to_path_buf()), &backend)
}

/// Load an RGB DICOM colour series into a [`LoadedVolume`] with `channels: 3`.
///
/// Routes through the substrate-free [`ritk_io::load_color_volume_flat_from_path`]
/// core: no tensor backend is constructed on this path.
fn load_dicom_color_volume(folder: &Path) -> Result<LoadedVolume> {
    info!(path = %folder.display(), "loading DICOM RGB colour volume");

    let (flat, dims, meta) = load_color_volume_flat_from_path(folder).with_context(|| {
        format!(
            "failed to load DICOM RGB series from '{}'",
            folder.display()
        )
    })?;

    Ok(loaded_volume_from_color_flat(
        flat,
        dims,
        meta,
        Some(folder.to_path_buf()),
    ))
}
