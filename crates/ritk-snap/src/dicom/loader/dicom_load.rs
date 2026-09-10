//! DICOM series loading into LoadedVolume.

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use coeus_core::SequentialBackend;
use ritk_io::{
    load_color_multiframe_flat, load_color_multiframe_flat_from_bytes, load_color_volume_flat,
    load_dicom_from_series, load_dicom_multiframe_flat, load_dicom_multiframe_flat_from_bytes,
    read_multiframe_info, read_multiframe_info_from_bytes,
};
use tracing::info;

use crate::render::GrayscalePresentation;
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
    if multiframe_count_for_series(&series)?.is_some() {
        return load_dicom_multiframe_volume_from_scanned_series(series);
    }

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

/// Return the frame count carried by a scanned member, using its retained
/// Part-10 bytes whenever available. The scanner's retained payload is the
/// same object whose metadata was validated, so frame admission cannot drift
/// from the subsequent decode.
fn multiframe_count(slice: &ritk_io::DicomSliceMetadata) -> Result<usize> {
    let info = match slice.part10_bytes.as_deref() {
        Some(bytes) => read_multiframe_info_from_bytes(&slice.path, bytes),
        None => read_multiframe_info(&slice.path),
    }?;
    Ok(info.n_frames)
}

/// Identify a single admitted multi-frame object and reject mixed layouts.
///
/// The current `LoadedVolume` contract has one spatial depth axis. A scanned
/// series containing more than one DICOM member is therefore a conventional
/// slice series; combining it with a multi-frame member would either duplicate
/// frames or silently discard all but frame zero.
fn multiframe_count_for_series(series: &ritk_io::ScannedDicomSeries) -> Result<Option<usize>> {
    let mut count = None;
    for slice in &series.metadata.slices {
        let current = multiframe_count(slice)?;
        if current == 1 {
            continue;
        }
        if series.metadata.slices.len() != 1 {
            bail!(
                "DICOM series mixes multi-frame and single-frame members; select one multi-frame object"
            );
        }
        if count.replace(current).is_some() {
            bail!("DICOM series contains more than one multi-frame object");
        }
    }
    Ok(count)
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
    image: ritk_image::Image<f32, SequentialBackend, 3>,
    meta: ritk_io::DicomReadMetadata,
    source: Option<std::path::PathBuf>,
    backend: &SequentialBackend,
) -> Result<LoadedVolume> {
    let shape = image.shape();
    let spacing = image.spacing().to_array();
    let origin = image.origin().to_array();
    let direction = image.direction().to_row_major();
    let pixels = image.data_cow_on(backend).into_owned();
    loaded_volume_from_scalar_data(pixels, shape, spacing, origin, direction, meta, source)
}

/// Convert a substrate-free scalar buffer and resolved DICOM metadata into a
/// viewer volume. This is shared by the conventional series reader and the
/// single-object multi-frame reader.
fn loaded_volume_from_scalar_data(
    pixels: Vec<f32>,
    shape: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
    direction: [f64; 9],
    meta: ritk_io::DicomReadMetadata,
    source: Option<std::path::PathBuf>,
) -> Result<LoadedVolume> {
    let modality = meta.modality;
    let patient_name = meta.patient_name.clone();
    let patient_id = meta.patient_id.clone();
    let study_date = meta.study_date;
    let series_description = meta.series_description.clone();
    let volume = LoadedVolume {
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
    };
    GrayscalePresentation::for_volume(&volume)
        .map_err(|error| anyhow::anyhow!("invalid DICOM grayscale presentation: {error}"))?;
    Ok(volume)
}

/// Load one scalar multi-frame object through the RITK multi-frame reader.
///
/// This path is deliberately selected before the ordinary series loader, whose
/// per-slice decoder requests frame zero by contract. It preserves every
/// spatial frame and adopts the multi-frame reader's geometry and rescale
/// decisions into the viewer's format-erased volume carrier.
fn load_dicom_multiframe_volume_from_scanned_series(
    series: ritk_io::ScannedDicomSeries,
) -> Result<LoadedVolume> {
    if series.metadata.slices.len() != 1 {
        bail!(
            "DICOM multi-frame loading requires exactly one object, got {} members",
            series.metadata.slices.len()
        );
    }
    let slice = series
        .metadata
        .slices
        .first()
        .context("DICOM multi-frame series has no member")?;
    let (info_samples, info_path) = match slice.part10_bytes.as_deref() {
        Some(bytes) => (
            read_multiframe_info_from_bytes(&slice.path, bytes)?,
            slice.path.clone(),
        ),
        None => (read_multiframe_info(&slice.path)?, slice.path.clone()),
    };
    if info_samples.samples_per_pixel == 3 {
        let color = match slice.part10_bytes.as_deref() {
            Some(bytes) => load_color_multiframe_flat_from_bytes(&slice.path, bytes),
            None => load_color_multiframe_flat(&slice.path),
        }
        .with_context(|| {
            format!(
                "failed to load DICOM RGB multi-frame object {:?}",
                slice.path
            )
        })?;
        let mut metadata = series.metadata;
        metadata.dimensions = [color.shape[1], color.shape[2], color.shape[0]];
        metadata.spacing = color.spacing.to_array();
        metadata.origin = color.origin.to_array();
        metadata.direction = color.direction.to_column_major();
        return Ok(loaded_volume_from_color_flat(
            color.data,
            color.shape,
            metadata,
            None,
        ));
    }
    if info_samples.samples_per_pixel != 1 {
        bail!(
            "DICOM multi-frame viewer loading currently accepts scalar SamplesPerPixel=1; {} declares SamplesPerPixel={}",
            info_path.display(),
            info_samples.samples_per_pixel
        );
    }

    let flat = match slice.part10_bytes.as_deref() {
        Some(bytes) => load_dicom_multiframe_flat_from_bytes(&slice.path, bytes),
        None => load_dicom_multiframe_flat(&slice.path),
    }
    .with_context(|| format!("failed to load DICOM multi-frame object {:?}", slice.path))?;

    let mut metadata = series.metadata;
    metadata.dimensions = [flat.shape[1], flat.shape[2], flat.shape[0]];
    metadata.spacing = flat.spacing.to_array();
    metadata.origin = flat.origin.to_array();
    let direction = flat.direction.to_row_major();
    metadata.direction = flat.direction.to_column_major();
    loaded_volume_from_scalar_data(
        flat.data,
        flat.shape,
        metadata.spacing,
        metadata.origin,
        direction,
        metadata,
        None,
    )
}

/// Load a scalar DICOM series from a pre-scanned series descriptor.
fn load_dicom_scalar_volume_from_scanned_series(
    series: ritk_io::ScannedDicomSeries,
) -> Result<LoadedVolume> {
    let backend = SequentialBackend;
    let (image, meta) = load_dicom_from_series(series, &backend)
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

/// Load an unambiguous DICOM directory, indexed DICOMDIR, or explicitly selected file.
///
/// The provider scans once and applies series identity before pixel decoding.
/// Unsupported or ambiguous inputs return the provider error without a color-probe fallback.
///
/// # Errors
/// Returns scan, decode, or geometry errors from the DICOM provider.
pub fn load_dicom_volume<P: AsRef<Path>>(path: P) -> Result<LoadedVolume> {
    let path = path.as_ref();
    info!(path = %path.display(), "loading DICOM volume");
    let series = ritk_io::scan_dicom_path(path)
        .with_context(|| format!("failed to scan DICOM input '{}'", path.display()))?;
    let mut volume = load_volume_from_scanned_series(series)?;
    volume.source = Some(path.to_path_buf());
    Ok(volume)
}
