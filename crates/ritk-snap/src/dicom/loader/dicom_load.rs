//! DICOM series loading into LoadedVolume.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
#[cfg(windows)]
use ritk_io::scan_dicom_files_with_budget;
use ritk_io::{
    decode_dicom_series, load_color_multiframe_flat, load_color_multiframe_flat_from_bytes,
    load_color_volume_flat, load_dicom_multiframe_flat, load_dicom_multiframe_flat_from_bytes,
    load_enhanced_pet_volumes_from_instances, read_multiframe_info,
    read_multiframe_info_from_bytes, EnhancedPetSelection,
};
use tracing::info;

use crate::LoadedVolume;

#[cfg(any(windows, target_arch = "wasm32"))]
const SERIES_PREVIEW_DECODED_BYTE_BUDGET: usize = 64 * 1024 * 1024;

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
    if is_enhanced_pet(&series) {
        return load_enhanced_pet_from_scanned_series(series);
    }
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

#[cfg(windows)]
pub(crate) fn load_volume_from_dicom_instance(path: &Path) -> Result<LoadedVolume> {
    let paths = [path.to_path_buf()];
    load_preview(|budget| scan_dicom_files_with_budget(&paths, budget))
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn load_volume_from_preview_bytes(name: &str, bytes: &[u8]) -> Result<LoadedVolume> {
    load_preview(|budget| ritk_io::scan_dicom_part10_bytes_with_budget(&[(name, bytes)], budget))
}

#[cfg(any(windows, target_arch = "wasm32"))]
fn load_preview(
    scan: impl FnOnce(&ritk_io::DicomReadBudget) -> Result<ritk_io::ScannedDicomSeries>,
) -> Result<LoadedVolume> {
    let budget = ritk_io::DicomReadBudget::try_new(
        ritk_io::DicomReadBudget::DEFAULT.parser(),
        SERIES_PREVIEW_DECODED_BYTE_BUDGET,
        SERIES_PREVIEW_DECODED_BYTE_BUDGET,
    )?;
    let series = scan(&budget).context("scan DICOM preview instance")?;
    let decoded = decode_dicom_series(series, &budget).context("decode DICOM preview instance")?;
    loaded_volume_from_decoded_series(decoded)
}

/// Return the frame count carried by a scanned member, using its retained
/// Part-10 bytes whenever available. The scanner's retained payload is the
/// same object whose metadata was validated, so frame admission cannot drift
/// from the subsequent decode.
fn is_enhanced_pet(series: &ritk_io::ScannedDicomSeries) -> bool {
    series
        .metadata
        .slices
        .first()
        .and_then(|slice| slice.sop_class_uid.as_deref())
        == Some("1.2.840.10008.5.1.4.1.1.130")
}

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
            anyhow::bail!(
                "DICOM series mixes multi-frame and single-frame members; select one multi-frame object"
            );
        }
        if count.replace(current).is_some() {
            anyhow::bail!("DICOM series contains more than one multi-frame object");
        }
    }
    Ok(count)
}

/// Transfer the reconstructed CPU pixels into the viewer without duplicating
/// the volume. The shape and physical coordinates come from the same decode.
fn loaded_volume_from_decoded_series(decoded: ritk_io::DecodedDicomSeries) -> Result<LoadedVolume> {
    loaded_volume_from_scalar_data(
        Arc::new(decoded.volume),
        decoded.shape,
        decoded.spacing.to_array(),
        decoded.origin.to_array(),
        decoded.direction.to_row_major(),
        decoded.metadata,
        None,
    )
}

/// Convert a substrate-free scalar buffer and resolved DICOM metadata into a
/// viewer volume. This is shared by the conventional series reader and the
/// single-object multi-frame reader.
fn loaded_volume_from_scalar_data(
    pixels: Arc<Vec<f32>>,
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
    let series_time = meta.series_time;
    let patient_weight_kg = meta.patient_weight_kg;
    let injected_dose_bq = meta.radionuclide_total_dose_bq;
    let radionuclide_half_life_s = meta.radionuclide_half_life_s;
    let radiopharmaceutical_start_time = meta.radiopharmaceutical_start_time;
    let decay_correction = meta.decay_correction;
    let volume = LoadedVolume {
        data: pixels,
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
        series_time,
        patient_weight_kg,
        injected_dose_bq,
        radionuclide_half_life_s,
        radiopharmaceutical_start_time,
        decay_correction,
    };
    crate::render::GrayscalePresentation::for_volume(&volume)
        .map_err(|error| anyhow::anyhow!("invalid DICOM grayscale presentation: {error}"))?;
    Ok(volume)
}

fn load_enhanced_pet_from_scanned_series(
    series: ritk_io::ScannedDicomSeries,
) -> Result<LoadedVolume> {
    let volumes = load_enhanced_pet_volumes_from_instances(
        &series.metadata.slices,
        &ritk_io::DicomReadBudget::DEFAULT,
    )?;
    let mut meta = series.metadata;
    let selection =
        EnhancedPetSelection::new(volumes).context("Enhanced PET has no spatial volume")?;
    let first = &selection.volumes()[0];
    let data = Arc::clone(&first.data);
    let shape = first.shape;
    let spacing = first.spacing.to_array();
    let origin = first.origin.to_array();
    let direction = first.direction.to_row_major();
    meta.enhanced_pet_selection = Some(selection);
    let mut loaded =
        loaded_volume_from_scalar_data(data, shape, spacing, origin, direction, meta, None)?;
    loaded
        .select_enhanced_pet_volume(0)
        .context("Enhanced PET initial volume is unavailable")?;
    Ok(loaded)
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
        anyhow::bail!(
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
        metadata.spacing = color.spacing.to_array().map(
            aequitas::systems::si::quantities::Length::from_unit::<
                aequitas::systems::si::units::Millimeter,
            >,
        );
        metadata.origin = color.origin.to_array().map(
            aequitas::systems::si::quantities::Length::from_unit::<
                aequitas::systems::si::units::Millimeter,
            >,
        );
        metadata.direction = color.direction.to_column_major();
        return Ok(loaded_volume_from_color_flat(
            color.data,
            color.shape,
            metadata,
            None,
        ));
    }
    if info_samples.samples_per_pixel != 1 {
        anyhow::bail!(
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
    metadata.enhanced_pet_suv = flat.suv_bw;
    metadata.dimensions = [flat.shape[1], flat.shape[2], flat.shape[0]];
    metadata.spacing = flat.spacing.to_array().map(
        aequitas::systems::si::quantities::Length::from_unit::<
            aequitas::systems::si::units::Millimeter,
        >,
    );
    metadata.origin = flat.origin.to_array().map(
        aequitas::systems::si::quantities::Length::from_unit::<
            aequitas::systems::si::units::Millimeter,
        >,
    );
    let direction = flat.direction.to_row_major();
    metadata.direction = flat.direction.to_column_major();
    loaded_volume_from_scalar_data(
        Arc::new(flat.data),
        flat.shape,
        metadata.spacing_mm(),
        metadata.origin_mm(),
        direction,
        metadata,
        None,
    )
}

/// Load a scalar DICOM series from a pre-scanned series descriptor.
fn load_dicom_scalar_volume_from_scanned_series(
    series: ritk_io::ScannedDicomSeries,
) -> Result<LoadedVolume> {
    let decoded = decode_dicom_series(series, &ritk_io::DicomReadBudget::DEFAULT)
        .with_context(|| "failed to load DICOM series from scanned instances")?;
    loaded_volume_from_decoded_series(decoded)
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
    let spacing = meta.spacing_mm();
    let origin = meta.origin_mm();
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
    let series_time = meta.series_time;
    let patient_weight_kg = meta.patient_weight_kg;
    let injected_dose_bq = meta.radionuclide_total_dose_bq;
    let radionuclide_half_life_s = meta.radionuclide_half_life_s;
    let radiopharmaceutical_start_time = meta.radiopharmaceutical_start_time;
    let decay_correction = meta.decay_correction;
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
        series_time,
        patient_weight_kg,
        injected_dose_bq,
        radionuclide_half_life_s,
        radiopharmaceutical_start_time,
        decay_correction,
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

/// Load one acquisition from a previously discovered series descriptor.
///
/// The exact member list and SeriesInstanceUID are re-scanned before pixels
/// are decoded. This keeps a user selection tied to the files and identity
/// that discovery reported, even if the directory changes between the two
/// operations.
pub(crate) fn load_volume_from_series_info(
    info: &ritk_io::DicomSeriesInfo,
) -> Result<LoadedVolume> {
    if info.file_paths.is_empty() {
        anyhow::bail!("selected DICOM series has no files");
    }
    let scanned = ritk_io::scan_dicom_files(&info.file_paths)
        .with_context(|| "failed to re-scan selected DICOM series members")?;
    if scanned.metadata.series_instance_uid.as_deref() != Some(info.series_instance_uid()) {
        anyhow::bail!("selected DICOM SeriesInstanceUID changed since discovery");
    }
    let mut expected: Vec<_> = info.file_paths.iter().collect();
    let mut actual: Vec<_> = scanned
        .metadata
        .slices
        .iter()
        .map(|slice| &slice.path)
        .collect();
    expected.sort();
    actual.sort();
    if actual != expected {
        anyhow::bail!("selected DICOM acquisition membership changed since discovery");
    }
    let mut volume = load_volume_from_scanned_series(scanned)?;
    volume.source = info.file_paths.first().cloned();
    Ok(volume)
}
