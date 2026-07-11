//! DICOM series loading into LoadedVolume.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use coeus_core::SequentialBackend;
use ritk_io::{
    is_rgb_dicom_series, load_dicom_color_from_series, load_dicom_color_series,
    load_native_dicom_from_series, load_native_dicom_series_with_metadata,
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

/// Load an RGB DICOM colour series from a pre-scanned series descriptor.
fn load_dicom_color_volume_from_scanned_series(
    series: ritk_io::ScannedDicomSeries,
) -> Result<LoadedVolume> {
    let backend = SequentialBackend;
    let (rgb_vol, meta) = load_dicom_color_from_series(series, &backend)
        .with_context(|| "failed to load DICOM RGB series from scanned instances")?;
    loaded_volume_from_color_image(rgb_vol, meta, None, &backend)
}

/// Convert a native RGB volume and DICOM metadata into viewer-owned storage.
fn loaded_volume_from_color_image(
    rgb_vol: ritk_image::native::RgbVolume<f32, SequentialBackend>,
    meta: ritk_io::DicomReadMetadata,
    source: Option<std::path::PathBuf>,
    backend: &SequentialBackend,
) -> Result<LoadedVolume> {
    let [depth, rows, cols] = rgb_vol.spatial_shape();
    let shape = [depth, rows, cols];
    let sp = rgb_vol.spacing();
    let spacing = [sp[0], sp[1], sp[2]];
    let orig = rgb_vol.origin();
    let origin = [orig.0[0], orig.0[1], orig.0[2]];
    let dir = rgb_vol.direction().0;
    let direction = [
        dir[(0, 0)],
        dir[(0, 1)],
        dir[(0, 2)],
        dir[(1, 0)],
        dir[(1, 1)],
        dir[(1, 2)],
        dir[(2, 0)],
        dir[(2, 1)],
        dir[(2, 2)],
    ];
    let pixels = rgb_vol.data_cow_on(backend).into_owned();
    let modality = meta.modality;
    let patient_name = meta.patient_name.clone();
    let patient_id = meta.patient_id.clone();
    let study_date = meta.study_date;
    let series_description = meta.series_description.clone();
    Ok(LoadedVolume {
        data: Arc::new(pixels),
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
    })
}

/// Load a DICOM series from `folder` into a [`LoadedVolume`].
///
/// # Algorithm
/// 1. Detects whether the series is RGB via [`is_rgb_dicom_series`].
/// 2. For RGB series: calls [`load_dicom_color_series`] and converts the native
///    RGB volume into a [`LoadedVolume`] with `channels: 3`.
/// 3. For scalar series: calls [`load_native_dicom_series_with_metadata`].
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
/// # Parameters
/// - `folder` — path to the DICOM series directory.
fn load_dicom_color_volume(folder: &Path) -> Result<LoadedVolume> {
    info!(path = %folder.display(), "loading DICOM RGB colour volume");

    let backend = SequentialBackend;
    let (rgb_vol, meta) = load_dicom_color_series(folder, &backend).with_context(|| {
        format!(
            "failed to load DICOM RGB series from '{}'",
            folder.display()
        )
    })?;
    loaded_volume_from_color_image(rgb_vol, meta, Some(folder.to_path_buf()), &backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_spatial::{Direction, Point, Spacing};

    #[test]
    fn native_scalar_volume_preserves_values_geometry_and_metadata() -> Result<()> {
        let backend = SequentialBackend;
        let pixels = vec![1.0_f32, 2.0, 3.0, 4.0];
        let image = ritk_image::native::Image::from_flat_on(
            pixels.clone(),
            [1, 2, 2],
            Point::new([3.0, 4.0, 5.0]),
            Spacing::new([0.25, 0.5, 0.75]),
            Direction::identity(),
            &backend,
        )?;
        let metadata = ritk_io::DicomReadMetadata {
            patient_name: Some("DOE^NATIVE".to_string()),
            series_description: Some("native scalar".to_string()),
            ..Default::default()
        };
        let source = std::path::PathBuf::from("native-scalar");

        let loaded =
            loaded_volume_from_scalar_image(image, metadata, Some(source.clone()), &backend)?;

        assert_eq!(loaded.data.as_ref(), &pixels);
        assert_eq!(loaded.shape, [1, 2, 2]);
        assert_eq!(loaded.channels, 1);
        assert_eq!(loaded.spacing, [0.25, 0.5, 0.75]);
        assert_eq!(loaded.origin, [3.0, 4.0, 5.0]);
        assert_eq!(
            loaded.direction,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(loaded.patient_name.as_deref(), Some("DOE^NATIVE"));
        assert_eq!(loaded.series_description.as_deref(), Some("native scalar"));
        assert_eq!(loaded.source.as_deref(), Some(source.as_path()));
        Ok(())
    }

    #[test]
    fn native_color_volume_preserves_values_geometry_and_metadata() -> Result<()> {
        let backend = SequentialBackend;
        let pixels = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let image = ritk_image::native::RgbVolume::from_flat_on(
            pixels.clone(),
            [1, 1, 2],
            Point::new([7.0, 8.0, 9.0]),
            Spacing::new([0.5, 1.5, 2.5]),
            Direction::identity(),
            &backend,
        )?;
        let metadata = ritk_io::DicomReadMetadata {
            patient_id: Some("patient-7".to_string()),
            series_description: Some("native RGB".to_string()),
            ..Default::default()
        };

        let loaded = loaded_volume_from_color_image(image, metadata, None, &backend)?;

        assert_eq!(loaded.data.as_ref(), &pixels);
        assert_eq!(loaded.shape, [1, 1, 2]);
        assert_eq!(loaded.channels, 3);
        assert_eq!(loaded.spacing, [0.5, 1.5, 2.5]);
        assert_eq!(loaded.origin, [7.0, 8.0, 9.0]);
        assert_eq!(
            loaded.direction,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(loaded.patient_id.as_deref(), Some("patient-7"));
        assert_eq!(loaded.series_description.as_deref(), Some("native RGB"));
        assert!(loaded.source.is_none());
        Ok(())
    }
}
