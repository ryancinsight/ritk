//! High-level DICOM series loading functions.
//!
//! Converts a pre-scanned `DicomSeriesInfo` into a reconstructed 3-D `Image`
//! with associated `DicomReadMetadata`. All pixel I/O and resampling logic is
//! contained here; callers only supply a scan result and a device handle.

use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use nalgebra::SMatrix;
use std::path::Path;

use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_dicom::TransferSyntaxKind;

use super::geometry::{
    analyze_slice_spacing, dot_3d, resample_frames_linear, slice_normal_from_iop,
};
use super::pixel::read_slice_pixels;
use super::scan::scan_dicom_directory;
use super::types::{DicomReadMetadata, DicomSeriesInfo};

/// Read a DICOM series and return the reconstructed 3-D image.
#[allow(dead_code)]
pub fn read_dicom_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let (image, _) = read_dicom_series_with_metadata(path, device)?;
    Ok(image)
}

/// Load a DICOM series, preserving metadata.
#[allow(dead_code)]
pub fn load_dicom_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    read_dicom_series(path, device)
}

/// Read a DICOM series and return both the image and metadata.
pub fn read_dicom_series_with_metadata<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<(Image<B, 3>, DicomReadMetadata)> {
    let series = scan_dicom_directory(path)?;
    load_from_series(series, device)
}

/// Load a DICOM series from a pre-scanned descriptor and return image plus metadata.
pub fn load_dicom_series_with_metadata<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<(Image<B, 3>, DicomReadMetadata)> {
    read_dicom_series_with_metadata(path, device)
}

pub(crate) fn load_from_series<B: Backend>(
    series: DicomSeriesInfo,
    device: &B::Device,
) -> Result<(Image<B, 3>, DicomReadMetadata)> {
    let mut metadata = series.metadata.clone();
    let slices = metadata.slices.clone();

    slices
        .first()
        .ok_or_else(|| anyhow!("DICOM series is empty"))?;

    // Guard: reject unsupported / big-endian transfer syntaxes before pixel decode.
    for slice in slices.iter() {
        if let Some(ref ts_uid) = slice.transfer_syntax_uid {
            let ts = TransferSyntaxKind::from_uid(ts_uid);
            if ts.is_compressed() && !ts.is_codec_supported() {
                bail!(
                    "DICOM series: compressed transfer syntax '{}' in slice {:?} is not \
                     supported (not natively decoded and no codec registered); \
                     decompress the series or use a supported transfer syntax",
                    ts_uid,
                    slice.path
                );
            }
            if ts.is_big_endian() {
                bail!(
                    "DICOM series: big-endian transfer syntax '{}' in slice {:?} is not \
                     supported; pixel decode requires little-endian byte order",
                    ts_uid,
                    slice.path
                );
            }
        }
    }

    let rows = metadata.dimensions[0];
    let cols = metadata.dimensions[1];
    let depth = metadata.dimensions[2];

    if rows == 0 || cols == 0 || depth == 0 {
        bail!("DICOM series has invalid zero dimensions");
    }

    // Geometry analysis: detect nonuniform or missing slices and resample when required.
    let iop = slices.first().and_then(|s| s.image_orientation_patient);
    let maybe_normal = iop.and_then(slice_normal_from_iop);

    let (needs_resample, final_spacing_z, resample_positions) = if let Some(normal) = maybe_normal {
        let proj: Vec<Option<f64>> = slices
            .iter()
            .map(|s| s.image_position_patient.map(|ipp| dot_3d(ipp, normal)))
            .collect();
        let missing_ipp = proj.iter().filter(|p| p.is_none()).count();
        if missing_ipp > 0 {
            tracing::warn!(
                missing_ipp_count = missing_ipp,
                total_slices = slices.len(),
                "DICOM series: {} of {} slices lack ImagePositionPatient; \
                 slice ordering may be incorrect",
                missing_ipp,
                slices.len()
            );
        }
        if proj.iter().all(|p| p.is_some()) {
            let positions: Vec<f64> = proj.into_iter().map(|p| p.unwrap()).collect();
            let report = analyze_slice_spacing(&positions);
            if report.is_nonuniform {
                tracing::warn!(
                    max_relative_deviation = report.max_relative_deviation,
                    nominal_spacing_mm = report.nominal_spacing,
                    n_slices = slices.len(),
                    "DICOM series: nonuniform slice spacing detected \
                     (max deviation {:.2}%); resampling to uniform grid \
                     with nominal spacing {:.4} mm",
                    report.max_relative_deviation * 100.0,
                    report.nominal_spacing,
                );
            }
            if report.has_missing_slices {
                tracing::warn!(
                    missing_between = ?report.missing_between,
                    nominal_spacing_mm = report.nominal_spacing,
                    "DICOM multiframe: {} gap(s) exceed 1.5x nominal spacing \
                     ({:.4} mm), indicating missing frames; \
                     resampling to fill gaps via linear interpolation",
                    report.missing_between.len(),
                    report.nominal_spacing,
                );
            }
            (
                report.is_nonuniform || report.has_missing_slices,
                report.nominal_spacing,
                Some(positions),
            )
        } else {
            (false, metadata.spacing[0], None)
        }
    } else {
        (false, metadata.spacing[0], None)
    };

    let frame_len = rows * cols;
    let (volume, final_depth) = if needs_resample {
        // Irregular z-spacing: decode to frame vectors then resample to uniform grid.
        #[cfg(not(target_arch = "wasm32"))]
        let decoded: Vec<Vec<f32>> = {
            use rayon::prelude::*;
            let decoded: Result<Vec<Vec<f32>>, anyhow::Error> = slices
                .par_iter()
                .map(|slice| {
                    let data = read_slice_pixels(slice).with_context(|| {
                        format!("failed to decode DICOM slice {:?}", slice.path)
                    })?;
                    if data.len() != frame_len {
                        bail!(
                            "DICOM slice size mismatch: expected {} pixels, got {}",
                            frame_len,
                            data.len()
                        );
                    }
                    Ok(data)
                })
                .collect();
            decoded?
        };

        #[cfg(target_arch = "wasm32")]
        let decoded: Vec<Vec<f32>> = {
            let mut decoded = Vec::with_capacity(depth);
            for slice in slices.iter() {
                let data = read_slice_pixels(slice)
                    .with_context(|| format!("failed to decode DICOM slice {:?}", slice.path))?;
                if data.len() != frame_len {
                    bail!(
                        "DICOM slice size mismatch: expected {} pixels, got {}",
                        frame_len,
                        data.len()
                    );
                }
                decoded.push(data);
            }
            decoded
        };

        let positions = resample_positions.as_ref().ok_or_else(|| {
            anyhow!("resample positions missing despite resample-required series")
        })?;
        let resampled = resample_frames_linear(&decoded, positions, final_spacing_z);
        let new_depth = resampled.len();
        let mut volume = vec![0f32; frame_len * new_depth];
        for (z, frame) in resampled.iter().enumerate() {
            let offset = z * frame_len;
            volume[offset..offset + frame_len].copy_from_slice(frame);
        }
        (volume, new_depth)
    } else {
        // Uniform z-spacing: decode directly into a preallocated contiguous volume.
        let mut volume = vec![0f32; frame_len * depth];

        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            volume
                .par_chunks_mut(frame_len)
                .zip(slices.par_iter())
                .try_for_each(|(dst, slice)| -> Result<()> {
                    let data = read_slice_pixels(slice).with_context(|| {
                        format!("failed to decode DICOM slice {:?}", slice.path)
                    })?;
                    if data.len() != frame_len {
                        bail!(
                            "DICOM slice size mismatch: expected {} pixels, got {}",
                            frame_len,
                            data.len()
                        );
                    }
                    dst.copy_from_slice(&data);
                    Ok(())
                })?;
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (z, slice) in slices.iter().enumerate() {
                let data = read_slice_pixels(slice)
                    .with_context(|| format!("failed to decode DICOM slice {:?}", slice.path))?;
                if data.len() != frame_len {
                    bail!(
                        "DICOM slice size mismatch: expected {} pixels, got {}",
                        frame_len,
                        data.len()
                    );
                }
                let offset = z * frame_len;
                volume[offset..offset + frame_len].copy_from_slice(&data);
            }
        }

        (volume, depth)
    };

    metadata.dimensions[2] = final_depth;
    metadata.spacing[0] = final_spacing_z.abs().max(1e-6);

    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(volume, Shape::new([final_depth, rows, cols])),
        device,
    );
    let image = Image::new(
        tensor,
        Point::new(metadata.origin),
        Spacing::new(metadata.spacing),
        Direction(SMatrix::<f64, 3, 3>::from_column_slice(&metadata.direction)),
    );

    Ok((image, metadata))
}
