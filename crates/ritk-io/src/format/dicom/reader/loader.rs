//! High-level DICOM series loading functions.
//!
//! Converts a pre-scanned `DicomSeriesInfo` into a reconstructed 3-D `Image`
//! with associated `DicomReadMetadata`. All pixel I/O and resampling logic is
//! contained here; callers only supply a scan result and a device handle.

use anyhow::{anyhow, bail, Context, Result};
use coeus_core::ComputeBackend;
use std::path::Path;

use ritk_dicom::TransferSyntaxKind;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

use super::geometry::{
    analyze_slice_spacing, dot, resample_frames_linear, resampled_frame_count,
    slice_normal_from_iop, SliceCoverage, SpacingUniformity,
};
use super::pixel::{read_slice_pixels, read_slice_pixels_from_bytes};
use super::scan::scan_dicom_path_with_budget;
use super::types::{DicomReadBudget, DicomReadMetadata, DicomSeriesInfo};

/// Read a DICOM series and return both the image and metadata.
pub fn read_dicom_series_with_metadata<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    read_dicom_series_with_metadata_with_budget(path, backend, &DicomReadBudget::DEFAULT)
}

/// Read a DICOM series with explicit parser, retained-byte, and decoded-buffer ceilings.
pub fn read_dicom_series_with_metadata_with_budget<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
    budget: &DicomReadBudget,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    let series = scan_dicom_path_with_budget(path, budget)?;
    load_from_series_with_budget(series, backend, budget)
}

/// Load a DICOM series from a pre-scanned descriptor and return image plus metadata.
pub fn load_dicom_series_with_metadata<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    load_dicom_series_with_metadata_with_budget(path, backend, &DicomReadBudget::DEFAULT)
}

/// Load a DICOM series with explicit parser, retained-byte, and decoded-buffer ceilings.
pub fn load_dicom_series_with_metadata_with_budget<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
    budget: &DicomReadBudget,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    read_dicom_series_with_metadata_with_budget(path, backend, budget)
}

/// Load a DICOM series from a pre-scanned [`DicomSeriesInfo`] and return image plus metadata.
///
/// This is the zero-disk counterpart of [`load_dicom_series_with_metadata`]:
/// callers that have already obtained a `DicomSeriesInfo` (e.g. via
/// [`scan_dicom_instances`](super::scan::scan_dicom_instances)) pass it directly
/// instead of re-scanning a directory. Pixel decode uses `part10_bytes` from the
/// slice metadata when present, falling back to file-path I/O otherwise.
pub fn load_dicom_from_series<B: ComputeBackend>(
    series: DicomSeriesInfo,
    backend: &B,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    load_from_series(series, backend)
}

/// Load a pre-scanned DICOM series with explicit decoded-buffer ceilings.
pub fn load_dicom_from_series_with_budget<B: ComputeBackend>(
    series: DicomSeriesInfo,
    backend: &B,
    budget: &DicomReadBudget,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    load_from_series_with_budget(series, backend, budget)
}

pub(crate) fn load_from_series<B: ComputeBackend>(
    series: DicomSeriesInfo,
    backend: &B,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    load_from_series_with_budget(series, backend, &DicomReadBudget::DEFAULT)
}

pub(crate) fn load_from_series_with_budget<B: ComputeBackend>(
    series: DicomSeriesInfo,
    backend: &B,
    budget: &DicomReadBudget,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    let decoded = decode_series(series, budget)?;
    let image = Image::from_flat_on(
        decoded.volume,
        decoded.shape,
        decoded.origin,
        decoded.spacing,
        decoded.direction,
        backend,
    )?;

    Ok((image, decoded.metadata))
}

struct DecodedDicomSeries {
    volume: Vec<f32>,
    shape: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
    metadata: DicomReadMetadata,
}

fn decode_series(series: DicomSeriesInfo, budget: &DicomReadBudget) -> Result<DecodedDicomSeries> {
    let mut metadata = series.metadata;
    let slices = std::mem::take(&mut metadata.slices);
    let parser_budget = budget.parser();

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
            .map(|s| s.image_position_patient.map(|ipp| dot(ipp, normal)))
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
        // Inter-slice spacing requires a pair of positions. A single-frame
        // image keeps its declared spacing; it has no interval to estimate.
        if proj.len() >= 2 && proj.iter().all(|p| p.is_some()) {
            let positions: Vec<f64> = proj
                .into_iter()
                .map(|p| p.expect("all slice positions verified non-None above"))
                .collect();
            let report = analyze_slice_spacing(&positions);
            if report.spacing_uniformity == SpacingUniformity::Nonuniform {
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
            if report.slice_coverage == SliceCoverage::HasMissingSlices {
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
                report.spacing_uniformity == SpacingUniformity::Nonuniform
                    || report.slice_coverage == SliceCoverage::HasMissingSlices,
                report.nominal_spacing,
                Some(positions),
            )
        } else {
            (false, metadata.spacing[0], None)
        }
    } else {
        (false, metadata.spacing[0], None)
    };

    let frame_len = dicom_frame_pixel_count(rows, cols)?;
    let target_depth = if needs_resample {
        let positions = resample_positions.as_ref().ok_or_else(|| {
            anyhow!("resample positions missing despite resample-required series")
        })?;
        resampled_frame_count(positions, final_spacing_z)
    } else {
        depth
    };
    let volume_len = dicom_volume_pixel_count(frame_len, target_depth)?;
    let source_workspace = decoded_source_workspace_bytes(frame_len, slices.len())?;
    let resampled_workspace = if needs_resample {
        decoded_frame_set_workspace_bytes(frame_len, target_depth)?
    } else {
        0
    };
    let volume_bytes = decoded_buffer_bytes(volume_len)?;
    let peak_workspace = source_workspace
        .checked_add(resampled_workspace)
        .and_then(|bytes| bytes.checked_add(volume_bytes))
        .context("DICOM decoded workspace byte count overflow")?;
    budget
        .checked_decoded_bytes(peak_workspace)
        .context("DICOM peak decoded workspace exceeds budget")?;
    let (volume, final_depth) = if needs_resample {
        // Irregular z-spacing: decode to frame vectors then resample to uniform grid.
        #[cfg(not(target_arch = "wasm32"))]
        let decoded: Vec<Vec<f32>> = {
            let decoded: Result<Vec<Vec<f32>>, anyhow::Error> =
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(slices.len(), |z| {
                    let slice = &slices[z];
                    let data = if let Some(ref bytes) = slice.part10_bytes {
                        read_slice_pixels_from_bytes(bytes, slice, &parser_budget)
                    } else {
                        read_slice_pixels(slice, &parser_budget)
                    }
                    .with_context(|| format!("failed to decode DICOM slice {:?}", slice.path))?;
                    if data.len() != frame_len {
                        bail!(
                            "DICOM slice size mismatch: expected {} pixels, got {}",
                            frame_len,
                            data.len()
                        );
                    }
                    Ok(data)
                })
                .into_iter()
                .collect();
            decoded?
        };

        #[cfg(target_arch = "wasm32")]
        let decoded: Vec<Vec<f32>> = {
            let mut decoded = Vec::with_capacity(depth);
            for slice in slices.iter() {
                let data = if let Some(ref bytes) = slice.part10_bytes {
                    read_slice_pixels_from_bytes(bytes, slice, &parser_budget)
                } else {
                    read_slice_pixels(slice, &parser_budget)
                }
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
        debug_assert_eq!(new_depth, target_depth);
        let mut volume = allocate_decoded_buffer(volume_len, budget, "DICOM resampled volume")?;
        for (z, frame) in resampled.iter().enumerate() {
            let offset = z * frame_len;
            volume[offset..offset + frame_len].copy_from_slice(frame);
        }
        (volume, new_depth)
    } else {
        // Uniform z-spacing: decode directly into a preallocated contiguous volume.
        let mut volume = allocate_decoded_buffer(volume_len, budget, "DICOM decoded volume")?;

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Decode slices in parallel (fallible), then write into the volume
            // sequentially (cheap memcpy) so the first decode error propagates.
            let decoded: Vec<Result<Vec<f32>>> =
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(slices.len(), |z| {
                    let slice = &slices[z];
                    let data = if let Some(ref bytes) = slice.part10_bytes {
                        read_slice_pixels_from_bytes(bytes, slice, &parser_budget)
                    } else {
                        read_slice_pixels(slice, &parser_budget)
                    }
                    .with_context(|| format!("failed to decode DICOM slice {:?}", slice.path))?;
                    if data.len() != frame_len {
                        bail!(
                            "DICOM slice size mismatch: expected {} pixels, got {}",
                            frame_len,
                            data.len()
                        );
                    }
                    Ok(data)
                });
            for (z, result) in decoded.into_iter().enumerate() {
                let data = result?;
                let offset = z * frame_len;
                volume[offset..offset + frame_len].copy_from_slice(&data);
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (z, slice) in slices.iter().enumerate() {
                let data = if let Some(ref bytes) = slice.part10_bytes {
                    read_slice_pixels_from_bytes(bytes, slice, &parser_budget)
                } else {
                    read_slice_pixels(slice, &parser_budget)
                }
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

    let shape = [final_depth, rows, cols];
    let origin = Point::new(metadata.origin);
    let spacing = Spacing::new(metadata.spacing);
    let direction = Direction::from_column_major(metadata.direction);
    metadata.slices = slices;
    Ok(DecodedDicomSeries {
        volume,
        shape,
        origin,
        spacing,
        direction,
        metadata,
    })
}

fn dicom_frame_pixel_count(rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| anyhow!("DICOM frame pixel count overflow: rows={rows}, cols={cols}"))
}

fn dicom_volume_pixel_count(frame_len: usize, depth: usize) -> Result<usize> {
    frame_len.checked_mul(depth).ok_or_else(|| {
        anyhow!("DICOM volume pixel count overflow: frame_len={frame_len}, depth={depth}")
    })
}

fn decoded_source_workspace_bytes(frame_len: usize, frame_count: usize) -> Result<usize> {
    let pixel_bytes = decoded_buffer_bytes(
        frame_len
            .checked_mul(frame_count)
            .context("DICOM decoded frame count overflow")?,
    )?;
    let frame_handles = frame_count
        .checked_mul(std::mem::size_of::<Vec<f32>>())
        .context("DICOM decoded frame handle count overflow")?;
    let result_handles = frame_count
        .checked_mul(std::mem::size_of::<Result<Vec<f32>>>())
        .context("DICOM decoded result handle count overflow")?;
    pixel_bytes
        .checked_add(frame_handles)
        .and_then(|bytes| bytes.checked_add(result_handles))
        .context("DICOM decoded source workspace byte count overflow")
}

fn decoded_frame_set_workspace_bytes(frame_len: usize, frame_count: usize) -> Result<usize> {
    let pixel_bytes = decoded_buffer_bytes(
        frame_len
            .checked_mul(frame_count)
            .context("DICOM resampled frame count overflow")?,
    )?;
    let frame_handles = frame_count
        .checked_mul(std::mem::size_of::<Vec<f32>>())
        .context("DICOM resampled frame handle count overflow")?;
    pixel_bytes
        .checked_add(frame_handles)
        .context("DICOM resampled workspace byte count overflow")
}

fn decoded_buffer_bytes(elements: usize) -> Result<usize> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .context("DICOM decoded buffer byte count overflow")
}

fn allocate_decoded_buffer(
    elements: usize,
    budget: &DicomReadBudget,
    what: &'static str,
) -> Result<Vec<f32>> {
    let bytes = decoded_buffer_bytes(elements)?;
    budget.checked_decoded_bytes(bytes)?;
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(elements).map_err(|_| {
        anyhow!("DICOM decoded buffer allocation for {what} requires {bytes} bytes")
    })?;
    buffer.resize(elements, 0.0);
    Ok(buffer)
}
