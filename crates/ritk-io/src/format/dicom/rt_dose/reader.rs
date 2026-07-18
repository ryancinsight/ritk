//! RT Dose reader â€” parse a DICOM RT Dose Storage file into [`RtDoseGrid`].

use anyhow::{bail, Context, Result};
use arrayvec::ArrayString;
use dicom::core::value::Value;
use dicom::core::Tag;
use ritk_dicom::{parse_file_with, DicomRsBackend};
use std::path::Path;

use super::types::{RtDoseGrid, RtDoseSummationType, RtDoseType, RT_DOSE_SOP_CLASS_UID};
use super::utils::{parse_ds_backslash, parse_frame_offsets};
use crate::format::dicom::reader::types::truncate_arraystring;

/// Read an RT Dose Storage DICOM file at `path` into an [`RtDoseGrid`].
///
/// # Errors
/// - `path` does not exist or is not readable.
/// - The file's `MediaStorageSOPClassUID` â‰  `1.2.840.10008.5.1.4.1.1.481.2`.
/// - Required tags (Rows, Columns, PixelData) are absent.
/// - `PixelData` length is not exactly `n_frames * rows * cols * 4` bytes.
/// - Present DS vector fields have the wrong component count or invalid components.
pub fn read_rt_dose<P: AsRef<Path>>(path: P) -> Result<RtDoseGrid> {
    let path = path.as_ref();
    let obj = parse_file_with::<DicomRsBackend, _>(path)
        .with_context(|| format!("open DICOM file: {}", path.display()))?;

    let sop = obj.meta().media_storage_sop_class_uid();
    let sop = sop.trim_end_matches('\0').trim();
    if sop != RT_DOSE_SOP_CLASS_UID {
        bail!(
            "SOP Class UID '{}' is not RT Dose Storage ({})",
            sop,
            RT_DOSE_SOP_CLASS_UID
        );
    }

    let rows: usize = obj
        .element(Tag(0x0028, 0x0010))
        .ok()
        .and_then(|e| e.to_str().ok().and_then(|s| s.trim().parse().ok()))
        .ok_or_else(|| anyhow::anyhow!("missing or unparseable Rows (0028,0010)"))?;

    let cols: usize = obj
        .element(Tag(0x0028, 0x0011))
        .ok()
        .and_then(|e| e.to_str().ok().and_then(|s| s.trim().parse().ok()))
        .ok_or_else(|| anyhow::anyhow!("missing or unparseable Columns (0028,0011)"))?;

    let n_frames: usize = obj
        .element(Tag(0x0028, 0x0008))
        .ok()
        .map(|e| {
            let value = e
                .to_int::<i32>()
                .context("unparseable NumberOfFrames (0028,0008)")?;
            if value <= 0 {
                bail!("NumberOfFrames (0028,0008) must be positive, got {value}");
            }
            Ok(value as usize)
        })
        .transpose()?
        .unwrap_or(1);

    let dose_grid_scaling: f64 = obj
        .element(Tag(0x3004, 0x000E))
        .ok()
        .and_then(|e| e.to_str().ok().and_then(|s| s.trim().parse().ok()))
        .unwrap_or(1.0);

    let dose_summation_type = obj
        .element(Tag(0x3004, 0x0002))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .map(|s| RtDoseSummationType::from_dicom_str(&s))
        .unwrap_or_else(|| RtDoseSummationType::Other(ArrayString::new()));

    let dose_type = obj
        .element(Tag(0x3004, 0x0004))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .map(|s| RtDoseType::from_dicom_str(&s))
        .unwrap_or(RtDoseType::Physical);

    let frame_offsets: Vec<f64> = obj
        .element(Tag(0x3004, 0x000C))
        .ok()
        .map(|e| {
            let raw = e
                .to_str()
                .context("Read GridFrameOffsetVector (3004,000C)")?;
            parse_frame_offsets(raw.trim(), n_frames)
        })
        .transpose()?
        .unwrap_or_else(|| (0..n_frames).map(|i| i as f64).collect());

    let px_bytes = obj
        .element(Tag(0x7FE0, 0x0010))
        .context("missing PixelData (7FE0,0010)")?
        .to_bytes()
        .context("PixelData to_bytes")?;

    let n_voxels = n_frames
        .checked_mul(rows)
        .and_then(|value| value.checked_mul(cols))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "RT Dose voxel count overflow: n_frames={} rows={} cols={}",
                n_frames,
                rows,
                cols
            )
        })?;
    let expected_bytes = n_voxels.checked_mul(4).ok_or_else(|| {
        anyhow::anyhow!(
            "RT Dose byte count overflow: {} voxels Ã— 4 bytes",
            n_voxels
        )
    })?;
    if px_bytes.len() != expected_bytes {
        bail!(
            "PixelData length mismatch for RT Dose: got {} bytes, expected {} ({} voxels Ã— 4 bytes)",
            px_bytes.len(),
            expected_bytes,
            n_voxels
        );
    }

    let dose_gy: Vec<f64> = px_bytes
        .chunks_exact(4)
        .take(n_voxels)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64 * dose_grid_scaling)
        .collect();

    let image_position: Option<[f64; 3]> = obj
        .element(Tag(0x0020, 0x0032))
        .ok()
        .map(|e| {
            let raw = e
                .to_str()
                .context("Read ImagePositionPatient (0020,0032)")?;
            parse_ds_backslash::<3>(&raw, "ImagePositionPatient (0020,0032)")
        })
        .transpose()?;

    let image_orientation: Option<[f64; 6]> = obj
        .element(Tag(0x0020, 0x0037))
        .ok()
        .map(|e| {
            let raw = e
                .to_str()
                .context("Read ImageOrientationPatient (0020,0037)")?;
            parse_ds_backslash::<6>(&raw, "ImageOrientationPatient (0020,0037)")
        })
        .transpose()?;

    let pixel_spacing: Option<[f64; 2]> = obj
        .element(Tag(0x0028, 0x0030))
        .ok()
        .map(|e| {
            let raw = e.to_str().context("Read PixelSpacing (0028,0030)")?;
            parse_ds_backslash::<2>(&raw, "PixelSpacing (0028,0030)")
        })
        .transpose()?;

    let referenced_rt_plan_sop_instance_uid: Option<ArrayString<64>> = obj
        .element(Tag(0x300C, 0x0002))
        .ok()
        .and_then(|e| match e.value() {
            Value::Sequence(seq) => seq.items().first().and_then(|item| {
                item.element(Tag(0x0008, 0x1155))
                    .ok()
                    .and_then(|el| el.to_str().ok().map(|s| s.trim().to_owned()))
                    .filter(|s| !s.is_empty())
                    .map(|s| match ArrayString::<64>::from(s.as_str()) {
                        Ok(v) => v,
                        Err(_) => {
                            tracing::warn!(
                                "ReferencedRTPlanSOPInstanceUID exceeds 64 chars, truncating: {}",
                                &s[..64]
                            );
                            truncate_arraystring::<64>(s.as_str())
                        }
                    })
            }),
            _ => None,
        });

    tracing::debug!(
        "read_rt_dose: rows={} cols={} n_frames={} scaling={} n_voxels={}",
        rows,
        cols,
        n_frames,
        dose_grid_scaling,
        n_voxels,
    );

    Ok(RtDoseGrid {
        rows,
        cols,
        n_frames,
        dose_type,
        dose_summation_type,
        dose_grid_scaling,
        frame_offsets,
        dose_gy,
        image_position,
        image_orientation,
        pixel_spacing,
        referenced_rt_plan_sop_instance_uid,
    })
}
