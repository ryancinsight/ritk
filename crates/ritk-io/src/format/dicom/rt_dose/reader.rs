//! RT Dose reader — parse a DICOM RT Dose Storage file into [`RtDoseGrid`].

use arrayvec::ArrayString;
use anyhow::{bail, Context, Result};
use dicom::core::value::Value;
use dicom::core::Tag;
use ritk_dicom::{parse_file_with, DicomRsBackend};
use std::path::Path;

use super::types::{RtDoseGrid, RT_DOSE_SOP_CLASS_UID};
use super::utils::parse_ds_backslash;

/// Read an RT Dose Storage DICOM file at `path` into an [`RtDoseGrid`].
///
/// # Errors
/// - `path` does not exist or is not readable.
/// - The file's `MediaStorageSOPClassUID` ≠ `1.2.840.10008.5.1.4.1.1.481.2`.
/// - Required tags (Rows, Columns, PixelData) are absent.
/// - `PixelData` length < `n_frames * rows * cols * 4` bytes.
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
        .and_then(|e| e.to_int::<i32>().ok())
        .map(|v| v.max(1) as usize)
        .unwrap_or(1);

    let dose_grid_scaling: f64 = obj
        .element(Tag(0x3004, 0x000E))
        .ok()
        .and_then(|e| e.to_str().ok().and_then(|s| s.trim().parse().ok()))
        .unwrap_or(1.0);

    let dose_summation_type: ArrayString<16> = obj
        .element(Tag(0x3004, 0x0002))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .map(|s| {
            match ArrayString::<16>::from(s.as_str()) {
                Ok(v) => v,
                Err(_) => {
                    tracing::warn!("DoseSummationType exceeds 16 chars, truncating: {}", &s[..16]);
                    ArrayString::from(&s[..16]).unwrap()
                }
            }
        })
        .unwrap_or_else(|| ArrayString::new());

    let dose_type: ArrayString<16> = obj
        .element(Tag(0x3004, 0x0004))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .map(|s| {
            match ArrayString::<16>::from(s.as_str()) {
                Ok(v) => v,
                Err(_) => {
                    tracing::warn!("DoseType exceeds 16 chars, truncating: {}", &s[..16]);
                    ArrayString::from(&s[..16]).unwrap()
                }
            }
        })
        .unwrap_or_else(|| match ArrayString::<16>::from("PHYSICAL") {
            Ok(v) => v,
            Err(_) => ArrayString::new(),
        });

    let frame_offsets: Vec<f64> = {
        let raw: Vec<f64> = obj
            .element(Tag(0x3004, 0x000C))
            .ok()
            .and_then(|e| e.to_str().ok())
            .map(|s| {
                s.trim()
                    .split('\\')
                    .filter_map(|t| t.trim().parse::<f64>().ok())
                    .collect()
            })
            .unwrap_or_default();
        if raw.is_empty() {
            (0..n_frames).map(|i| i as f64).collect()
        } else {
            raw
        }
    };

    let px_bytes = obj
        .element(Tag(0x7FE0, 0x0010))
        .context("missing PixelData (7FE0,0010)")?
        .to_bytes()
        .context("PixelData to_bytes")?;

    let n_voxels = n_frames * rows * cols;
    let expected_bytes = n_voxels * 4;
    if px_bytes.len() < expected_bytes {
        bail!(
            "PixelData too short for RT Dose: got {} bytes, need {} ({} voxels × 4 bytes)",
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
        .and_then(|e| e.to_str().ok())
        .and_then(|s| parse_ds_backslash::<3>(&s));

    let image_orientation: Option<[f64; 6]> = obj
        .element(Tag(0x0020, 0x0037))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| parse_ds_backslash::<6>(&s));

    let pixel_spacing: Option<[f64; 2]> = obj
        .element(Tag(0x0028, 0x0030))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| parse_ds_backslash::<2>(&s));

    let referenced_rt_plan_sop_instance_uid: Option<ArrayString<64>> = obj
        .element(Tag(0x300C, 0x0002))
        .ok()
        .and_then(|e| match e.value() {
            Value::Sequence(seq) => seq.items().first().and_then(|item| {
                item.element(Tag(0x0008, 0x1155))
                    .ok()
                    .and_then(|el| el.to_str().ok().map(|s| s.trim().to_owned()))
                    .filter(|s| !s.is_empty())
                    .and_then(|s| {
                        match ArrayString::<64>::from(s.as_str()) {
                            Ok(v) => Some(v),
                            Err(_) => {
                                tracing::warn!("ReferencedRTPlanSOPInstanceUID exceeds 64 chars, truncating: {}", &s[..64]);
                                Some(ArrayString::from(&s[..64]).unwrap())
                            }
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
