//! RT Dose Storage (SOP Class 1.2.840.10008.5.1.4.1.1.481.2) reader.
//!
//! # Specification
//!
//! An RT Dose file contains a 3-D dose grid:
//! - (3004,000E) DoseGridScaling: multiply raw pixel values to get dose in Gy.
//! - (3004,0002) DoseSummationType: PLAN, BEAM, FRACTION, CONTROL_PT, etc.
//! - (3004,0004) DoseType: PHYSICAL, EFFECTIVE, or ERROR.
//! - (3004,000C) GridFrameOffsetVector: z-positions of each dose plane (DS, multi-value).
//! - (0028,0010) Rows, (0028,0011) Columns, (0028,0008) NumberOfFrames.
//! - (7FE0,0010) PixelData: Uint32 LE voxel values (BitsAllocated = 32).
//! - Dose(x) = PixelValue(x) * DoseGridScaling.
//!
//! ## Dose computation invariant
//!
//! For voxel index `k = frame * rows * cols + row * cols + col`:
//!   raw_u32 = u32::from_le_bytes(pixel_bytes[k*4 .. k*4+4])
//!   dose_gy[k] = raw_u32 as f64 * dose_grid_scaling

use anyhow::{bail, Context, Result};
use dicom::core::smallvec::SmallVec;
use dicom::core::Tag;
use dicom::core::{DataElement, PrimitiveValue, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::open_file;
use dicom::object::InMemDicomObject;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// SOP Class UID for RT Dose Storage.
pub const RT_DOSE_SOP_CLASS_UID: &str = "1.2.840.10008.5.1.4.1.1.481.2";

// ── Domain types ─────────────────────────────────────────────────────────────

/// In-memory representation of an RT Dose grid.
///
/// # Invariants
/// - `dose_gy.len() == n_frames * rows * cols`
/// - `frame_offsets.len() == n_frames`
/// - `dose_gy[frame * rows * cols + row * cols + col]`
///   = `raw_pixel_u32 * dose_grid_scaling`
#[derive(Debug, Clone)]
pub struct RtDoseGrid {
    /// Grid rows (0028,0010).
    pub rows: usize,
    /// Grid columns (0028,0011).
    pub cols: usize,
    /// Number of dose planes (0028,0008); defaults to 1 when absent.
    pub n_frames: usize,
    /// DoseType (3004,0004): PHYSICAL, EFFECTIVE, or ERROR.
    pub dose_type: String,
    /// DoseSummationType (3004,0002): PLAN, BEAM, FRACTION, CONTROL_PT, etc.
    pub dose_summation_type: String,
    /// DoseGridScaling (3004,000E): factor converting raw pixel values to Gy.
    pub dose_grid_scaling: f64,
    /// Z-offset per frame (mm) from GridFrameOffsetVector (3004,000C).
    /// Length == n_frames. When the tag is absent, offsets are 0.0, 1.0, 2.0, …
    pub frame_offsets: Vec<f64>,
    /// Dose values in Gy per voxel. Length = n_frames * rows * cols.
    /// Flat index: frame * rows * cols + row * cols + col.
    pub dose_gy: Vec<f64>,
    /// ImagePositionPatient (0020,0032) — origin of the dose grid in mm.
    pub image_position: Option<[f64; 3]>,
    /// ImageOrientationPatient (0020,0037) — 6-component direction cosines.
    pub image_orientation: Option<[f64; 6]>,
    /// PixelSpacing (0028,0030) — [row_spacing, col_spacing] in mm.
    pub pixel_spacing: Option<[f64; 2]>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Read an RT Dose Storage DICOM file at `path` into an [`RtDoseGrid`].
///
/// # Errors
/// - `path` does not exist or is not readable.
/// - The file's `MediaStorageSOPClassUID` ≠ `1.2.840.10008.5.1.4.1.1.481.2`.
/// - Required tags (Rows, Columns, PixelData) are absent.
/// - `PixelData` length < `n_frames * rows * cols * 4` bytes.
pub fn read_rt_dose<P: AsRef<Path>>(path: P) -> Result<RtDoseGrid> {
    let path = path.as_ref();
    let obj = open_file(path).with_context(|| format!("open DICOM file: {}", path.display()))?;

    // Validate SOP Class UID.
    let sop = obj.meta().media_storage_sop_class_uid();
    let sop = sop.trim_end_matches('\0').trim();
    if sop != RT_DOSE_SOP_CLASS_UID {
        bail!(
            "SOP Class UID '{}' is not RT Dose Storage ({})",
            sop,
            RT_DOSE_SOP_CLASS_UID
        );
    }

    // ── Scalar header fields ─────────────────────────────────────────────────

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

    let dose_summation_type: String = obj
        .element(Tag(0x3004, 0x0002))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .unwrap_or_default();

    let dose_type: String = obj
        .element(Tag(0x3004, 0x0004))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .unwrap_or_else(|| "PHYSICAL".to_owned());

    // ── GridFrameOffsetVector (3004,000C) ────────────────────────────────────
    // DS, backslash-separated. Defaults to 0.0, 1.0, … when absent or empty.
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

    // ── PixelData (7FE0,0010) ────────────────────────────────────────────────
    // RT Dose uses BitsAllocated = 32: each voxel is a u32 LE value.
    // dose_gy[k] = raw_u32[k] as f64 * dose_grid_scaling.
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

    // ── Spatial metadata ─────────────────────────────────────────────────────

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
    })
}

// ── Public writer ─────────────────────────────────────────────────────────────

/// Write an [`RtDoseGrid`] to a DICOM RT Dose Storage file at `path`.
///
/// # Write/Read Invariant
///
/// For every voxel index `k`:
///   `dose_gy[k] = raw_u32[k] as f64 * dose_grid_scaling`
///
/// Encoding: `raw_u32[k] = (dose_gy[k] / dose_grid_scaling).round().clamp(0.0, u32::MAX as f64) as u32`.
/// Quantization error: `|dose_gy[k] - reconstructed[k]| ≤ dose_grid_scaling / 2`.
///
/// # Errors
/// - `grid.dose_gy.len() != grid.n_frames * grid.rows * grid.cols`
/// - `grid.frame_offsets.len() != grid.n_frames`
/// - `!grid.dose_grid_scaling.is_finite() || grid.dose_grid_scaling <= 0.0`
/// - File cannot be created or written at `path`.
pub fn write_rt_dose<P: AsRef<Path>>(path: P, grid: &RtDoseGrid) -> Result<()> {
    let path = path.as_ref();

    // ── Validation ────────────────────────────────────────────────────────────
    let expected_voxels = grid.n_frames * grid.rows * grid.cols;
    if grid.dose_gy.len() != expected_voxels {
        bail!(
            "dose_gy length {} does not match n_frames={} * rows={} * cols={} = {} voxels",
            grid.dose_gy.len(),
            grid.n_frames,
            grid.rows,
            grid.cols,
            expected_voxels,
        );
    }
    if grid.frame_offsets.len() != grid.n_frames {
        bail!(
            "frame_offsets length {} does not match n_frames={}",
            grid.frame_offsets.len(),
            grid.n_frames,
        );
    }
    if !grid.dose_grid_scaling.is_finite() || grid.dose_grid_scaling <= 0.0 {
        bail!(
            "dose_grid_scaling must be finite and positive; got {}",
            grid.dose_grid_scaling,
        );
    }

    // ── SOP Instance UID ──────────────────────────────────────────────────────
    static RT_DOSE_UID_COUNTER: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let n = RT_DOSE_UID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let sop_instance_uid = format!("2.25.{}.{}", t, n);

    // ── Pixel encoding ────────────────────────────────────────────────────────
    // Invariant: dose_gy[k] = raw_u32[k] as f64 * dose_grid_scaling
    // Encoding:  raw_u32[k] = (dose_gy[k] / dose_grid_scaling).round().clamp(0, u32::MAX)
    let inv_scaling = 1.0 / grid.dose_grid_scaling;
    let mut pixel_bytes: Vec<u8> = Vec::with_capacity(expected_voxels * 4);
    for &v in &grid.dose_gy {
        let raw = (v * inv_scaling).round().clamp(0.0, u32::MAX as f64) as u32;
        pixel_bytes.extend_from_slice(&raw.to_le_bytes());
    }

    // ── Build DICOM object ────────────────────────────────────────────────────
    let mut obj = InMemDicomObject::new_empty();

    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from(RT_DOSE_SOP_CLASS_UID),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from(sop_instance_uid.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from("RTDOSE"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from("MONOCHROME2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from(grid.n_frames.to_string().as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(grid.rows as u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(grid.cols as u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(32u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0101),
        VR::US,
        PrimitiveValue::from(32u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0102),
        VR::US,
        PrimitiveValue::from(31u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0u16),
    ));
    obj.put(DataElement::new(
        Tag(0x3004, 0x0002),
        VR::CS,
        PrimitiveValue::from(grid.dose_summation_type.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x3004, 0x0004),
        VR::CS,
        PrimitiveValue::from(grid.dose_type.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x3004, 0x000E),
        VR::DS,
        PrimitiveValue::from(format!("{}", grid.dose_grid_scaling).as_str()),
    ));

    let offset_str = grid
        .frame_offsets
        .iter()
        .map(|v| format!("{}", v))
        .collect::<Vec<_>>()
        .join("\\");
    obj.put(DataElement::new(
        Tag(0x3004, 0x000C),
        VR::DS,
        PrimitiveValue::from(offset_str.as_str()),
    ));

    if let Some(pos) = grid.image_position {
        let s = format!("{}\\{}\\{}", pos[0], pos[1], pos[2]);
        obj.put(DataElement::new(
            Tag(0x0020, 0x0032),
            VR::DS,
            PrimitiveValue::from(s.as_str()),
        ));
    }
    if let Some(ori) = grid.image_orientation {
        let s = ori
            .iter()
            .map(|v| format!("{}", v))
            .collect::<Vec<_>>()
            .join("\\");
        obj.put(DataElement::new(
            Tag(0x0020, 0x0037),
            VR::DS,
            PrimitiveValue::from(s.as_str()),
        ));
    }
    if let Some(ps) = grid.pixel_spacing {
        let s = format!("{}\\{}", ps[0], ps[1]);
        obj.put(DataElement::new(
            Tag(0x0028, 0x0030),
            VR::DS,
            PrimitiveValue::from(s.as_str()),
        ));
    }

    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U8(SmallVec::from_vec(pixel_bytes)),
    ));

    // ── Write ─────────────────────────────────────────────────────────────────
    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(RT_DOSE_SOP_CLASS_UID)
            .media_storage_sop_instance_uid(sop_instance_uid.as_str())
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .with_context(|| "build RT Dose file meta")?
    .write_to_file(path)
    .with_context(|| format!("write RT Dose to {}", path.display()))?;

    Ok(())
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Parse a `\`-separated DICOM Decimal String into a fixed-size `f64` array.
///
/// Returns `None` when fewer than `N` parseable values are present.
fn parse_ds_backslash<const N: usize>(s: &str) -> Option<[f64; N]> {
    let parts: Vec<f64> = s
        .trim()
        .split('\\')
        .filter_map(|p| p.trim().parse::<f64>().ok())
        .collect();
    if parts.len() >= N {
        let mut arr = [0.0_f64; N];
        arr.copy_from_slice(&parts[..N]);
        Some(arr)
    } else {
        None
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
    use dicom::object::meta::FileMetaTableBuilder;
    use dicom::object::InMemDicomObject;

    // ── Test helpers ─────────────────────────────────────────────────────────

    /// Write a minimal RT Dose DICOM Part-10 file carrying the given object.
    fn write_rt_dose_file(obj: InMemDicomObject, path: &std::path::Path) {
        obj.with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(RT_DOSE_SOP_CLASS_UID)
                .media_storage_sop_instance_uid("2.25.1")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("meta build")
        .write_to_file(path)
        .expect("write RT Dose file");
    }

    /// Write a DICOM file with an arbitrary SOP Class UID (no RT Dose tags).
    fn write_wrong_sop_file(sop: &str, path: &std::path::Path) {
        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from(sop),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.99"),
        ));
        // Minimal PixelData to satisfy Part-10 writer.
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U8(dicom::core::smallvec::SmallVec::new()),
        ));
        obj.with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(sop)
                .media_storage_sop_instance_uid("2.25.99")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("meta")
        .write_to_file(path)
        .expect("write wrong-SOP file");
    }

    /// Build a synthetic RT Dose InMemDicomObject.
    ///
    /// All `rows * cols * n_frames` voxels are set to `pixel_val` (u32 LE).
    fn build_rt_dose_obj(
        rows: u16,
        cols: u16,
        n_frames: u32,
        dose_grid_scaling: f64,
        pixel_val: u32,
    ) -> InMemDicomObject {
        let mut obj = InMemDicomObject::new_empty();

        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(rows),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(cols),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0008),
            VR::IS,
            PrimitiveValue::from(n_frames.to_string().as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(32u16),
        ));
        obj.put(DataElement::new(
            Tag(0x3004, 0x000E),
            VR::DS,
            PrimitiveValue::from(format!("{}", dose_grid_scaling).as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x3004, 0x0002),
            VR::CS,
            PrimitiveValue::from("PLAN"),
        ));
        obj.put(DataElement::new(
            Tag(0x3004, 0x0004),
            VR::CS,
            PrimitiveValue::from("PHYSICAL"),
        ));

        // GridFrameOffsetVector: "0\1\2…" for n_frames planes.
        let offset_str: String = (0..n_frames as usize)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\\");
        obj.put(DataElement::new(
            Tag(0x3004, 0x000C),
            VR::DS,
            PrimitiveValue::from(offset_str.as_str()),
        ));

        // PixelData: n_frames * rows * cols u32 LE values, all equal to pixel_val.
        let n_voxels = n_frames as usize * rows as usize * cols as usize;
        let mut pixel_bytes: Vec<u8> = Vec::with_capacity(n_voxels * 4);
        for _ in 0..n_voxels {
            pixel_bytes.extend_from_slice(&pixel_val.to_le_bytes());
        }
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U8(dicom::core::smallvec::SmallVec::from_vec(pixel_bytes)),
        ));

        obj
    }

    // ── Test A: missing file ─────────────────────────────────────────────────

    /// Invariant: a nonexistent path must produce Err mentioning the path or open failure.
    #[test]
    fn test_read_rt_dose_missing_file_returns_error() {
        let result = read_rt_dose("/nonexistent/path.dcm");
        assert!(result.is_err(), "nonexistent path must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("nonexistent") || msg.contains("open"),
            "error must mention path or open failure; got: {msg}"
        );
    }

    // ── Test B: wrong SOP class ──────────────────────────────────────────────

    /// Invariant: a file whose SOP Class UID ≠ RT Dose must produce Err
    /// containing the rejected UID in the message.
    #[test]
    fn test_read_rt_dose_wrong_sop_class_returns_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("ct.dcm");
        write_wrong_sop_file("1.2.840.10008.5.1.4.1.1.2", &path);

        let result = read_rt_dose(&path);
        assert!(result.is_err(), "wrong SOP class must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("1.2.840.10008.5.1.4.1.1.2"),
            "error must contain the rejected SOP UID; got: {msg}"
        );
    }

    // ── Test C: synthetic 4×4×2 grid ─────────────────────────────────────────

    /// Invariant: for pixel_val = 1000 and dose_grid_scaling = 0.001,
    /// every dose_gy[i] = 1000 × 0.001 = 1.0 Gy (exact in f64 representation).
    /// Geometry tags rows=4, cols=4, n_frames=2 must be preserved.
    /// frame_offsets must equal [0.0, 1.0] from GridFrameOffsetVector "0\1".
    #[test]
    fn test_read_rt_dose_synthetic_grid() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("rt_dose.dcm");

        // 4×4 grid, 2 frames, all voxels = 1000, scaling = 0.001 → dose = 1.0 Gy.
        let obj = build_rt_dose_obj(4, 4, 2, 0.001, 1000);
        write_rt_dose_file(obj, &path);

        let grid = read_rt_dose(&path).expect("read_rt_dose synthetic");

        assert_eq!(grid.rows, 4, "rows");
        assert_eq!(grid.cols, 4, "cols");
        assert_eq!(grid.n_frames, 2, "n_frames");
        assert_eq!(grid.dose_gy.len(), 4 * 4 * 2, "dose_gy length");

        // 1000 × 0.001 = 1.0 is exact in binary f64.
        for (i, &v) in grid.dose_gy.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-12, "dose_gy[{i}] = {v}, expected 1.0");
        }

        assert_eq!(grid.dose_grid_scaling, 0.001, "dose_grid_scaling");
        assert_eq!(grid.dose_summation_type, "PLAN", "dose_summation_type");
        assert_eq!(grid.dose_type, "PHYSICAL", "dose_type");

        assert_eq!(grid.frame_offsets.len(), 2, "frame_offsets length");
        assert!(
            (grid.frame_offsets[0] - 0.0).abs() < 1e-12,
            "frame_offsets[0] = {}, expected 0.0",
            grid.frame_offsets[0]
        );
        assert!(
            (grid.frame_offsets[1] - 1.0).abs() < 1e-12,
            "frame_offsets[1] = {}, expected 1.0",
            grid.frame_offsets[1]
        );
    }

    // ── Test D: validation rejects mismatched voxel count ────────────────────

    /// Invariant: write_rt_dose must return Err when dose_gy.len() ≠ n_frames * rows * cols.
    /// Grid: rows=2, cols=2, n_frames=1 → expected 4 voxels; dose_gy has 5.
    #[test]
    fn test_write_rt_dose_rejects_mismatched_voxel_count() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("dose_bad.dcm");
        let grid = RtDoseGrid {
            rows: 2,
            cols: 2,
            n_frames: 1,
            dose_type: "PHYSICAL".to_owned(),
            dose_summation_type: "PLAN".to_owned(),
            dose_grid_scaling: 0.001,
            frame_offsets: vec![0.0],
            dose_gy: vec![0.0, 0.001, 0.002, 0.003, 0.004], // 5 values; expected 4
            image_position: None,
            image_orientation: None,
            pixel_spacing: None,
        };
        let result = write_rt_dose(&path, &grid);
        assert!(result.is_err(), "mismatched voxel count must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("dose_gy")
                || msg.contains("voxel")
                || msg.contains('5')
                || msg.contains('4'),
            "error message must reference dose_gy, voxel, or the count mismatch; got: {msg}"
        );
    }

    // ── Test E: round-trip write/read preserves all fields ───────────────────

    /// Invariant: write_rt_dose followed by read_rt_dose reconstructs:
    /// - rows, cols, n_frames exactly
    /// - dose_gy[k] within 1e-12 (values are exact multiples of 0.001: raw_u32 = k, scaling = 0.001)
    /// - dose_grid_scaling exactly
    /// - dose_summation_type exactly
    /// - frame_offsets[i] within 1e-12
    /// - image_position components within 1e-6
    /// - pixel_spacing components within 1e-6
    #[test]
    fn test_write_rt_dose_round_trip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("dose_rt.dcm");

        // dose_gy: [0.000, 0.001, ..., 0.031] — 32 values, all exact multiples of 0.001.
        // raw_u32[k] = k (0..31); reconstruction = k * 0.001 is exact in f64.
        let dose_gy: Vec<f64> = (0..32).map(|i| i as f64 * 0.001).collect();
        let grid = RtDoseGrid {
            rows: 4,
            cols: 4,
            n_frames: 2,
            dose_type: "PHYSICAL".to_owned(),
            dose_summation_type: "BEAM".to_owned(),
            dose_grid_scaling: 0.001,
            frame_offsets: vec![0.0, 5.0],
            dose_gy: dose_gy.clone(),
            image_position: Some([10.0, 20.0, 30.0]),
            image_orientation: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            pixel_spacing: Some([2.5, 2.5]),
        };

        write_rt_dose(&path, &grid).expect("write_rt_dose round-trip");
        let back = read_rt_dose(&path).expect("read_rt_dose round-trip");

        assert_eq!(back.rows, 4, "rows");
        assert_eq!(back.cols, 4, "cols");
        assert_eq!(back.n_frames, 2, "n_frames");
        assert_eq!(back.dose_gy.len(), 32, "dose_gy.len");

        for i in 0..32 {
            assert!(
                (back.dose_gy[i] - dose_gy[i]).abs() < 1e-12,
                "dose_gy[{i}]: got {}, expected {}",
                back.dose_gy[i],
                dose_gy[i]
            );
        }

        assert_eq!(back.dose_grid_scaling, 0.001, "dose_grid_scaling");
        assert_eq!(back.dose_summation_type, "BEAM", "dose_summation_type");

        assert!(
            (back.frame_offsets[0] - 0.0).abs() < 1e-12,
            "frame_offsets[0] = {}, expected 0.0",
            back.frame_offsets[0]
        );
        assert!(
            (back.frame_offsets[1] - 5.0).abs() < 1e-12,
            "frame_offsets[1] = {}, expected 5.0",
            back.frame_offsets[1]
        );

        let pos = back.image_position.expect("image_position must be Some");
        assert!(
            (pos[0] - 10.0).abs() < 1e-6,
            "image_position[0] = {}, expected 10.0",
            pos[0]
        );
        assert!(
            (pos[1] - 20.0).abs() < 1e-6,
            "image_position[1] = {}, expected 20.0",
            pos[1]
        );
        assert!(
            (pos[2] - 30.0).abs() < 1e-6,
            "image_position[2] = {}, expected 30.0",
            pos[2]
        );

        let ps = back.pixel_spacing.expect("pixel_spacing must be Some");
        assert!(
            (ps[0] - 2.5).abs() < 1e-6,
            "pixel_spacing[0] = {}, expected 2.5",
            ps[0]
        );
        assert!(
            (ps[1] - 2.5).abs() < 1e-6,
            "pixel_spacing[1] = {}, expected 2.5",
            ps[1]
        );
    }
}
