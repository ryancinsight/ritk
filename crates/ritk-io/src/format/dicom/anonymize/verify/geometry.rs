//! Geometry coherence of one parsed object: item 4 of the gate.

use super::{FileVerifyReport, VerifyIssue};
use ritk_dicom::dataset::DicomFile;
use ritk_dicom::dataset::DicomTag;

/// Float Pixel Data, Double Float Pixel Data and Pixel Data.
const PIXEL_DATA_ELEMENTS: [DicomTag; 3] = [
    DicomTag::new(0x7FE0, 0x0008),
    DicomTag::new(0x7FE0, 0x0009),
    DicomTag::new(0x7FE0, 0x0010),
];

pub(super) fn check_geometry(obj: &DicomFile, report: &mut FileVerifyReport) {
    if PIXEL_DATA_ELEMENTS
        .iter()
        .all(|tag| obj.element(*tag).is_err())
    {
        return;
    }
    let rows = obj
        .element(DicomTag::new(0x0028, 0x0010))
        .ok()
        .and_then(|e| e.to_num::<u32>().ok());
    let cols = obj
        .element(DicomTag::new(0x0028, 0x0011))
        .ok()
        .and_then(|e| e.to_num::<u32>().ok());
    let bits = obj
        .element(DicomTag::new(0x0028, 0x0100))
        .ok()
        .and_then(|e| e.to_num::<u32>().ok());
    // An absent SamplesPerPixel counts as 1; one that is present must be a
    // number.
    let samples = obj
        .element(DicomTag::new(0x0028, 0x0002))
        .map_or(Some(1), |e| e.to_num::<u32>().ok());
    let (Some(rows), Some(cols), Some(bits)) = (rows, cols, bits) else {
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: "Rows/Columns/BitsAllocated must all be present".to_owned(),
        });
        return;
    };

    let Some(samples) = samples else {
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: "SamplesPerPixel is present and is not a number".to_owned(),
        });
        return;
    };

    if rows == 0 || cols == 0 || samples == 0 || bits == 0 {
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: format!(
                "non-positive geometry rows={rows} cols={cols} samples={samples} bits={bits}"
            ),
        });
        return;
    }

    if let Err(error) = ritk_dicom::ScalarEncoding::from_dataset(obj) {
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: error.to_string(),
        });
        return;
    }
    if bits != 1 && !bits.is_multiple_of(8) {
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: format!("invalid DICOM BitsAllocated={bits}"),
        });
        return;
    }
    let frames = obj
        .get(DicomTag::new(0x0028, 0x0008))
        .map_or(Ok(1), |e| e.to_num::<u64>());
    let Ok(frames @ 1..) = frames else {
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: "NumberOfFrames must be a positive integer".into(),
        });
        return;
    };
    // Native YBR 4:2:2 stores two luma and two chroma components per pair.
    let subsampled = obj
        .get(DicomTag::new(0x0028, 0x0004))
        .and_then(|e| e.to_str().ok())
        .is_some_and(|p| matches!(p.trim(), "YBR_FULL_422" | "YBR_PARTIAL_422"));
    let stored_components = if subsampled && samples == 3 {
        2
    } else {
        samples
    };
    let expected = u64::from(rows)
        .checked_mul(u64::from(cols))
        .and_then(|n| n.checked_mul(u64::from(stored_components)))
        .and_then(|n| n.checked_mul(frames))
        .and_then(|n| {
            if bits == 1 {
                Some(n.div_ceil(8))
            } else {
                n.checked_mul(u64::from(bits / 8))
            }
        });
    let Some(expected) = expected else {
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: format!(
                "frame size exceeds 64 bits \
                 (rows={rows} cols={cols} samples={samples} bits={bits})"
            ),
        });
        return;
    };
    let pixel_bytes = PIXEL_DATA_ELEMENTS
        .iter()
        .find_map(|tag| obj.get(*tag))
        .and_then(|e| e.to_bytes().ok())
        .map(|bytes| bytes.len());
    if let Some(len) = pixel_bytes
        && u64::try_from(len).is_ok_and(|len| len < expected)
    {
        let extent = if frames == 1 {
            "single-frame".to_owned()
        } else {
            format!("{frames}-frame")
        };
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: format!("PixelData length {len} < expected {extent} {expected} (rows={rows} cols={cols} samples={samples} bits={bits})")
        });
    }
}
