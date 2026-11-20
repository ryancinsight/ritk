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
    let samples = obj
        .element(DicomTag::new(0x0028, 0x0002))
        .ok()
        .and_then(|e| e.to_num::<u32>().ok())
        .unwrap_or(1);
    let pixel_bytes = obj
        .element(DicomTag::new(0x7FE0, 0x0010))
        .ok()
        .and_then(|e| e.to_bytes().ok())
        .map(|b| b.len());

    let (Some(rows), Some(cols), Some(bits)) = (rows, cols, bits) else {
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: "Rows/Columns/BitsAllocated must all be present".to_owned(),
        });
        return;
    };

    if rows == 0 || cols == 0 || bits == 0 {
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: format!("non-positive geometry rows={rows} cols={cols} bits={bits}"),
        });
        return;
    }

    let expected_bytes_per_frame =
        rows as usize * cols as usize * samples as usize * (bits as usize).div_ceil(8);
    if let Some(len) = pixel_bytes
        && len < expected_bytes_per_frame
    {
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: format!(
                "PixelData length {len} < expected single-frame {expected_bytes_per_frame} \
                     (rows={rows} cols={cols} samples={samples} bits={bits})"
            ),
        });
    }
}
