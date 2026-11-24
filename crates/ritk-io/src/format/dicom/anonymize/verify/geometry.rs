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

    // PS3.5 section 8.1.1 allows BitsAllocated 1 beside the multiples of 8,
    // and section 8.2 concatenates the bits of successive pixel cells, so
    // the first 1-bit frame ends within its sample count in bits, rounded up
    // to a byte.
    let expected_bytes_per_frame = u64::from(rows)
        .checked_mul(u64::from(cols))
        .and_then(|cells| cells.checked_mul(u64::from(samples)))
        .and_then(|frame_samples| {
            if bits == 1 {
                Some(frame_samples.div_ceil(8))
            } else {
                frame_samples.checked_mul(u64::from(bits.div_ceil(8)))
            }
        });
    let Some(expected_bytes_per_frame) = expected_bytes_per_frame else {
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: format!(
                "frame size exceeds 64 bits \
                 (rows={rows} cols={cols} samples={samples} bits={bits})"
            ),
        });
        return;
    };
    // A length beyond 64 bits cannot be short of a 64-bit frame size.
    if let Some(len) = pixel_bytes
        && u64::try_from(len).is_ok_and(|len| len < expected_bytes_per_frame)
    {
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: format!(
                "PixelData length {len} < expected single-frame {expected_bytes_per_frame} \
                     (rows={rows} cols={cols} samples={samples} bits={bits})"
            ),
        });
    }
}
