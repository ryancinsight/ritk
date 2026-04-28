//! DICOM RT Structure Set reader.
//!
//! # Specification
//!
//! SOP Class UID: `1.2.840.10008.5.1.4.1.1.481.3` (RT Structure Set Storage)
//!
//! ## DICOM tags consumed
//! - `(3006,0002)` StructureSetLabel LO
//! - `(3006,0004)` StructureSetName LO (optional)
//! - `(3006,0020)` StructureSetROISequence SQ
//!   - `(3006,0022)` ROINumber IS
//!   - `(3006,0026)` ROIName LO
//!   - `(3006,0028)` ROIDescription ST (optional)
//! - `(3006,0080)` RTROIObservationsSequence SQ
//!   - `(3006,0084)` ReferencedROINumber IS
//!   - `(3006,00A4)` RTROIInterpretedType CS (optional)
//! - `(3006,0039)` ROIContourSequence SQ
//!   - `(3006,0084)` ReferencedROINumber IS
//!   - `(3006,002A)` ROIDisplayColor IS (3 values: R\G\B)
//!   - `(3006,0040)` ContourSequence SQ
//!     - `(3006,0042)` ContourGeometricType CS
//!     - `(3006,0050)` ContourData DS (X\Y\Z\X\Y\Z…)
//!
//! ## Invariants
//! 1. The file must carry SOP Class UID `1.2.840.10008.5.1.4.1.1.481.3`.
//! 2. ROIs are de-duplicated by ROINumber; the result is sorted ascending.
//! 3. Contour data length is always a multiple of 3 (X, Y, Z per point).
//! 4. `display_color` is `Some([r,g,b])` when present and parseable as three u8 values.
//! 5. `rt_roi_to_polydata` maps geometric type to the correct VTK cell bucket;
//!    unknown types fall back to `lines`.

use anyhow::{bail, Context, Result};
use dicom::core::value::Value;
use dicom::core::Tag;
use dicom::object::open_file;
use std::collections::HashMap;
use std::path::Path;

use crate::domain::vtk_data_object::VtkPolyData;

/// SOP Class UID for RT Structure Set Storage.
pub const RT_STRUCT_SOP_CLASS_UID: &str = "1.2.840.10008.5.1.4.1.1.481.3";

// ─── Domain types ───────────────────────────────────────────────────────────

/// A single contour slice within an RT ROI.
///
/// # Mathematical specification
///
/// A contour is a sequence of N ≥ 1 points in patient coordinate space (mm):
/// - `CLOSED_PLANAR`: N points lie on one plane; the polygon is implicitly closed.
/// - `OPEN_PLANAR`:   N ≥ 2 points on one plane; the polyline is open.
/// - `POINT`:         N = 1 control point.
///
/// Encoding: `X₀\Y₀\Z₀\X₁\Y₁\Z₁\…\X_{N-1}\Y_{N-1}\Z_{N-1}`
#[derive(Debug, Clone)]
pub struct RtContour {
    /// Geometric type: `"POINT"` | `"OPEN_PLANAR"` | `"CLOSED_PLANAR"`.
    pub geometric_type: String,
    /// Patient-coordinate points `[X, Y, Z]` in mm.
    pub points: Vec<[f64; 3]>,
}

/// ROI metadata and contours extracted from an RT Structure Set.
#[derive(Debug, Clone)]
pub struct RtRoiInfo {
    /// Unique ROI identifier within the structure set.
    pub roi_number: u32,
    /// Human-readable name.
    pub roi_name: String,
    /// Optional free-text description.
    pub roi_description: Option<String>,
    /// RT ROI Interpreted Type from `(3006,00A4)`, e.g. `"GTV"`, `"CTV"`, `"PTV"`.
    pub roi_interpreted_type: Option<String>,
    /// Display color `[R, G, B]` from `(3006,002A)`.
    pub display_color: Option<[u8; 3]>,
    /// Contour slices.
    pub contours: Vec<RtContour>,
}

/// Parsed representation of a DICOM RT Structure Set file.
#[derive(Debug, Clone)]
pub struct RtStructureSet {
    /// Structure Set Label `(3006,0002)`.
    pub structure_set_label: String,
    /// Structure Set Name `(3006,0004)` — optional.
    pub structure_set_name: Option<String>,
    /// ROIs sorted ascending by `roi_number`.
    pub rois: Vec<RtRoiInfo>,
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Parse a `\`-separated DS contour-data string into 3-D point triples.
///
/// # Invariant
/// Output length = `floor(parseable_token_count / 3)`.
/// Non-numeric tokens are silently discarded. The result always contains
/// complete `[X, Y, Z]` triples; partial trailing values are dropped.
fn parse_contour_data(s: &str) -> Vec<[f64; 3]> {
    let vals: Vec<f64> = s
        .split('\\')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    vals.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Parse a `\`-separated IS color string into an `[R, G, B]` u8 triple.
///
/// # Invariant
/// Returns `Some([r, g, b])` iff the string contains at least 3 parseable u8 values.
fn parse_color(s: &str) -> Option<[u8; 3]> {
    let v: Vec<u8> = s
        .split('\\')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    if v.len() >= 3 {
        Some([v[0], v[1], v[2]])
    } else {
        None
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Read and parse a DICOM RT Structure Set file.
///
/// # Errors
/// - Returns `Err` when the path does not exist or is unreadable.
/// - Returns `Err` when the SOP Class UID ≠ `1.2.840.10008.5.1.4.1.1.481.3`.
///
/// # Invariants
/// 1. ROIs are sorted ascending by `roi_number`.
/// 2. All contour data is fully parsed; partial triples are discarded.
/// 3. `structure_set_name` is `None` when the tag is absent or empty.
pub fn read_rt_struct<P: AsRef<Path>>(path: P) -> Result<RtStructureSet> {
    let path = path.as_ref();
    let obj = open_file(path).with_context(|| format!("open DICOM file: {}", path.display()))?;

    // Verify SOP Class UID (trim NUL padding and whitespace).
    let sop = obj.meta().media_storage_sop_class_uid();
    let sop = sop.trim_end_matches('\0').trim();
    if sop != RT_STRUCT_SOP_CLASS_UID {
        bail!(
            "SOP Class UID '{}' is not RT Structure Set Storage ({})",
            sop,
            RT_STRUCT_SOP_CLASS_UID
        );
    }

    // Structure Set Label (3006,0002) — mandatory per IOD; default to empty when absent.
    let structure_set_label = obj
        .element(Tag(0x3006, 0x0002))
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    // Structure Set Name (3006,0004) — optional.
    let structure_set_name = obj
        .element(Tag(0x3006, 0x0004))
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    // ── Step 1: Build ROI map from StructureSetROISequence (3006,0020) ──────
    let mut roi_map: HashMap<u32, RtRoiInfo> = HashMap::new();
    if let Ok(seq_elem) = obj.element(Tag(0x3006, 0x0020)) {
        if let Value::Sequence(seq) = seq_elem.value() {
            for item in seq.items() {
                let roi_number: u32 = item
                    .element(Tag(0x3006, 0x0022))
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);

                let roi_name = item
                    .element(Tag(0x3006, 0x0026))
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();

                let roi_description = item
                    .element(Tag(0x3006, 0x0028))
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());

                roi_map.insert(
                    roi_number,
                    RtRoiInfo {
                        roi_number,
                        roi_name,
                        roi_description,
                        roi_interpreted_type: None,
                        display_color: None,
                        contours: Vec::new(),
                    },
                );
            }
        }
    }

    // ── Step 2: Update roi_interpreted_type from RTROIObservationsSequence (3006,0080) ──
    if let Ok(seq_elem) = obj.element(Tag(0x3006, 0x0080)) {
        if let Value::Sequence(seq) = seq_elem.value() {
            for item in seq.items() {
                let ref_roi: u32 = item
                    .element(Tag(0x3006, 0x0084))
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);

                let interpreted_type = item
                    .element(Tag(0x3006, 0x00A4))
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());

                if let Some(roi) = roi_map.get_mut(&ref_roi) {
                    roi.roi_interpreted_type = interpreted_type;
                }
            }
        }
    }

    // ── Step 3: Update display_color + contours from ROIContourSequence (3006,0039) ──
    if let Ok(seq_elem) = obj.element(Tag(0x3006, 0x0039)) {
        if let Value::Sequence(seq) = seq_elem.value() {
            for item in seq.items() {
                let ref_roi: u32 = item
                    .element(Tag(0x3006, 0x0084))
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);

                // ROIDisplayColor (3006,002A) — IS, three backslash-separated values.
                let display_color = item
                    .element(Tag(0x3006, 0x002A))
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .and_then(|s| parse_color(s.trim()));

                // ContourSequence (3006,0040) — nested SQ inside each ROI contour item.
                let mut contours: Vec<RtContour> = Vec::new();
                if let Ok(cs_elem) = item.element(Tag(0x3006, 0x0040)) {
                    if let Value::Sequence(cs) = cs_elem.value() {
                        for ci in cs.items() {
                            let geometric_type = ci
                                .element(Tag(0x3006, 0x0042))
                                .ok()
                                .and_then(|e| e.to_str().ok())
                                .map(|s| s.trim().to_string())
                                .unwrap_or_default();

                            let points = ci
                                .element(Tag(0x3006, 0x0050))
                                .ok()
                                .and_then(|e| e.to_str().ok())
                                .map(|s| parse_contour_data(s.trim()))
                                .unwrap_or_default();

                            contours.push(RtContour {
                                geometric_type,
                                points,
                            });
                        }
                    }
                }

                if let Some(roi) = roi_map.get_mut(&ref_roi) {
                    if display_color.is_some() {
                        roi.display_color = display_color;
                    }
                    roi.contours = contours;
                }
            }
        }
    }

    // Collect, sort ascending by roi_number, and return.
    let mut rois: Vec<RtRoiInfo> = roi_map.into_values().collect();
    rois.sort_by_key(|r| r.roi_number);

    Ok(RtStructureSet {
        structure_set_label,
        structure_set_name,
        rois,
    })
}

/// Convert an [`RtRoiInfo`] to a [`VtkPolyData`].
///
/// # Geometric type mapping
/// | DICOM geometric type | VTK cell bucket        |
/// |----------------------|------------------------|
/// | `CLOSED_PLANAR`      | `poly.polygons`        |
/// | `OPEN_PLANAR`        | `poly.lines`           |
/// | `POINT`              | `poly.vertices`        |
/// | (unknown)            | `poly.lines` (fallback)|
///
/// # Invariants
/// - `poly.points` accumulates all contour points in contour order.
/// - Each cell references point indices computed from a running offset.
/// - Patient coordinates (`f64`, mm) are cast to `f32` for VTK storage.
pub fn rt_roi_to_polydata(roi: &RtRoiInfo) -> VtkPolyData {
    let mut poly = VtkPolyData::default();
    let mut offset: u32 = 0;

    for contour in &roi.contours {
        let n = contour.points.len() as u32;

        // Accumulate points (f64 patient coords → f32 VTK coords).
        for p in &contour.points {
            poly.points.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        }

        // Build the contiguous index slice for this cell.
        let indices: Vec<u32> = (offset..offset + n).collect();

        match contour.geometric_type.as_str() {
            "CLOSED_PLANAR" => poly.polygons.push(indices),
            "OPEN_PLANAR" => poly.lines.push(indices),
            "POINT" => poly.vertices.push(indices),
            _ => poly.lines.push(indices),
        }

        offset += n;
    }

    poly
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::core::header::Length;
    use dicom::core::value::DataSetSequence;
    use dicom::core::value::Value as DicomValue;
    use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
    use dicom::object::meta::FileMetaTableBuilder;
    use dicom::object::InMemDicomObject;

    // ── Test helpers ─────────────────────────────────────────────────────────

    /// Write a minimal RT Structure Set DICOM file carrying the given object.
    fn write_rt_struct_file(obj: InMemDicomObject, path: &std::path::Path) {
        obj.with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.481.3")
                .media_storage_sop_instance_uid("2.25.2")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("meta")
        .write_to_file(path)
        .expect("write");
    }

    /// Write a DICOM file with an arbitrary SOP Class UID and no RT tags.
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
        obj.with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(sop)
                .media_storage_sop_instance_uid("2.25.99")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("meta")
        .write_to_file(path)
        .expect("write");
    }

    /// Build a minimal RT Structure Set InMemDicomObject for one ROI with one contour.
    ///
    /// - Structure Set Label = "TestPlan"
    /// - ROI number=1, name="GTV", color="255\0\0"
    /// - 1 CLOSED_PLANAR contour: [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
    fn build_single_roi_obj() -> InMemDicomObject {
        let contour_data = "0.0\\0.0\\0.0\\1.0\\0.0\\0.0\\1.0\\1.0\\0.0\\0.0\\1.0\\0.0";

        // Contour item: (3006,0042) + (3006,0050)
        let mut contour_item = InMemDicomObject::new_empty();
        contour_item.put(DataElement::new(
            Tag(0x3006, 0x0042),
            VR::CS,
            PrimitiveValue::from("CLOSED_PLANAR"),
        ));
        contour_item.put(DataElement::new(
            Tag(0x3006, 0x0050),
            VR::DS,
            PrimitiveValue::from(contour_data),
        ));
        let contour_seq = DataSetSequence::new(vec![contour_item], Length::UNDEFINED);

        // ROI contour item: (3006,0084) + (3006,002A) + (3006,0040)
        let mut roi_contour_item = InMemDicomObject::new_empty();
        roi_contour_item.put(DataElement::new(
            Tag(0x3006, 0x0084),
            VR::IS,
            PrimitiveValue::from("1"),
        ));
        roi_contour_item.put(DataElement::new(
            Tag(0x3006, 0x002A),
            VR::IS,
            PrimitiveValue::from("255\\0\\0"),
        ));
        roi_contour_item.put(DataElement::new(
            Tag(0x3006, 0x0040),
            VR::SQ,
            DicomValue::from(contour_seq),
        ));
        let roi_contour_seq = DataSetSequence::new(vec![roi_contour_item], Length::UNDEFINED);

        // StructureSetROI item: (3006,0022) + (3006,0026)
        let mut roi_item = InMemDicomObject::new_empty();
        roi_item.put(DataElement::new(
            Tag(0x3006, 0x0022),
            VR::IS,
            PrimitiveValue::from("1"),
        ));
        roi_item.put(DataElement::new(
            Tag(0x3006, 0x0026),
            VR::LO,
            PrimitiveValue::from("GTV"),
        ));
        let roi_seq = DataSetSequence::new(vec![roi_item], Length::UNDEFINED);

        // Root object
        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x3006, 0x0002),
            VR::LO,
            PrimitiveValue::from("TestPlan"),
        ));
        obj.put(DataElement::new(
            Tag(0x3006, 0x0020),
            VR::SQ,
            DicomValue::from(roi_seq),
        ));
        obj.put(DataElement::new(
            Tag(0x3006, 0x0039),
            VR::SQ,
            DicomValue::from(roi_contour_seq),
        ));
        obj
    }

    // ── Tests 1–2: error paths ────────────────────────────────────────────────

    /// Invariant: a nonexistent path must produce Err.
    #[test]
    fn test_read_rt_struct_missing_file_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nonexistent.dcm");
        let result = read_rt_struct(&path);
        assert!(result.is_err(), "nonexistent path must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("nonexistent") || msg.contains("open") || msg.contains("No such"),
            "error must mention the open failure; got: {msg}"
        );
    }

    /// Invariant: a file whose SOP Class UID ≠ RT Structure Set must produce Err
    /// containing the rejected UID in the message.
    #[test]
    fn test_read_rt_struct_wrong_sop_class_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("wrong_sop.dcm");
        write_wrong_sop_file("1.2.840.10008.5.1.4.1.1.2", &path);

        let result = read_rt_struct(&path);
        assert!(result.is_err(), "wrong SOP class must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("1.2.840.10008.5.1.4.1.1.2"),
            "error must contain the rejected SOP UID; got: {msg}"
        );
    }

    // ── Test 3: single ROI, CLOSED_PLANAR ────────────────────────────────────

    /// Invariant: a single-ROI RT Structure Set with one CLOSED_PLANAR contour
    /// must parse the label, roi_name, display_color, and all four contour points.
    #[test]
    fn test_read_rt_struct_single_roi_closed_planar() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("single_roi.dcm");
        write_rt_struct_file(build_single_roi_obj(), &path);

        let ss = read_rt_struct(&path).expect("read_rt_struct must succeed");

        assert_eq!(
            ss.structure_set_label, "TestPlan",
            "structure_set_label must be 'TestPlan'"
        );
        assert_eq!(ss.rois.len(), 1, "must contain exactly 1 ROI");

        let roi = &ss.rois[0];
        assert_eq!(roi.roi_name, "GTV", "roi_name must be 'GTV'");
        assert_eq!(
            roi.display_color,
            Some([255, 0, 0]),
            "display_color must be [255, 0, 0]"
        );
        assert_eq!(roi.contours.len(), 1, "must contain 1 contour");

        let contour = &roi.contours[0];
        assert_eq!(
            contour.geometric_type, "CLOSED_PLANAR",
            "geometric_type must be CLOSED_PLANAR"
        );
        assert_eq!(contour.points.len(), 4, "contour must contain 4 points");
        assert_eq!(
            contour.points[0],
            [0.0, 0.0, 0.0],
            "first point must be [0, 0, 0]"
        );
        assert_eq!(
            contour.points[1],
            [1.0, 0.0, 0.0],
            "second point must be [1, 0, 0]"
        );
        assert_eq!(
            contour.points[2],
            [1.0, 1.0, 0.0],
            "third point must be [1, 1, 0]"
        );
        assert_eq!(
            contour.points[3],
            [0.0, 1.0, 0.0],
            "fourth point must be [0, 1, 0]"
        );
    }

    // ── Test 4: two ROIs, sorted by number ───────────────────────────────────

    /// Invariant: when two ROIs are present, both are returned sorted ascending
    /// by roi_number regardless of insertion order in the DICOM sequence.
    #[test]
    fn test_read_rt_struct_two_rois() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("two_rois.dcm");

        // ROI items in reverse number order to exercise the sort.
        let mut roi_item2 = InMemDicomObject::new_empty();
        roi_item2.put(DataElement::new(
            Tag(0x3006, 0x0022),
            VR::IS,
            PrimitiveValue::from("2"),
        ));
        roi_item2.put(DataElement::new(
            Tag(0x3006, 0x0026),
            VR::LO,
            PrimitiveValue::from("PTV"),
        ));

        let mut roi_item1 = InMemDicomObject::new_empty();
        roi_item1.put(DataElement::new(
            Tag(0x3006, 0x0022),
            VR::IS,
            PrimitiveValue::from("1"),
        ));
        roi_item1.put(DataElement::new(
            Tag(0x3006, 0x0026),
            VR::LO,
            PrimitiveValue::from("GTV"),
        ));

        // Insert ROI 2 first to exercise sort correctness.
        let roi_seq = DataSetSequence::new(vec![roi_item2, roi_item1], Length::UNDEFINED);

        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x3006, 0x0002),
            VR::LO,
            PrimitiveValue::from("TwoPlan"),
        ));
        obj.put(DataElement::new(
            Tag(0x3006, 0x0020),
            VR::SQ,
            DicomValue::from(roi_seq),
        ));

        write_rt_struct_file(obj, &path);

        let ss = read_rt_struct(&path).expect("read_rt_struct must succeed");
        assert_eq!(ss.rois.len(), 2, "must contain 2 ROIs");
        assert_eq!(ss.rois[0].roi_number, 1, "first ROI must have number 1");
        assert_eq!(ss.rois[0].roi_name, "GTV", "first ROI must be GTV");
        assert_eq!(ss.rois[1].roi_number, 2, "second ROI must have number 2");
        assert_eq!(ss.rois[1].roi_name, "PTV", "second ROI must be PTV");
    }

    // ── Test 5: roi_interpreted_type from observations sequence ──────────────

    /// Invariant: ROI Observations Sequence (3006,0080) must populate
    /// `roi_interpreted_type` on the matching ROI.
    #[test]
    fn test_read_rt_struct_roi_interpreted_type() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("roi_type.dcm");

        // StructureSetROI item.
        let mut roi_item = InMemDicomObject::new_empty();
        roi_item.put(DataElement::new(
            Tag(0x3006, 0x0022),
            VR::IS,
            PrimitiveValue::from("1"),
        ));
        roi_item.put(DataElement::new(
            Tag(0x3006, 0x0026),
            VR::LO,
            PrimitiveValue::from("GTV"),
        ));
        let roi_seq = DataSetSequence::new(vec![roi_item], Length::UNDEFINED);

        // RT ROI Observations item.
        let mut obs_item = InMemDicomObject::new_empty();
        obs_item.put(DataElement::new(
            Tag(0x3006, 0x0084),
            VR::IS,
            PrimitiveValue::from("1"),
        ));
        obs_item.put(DataElement::new(
            Tag(0x3006, 0x00A4),
            VR::CS,
            PrimitiveValue::from("GTV"),
        ));
        let obs_seq = DataSetSequence::new(vec![obs_item], Length::UNDEFINED);

        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x3006, 0x0002),
            VR::LO,
            PrimitiveValue::from("ObsPlan"),
        ));
        obj.put(DataElement::new(
            Tag(0x3006, 0x0020),
            VR::SQ,
            DicomValue::from(roi_seq),
        ));
        obj.put(DataElement::new(
            Tag(0x3006, 0x0080),
            VR::SQ,
            DicomValue::from(obs_seq),
        ));

        write_rt_struct_file(obj, &path);

        let ss = read_rt_struct(&path).expect("read_rt_struct must succeed");
        assert_eq!(ss.rois.len(), 1, "must contain 1 ROI");
        assert_eq!(
            ss.rois[0].roi_interpreted_type,
            Some("GTV".to_string()),
            "roi_interpreted_type must be Some(\"GTV\")"
        );
    }

    // ── Tests 6–8: rt_roi_to_polydata ─────────────────────────────────────────

    /// Invariant: a single CLOSED_PLANAR contour (unit square, 4 points) must
    /// produce exactly 1 polygon cell containing 4 point indices, 4 points total,
    /// with no lines or vertex cells.
    #[test]
    fn test_rt_roi_to_polydata_closed_planar() {
        let roi = RtRoiInfo {
            roi_number: 1,
            roi_name: "GTV".to_string(),
            roi_description: None,
            roi_interpreted_type: None,
            display_color: None,
            contours: vec![RtContour {
                geometric_type: "CLOSED_PLANAR".to_string(),
                points: vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
            }],
        };

        let poly = rt_roi_to_polydata(&roi);

        assert_eq!(poly.points.len(), 4, "poly.points must contain 4 points");
        assert_eq!(poly.polygons.len(), 1, "must have exactly 1 polygon");
        assert_eq!(
            poly.polygons[0].len(),
            4,
            "polygon cell must reference 4 indices"
        );
        assert_eq!(poly.lines.len(), 0, "lines must be empty");
        assert_eq!(poly.vertices.len(), 0, "vertices must be empty");

        // Indices must be a contiguous range starting at 0.
        assert_eq!(
            poly.polygons[0],
            vec![0, 1, 2, 3],
            "polygon indices must be [0, 1, 2, 3]"
        );

        // Verify point coordinate fidelity (f64 → f32 cast).
        assert_eq!(
            poly.points[0],
            [0.0_f32, 0.0, 0.0],
            "first point must be [0, 0, 0]"
        );
        assert_eq!(
            poly.points[2],
            [1.0_f32, 1.0, 0.0],
            "third point must be [1, 1, 0]"
        );
    }

    /// Invariant: a single OPEN_PLANAR contour (3 points) must produce exactly
    /// 1 line cell and zero polygon or vertex cells.
    #[test]
    fn test_rt_roi_to_polydata_open_planar() {
        let roi = RtRoiInfo {
            roi_number: 2,
            roi_name: "Wire".to_string(),
            roi_description: None,
            roi_interpreted_type: None,
            display_color: None,
            contours: vec![RtContour {
                geometric_type: "OPEN_PLANAR".to_string(),
                points: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0]],
            }],
        };

        let poly = rt_roi_to_polydata(&roi);

        assert_eq!(poly.points.len(), 3, "poly.points must contain 3 points");
        assert_eq!(poly.lines.len(), 1, "must have exactly 1 line");
        assert_eq!(poly.lines[0].len(), 3, "line cell must reference 3 indices");
        assert_eq!(poly.polygons.len(), 0, "polygons must be empty");
        assert_eq!(poly.vertices.len(), 0, "vertices must be empty");
        assert_eq!(
            poly.lines[0],
            vec![0, 1, 2],
            "line indices must be [0, 1, 2]"
        );
    }

    /// Invariant: an ROI with 1 CLOSED_PLANAR + 1 OPEN_PLANAR contour must
    /// populate both poly.polygons and poly.lines, with correct running offsets.
    #[test]
    fn test_rt_roi_to_polydata_mixed_contours() {
        let roi = RtRoiInfo {
            roi_number: 3,
            roi_name: "Mixed".to_string(),
            roi_description: None,
            roi_interpreted_type: None,
            display_color: None,
            contours: vec![
                RtContour {
                    geometric_type: "CLOSED_PLANAR".to_string(),
                    points: vec![
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                },
                RtContour {
                    geometric_type: "OPEN_PLANAR".to_string(),
                    points: vec![[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [7.0, 1.0, 0.0]],
                },
            ],
        };

        let poly = rt_roi_to_polydata(&roi);

        // Total points: 4 (CLOSED_PLANAR) + 3 (OPEN_PLANAR) = 7.
        assert_eq!(
            poly.points.len(),
            7,
            "poly.points must contain 7 total points"
        );

        // One polygon cell covering the first 4 points.
        assert_eq!(poly.polygons.len(), 1, "must have 1 polygon");
        assert_eq!(
            poly.polygons[0],
            vec![0, 1, 2, 3],
            "polygon indices must be [0, 1, 2, 3]"
        );

        // One line cell covering the next 3 points (offset = 4).
        assert_eq!(poly.lines.len(), 1, "must have 1 line");
        assert_eq!(
            poly.lines[0],
            vec![4, 5, 6],
            "line indices must be [4, 5, 6] (offset by 4)"
        );

        assert_eq!(poly.vertices.len(), 0, "vertices must be empty");

        // Spot-check point coordinates.
        assert_eq!(
            poly.points[4],
            [5.0_f32, 0.0, 0.0],
            "point[4] must be [5, 0, 0] from the OPEN_PLANAR contour"
        );
    }
}
