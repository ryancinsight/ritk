//! RT Plan Storage (SOP Class 1.2.840.10008.5.1.4.1.1.481.5) reader.
//!
//! # Specification
//!
//! An RT Plan file contains radiotherapy treatment planning metadata.
//! Key structures:
//! - (300A,0010) DoseReferenceSequence: prescribed dose constraints.
//! - (300A,0070) FractionGroupSequence: fractions and beam references.
//! - (300A,00B0) BeamSequence: treatment beams with geometry and MUs.
//!
//! ## Tags consumed
//! - (300A,0002) RTPlanLabel LO
//! - (300A,0003) RTPlanName LO (optional)
//! - (300A,0004) RTPlanDescription ST (optional)
//! - (300A,000A) PlanIntent CS (optional)
//! - (300A,00B0) BeamSequence SQ
//!   - (300A,00C0) BeamNumber IS
//!   - (300A,00C2) BeamName LO
//!   - (300A,00C3) BeamDescription ST (optional)
//!   - (300A,00C6) RadiationType CS
//!   - (300A,00CE) TreatmentDeliveryType CS
//!   - (300A,0110) NumberOfControlPoints IS
//! - (300A,0070) FractionGroupSequence SQ
//!   - (300A,0071) FractionGroupNumber IS
//!   - (300A,0078) NumberOfFractionsPlanned IS
//!   - (300A,00B6) ReferencedBeamSequence SQ
//!     - (300A,00C0) ReferencedBeamNumber IS

use anyhow::{bail, Context, Result};
use dicom::core::header::Length;
use dicom::core::value::DataSetSequence;
use dicom::core::value::Value;
use dicom::core::Tag;
use dicom::core::{DataElement, PrimitiveValue, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::open_file;
use dicom::object::InMemDicomObject;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// SOP Class UID for RT Plan Storage.
pub const RT_PLAN_SOP_CLASS_UID: &str = "1.2.840.10008.5.1.4.1.1.481.5";

// ── Domain types ──────────────────────────────────────────────────────────────

/// Metadata for a single treatment beam from BeamSequence (300A,00B0).
#[derive(Debug, Clone)]
pub struct RtBeamInfo {
    /// BeamNumber (300A,00C0).
    pub beam_number: u32,
    /// BeamName (300A,00C2).
    pub beam_name: String,
    /// BeamDescription (300A,00C3).
    pub beam_description: String,
    /// RadiationType (300A,00C6): PHOTON, ELECTRON, NEUTRON, PROTON, etc.
    pub radiation_type: String,
    /// TreatmentDeliveryType (300A,00CE): TREATMENT, DRR, CONTINUATION, etc.
    pub treatment_delivery_type: String,
    /// NumberOfControlPoints (300A,0110).
    pub n_control_points: u32,
}

/// A fraction group from FractionGroupSequence (300A,0070).
#[derive(Debug, Clone)]
pub struct RtFractionGroup {
    /// FractionGroupNumber (300A,0071).
    pub fraction_group_number: u32,
    /// NumberOfFractionsPlanned (300A,0078).
    pub n_fractions_planned: u32,
    /// BeamNumbers referenced via ReferencedBeamSequence (300A,00B6) items,
    /// each carrying ReferencedBeamNumber (300A,00C0).
    pub referenced_beam_numbers: Vec<u32>,
}

/// Parsed representation of a DICOM RT Plan file.
///
/// # Invariants
/// 1. SOP Class UID must equal `RT_PLAN_SOP_CLASS_UID`.
/// 2. `beams` order follows encounter order in BeamSequence (300A,00B0).
/// 3. `fraction_groups` order follows encounter order in FractionGroupSequence (300A,0070).
#[derive(Debug, Clone)]
pub struct RtPlanInfo {
    /// RTPlanLabel (300A,0002).
    pub rt_plan_label: String,
    /// RTPlanName (300A,0003).
    pub rt_plan_name: String,
    /// RTPlanDescription (300A,0004).
    pub rt_plan_description: String,
    /// PlanIntent (300A,000A): CURATIVE, PALLIATIVE, PROPHYLACTIC, VERIFICATION, etc.
    pub plan_intent: String,
    /// Beams from BeamSequence (300A,00B0).
    pub beams: Vec<RtBeamInfo>,
    /// Fraction groups from FractionGroupSequence (300A,0070).
    pub fraction_groups: Vec<RtFractionGroup>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Read an RT Plan Storage DICOM file at `path` into an [`RtPlanInfo`].
///
/// # Errors
/// - `path` does not exist or is not readable.
/// - The file's `MediaStorageSOPClassUID` ≠ `1.2.840.10008.5.1.4.1.1.481.5`.
pub fn read_rt_plan<P: AsRef<Path>>(path: P) -> Result<RtPlanInfo> {
    let path = path.as_ref();
    let obj = open_file(path).with_context(|| format!("open DICOM file: {}", path.display()))?;

    // Validate SOP Class UID.
    let sop = obj.meta().media_storage_sop_class_uid();
    let sop = sop.trim_end_matches('\0').trim();
    if sop != RT_PLAN_SOP_CLASS_UID {
        bail!(
            "SOP Class UID '{}' is not RT Plan Storage ({})",
            sop,
            RT_PLAN_SOP_CLASS_UID
        );
    }

    // ── Plan-level string fields ─────────────────────────────────────────────

    let rt_plan_label = obj
        .element(Tag(0x300A, 0x0002))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .unwrap_or_default();

    let rt_plan_name = obj
        .element(Tag(0x300A, 0x0003))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .unwrap_or_default();

    let rt_plan_description = obj
        .element(Tag(0x300A, 0x0004))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .unwrap_or_default();

    let plan_intent = obj
        .element(Tag(0x300A, 0x000A))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .unwrap_or_default();

    // ── BeamSequence (300A,00B0) ─────────────────────────────────────────────

    let beams: Vec<RtBeamInfo> = match obj.element(Tag(0x300A, 0x00B0)) {
        Ok(elem) => match elem.value() {
            Value::Sequence(seq) => seq
                .items()
                .iter()
                .map(|item| {
                    let beam_number: u32 = item
                        .element(Tag(0x300A, 0x00C0))
                        .ok()
                        .and_then(|e| e.to_str().ok().and_then(|s| s.trim().parse().ok()))
                        .unwrap_or(0);
                    let beam_name = item
                        .element(Tag(0x300A, 0x00C2))
                        .ok()
                        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
                        .unwrap_or_default();
                    let beam_description = item
                        .element(Tag(0x300A, 0x00C3))
                        .ok()
                        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
                        .unwrap_or_default();
                    let radiation_type = item
                        .element(Tag(0x300A, 0x00C6))
                        .ok()
                        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
                        .unwrap_or_default();
                    let treatment_delivery_type = item
                        .element(Tag(0x300A, 0x00CE))
                        .ok()
                        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
                        .unwrap_or_default();
                    let n_control_points: u32 = item
                        .element(Tag(0x300A, 0x0110))
                        .ok()
                        .and_then(|e| e.to_str().ok().and_then(|s| s.trim().parse().ok()))
                        .unwrap_or(0);
                    RtBeamInfo {
                        beam_number,
                        beam_name,
                        beam_description,
                        radiation_type,
                        treatment_delivery_type,
                        n_control_points,
                    }
                })
                .collect(),
            _ => Vec::new(),
        },
        Err(_) => Vec::new(),
    };

    // ── FractionGroupSequence (300A,0070) ────────────────────────────────────

    let fraction_groups: Vec<RtFractionGroup> = match obj.element(Tag(0x300A, 0x0070)) {
        Ok(elem) => match elem.value() {
            Value::Sequence(seq) => seq
                .items()
                .iter()
                .map(|item| {
                    let fraction_group_number: u32 = item
                        .element(Tag(0x300A, 0x0071))
                        .ok()
                        .and_then(|e| e.to_str().ok().and_then(|s| s.trim().parse().ok()))
                        .unwrap_or(0);
                    let n_fractions_planned: u32 = item
                        .element(Tag(0x300A, 0x0078))
                        .ok()
                        .and_then(|e| e.to_str().ok().and_then(|s| s.trim().parse().ok()))
                        .unwrap_or(0);

                    // ReferencedBeamSequence (300A,00B6) → ReferencedBeamNumber (300A,00C0)
                    let referenced_beam_numbers: Vec<u32> = item
                        .element(Tag(0x300A, 0x00B6))
                        .ok()
                        .and_then(|e| match e.value() {
                            Value::Sequence(s) => Some(
                                s.items()
                                    .iter()
                                    .map(|bi| {
                                        bi.element(Tag(0x300A, 0x00C0))
                                            .ok()
                                            .and_then(|be| {
                                                be.to_str()
                                                    .ok()
                                                    .and_then(|sv| sv.trim().parse().ok())
                                            })
                                            .unwrap_or(0)
                                    })
                                    .collect(),
                            ),
                            _ => None,
                        })
                        .unwrap_or_default();

                    RtFractionGroup {
                        fraction_group_number,
                        n_fractions_planned,
                        referenced_beam_numbers,
                    }
                })
                .collect(),
            _ => Vec::new(),
        },
        Err(_) => Vec::new(),
    };

    tracing::debug!(
        "read_rt_plan: label='{}' n_beams={} n_fraction_groups={}",
        rt_plan_label,
        beams.len(),
        fraction_groups.len(),
    );

    Ok(RtPlanInfo {
        rt_plan_label,
        rt_plan_name,
        rt_plan_description,
        plan_intent,
        beams,
        fraction_groups,
    })
}

// ── Public writer ─────────────────────────────────────────────────────────────

/// Write an [`RtPlanInfo`] to a DICOM RT Plan Storage file at `path`.
///
/// # Write/Read Invariant
///
/// All plan-level strings (`rt_plan_label`, `rt_plan_name`, `rt_plan_description`,
/// `plan_intent`) and all beam and fraction group fields are preserved through
/// the DICOM write-read cycle without loss.
///
/// # Errors
/// - File cannot be created or written at `path`.
pub fn write_rt_plan<P: AsRef<Path>>(path: P, plan: &RtPlanInfo) -> Result<()> {
    let path = path.as_ref();

    // ── SOP Instance UID ──────────────────────────────────────────────────────
    static RT_PLAN_UID_COUNTER: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let n = RT_PLAN_UID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let sop_instance_uid = format!("2.25.{}.{}", t, n);

    // ── Beam items ────────────────────────────────────────────────────────────
    let beam_items: Vec<InMemDicomObject> = plan
        .beams
        .iter()
        .map(|beam| {
            let mut item = InMemDicomObject::new_empty();
            item.put(DataElement::new(
                Tag(0x300A, 0x00C0),
                VR::IS,
                PrimitiveValue::from(beam.beam_number.to_string().as_str()),
            ));
            item.put(DataElement::new(
                Tag(0x300A, 0x00C2),
                VR::LO,
                PrimitiveValue::from(beam.beam_name.as_str()),
            ));
            item.put(DataElement::new(
                Tag(0x300A, 0x00C3),
                VR::ST,
                PrimitiveValue::from(beam.beam_description.as_str()),
            ));
            item.put(DataElement::new(
                Tag(0x300A, 0x00C6),
                VR::CS,
                PrimitiveValue::from(beam.radiation_type.as_str()),
            ));
            item.put(DataElement::new(
                Tag(0x300A, 0x00CE),
                VR::CS,
                PrimitiveValue::from(beam.treatment_delivery_type.as_str()),
            ));
            item.put(DataElement::new(
                Tag(0x300A, 0x0110),
                VR::IS,
                PrimitiveValue::from(beam.n_control_points.to_string().as_str()),
            ));
            item
        })
        .collect();

    // ── Fraction group items ──────────────────────────────────────────────────
    let fg_items: Vec<InMemDicomObject> = plan
        .fraction_groups
        .iter()
        .map(|fg| {
            let ref_beam_items: Vec<InMemDicomObject> = fg
                .referenced_beam_numbers
                .iter()
                .map(|&bn| {
                    let mut ref_item = InMemDicomObject::new_empty();
                    ref_item.put(DataElement::new(
                        Tag(0x300A, 0x00C0),
                        VR::IS,
                        PrimitiveValue::from(bn.to_string().as_str()),
                    ));
                    ref_item
                })
                .collect();

            let mut item = InMemDicomObject::new_empty();
            item.put(DataElement::new(
                Tag(0x300A, 0x0071),
                VR::IS,
                PrimitiveValue::from(fg.fraction_group_number.to_string().as_str()),
            ));
            item.put(DataElement::new(
                Tag(0x300A, 0x0078),
                VR::IS,
                PrimitiveValue::from(fg.n_fractions_planned.to_string().as_str()),
            ));
            item.put(DataElement::new(
                Tag(0x300A, 0x00B6),
                VR::SQ,
                Value::from(DataSetSequence::new(ref_beam_items, Length::UNDEFINED)),
            ));
            item
        })
        .collect();

    // ── Root object ───────────────────────────────────────────────────────────
    let mut obj = InMemDicomObject::new_empty();

    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from(RT_PLAN_SOP_CLASS_UID),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from(sop_instance_uid.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from("RTPLAN"),
    ));
    obj.put(DataElement::new(
        Tag(0x300A, 0x0002),
        VR::LO,
        PrimitiveValue::from(plan.rt_plan_label.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x300A, 0x0003),
        VR::LO,
        PrimitiveValue::from(plan.rt_plan_name.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x300A, 0x0004),
        VR::ST,
        PrimitiveValue::from(plan.rt_plan_description.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x300A, 0x000A),
        VR::CS,
        PrimitiveValue::from(plan.plan_intent.as_str()),
    ));

    if !beam_items.is_empty() {
        obj.put(DataElement::new(
            Tag(0x300A, 0x00B0),
            VR::SQ,
            Value::from(DataSetSequence::new(beam_items, Length::UNDEFINED)),
        ));
    }
    if !fg_items.is_empty() {
        obj.put(DataElement::new(
            Tag(0x300A, 0x0070),
            VR::SQ,
            Value::from(DataSetSequence::new(fg_items, Length::UNDEFINED)),
        ));
    }

    // ── Write ─────────────────────────────────────────────────────────────────
    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(RT_PLAN_SOP_CLASS_UID)
            .media_storage_sop_instance_uid(sop_instance_uid.as_str())
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .with_context(|| "build RT Plan file meta")?
    .write_to_file(path)
    .with_context(|| format!("write RT Plan to {}", path.display()))?;

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::core::header::Length;
    use dicom::core::value::DataSetSequence;
    use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
    use dicom::object::meta::FileMetaTableBuilder;
    use dicom::object::InMemDicomObject;

    // ── Test helpers ─────────────────────────────────────────────────────────

    /// Write a minimal RT Plan DICOM Part-10 file carrying the given object.
    fn write_rt_plan_file(obj: InMemDicomObject, path: &std::path::Path) {
        obj.with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(RT_PLAN_SOP_CLASS_UID)
                .media_storage_sop_instance_uid("2.25.1")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("meta build")
        .write_to_file(path)
        .expect("write RT Plan file");
    }

    /// Write a DICOM file with an arbitrary SOP Class UID (no RT Plan tags).
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
        .expect("write wrong-SOP file");
    }

    /// Build a beam item for BeamSequence (300A,00B0).
    fn make_beam_item(
        beam_number: u32,
        beam_name: &str,
        radiation_type: &str,
        n_control_points: u32,
    ) -> InMemDicomObject {
        let mut item = InMemDicomObject::new_empty();
        item.put(DataElement::new(
            Tag(0x300A, 0x00C0),
            VR::IS,
            PrimitiveValue::from(beam_number.to_string().as_str()),
        ));
        item.put(DataElement::new(
            Tag(0x300A, 0x00C2),
            VR::LO,
            PrimitiveValue::from(beam_name),
        ));
        item.put(DataElement::new(
            Tag(0x300A, 0x00C3),
            VR::ST,
            PrimitiveValue::from(""),
        ));
        item.put(DataElement::new(
            Tag(0x300A, 0x00C6),
            VR::CS,
            PrimitiveValue::from(radiation_type),
        ));
        item.put(DataElement::new(
            Tag(0x300A, 0x00CE),
            VR::CS,
            PrimitiveValue::from("TREATMENT"),
        ));
        item.put(DataElement::new(
            Tag(0x300A, 0x0110),
            VR::IS,
            PrimitiveValue::from(n_control_points.to_string().as_str()),
        ));
        item
    }

    /// Build a fraction group item for FractionGroupSequence (300A,0070),
    /// including a nested ReferencedBeamSequence (300A,00B6).
    fn make_fraction_group_item(
        fg_number: u32,
        n_fractions: u32,
        ref_beam_numbers: &[u32],
    ) -> InMemDicomObject {
        let ref_beam_items: Vec<InMemDicomObject> = ref_beam_numbers
            .iter()
            .map(|&bn| {
                let mut ref_item = InMemDicomObject::new_empty();
                ref_item.put(DataElement::new(
                    Tag(0x300A, 0x00C0),
                    VR::IS,
                    PrimitiveValue::from(bn.to_string().as_str()),
                ));
                ref_item
            })
            .collect();
        let ref_beam_seq = DataSetSequence::new(ref_beam_items, Length::UNDEFINED);

        let mut item = InMemDicomObject::new_empty();
        item.put(DataElement::new(
            Tag(0x300A, 0x0071),
            VR::IS,
            PrimitiveValue::from(fg_number.to_string().as_str()),
        ));
        item.put(DataElement::new(
            Tag(0x300A, 0x0078),
            VR::IS,
            PrimitiveValue::from(n_fractions.to_string().as_str()),
        ));
        item.put(DataElement::new(
            Tag(0x300A, 0x00B6),
            VR::SQ,
            Value::from(ref_beam_seq),
        ));
        item
    }

    // ── Test A: missing file ─────────────────────────────────────────────────

    /// Invariant: a nonexistent path must produce Err mentioning the path or open failure.
    #[test]
    fn test_read_rt_plan_missing_file_returns_error() {
        let result = read_rt_plan("/nonexistent/plan.dcm");
        assert!(result.is_err(), "nonexistent path must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("nonexistent") || msg.contains("open"),
            "error must mention path or open failure; got: {msg}"
        );
    }

    // ── Test B: wrong SOP class ──────────────────────────────────────────────

    /// Invariant: a file whose SOP Class UID ≠ RT Plan must produce Err
    /// containing the rejected UID in the message.
    #[test]
    fn test_read_rt_plan_wrong_sop_class_returns_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("wrong.dcm");
        write_wrong_sop_file("1.2.840.10008.5.1.4.1.1.2", &path);

        let result = read_rt_plan(&path);
        assert!(result.is_err(), "wrong SOP class must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("1.2.840.10008.5.1.4.1.1.2"),
            "error must contain the rejected SOP UID; got: {msg}"
        );
    }

    // ── Test C: synthetic plan roundtrip ─────────────────────────────────────

    /// Invariant: rt_plan_label, beam count, beam names, radiation type,
    /// fraction count, and referenced beam numbers are all preserved through
    /// the DICOM write-read cycle.
    ///
    /// Reference values:
    /// - 2 beams: numbers 1 and 2, names "FIELD_1" and "FIELD_2", type "PHOTON"
    /// - 1 fraction group: 30 fractions planned, referencing beams [1, 2]
    #[test]
    fn test_read_rt_plan_synthetic_plan() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("rt_plan.dcm");

        let beam1 = make_beam_item(1, "FIELD_1", "PHOTON", 2);
        let beam2 = make_beam_item(2, "FIELD_2", "PHOTON", 2);
        let beam_seq = DataSetSequence::new(vec![beam1, beam2], Length::UNDEFINED);

        let fg = make_fraction_group_item(1, 30, &[1, 2]);
        let fg_seq = DataSetSequence::new(vec![fg], Length::UNDEFINED);

        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x300A, 0x0002),
            VR::LO,
            PrimitiveValue::from("PLAN_A"),
        ));
        obj.put(DataElement::new(
            Tag(0x300A, 0x0003),
            VR::LO,
            PrimitiveValue::from(""),
        ));
        obj.put(DataElement::new(
            Tag(0x300A, 0x0004),
            VR::ST,
            PrimitiveValue::from(""),
        ));
        obj.put(DataElement::new(
            Tag(0x300A, 0x000A),
            VR::CS,
            PrimitiveValue::from("CURATIVE"),
        ));
        obj.put(DataElement::new(
            Tag(0x300A, 0x00B0),
            VR::SQ,
            Value::from(beam_seq),
        ));
        obj.put(DataElement::new(
            Tag(0x300A, 0x0070),
            VR::SQ,
            Value::from(fg_seq),
        ));
        write_rt_plan_file(obj, &path);

        let plan = read_rt_plan(&path).expect("read_rt_plan synthetic");

        assert_eq!(plan.rt_plan_label, "PLAN_A", "rt_plan_label");
        assert_eq!(plan.plan_intent, "CURATIVE", "plan_intent");
        assert_eq!(plan.beams.len(), 2, "beam count");

        assert_eq!(plan.beams[0].beam_number, 1, "beam 0 number");
        assert_eq!(plan.beams[0].beam_name, "FIELD_1", "beam 0 name");
        assert_eq!(
            plan.beams[0].radiation_type, "PHOTON",
            "beam 0 radiation_type"
        );
        assert_eq!(
            plan.beams[0].treatment_delivery_type, "TREATMENT",
            "beam 0 delivery type"
        );
        assert_eq!(plan.beams[0].n_control_points, 2, "beam 0 control points");

        assert_eq!(plan.beams[1].beam_number, 2, "beam 1 number");
        assert_eq!(plan.beams[1].beam_name, "FIELD_2", "beam 1 name");
        assert_eq!(
            plan.beams[1].radiation_type, "PHOTON",
            "beam 1 radiation_type"
        );

        assert_eq!(plan.fraction_groups.len(), 1, "fraction group count");
        assert_eq!(
            plan.fraction_groups[0].fraction_group_number, 1,
            "fraction_group_number"
        );
        assert_eq!(
            plan.fraction_groups[0].n_fractions_planned, 30,
            "n_fractions_planned"
        );
        assert_eq!(
            plan.fraction_groups[0].referenced_beam_numbers,
            vec![1u32, 2u32],
            "referenced_beam_numbers"
        );
    }

    // ── Test D: empty plan writes and reads back ──────────────────────────────

    /// Invariant: write_rt_plan with empty beams and fraction_groups must succeed.
    /// rt_plan_label must be preserved exactly; beams and fraction_groups remain empty.
    #[test]
    fn test_write_rt_plan_rejects_nothing_but_writes_empty() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("plan_empty.dcm");
        let plan = RtPlanInfo {
            rt_plan_label: "EMPTY_PLAN".to_owned(),
            rt_plan_name: "".to_owned(),
            rt_plan_description: "".to_owned(),
            plan_intent: "".to_owned(),
            beams: vec![],
            fraction_groups: vec![],
        };
        let result = write_rt_plan(&path, &plan);
        assert!(
            result.is_ok(),
            "empty plan write must succeed; got: {:?}",
            result.err()
        );
        let back = read_rt_plan(&path).expect("read_rt_plan empty round-trip");
        assert_eq!(back.rt_plan_label, "EMPTY_PLAN", "rt_plan_label");
        assert!(back.beams.is_empty(), "beams must be empty");
        assert!(
            back.fraction_groups.is_empty(),
            "fraction_groups must be empty"
        );
    }

    // ── Test E: full round-trip ───────────────────────────────────────────────

    /// Invariant: all plan fields, beam fields, and fraction group fields are
    /// preserved through the DICOM write-read cycle without loss.
    #[test]
    fn test_write_rt_plan_round_trip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("plan_rt.dcm");

        let plan = RtPlanInfo {
            rt_plan_label: "PLAN_B".to_owned(),
            rt_plan_name: "Full Plan".to_owned(),
            rt_plan_description: "Test description".to_owned(),
            plan_intent: "CURATIVE".to_owned(),
            beams: vec![
                RtBeamInfo {
                    beam_number: 10,
                    beam_name: "BEAM_A".to_owned(),
                    beam_description: "first beam".to_owned(),
                    radiation_type: "PHOTON".to_owned(),
                    treatment_delivery_type: "TREATMENT".to_owned(),
                    n_control_points: 5,
                },
                RtBeamInfo {
                    beam_number: 20,
                    beam_name: "BEAM_B".to_owned(),
                    beam_description: "second beam".to_owned(),
                    radiation_type: "ELECTRON".to_owned(),
                    treatment_delivery_type: "TREATMENT".to_owned(),
                    n_control_points: 3,
                },
            ],
            fraction_groups: vec![RtFractionGroup {
                fraction_group_number: 1,
                n_fractions_planned: 25,
                referenced_beam_numbers: vec![10, 20],
            }],
        };

        write_rt_plan(&path, &plan).expect("write_rt_plan round-trip");
        let back = read_rt_plan(&path).expect("read_rt_plan round-trip");

        assert_eq!(back.rt_plan_label, "PLAN_B", "rt_plan_label");
        assert_eq!(back.rt_plan_name, "Full Plan", "rt_plan_name");
        assert_eq!(back.plan_intent, "CURATIVE", "plan_intent");
        assert_eq!(back.beams.len(), 2, "beams.len");

        assert_eq!(back.beams[0].beam_number, 10, "beam[0].beam_number");
        assert_eq!(back.beams[0].beam_name, "BEAM_A", "beam[0].beam_name");
        assert_eq!(
            back.beams[0].radiation_type, "PHOTON",
            "beam[0].radiation_type"
        );
        assert_eq!(
            back.beams[0].n_control_points, 5,
            "beam[0].n_control_points"
        );

        assert_eq!(back.beams[1].beam_number, 20, "beam[1].beam_number");
        assert_eq!(back.beams[1].beam_name, "BEAM_B", "beam[1].beam_name");
        assert_eq!(
            back.beams[1].radiation_type, "ELECTRON",
            "beam[1].radiation_type"
        );

        assert_eq!(back.fraction_groups.len(), 1, "fraction_groups.len");
        assert_eq!(
            back.fraction_groups[0].n_fractions_planned, 25,
            "n_fractions_planned"
        );
        assert_eq!(
            back.fraction_groups[0].referenced_beam_numbers,
            vec![10u32, 20u32],
            "referenced_beam_numbers"
        );
    }
}
