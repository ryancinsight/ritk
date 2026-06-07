use super::*;
use arrayvec::ArrayString;
use dicom::core::header::Length;
use dicom::core::value::DataSetSequence;
use dicom::core::value::Value;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;

// ── Test helpers ──────────────────────────────────────────────────────────────

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

// ── Test A: missing file ──────────────────────────────────────────────────────

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

// ── Test B: wrong SOP class ───────────────────────────────────────────────────

/// Invariant: a file whose SOP Class UID ≠ RT Plan must produce Err
/// containing the rejected UID.
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

// ── Test C: synthetic plan roundtrip ─────────────────────────────────────────

/// Invariant: rt_plan_label, beam count, beam names, radiation type,
/// fraction count, and referenced beam numbers preserved through write-read cycle.
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
    assert_eq!(plan.plan_intent.as_str(), "CURATIVE", "plan_intent");
    assert_eq!(plan.beams.len(), 2, "beam count");

    assert_eq!(plan.beams[0].beam_number, 1, "beam 0 number");
    assert_eq!(plan.beams[0].beam_name, "FIELD_1", "beam 0 name");
    assert_eq!(
        plan.beams[0].radiation_type.as_str(),
        "PHOTON",
        "beam 0 radiation_type"
    );
    assert_eq!(
        plan.beams[0].treatment_delivery_type.as_str(),
        "TREATMENT",
        "beam 0 delivery type"
    );
    assert_eq!(plan.beams[0].n_control_points, 2, "beam 0 control points");

    assert_eq!(plan.beams[1].beam_number, 2, "beam 1 number");
    assert_eq!(plan.beams[1].beam_name, "FIELD_2", "beam 1 name");
    assert_eq!(
        plan.beams[1].radiation_type.as_str(),
        "PHOTON",
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

// ── Test D: empty plan writes and reads back ──────────────────────────────────

/// Invariant: write_rt_plan with empty beams and fraction_groups must succeed;
/// rt_plan_label must be preserved.
#[test]
fn test_write_rt_plan_rejects_nothing_but_writes_empty() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("plan_empty.dcm");
    let plan = RtPlanInfo {
        sop_instance_uid: ArrayString::new(),
        rt_plan_label: "EMPTY_PLAN".to_owned(),
        rt_plan_name: "".to_owned(),
        rt_plan_description: "".to_owned(),
        plan_intent: ArrayString::new(),
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

// ── Test E: full round-trip ───────────────────────────────────────────────────

/// Invariant: all plan fields, beam fields, and fraction group fields preserved
/// through the DICOM write-read cycle.
#[test]
fn test_write_rt_plan_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("plan_rt.dcm");

    let plan = RtPlanInfo {
        sop_instance_uid: ArrayString::new(),
        rt_plan_label: "PLAN_B".to_owned(),
        rt_plan_name: "Full Plan".to_owned(),
        rt_plan_description: "Test description".to_owned(),
        plan_intent: ArrayString::from("CURATIVE").unwrap(),
        beams: vec![
            RtBeamInfo {
                beam_number: 10,
                beam_name: "BEAM_A".to_owned(),
                beam_description: "first beam".to_owned(),
                radiation_type: ArrayString::from("PHOTON").unwrap(),
                treatment_delivery_type: ArrayString::from("TREATMENT").unwrap(),
                n_control_points: 5,
            },
            RtBeamInfo {
                beam_number: 20,
                beam_name: "BEAM_B".to_owned(),
                beam_description: "second beam".to_owned(),
                radiation_type: ArrayString::from("ELECTRON").unwrap(),
                treatment_delivery_type: ArrayString::from("TREATMENT").unwrap(),
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

    assert!(
        !back.sop_instance_uid.is_empty(),
        "sop_instance_uid must be present"
    );
    assert_eq!(back.rt_plan_label, "PLAN_B", "rt_plan_label");
    assert_eq!(back.rt_plan_name, "Full Plan", "rt_plan_name");
    assert_eq!(back.plan_intent.as_str(), "CURATIVE", "plan_intent");
    assert_eq!(back.beams.len(), 2, "beams.len");

    assert_eq!(back.beams[0].beam_number, 10, "beam[0].beam_number");
    assert_eq!(back.beams[0].beam_name, "BEAM_A", "beam[0].beam_name");
    assert_eq!(
        back.beams[0].radiation_type.as_str(),
        "PHOTON",
        "beam[0].radiation_type"
    );
    assert_eq!(
        back.beams[0].n_control_points, 5,
        "beam[0].n_control_points"
    );

    assert_eq!(back.beams[1].beam_number, 20, "beam[1].beam_number");
    assert_eq!(back.beams[1].beam_name, "BEAM_B", "beam[1].beam_name");
    assert_eq!(
        back.beams[1].radiation_type.as_str(),
        "ELECTRON",
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
