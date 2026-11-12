use super::*;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{meta::FileMetaTableBuilder, open_file, InMemDicomObject};

fn fields() -> EditorFields {
    EditorFields {
        patient_id: "SUBJECT-42".into(),
        patient_name: "Study^Subject".into(),
        sex: "O".into(),
        age_years: "42".into(),
        birth_date: "1984-02-29".into(),
        height_cm: "175".into(),
        weight_kg: "62.5".into(),
        study_date: "2024-06-10".into(),
        study_time: "09:35:12".into(),
        study_id: "STUDY".into(),
        accession_number: "ACCESSION".into(),
    }
}

fn fixture(path: &std::path::Path) {
    let mut obj = InMemDicomObject::new_empty();
    for (tag, vr, value) in [
        (Tag(0x0010, 0x0010), VR::PN, "Original^Synthetic"),
        (Tag(0x0010, 0x0020), VR::LO, "SOURCE-ID"),
        (Tag(0x0008, 0x0018), VR::UI, "1.2.3.4.5"),
        (Tag(0x0008, 0x0016), VR::UI, "1.2.840.10008.5.1.4.1.1.2"),
        (Tag(0x0020, 0x000d), VR::UI, "1.2.3.4"),
        (Tag(0x0020, 0x000e), VR::UI, "1.2.3.4.1"),
        (Tag(0x0011, 0x0010), VR::LO, "PRIVATE"),
    ] {
        obj.put(DataElement::new(tag, vr, PrimitiveValue::from(value)));
    }
    obj.put(DataElement::new(
        Tag(0x7fe0, 0x0010),
        VR::OB,
        PrimitiveValue::from(vec![1u8, 2, 3, 4]),
    ));
    obj.with_meta(FileMetaTableBuilder::new().transfer_syntax("1.2.840.10008.1.2.1"))
        .unwrap()
        .write_to_file(path)
        .unwrap();
}

#[test]
fn fields_normalize_dicom_units_and_vrs() {
    let list = fields().elements().unwrap();
    assert_eq!(list[3].2, "042Y");
    assert_eq!(list[4].2, "19840229");
    assert_eq!(list[5].2, "1.75");
    assert_eq!(list[8].2, "093512");
}
#[test]
fn invalid_fields_rejected() {
    let mut f = fields();
    f.birth_date = "2023-02-29".into();
    assert!(f.validate().is_err());
    f = fields();
    f.study_time = "24:00".into();
    assert!(f.validate().is_err());
    f = fields();
    f.age_years = "1000".into();
    assert!(f.validate().is_err());
    f = fields();
    f.height_cm = "NaN".into();
    assert!(f.validate().is_err());
    f = fields();
    f.weight_kg = "-1".into();
    assert!(f.validate().is_err());
    f = fields();
    f.patient_id = "one\\two".into();
    assert!(f.validate().is_err());
    f = fields();
    f.patient_id = "".into();
    assert!(f.validate().is_err());
    f = fields();
    f.sex = "Unknown".into();
    assert!(f.validate().is_err());
}
#[test]
fn output_round_trips_without_modifying_input() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("source.dcm");
    fixture(&input);
    let original = std::fs::read(&input).unwrap();
    let output = temp.path().join("export");
    let report = export_batch(&[input.clone()], &output, &fields(), "test-salt").unwrap();
    assert_eq!(report.file_count, 1);
    assert_eq!(std::fs::read(input).unwrap(), original);
    assert!(!output.join("INCOMPLETE").exists());
    let obj = open_file(output.join("instance-000001.dcm")).unwrap();
    assert_eq!(
        obj.element(Tag(0x0010, 0x0020)).unwrap().to_str().unwrap(),
        "SUBJECT-42"
    );
    assert_eq!(
        obj.element(Tag(0x0010, 0x1020)).unwrap().to_str().unwrap(),
        "1.75"
    );
    assert!(obj.element(Tag(0x0011, 0x0010)).is_err());
    assert!(
        obj.element(Tag(0x0012, 0x0063))
            .unwrap()
            .to_str()
            .unwrap()
            .len()
            <= 64
    );
    assert_eq!(
        obj.element(Tag(0x0012, 0x0062)).unwrap().to_str().unwrap(),
        "NO"
    );
    assert_ne!(
        obj.element(Tag(0x0008, 0x0018)).unwrap().to_str().unwrap(),
        "1.2.3.4.5"
    );
}
#[test]
fn existing_destination_is_never_overwritten() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("exists");
    std::fs::create_dir(&output).unwrap();
    std::fs::write(output.join("keep"), b"unchanged").unwrap();
    assert!(export_batch(&[temp.path().join("missing")], &output, &fields(), "salt").is_err());
    assert_eq!(std::fs::read(output.join("keep")).unwrap(), b"unchanged");
}
#[test]
fn failure_after_one_file_cleans_batch() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("source.dcm");
    fixture(&input);
    let output = temp.path().join("export");
    assert!(export_batch(
        &[input, temp.path().join("missing")],
        &output,
        &fields(),
        "salt"
    )
    .is_err());
    assert!(!output.exists());
}
#[test]
fn duplicate_instance_aborts_batch() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("source.dcm");
    fixture(&input);
    let output = temp.path().join("export");
    assert!(export_batch(&[input.clone(), input], &output, &fields(), "salt").is_err());
    assert!(!output.exists());
}

#[test]
fn empty_optional_fields_are_written_empty() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("source.dcm");
    fixture(&input);
    let output = temp.path().join("export");
    let f = EditorFields {
        patient_id: "ANON".into(),
        ..Default::default()
    };
    export_batch(&[input], &output, &f, "salt").unwrap();
    let obj = open_file(output.join("instance-000001.dcm")).unwrap();
    for (tag, _, value) in f.elements().unwrap() {
        assert_eq!(obj.element(tag).unwrap().to_str().unwrap(), value);
    }
}

#[test]
fn malformed_time_separators_never_panic() {
    for time in [
        "1::2::34",
        "12::3",
        "12:3:",
        "12:34::5",
        "12:3::56",
        "::3456",
        "１２:３４",
    ] {
        let mut f = fields();
        f.study_time = time.into();
        assert!(f.validate().is_err(), "accepted {time}");
    }
}
