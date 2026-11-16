//! Study-level de-identification over a folder tree.

use super::verify::VerifyIssue;
use super::{anonymize_dicom_study, AnonymizationProfile, AnonymizeOptions, StudyAnonymizeError};
use crate::format::dicom::export_tree::ExportTreeError;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{meta::FileMetaTableBuilder, open_file, FileDicomObject, InMemDicomObject};
use std::path::Path;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const STUDY: &str = "1.2.826.0.1.3680043.8.498.1";
const CT_IMAGE_STORAGE: &str = "1.2.840.10008.5.1.4.1.1.2";
const PATIENT_NAME: Tag = Tag(0x0010, 0x0010);
const PATIENT_ID: Tag = Tag(0x0010, 0x0020);
const SERIES_UID: Tag = Tag(0x0020, 0x000E);
const CONTRAST_AGENT: Tag = Tag(0x0018, 0x0010);

fn text(tag: Tag, vr: VR, value: &str) -> DataElement<InMemDicomObject> {
    DataElement::new(tag, vr, PrimitiveValue::from(value))
}

fn instance(
    series: &str,
    sop: &str,
    sop_class: &str,
) -> Result<FileDicomObject<InMemDicomObject>, Box<dyn std::error::Error>> {
    let mut object = InMemDicomObject::new_empty();
    object.put(text(Tag(0x0008, 0x0016), VR::UI, sop_class));
    object.put(text(Tag(0x0008, 0x0018), VR::UI, sop));
    object.put(text(Tag(0x0020, 0x000D), VR::UI, STUDY));
    object.put(text(SERIES_UID, VR::UI, series));
    object.put(text(PATIENT_NAME, VR::PN, "Doe^John"));
    object.put(text(PATIENT_ID, VR::LO, "PAT001"));
    for (tag, value) in [
        (Tag(0x0028, 0x0010), 2_u16),
        (Tag(0x0028, 0x0011), 2),
        (Tag(0x0028, 0x0100), 8),
        (Tag(0x0028, 0x0002), 1),
    ] {
        object.put(DataElement::new(tag, VR::US, PrimitiveValue::from(value)));
    }
    object.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OB,
        PrimitiveValue::from(vec![1_u8, 2, 3, 4]),
    ));
    Ok(object.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(sop_class)
            .media_storage_sop_instance_uid(sop)
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )?)
}

fn write(object: &FileDicomObject<InMemDicomObject>, path: &Path) -> TestResult {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    object.write_to_file(path)?;
    Ok(())
}

fn value(path: &Path, tag: Tag) -> Result<String, Box<dyn std::error::Error>> {
    Ok(open_file(path)?
        .element(tag)?
        .to_str()?
        .trim_end_matches(['\0', ' '])
        .to_owned())
}

fn options() -> AnonymizeOptions {
    AnonymizeOptions {
        profile: AnonymizationProfile::BasicReplaceUids,
        uid_salt: String::from("run-salt"),
        ..AnonymizeOptions::default()
    }
}

/// Two series folders, one stray text file and one DICOMDIR-class object.
fn study_tree(root: &Path) -> TestResult {
    for (folder, series, sop) in [
        ("s1/a.dcm", "1.2.3.10", "1.2.3.10.1"),
        ("s1/b.dcm", "1.2.3.10", "1.2.3.10.2"),
        ("s2/deep/c.dcm", "1.2.3.20", "1.2.3.20.1"),
    ] {
        write(
            &instance(series, sop, CT_IMAGE_STORAGE)?,
            &root.join(folder),
        )?;
    }
    std::fs::write(root.join("notes.txt"), "not DICOM")?;
    write(
        &instance("1.2.3.30", "1.2.3.30.1", "1.2.840.10008.1.3.10")?,
        &root.join("DICOMDIR"),
    )
}

#[test]
fn a_study_tree_is_mirrored_with_identifiers_replaced_and_series_kept_together() -> TestResult {
    let source = tempfile::tempdir()?;
    let parent = tempfile::tempdir()?;
    let output = parent.path().join("export");
    study_tree(source.path())?;
    let original = std::fs::read(source.path().join("s1/a.dcm"))?;

    let report = anonymize_dicom_study(source.path(), &output, &options())?;

    assert_eq!(report.instances_written, 3);
    assert_eq!(report.non_dicom_skipped, 1);
    assert_eq!(report.indexes_omitted, 1);
    for file in ["s1/a.dcm", "s1/b.dcm", "s2/deep/c.dcm"] {
        assert_eq!(value(&output.join(file), PATIENT_NAME)?, "ANONYMOUS");
        assert_eq!(value(&output.join(file), PATIENT_ID)?, "ANON001");
    }
    let first = value(&output.join("s1/a.dcm"), SERIES_UID)?;
    assert_eq!(first, value(&output.join("s1/b.dcm"), SERIES_UID)?);
    assert_ne!(first, "1.2.3.10");
    assert_ne!(first, value(&output.join("s2/deep/c.dcm"), SERIES_UID)?);
    let exported = open_file(output.join("s1/a.dcm"))?;
    let meta_uid = exported.meta().media_storage_sop_instance_uid();
    let data_set_uid = value(&output.join("s1/a.dcm"), Tag(0x0008, 0x0018))?;
    assert_eq!(meta_uid, data_set_uid);
    assert_ne!(data_set_uid, "1.2.3.10.1");
    assert!(!output.join("notes.txt").exists());
    assert!(!output.join("DICOMDIR").exists());
    assert_eq!(std::fs::read(source.path().join("s1/a.dcm"))?, original);
    Ok(())
}

#[test]
fn an_output_folder_the_planner_refuses_is_reported_as_a_tree_error() -> TestResult {
    let source = tempfile::tempdir()?;
    study_tree(source.path())?;

    let error = anonymize_dicom_study(source.path(), source.path().join("export"), &options())
        .expect_err("invariant: an output inside the source overlaps it");

    assert!(matches!(
        error,
        StudyAnonymizeError::Tree(ExportTreeError::OutputOverlapsInput { .. })
    ));
    Ok(())
}

#[test]
fn a_source_without_instances_is_reported() -> TestResult {
    let source = tempfile::tempdir()?;
    let output = tempfile::tempdir()?;
    std::fs::write(source.path().join("notes.txt"), "not DICOM")?;

    let error = anonymize_dicom_study(source.path(), output.path(), &options())
        .expect_err("invariant: the source holds no DICOM instance");

    assert!(matches!(error, StudyAnonymizeError::NoInstances(_)));
    Ok(())
}

#[test]
fn a_patient_name_surviving_in_an_unlisted_element_removes_the_whole_export() -> TestResult {
    let source = tempfile::tempdir()?;
    let output = tempfile::tempdir()?;
    study_tree(source.path())?;
    let mut leaking = instance("1.2.3.10", "1.2.3.10.9", CT_IMAGE_STORAGE)?;
    leaking.put(text(CONTRAST_AGENT, VR::LO, "given to Doe^John"));
    write(&leaking, &source.path().join("s1/z.dcm"))?;

    let error = anonymize_dicom_study(source.path(), output.path(), &options())
        .expect_err("invariant: the Basic profile leaves ContrastBolusAgent in place");

    let StudyAnonymizeError::Verification(issues) = error else {
        return Err(format!("expected a verification failure, got {error}").into());
    };
    assert_eq!(issues.len(), 1);
    assert!(issues[0].0.ends_with("z.dcm"));
    assert_eq!(
        issues[0].1,
        VerifyIssue::PatientLeak {
            value: String::from("Doe^John")
        }
    );
    for file in ["s1/a.dcm", "s1/b.dcm", "s1/z.dcm", "s2/deep/c.dcm"] {
        assert!(!output.path().join(file).exists(), "{file} was left behind");
    }
    Ok(())
}
