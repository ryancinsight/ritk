//! Study-level de-identification over a folder tree.

use super::verify::VerifyIssue;
use super::{
    anonymize_dicom_selection, anonymize_dicom_study, AnonymizationProfile, AnonymizeOptions,
    StudyAnonymizeError,
};
use crate::format::dicom::export_tree::ExportTreeError;
use ritk_dicom::dataset::{DataElement, DicomTag, PrimitiveValue, VR};
use ritk_dicom::dataset::{Dataset, DicomFile, FileMetaBuilder};
use std::path::Path;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const STUDY: &str = "1.2.826.0.1.3680043.8.498.1";
const CT_IMAGE_STORAGE: &str = "1.2.840.10008.5.1.4.1.1.2";
const PATIENT_NAME: DicomTag = DicomTag::new(0x0010, 0x0010);
const PATIENT_ID: DicomTag = DicomTag::new(0x0010, 0x0020);
const SERIES_UID: DicomTag = DicomTag::new(0x0020, 0x000E);
const CONTRAST_AGENT: DicomTag = DicomTag::new(0x0018, 0x0010);

fn text(tag: DicomTag, vr: VR, value: &str) -> DataElement {
    DataElement::new(tag, vr, PrimitiveValue::from(value))
}

fn instance(
    series: &str,
    sop: &str,
    sop_class: &str,
) -> Result<DicomFile, Box<dyn std::error::Error>> {
    let mut object = Dataset::new();
    object.put(text(DicomTag::new(0x0008, 0x0016), VR::UI, sop_class));
    object.put(text(DicomTag::new(0x0008, 0x0018), VR::UI, sop));
    object.put(text(DicomTag::new(0x0020, 0x000D), VR::UI, STUDY));
    object.put(text(SERIES_UID, VR::UI, series));
    object.put(text(PATIENT_NAME, VR::PN, "Doe^John"));
    object.put(text(PATIENT_ID, VR::LO, "PAT001"));
    for (tag, value) in [
        (DicomTag::new(0x0028, 0x0010), 2_u16),
        (DicomTag::new(0x0028, 0x0011), 2),
        (DicomTag::new(0x0028, 0x0100), 8),
        (DicomTag::new(0x0028, 0x0002), 1),
    ] {
        object.put(DataElement::new(tag, VR::US, PrimitiveValue::from(value)));
    }
    object.put(DataElement::new(
        DicomTag::new(0x7FE0, 0x0010),
        VR::OB,
        PrimitiveValue::from(vec![1_u8, 2, 3, 4]),
    ));
    Ok(object.with_meta(
        FileMetaBuilder::new()
            .media_storage_sop_class_uid(sop_class)
            .media_storage_sop_instance_uid(sop)
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )?)
}

fn write(object: &DicomFile, path: &Path) -> TestResult {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    object.write_to_file(path)?;
    Ok(())
}

fn value(path: &Path, tag: DicomTag) -> Result<String, Box<dyn std::error::Error>> {
    Ok(ritk_dicom::dataset::DicomFile::open(path)?
        .element(tag)?
        .to_str()?
        .trim_end_matches(['\0', ' '])
        .to_owned())
}

/// Every file and folder below `root`, relative to it with `/` separators,
/// sorted as strings.
fn listing(root: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut names = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(folder) = pending.pop() {
        for entry in std::fs::read_dir(&folder)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                pending.push(entry.path());
            }
            let below = entry.path().strip_prefix(root)?.to_path_buf();
            names.push(below.to_string_lossy().replace('\\', "/"));
        }
    }
    names.sort();
    Ok(names)
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
fn a_study_tree_is_exported_under_neutral_names_with_series_kept_together() -> TestResult {
    let source = tempfile::tempdir()?;
    let parent = tempfile::tempdir()?;
    let output = parent.path().join("export");
    study_tree(source.path())?;
    let original = std::fs::read(source.path().join("s1/a.dcm"))?;

    let report = anonymize_dicom_study(source.path(), &output, &options())?;

    assert_eq!(report.instances_written, 3);
    assert_eq!(report.non_dicom_skipped, 1);
    assert_eq!(report.indexes_omitted, 1);
    // Sources sort as DICOMDIR, notes.txt, s1/a.dcm, s1/b.dcm, s2/deep/c.dcm;
    // the first two are left out and take no number.
    let exported = ["000001.dcm", "000002.dcm", "000003.dcm"];
    assert_eq!(listing(&output)?, exported);
    for file in exported {
        assert_eq!(value(&output.join(file), PATIENT_NAME)?, "ANONYMOUS");
        assert_eq!(value(&output.join(file), PATIENT_ID)?, "ANON001");
    }
    let first = value(&output.join("000001.dcm"), SERIES_UID)?;
    assert_eq!(first, value(&output.join("000002.dcm"), SERIES_UID)?);
    assert_ne!(first, "1.2.3.10");
    assert_ne!(first, value(&output.join("000003.dcm"), SERIES_UID)?);
    let exported = ritk_dicom::dataset::DicomFile::open(output.join("000001.dcm"))?;
    let meta_uid = exported.meta().media_storage_sop_instance_uid();
    let data_set_uid = value(&output.join("000001.dcm"), DicomTag::new(0x0008, 0x0018))?;
    assert_eq!(meta_uid, data_set_uid);
    assert_ne!(data_set_uid, "1.2.3.10.1");
    assert_eq!(std::fs::read(source.path().join("s1/a.dcm"))?, original);
    Ok(())
}

#[test]
fn a_failed_selected_instance_removes_the_completed_files() -> TestResult {
    let source = tempfile::tempdir()?;
    let output = tempfile::tempdir()?;
    study_tree(source.path())?;
    let broken = source.path().join("z.dcm");
    std::fs::write(&broken, "invalid DICOM")?;
    let error = anonymize_dicom_selection(
        source.path(),
        &[&source.path().join("s1/a.dcm"), &broken],
        output.path(),
        &options(),
    )
    .expect_err("an unreadable selected file must abort the export");
    assert!(
        matches!(error, StudyAnonymizeError::SelectedInstance(path) if path == broken.canonicalize()?)
    );
    assert_eq!(listing(output.path())?, Vec::<String>::new());
    Ok(())
}

#[test]
fn a_selected_instance_outside_the_source_is_rejected() -> TestResult {
    let source = tempfile::tempdir()?;
    let other = tempfile::tempdir()?;
    let output = tempfile::tempdir()?;
    study_tree(other.path())?;
    let outside = other.path().join("s1/a.dcm");
    let error = anonymize_dicom_selection(source.path(), &[&outside], output.path(), &options())
        .expect_err("an outside file must abort the export");
    assert!(
        matches!(error, StudyAnonymizeError::Tree(ExportTreeError::Io {
        action: "place", path, ..
    }) if path == outside.canonicalize()?)
    );
    assert_eq!(listing(output.path())?, Vec::<String>::new());
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
    assert_eq!(
        issues[0].0,
        source.path().canonicalize()?.join("s1").join("z.dcm")
    );
    assert_eq!(
        issues[0].1,
        VerifyIssue::PatientLeak {
            value: String::from("Doe^John")
        }
    );
    assert_eq!(listing(output.path())?, [""; 0]);
    Ok(())
}

#[test]
fn identifiers_in_source_folder_and_file_names_do_not_reach_the_export() -> TestResult {
    let root = tempfile::tempdir()?;
    let named = root.path().join("Doe^John").join("PAT001");
    for (file, sop) in [
        ("1.2.3.10.2.dcm", "1.2.3.10.2"),
        ("1.2.3.10.1", "1.2.3.10.1"),
    ] {
        write(
            &instance("1.2.3.10", sop, CT_IMAGE_STORAGE)?,
            &named.join("1.2.3.10").join(file),
        )?;
    }

    let folder = root.path().join("from-folder");
    let report = anonymize_dicom_study(root.path().join("Doe^John"), &folder, &options())?;
    assert_eq!(report.instances_written, 2);
    assert_eq!(listing(&folder)?, ["000001.dcm", "000002.dcm"]);
    // `1.2.3.10.1` sorts before `1.2.3.10.2.dcm`.
    let salted = |sop: &str| {
        let source = instance("1.2.3.10", sop, CT_IMAGE_STORAGE)?;
        let (anonymized, _) = super::anonymize_object(source, &options())?;
        Ok::<_, Box<dyn std::error::Error>>(
            anonymized
                .element(DicomTag::new(0x0008, 0x0018))?
                .to_str()?
                .trim_end_matches(['\0', ' '])
                .to_owned(),
        )
    };
    assert_eq!(
        value(&folder.join("000001.dcm"), DicomTag::new(0x0008, 0x0018))?,
        salted("1.2.3.10.1")?
    );
    assert_eq!(
        value(&folder.join("000002.dcm"), DicomTag::new(0x0008, 0x0018))?,
        salted("1.2.3.10.2")?
    );

    let single = root.path().join("from-file");
    let source = named.join("1.2.3.10").join("1.2.3.10.2.dcm");
    let report = anonymize_dicom_study(&source, &single, &options())?;
    assert_eq!(report.instances_written, 1);
    assert_eq!(listing(&single)?, ["000001.dcm"]);
    Ok(())
}

#[test]
fn an_object_without_pixel_data_is_exported_with_the_images() -> TestResult {
    let root = tempfile::tempdir()?;
    let (input, output) = (root.path().join("study"), root.path().join("export"));
    write(
        &instance("1.2.3.10", "1.2.3.10.1", CT_IMAGE_STORAGE)?,
        &input.join("image.dcm"),
    )?;
    let mut report_object = instance("1.2.3.40", "1.2.3.40.1", "1.2.840.10008.5.1.4.1.1.88.11")?;
    for tag in [
        DicomTag::new(0x0028, 0x0010),
        DicomTag::new(0x0028, 0x0011),
        DicomTag::new(0x0028, 0x0100),
        DicomTag::new(0x0028, 0x0002),
        DicomTag::new(0x7FE0, 0x0010),
    ] {
        assert!(report_object.remove_element(tag).is_some());
    }
    write(&report_object, &input.join("report.dcm"))?;

    let report = anonymize_dicom_study(&input, &output, &options())?;

    assert_eq!(report.instances_written, 2);
    // `image.dcm` sorts first, so the report is the second export.
    assert_eq!(
        [
            value(&output.join("000001.dcm"), DicomTag::new(0x0008, 0x0016))?,
            value(&output.join("000002.dcm"), DicomTag::new(0x0008, 0x0016))?,
        ],
        [CT_IMAGE_STORAGE, "1.2.840.10008.5.1.4.1.1.88.11"]
    );
    assert_eq!(
        value(&output.join("000002.dcm"), PATIENT_NAME)?,
        "ANONYMOUS"
    );
    assert_eq!(value(&output.join("000002.dcm"), PATIENT_ID)?, "ANON001");
    let exported_series = value(&output.join("000002.dcm"), SERIES_UID)?;
    assert_ne!(exported_series, "1.2.3.40");
    assert_ne!(
        exported_series,
        value(&output.join("000001.dcm"), SERIES_UID)?
    );
    Ok(())
}
