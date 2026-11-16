//! A repaired copy of a study folder.

use super::{repair_dicom_study, StudyRepairError, StudyRepairReport};
use crate::format::dicom::export_tree::ExportTreeError;
use crate::format::dicom::repair::RepairSummary;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{meta::FileMetaTableBuilder, open_file, InMemDicomObject};
use std::path::Path;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const CT_IMAGE_STORAGE: &str = "1.2.840.10008.5.1.4.1.1.2";
const MEDIA_STORAGE_DIRECTORY: &str = "1.2.840.10008.1.3.10";
const SOP_INSTANCE_UID: Tag = Tag(0x0008, 0x0018);
const STUDY_DATE: Tag = Tag(0x0008, 0x0020);
const STUDY_TIME: Tag = Tag(0x0008, 0x0030);

/// Write an instance whose data set holds `sop`, `date` and `time` as given
/// and whose file meta group names `meta_sop`.
fn write(
    path: &Path,
    sop_class: &str,
    (sop, meta_sop): (&str, &str),
    (date, time): (&str, &str),
) -> TestResult {
    let mut object = InMemDicomObject::new_empty();
    for (tag, vr, value) in [
        (Tag(0x0008, 0x0016), VR::UI, sop_class),
        (SOP_INSTANCE_UID, VR::UI, sop),
        (STUDY_DATE, VR::DA, date),
        (STUDY_TIME, VR::TM, time),
    ] {
        object.put(DataElement::new(tag, vr, PrimitiveValue::from(value)));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = object.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(sop_class)
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )?;
    // `with_meta` takes the instance UID from the data set.
    file.update_meta(|meta| meta.media_storage_sop_instance_uid = meta_sop.to_owned());
    file.write_to_file(path)?;
    Ok(())
}

/// The stored SOP Instance UID, study date, study time and file meta UID.
fn values(path: &Path) -> Result<[String; 4], Box<dyn std::error::Error>> {
    let object = open_file(path)?;
    let text = |tag| -> Result<String, Box<dyn std::error::Error>> {
        Ok(object.element(tag)?.to_str()?.into_owned())
    };
    Ok([
        text(SOP_INSTANCE_UID)?,
        text(STUDY_DATE)?,
        text(STUDY_TIME)?,
        object.meta().media_storage_sop_instance_uid().to_owned(),
    ])
}

#[test]
fn a_study_copy_is_repaired_and_the_source_is_untouched() -> TestResult {
    let root = tempfile::tempdir()?;
    let (input, output) = (root.path().join("study"), root.path().join("copy"));
    let defective = input.join("s1/a.dcm");
    write(
        &defective,
        CT_IMAGE_STORAGE,
        ("1.2.3.04", "1.2.3.99"),
        ("1994.07.21", "10:20:30"),
    )?;
    write(
        &input.join("s1/b.dcm"),
        CT_IMAGE_STORAGE,
        ("1.2.3.5", "1.2.3.5"),
        ("19940721", "102030"),
    )?;
    write(
        &input.join("DICOMDIR"),
        MEDIA_STORAGE_DIRECTORY,
        ("1.2.3.6", "1.2.3.6"),
        ("19940721", "102030"),
    )?;
    std::fs::write(input.join("notes.txt"), "not DICOM")?;
    let source_bytes = std::fs::read(&defective)?;

    let report = repair_dicom_study(&input, &output)?;

    let mut expected = StudyRepairReport::default();
    expected.instances_written = 2;
    expected.instances_repaired = 1;
    expected.values = RepairSummary {
        dates: 1,
        times: 1,
        uids: 1,
    };
    expected.file_meta_uids_aligned = 1;
    expected.non_dicom_skipped = 1;
    expected.indexes_omitted = 1;
    assert_eq!(report, expected);
    assert_eq!(
        values(&output.join("s1/a.dcm"))?,
        ["1.2.3.4", "19940721", "102030", "1.2.3.4"]
    );
    assert_eq!(
        values(&output.join("s1/b.dcm"))?,
        ["1.2.3.5", "19940721", "102030", "1.2.3.5"]
    );
    assert!(!output.join("DICOMDIR").exists());
    assert!(!output.join("notes.txt").exists());
    assert_eq!(std::fs::read(&defective)?, source_bytes);
    assert_eq!(
        values(&defective)?,
        ["1.2.3.04", "1994.07.21", "10:20:30", "1.2.3.99"]
    );
    Ok(())
}

#[test]
fn a_folder_without_instances_produces_no_copy() -> TestResult {
    let root = tempfile::tempdir()?;
    let (input, output) = (root.path().join("study"), root.path().join("copy"));
    std::fs::create_dir_all(&input)?;
    std::fs::write(input.join("notes.txt"), "not DICOM")?;

    let error = repair_dicom_study(&input, &output).expect_err("nothing to repair");

    assert!(
        matches!(&error, StudyRepairError::NoInstances(path) if path.ends_with("study")),
        "{error:?}"
    );
    Ok(())
}

#[test]
fn an_output_folder_inside_the_study_is_refused() -> TestResult {
    let root = tempfile::tempdir()?;
    let input = root.path().join("study");
    write(
        &input.join("a.dcm"),
        CT_IMAGE_STORAGE,
        ("1.2.3.4", "1.2.3.4"),
        ("19940721", "102030"),
    )?;

    let error = repair_dicom_study(&input, input.join("copy")).expect_err("overlapping output");

    assert!(
        matches!(
            error,
            StudyRepairError::Tree(ExportTreeError::OutputOverlapsInput { .. })
        ),
        "{error:?}"
    );
    assert!(!input.join("copy").exists());
    Ok(())
}
