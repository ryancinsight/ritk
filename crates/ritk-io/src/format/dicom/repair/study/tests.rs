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
const SOP_CLASS_UID: Tag = Tag(0x0008, 0x0016);
const SOP_INSTANCE_UID: Tag = Tag(0x0008, 0x0018);
const STUDY_DATE: Tag = Tag(0x0008, 0x0020);
const SERIES_DATE: Tag = Tag(0x0008, 0x0021);
const STUDY_TIME: Tag = Tag(0x0008, 0x0030);
const STUDY_UID: Tag = Tag(0x0020, 0x000D);
const SERIES_UID: Tag = Tag(0x0020, 0x000E);

/// Write an instance holding `elements`, with a file meta group naming
/// `meta_class` and `meta_sop`.
fn write(
    path: &Path,
    (meta_class, meta_sop): (&str, &str),
    elements: &[(Tag, VR, &str)],
) -> TestResult {
    let mut object = InMemDicomObject::new_empty();
    for (tag, vr, value) in elements {
        object.put(DataElement::new(*tag, *vr, PrimitiveValue::from(*value)));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = object.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(meta_class)
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )?;
    // `with_meta` takes the instance UID from the data set.
    file.update_meta(|meta| meta.media_storage_sop_instance_uid = meta_sop.to_owned());
    file.write_to_file(path)?;
    Ok(())
}

/// The file meta class and instance UIDs, then the stored text of `tags`.
fn stored(path: &Path, tags: &[Tag]) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let object = open_file(path)?;
    let mut values = vec![
        object.meta().media_storage_sop_class_uid().to_owned(),
        object.meta().media_storage_sop_instance_uid().to_owned(),
    ];
    for tag in tags {
        values.push(object.element(*tag)?.to_str()?.into_owned());
    }
    Ok(values)
}

#[test]
fn a_study_copy_is_repaired_and_the_source_is_untouched() -> TestResult {
    let root = tempfile::tempdir()?;
    let (input, output) = (root.path().join("study"), root.path().join("copy"));
    // Value defects only: two dates and one time.
    let values_only = input.join("s1/a.dcm");
    write(
        &values_only,
        (CT_IMAGE_STORAGE, "1.2.3.1"),
        &[
            (SOP_CLASS_UID, VR::UI, CT_IMAGE_STORAGE),
            (SOP_INSTANCE_UID, VR::UI, "1.2.3.1"),
            (STUDY_DATE, VR::DA, "1994.07.21"),
            (SERIES_DATE, VR::DA, "1994.07.22"),
            (STUDY_TIME, VR::TM, "10:20:30"),
        ],
    )?;
    // A stale file meta instance UID only.
    write(
        &input.join("s1/b.dcm"),
        (CT_IMAGE_STORAGE, "1.2.3.99"),
        &[
            (SOP_CLASS_UID, VR::UI, CT_IMAGE_STORAGE),
            (SOP_INSTANCE_UID, VR::UI, "1.2.3.2"),
            (STUDY_DATE, VR::DA, "19940721"),
        ],
    )?;
    // Four zero-led UIDs, two of them repeated in the file meta group, and a
    // date.
    write(
        &input.join("s2/c.dcm"),
        ("1.2.840.10008.5.1.4.1.1.02", "1.2.3.04"),
        &[
            (SOP_CLASS_UID, VR::UI, "1.2.840.10008.5.1.4.1.1.02"),
            (SOP_INSTANCE_UID, VR::UI, "1.2.3.04"),
            (STUDY_DATE, VR::DA, "1994.07.23"),
            (STUDY_UID, VR::UI, "1.02.3"),
            (SERIES_UID, VR::UI, "1.2.03.7"),
        ],
    )?;
    // Nothing to repair.
    write(
        &input.join("s2/d.dcm"),
        (CT_IMAGE_STORAGE, "1.2.3.5"),
        &[
            (SOP_CLASS_UID, VR::UI, CT_IMAGE_STORAGE),
            (SOP_INSTANCE_UID, VR::UI, "1.2.3.5"),
            (STUDY_DATE, VR::DA, "19940721"),
        ],
    )?;
    write(
        &input.join("DICOMDIR"),
        (MEDIA_STORAGE_DIRECTORY, "1.2.3.6"),
        &[(SOP_INSTANCE_UID, VR::UI, "1.2.3.6")],
    )?;
    std::fs::write(input.join("notes.txt"), "not DICOM")?;
    std::fs::write(input.join("s1/report.txt"), "not DICOM either")?;
    let source_bytes = std::fs::read(&values_only)?;

    let report = repair_dicom_study(&input, &output)?;

    assert_eq!(
        report,
        StudyRepairReport {
            instances_written: 4,
            instances_repaired: 3,
            values: RepairSummary {
                dates: 3,
                times: 1,
                uids: 4,
            },
            file_meta_uids_aligned: 2,
            non_dicom_skipped: 2,
            indexes_omitted: 1,
        }
    );
    assert_eq!(
        stored(
            &output.join("s1/a.dcm"),
            &[STUDY_DATE, SERIES_DATE, STUDY_TIME]
        )?,
        [
            CT_IMAGE_STORAGE,
            "1.2.3.1",
            "19940721",
            "19940722",
            "102030"
        ]
    );
    assert_eq!(
        stored(&output.join("s1/b.dcm"), &[SOP_INSTANCE_UID, STUDY_DATE])?,
        [CT_IMAGE_STORAGE, "1.2.3.2", "1.2.3.2", "19940721"]
    );
    assert_eq!(
        stored(
            &output.join("s2/c.dcm"),
            &[
                SOP_CLASS_UID,
                SOP_INSTANCE_UID,
                STUDY_DATE,
                STUDY_UID,
                SERIES_UID
            ]
        )?,
        [
            CT_IMAGE_STORAGE,
            "1.2.3.4",
            CT_IMAGE_STORAGE,
            "1.2.3.4",
            "19940723",
            "1.2.3",
            "1.2.3.7"
        ]
    );
    assert_eq!(
        stored(&output.join("s2/d.dcm"), &[SOP_INSTANCE_UID, STUDY_DATE])?,
        [CT_IMAGE_STORAGE, "1.2.3.5", "1.2.3.5", "19940721"]
    );
    for left_out in ["DICOMDIR", "notes.txt", "s1/report.txt"] {
        assert!(!output.join(left_out).exists(), "{left_out}");
    }
    assert_eq!(std::fs::read(&values_only)?, source_bytes);
    assert_eq!(
        stored(&values_only, &[STUDY_DATE, SERIES_DATE, STUDY_TIME])?,
        [
            CT_IMAGE_STORAGE,
            "1.2.3.1",
            "1994.07.21",
            "1994.07.22",
            "10:20:30"
        ]
    );
    Ok(())
}

#[test]
fn an_invalid_instance_uid_leaves_the_file_meta_group_as_stored() -> TestResult {
    let root = tempfile::tempdir()?;
    let (input, output) = (root.path().join("study"), root.path().join("copy"));
    write(
        &input.join("a.dcm"),
        (CT_IMAGE_STORAGE, "1.2.3.9"),
        &[
            (SOP_CLASS_UID, VR::UI, CT_IMAGE_STORAGE),
            (SOP_INSTANCE_UID, VR::UI, "1..2"),
        ],
    )?;

    let report = repair_dicom_study(&input, &output)?;

    assert_eq!(
        report,
        StudyRepairReport {
            instances_written: 1,
            ..StudyRepairReport::default()
        }
    );
    assert_eq!(
        stored(&output.join("a.dcm"), &[SOP_INSTANCE_UID])?,
        [CT_IMAGE_STORAGE, "1.2.3.9", "1..2"]
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
fn a_missing_source_and_an_output_inside_the_study_are_refused() -> TestResult {
    let root = tempfile::tempdir()?;
    let input = root.path().join("study");

    let missing = repair_dicom_study(&input, root.path().join("copy"))
        .expect_err("the source does not exist");
    assert!(
        matches!(
            &missing,
            StudyRepairError::Tree(ExportTreeError::Io { action: "resolve", path, .. })
                if path == &input
        ),
        "{missing:?}"
    );

    write(
        &input.join("a.dcm"),
        (CT_IMAGE_STORAGE, "1.2.3.4"),
        &[
            (SOP_CLASS_UID, VR::UI, CT_IMAGE_STORAGE),
            (SOP_INSTANCE_UID, VR::UI, "1.2.3.4"),
        ],
    )?;
    let overlapping =
        repair_dicom_study(&input, input.join("copy")).expect_err("overlapping output");
    assert!(
        matches!(
            overlapping,
            StudyRepairError::Tree(ExportTreeError::OutputOverlapsInput { .. })
        ),
        "{overlapping:?}"
    );
    assert!(!input.join("copy").exists());
    Ok(())
}
