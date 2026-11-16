//! De-identified export of the open study from the native shell.

use super::*;
use crate::presentation::native_session::deidentify::Deidentification;
use ritk_io::format::dicom::anonymize::verify::{
    verify_dicom_instances, VerifyIssue, VerifyOptions,
};
use ritk_io::format::dicom::anonymize::StudyAnonymizeError;
use ritk_io::format::dicom::ExportTreeError;
use std::path::{Path, PathBuf};

const STUDY_DATE: &str = "20260905";
/// The fixture study's three instances, as `fixtures::write_study` names them.
fn instances(root: &Path) -> Vec<PathBuf> {
    ["a.dcm", "b.dcm", "c.dcm"]
        .iter()
        .map(|name| root.join(format!("{}_{name}", fixtures::SERIES_UID)))
        .collect()
}

/// The prohibited values the export gate finds under `root`, one per leaking
/// element; any other issue appears as its message.
fn leaks(root: &Path, values: &[&str]) -> Vec<String> {
    let options = VerifyOptions {
        prohibited_values: values.iter().map(|value| (*value).to_owned()).collect(),
        ..VerifyOptions::default()
    };
    verify_dicom_instances(&instances(root), &options)
        .into_iter()
        .map(|(_, issue)| match issue {
            VerifyIssue::PatientLeak { value } => value,
            other => other.to_string(),
        })
        .collect()
}

/// A session showing the fixture study, opened from its folder.
fn session_on_study() -> (NativeViewerSession, tempfile::TempDir) {
    let (mut session, root) = session();
    session
        .open_study_path(root.path())
        .expect("open the fixture study from its folder");
    (session, root)
}

#[test]
fn native_anonymize_keeping_dates_replaces_identifiers_and_instance_uids() {
    let (mut session, root) = session_on_study();
    let parent = tempfile::tempdir().expect("export parent");
    let output = parent.path().join("export");
    let source = instances(root.path()).swap_remove(0);
    let source_bytes = std::fs::read(&source).expect("read source instance");
    let identifiers = ["FIXTURE^PATIENT", "RITK-TEST-001", fixtures::SERIES_UID];
    assert_eq!(
        leaks(root.path(), &identifiers),
        identifiers.repeat(3),
        "the gate must see the source identifiers"
    );

    session
        .anonymize_study_to(&output, Deidentification::KeepDates)
        .expect("export a de-identified copy");

    assert_eq!(leaks(&output, &identifiers), [""; 0]);
    assert_eq!(leaks(&output, &[STUDY_DATE]), [STUDY_DATE; 3]);
    assert_eq!(
        session.app.status_message,
        format!(
            "De-identified 3 image(s) to {} (dates and descriptions kept). \
             Text burned into pixels is not removed.",
            output.display()
        )
    );
    assert_eq!(
        std::fs::read(&source).expect("re-read source instance"),
        source_bytes
    );
}

#[test]
fn native_anonymize_strict_also_empties_the_study_date() {
    let (mut session, _root) = session_on_study();
    let parent = tempfile::tempdir().expect("export parent");
    let output = parent.path().join("export");

    session
        .anonymize_study_to(&output, Deidentification::Strict)
        .expect("export a strictly de-identified copy");

    assert_eq!(
        leaks(&output, &["FIXTURE^PATIENT", "RITK-TEST-001", STUDY_DATE]),
        [""; 0]
    );
    assert_eq!(
        session.app.status_message,
        format!(
            "De-identified 3 image(s) to {} (dates, descriptions and private elements removed). \
             Text burned into pixels is not removed.",
            output.display()
        )
    );
}

#[test]
fn native_anonymize_refuses_an_output_folder_inside_the_study() {
    let (mut session, root) = session_on_study();
    let shape = session.app.loaded.as_ref().expect("open study").shape;

    let error = session
        .anonymize_study_to(&root.path().join("export"), Deidentification::KeepDates)
        .expect_err("an output inside the source overlaps it");

    assert!(matches!(
        error.downcast_ref::<StudyAnonymizeError>(),
        Some(StudyAnonymizeError::Tree(
            ExportTreeError::OutputOverlapsInput { .. }
        ))
    ));
    assert!(!root.path().join("export").exists());
    assert_eq!(
        session.app.loaded.as_ref().expect("open study").shape,
        shape
    );
}

#[test]
fn native_anonymize_without_a_study_folder_says_so_and_opens_no_dialog() {
    let (mut session, _root) = session();
    assert_eq!(session.study_root, None);

    let changed = session
        .anonymize_study_from_dialog(Deidentification::KeepDates)
        .expect("report the missing study folder");

    assert!(changed);
    assert_eq!(
        session.app.status_message,
        "Open a study from disk before exporting a de-identified copy."
    );
}

#[test]
fn native_anonymize_source_follows_the_study_that_is_displayed() {
    let (mut session, root) = session_on_study();
    let empty = tempfile::tempdir().expect("empty folder");

    session
        .open_study_path(empty.path())
        .expect_err("an empty folder holds no study");

    assert_eq!(session.study_root.as_deref(), Some(root.path()));
}
