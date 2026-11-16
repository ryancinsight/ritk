//! Repaired copy of the open study from the native shell.

use super::*;
use crate::presentation::native_session::repair::repair_status;
use ritk_io::format::dicom::repair::{StudyRepairError, StudyRepairReport};
use ritk_io::format::dicom::ExportTreeError;
use std::path::{Path, PathBuf};

fn first_instance(root: &Path) -> PathBuf {
    root.join(format!("{}_a.dcm", fixtures::SERIES_UID))
}

/// StudyDate (0008,0020) as the fixture's explicit VR little endian writer
/// stores it: tag, `DA`, a two-byte length, then the even-length text.
fn study_date_element(text: &str) -> Vec<u8> {
    let length = u16::try_from(text.len()).expect("a date fits a short length");
    let mut bytes = vec![0x08, 0x00, 0x20, 0x00, b'D', b'A'];
    bytes.extend(length.to_le_bytes());
    bytes.extend(text.bytes());
    bytes
}

fn occurrences(bytes: &[u8], element: &[u8]) -> usize {
    bytes
        .windows(element.len())
        .filter(|window| *window == element)
        .count()
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
fn native_repair_writes_a_corrected_copy_and_leaves_the_source() {
    let (mut session, root) = session_on_study();
    let defective = first_instance(root.path());
    let (conformant, acr_nema) = (
        study_date_element("20260905"),
        study_date_element("2026.09.05"),
    );
    let stored = std::fs::read(&defective).expect("read the first instance");
    assert_eq!(occurrences(&stored, &conformant), 1);
    let at = stored
        .windows(conformant.len())
        .position(|window| window == conformant)
        .expect("the fixture stores StudyDate");
    let mut source_bytes = stored[..at].to_vec();
    source_bytes.extend(&acr_nema);
    source_bytes.extend(&stored[at + conformant.len()..]);
    std::fs::write(&defective, &source_bytes).expect("store the ACR-NEMA date");
    let parent = tempfile::tempdir().expect("copy parent");
    let output = parent.path().join("copy");

    session
        .repair_study_to(&output)
        .expect("write a repaired copy");

    let copy = std::fs::read(first_instance(&output)).expect("read the repaired copy");
    assert_eq!(occurrences(&copy, &conformant), 1);
    assert_eq!(occurrences(&copy, &acr_nema), 0);
    assert_eq!(
        std::fs::read(&defective).expect("re-read the defective instance"),
        source_bytes
    );
    assert_eq!(
        session.app.status_message,
        format!(
            "Wrote 3 instance(s) to {}; 1 changed: 1 date(s), 0 time(s) and 0 UID(s) rewritten, \
             file meta UIDs aligned in 0 instance(s).",
            output.display()
        )
    );
}

#[test]
fn native_repair_status_reports_each_count_under_its_own_name() {
    let mut report = StudyRepairReport::default();
    report.instances_written = 9;
    report.instances_repaired = 8;
    report.values.dates = 7;
    report.values.times = 6;
    report.values.uids = 5;
    report.file_meta_uids_aligned = 4;
    report.non_dicom_skipped = 2;
    report.indexes_omitted = 1;

    assert_eq!(
        repair_status(report, Path::new("out")),
        "Wrote 9 instance(s) to out; 8 changed: 7 date(s), 6 time(s) and 5 UID(s) rewritten, \
         file meta UIDs aligned in 4 instance(s). 3 file(s) left out (not DICOM, or a DICOMDIR)."
    );
}

#[test]
fn native_repair_refuses_an_output_folder_inside_the_study() {
    let (mut session, root) = session_on_study();

    let error = session
        .repair_study_to(&root.path().join("copy"))
        .expect_err("an output inside the source overlaps it");

    assert!(matches!(
        error.downcast_ref::<StudyRepairError>(),
        Some(StudyRepairError::Tree(
            ExportTreeError::OutputOverlapsInput { .. }
        ))
    ));
    assert!(!root.path().join("copy").exists());
}

#[test]
fn native_repair_without_a_study_folder_says_so_and_opens_no_dialog() {
    let (mut session, _root) = session();

    let changed = session
        .repair_study_from_dialog()
        .expect("no dialog is shown without a study folder");

    assert!(changed);
    assert_eq!(
        session.app.status_message,
        "Open a study from disk before writing a repaired copy."
    );
}
