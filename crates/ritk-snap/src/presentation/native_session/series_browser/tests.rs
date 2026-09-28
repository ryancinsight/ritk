use super::*;
use crate::dicom::loader::tests::fixtures;
use tempfile::tempdir;

#[test]
fn browser_retains_every_series_and_selects_by_uid() {
    let root = tempdir().expect("study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write CT series");
    fixtures::write_study(root.path(), "MR", "2.25.20260905002").expect("write MR series");
    let tree = crate::dicom::loader::scan_folder_for_series(root.path()).expect("scan study");

    let browser = SeriesBrowser::from_tree(&tree, Some("2.25.20260905002")).expect("browser");

    assert_eq!(browser.len(), 2);
    assert_eq!(browser.study_count(), 1);
    assert_eq!(tree.patients.len(), 1);
    assert_eq!(tree.patients[0].patient_name, "FIXTURE^PATIENT");
    assert_eq!(tree.patients[0].studies.len(), 1);
    assert_eq!(
        tree.patients[0].studies[0].study_uid.as_deref(),
        Some("2.25.20260905")
    );
    assert_eq!(
        tree.patients[0].studies[0].study_date.as_deref(),
        Some("20260905")
    );
    assert_eq!(browser.active_index(), 1);
    assert_eq!(
        browser.choice(0).expect("first series").modality.as_ref(),
        "CT"
    );
    assert_eq!(
        browser.choice(1).expect("second series").modality.as_ref(),
        "MR"
    );
    assert_eq!(
        browser
            .choice(1)
            .expect("second series")
            .description
            .as_ref(),
        "Series 2"
    );
    assert_eq!(browser.choice(1).expect("second series").instance_count, 3);
}

#[test]
fn browser_rejects_a_series_uid_outside_the_discovered_study() {
    let root = tempdir().expect("study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write CT series");
    let tree = crate::dicom::loader::scan_folder_for_series(root.path()).expect("scan study");

    let result = SeriesBrowser::from_tree(&tree, Some("2.25.999"));

    assert_eq!(
        result.err().map(|error| error.to_string()),
        Some("selected SeriesInstanceUID is absent from study".into())
    );
}

#[test]
fn active_series_stays_visible_after_selection_and_scroll_clamps() {
    let root = tempdir().expect("study root");
    for index in 1..=4 {
        fixtures::write_study(root.path(), "MR", &format!("2.25.2026090500{index}"))
            .expect("write series");
    }
    let tree = crate::dicom::loader::scan_folder_for_series(root.path()).expect("scan study");
    let mut browser = SeriesBrowser::from_tree(&tree, None).expect("browser");

    assert!(browser.scroll_series(2, 2));
    assert_eq!(browser.first_visible(), 2);
    assert!(browser.set_active(0));
    assert_eq!(browser.first_visible(), 0);
    assert!(browser.set_active(3));
    assert_eq!(browser.first_visible(), 3);
    assert!(browser.scroll_series(1, 2));
    assert_eq!(browser.first_visible(), 2);
    assert!(browser.scroll_series(-2, 2));
    assert_eq!(browser.first_visible(), 0);
}
