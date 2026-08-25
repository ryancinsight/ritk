//! Tests for the sidebar UI module.

<<<<<<< HEAD
#[allow(unused_imports, reason = "ratchet RITK-LINT-1")]
=======
>>>>>>> origin/main
use super::*;
use crate::dicom::metadata_table::{MetadataRow, MetadataScope};
use std::borrow::Cow;

/// Test helper: build a MetadataRow from raw string slices.
fn make_row<'a>(tag: &'a str, keyword: &'a str, value: &'a str) -> MetadataRow<'a> {
    MetadataRow {
        scope: MetadataScope::Series,
        tag: Cow::Borrowed(tag),
        keyword: Cow::Borrowed(keyword),
        vr: Cow::Borrowed("LO"),
        value: Cow::Borrowed(value),
    }
}

/// Test helper: filter rows by case-insensitive needle matching tag, keyword, or value.
fn filter_rows<'a>(rows: &'a [MetadataRow<'a>], needle: &str) -> Vec<&'a MetadataRow<'a>> {
    let needle_lc = needle.to_ascii_lowercase();
    rows.iter()
        .filter(|r| {
            needle.is_empty()
                || r.tag.to_ascii_lowercase().contains(&needle_lc)
                || r.keyword.to_ascii_lowercase().contains(&needle_lc)
                || r.value.to_ascii_lowercase().contains(&needle_lc)
        })
        .collect()
}
/// `SidebarTab::default()` must be `SidebarTab::Series`.
///
/// The `#[default]` attribute on `Series` drives the derived `Default`
/// impl; this test verifies the attribute is in place.
#[test]
fn test_sidebar_tab_default_is_series() {
    let tab = SidebarTab::default();
    assert_eq!(
        tab,
        SidebarTab::Series,
        "default SidebarTab must be Series, got {tab:?}"
    );
}

/// `SidebarTab::Series` and `SidebarTab::Metadata` must be distinct values.
///
/// The two tab variants differ in their discriminant; the derived `PartialEq`
/// and `Eq` impls must reflect that inequality.
#[test]
fn test_sidebar_tab_variants_are_distinct() {
    assert_ne!(
        SidebarTab::Series,
        SidebarTab::Metadata,
        "SidebarTab::Series and SidebarTab::Metadata must be distinct variants"
    );
}

/// Needle "patient" (case-insensitive) must match PatientName and PatientID.
#[test]
fn test_tag_filter_keyword_case_insensitive() {
    let rows = vec![
        make_row("(0010,0010)", "PatientName", "Doe^John"),
        make_row("(0010,0020)", "PatientID", "MR12345"),
        make_row("(0008,0060)", "Modality", "CT"),
    ];
    let filtered = filter_rows(&rows, "patient");
    assert_eq!(
        filtered.len(),
        2,
        "needle 'patient' must match PatientName and PatientID"
    );
    assert_eq!(filtered[0].keyword, "PatientName");
    assert_eq!(filtered[1].keyword, "PatientID");
}

/// Needle matching a tag hex code must return the matching row.
#[test]
fn test_tag_filter_by_hex_tag() {
    let rows = vec![
        make_row("(0010,0010)", "PatientName", "Doe^John"),
        make_row("(0008,0060)", "Modality", "CT"),
    ];
    let filtered = filter_rows(&rows, "0008,0060");
    assert_eq!(
        filtered.len(),
        1,
        "tag hex search must match exactly one row"
    );
    assert_eq!(filtered[0].keyword, "Modality");
}

/// Needle matching a value must return the correct row.
#[test]
fn test_tag_filter_by_value() {
    let rows = vec![
        make_row("(0010,0010)", "PatientName", "Doe^John"),
        make_row("(0008,0060)", "Modality", "CT"),
        make_row("(0020,0013)", "InstanceNumber", "42"),
    ];
    let filtered = filter_rows(&rows, "doe^john");
    assert_eq!(filtered.len(), 1, "value search must match exactly one row");
    assert_eq!(filtered[0].keyword, "PatientName");
}

/// Needle that matches no field must return empty.
#[test]
fn test_tag_filter_no_match_returns_empty() {
    let rows = vec![
        make_row("(0010,0010)", "PatientName", "Doe^John"),
        make_row("(0008,0060)", "Modality", "CT"),
    ];
    let filtered = filter_rows(&rows, "xxxxnonexistent");
    assert_eq!(filtered.len(), 0, "unmatched needle must return empty");
}

/// Empty string needle: all rows pass through ("" is contained in every string).
#[test]
fn test_tag_filter_empty_needle_passes_all() {
    let rows = vec![
        make_row("(0010,0010)", "PatientName", "Doe^John"),
        make_row("(0008,0060)", "Modality", "CT"),
    ];
    let filtered = filter_rows(&rows, "");
    assert_eq!(filtered.len(), rows.len(), "empty needle: all rows pass");
}
