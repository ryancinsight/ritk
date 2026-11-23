use super::*;
use ritk_io::{DicomObjectNode, DicomSliceMetadata};
use std::path::PathBuf;

fn acquisition(
    uid: &str,
    modality: &str,
    number: Option<&str>,
    names: &[&str],
    patient: &str,
    study: &str,
) -> DiscoveredSeries {
    let paths: Vec<PathBuf> = names.iter().map(PathBuf::from).collect();
    let mut slice = DicomSliceMetadata {
        path: paths.iter().min().expect("member").clone(),
        ..Default::default()
    };
    if let Some(number) = number {
        slice.preservation.object.insert(DicomObjectNode::text(
            DicomTag::new(0x0020, 0x0011),
            "IS",
            number,
        ));
    }
    let metadata = DicomReadMetadata {
        study_instance_uid: Some(study.try_into().expect("study UID")),
        slices: vec![slice],
        ..Default::default()
    };
    DiscoveredSeries::new(
        DicomSeriesInfo::new(uid, String::new(), modality, patient.into(), paths),
        &metadata,
    )
}

#[test]
fn discovery_orders_signed_numbers_ties_and_missing_numbers_without_uids() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let cases: &[(&str, Option<&str>, &str, &[&str])] = &[
        ("6", None, "CT", &["e"]),
        ("5", Some(""), "CT", &["d"]),
        ("4", Some("10"), "CT", &["c"]),
        ("3", Some(" +2 "), "MR", &["b"]),
        ("2", Some("2"), "CT", &["z"]),
        ("1", Some("2"), "CT", &["y", "a"]),
        ("9", Some("-1"), "CT", &["w"]),
        ("8", Some("invalid"), "MR", &["g"]),
    ];
    for reversed in [false, true] {
        let mut input = tracing::info_span!("arrange").in_scope(|| {
            cases
                .iter()
                .map(|(uid, n, m, files)| acquisition(uid, m, *n, files, "P", "1"))
                .collect::<Vec<_>>()
        });
        if reversed {
            input.reverse();
        }
        let tree = tracing::info_span!("act").in_scope(|| SeriesTree::from_discovery(input));
        let order: Vec<_> = tree
            .iter_series()
            .map(|s| s.acquisition.series_instance_uid())
            .collect();
        tracing::info_span!("assert")
            .in_scope(|| assert_eq!(order, ["9", "1", "2", "3", "4", "5", "6", "8"]));
    }
}

#[test]
fn discovery_keeps_patients_and_studies_grouped_in_a_collection() {
    let tree = SeriesTree::from_discovery(vec![
        acquisition("3", "MR", None, &["z/path/slice.dcm"], "P2", "2"),
        acquisition("2", "CT", None, &["b/path/slice.dcm"], "P1", "1"),
        acquisition("1", "CT", None, &["a/path/slice.dcm"], "P1", "1"),
    ]);
    assert_eq!(tree.patients.len(), 2);
    let order: Vec<_> = tree
        .iter_series()
        .map(|s| s.acquisition.series_instance_uid())
        .collect();
    assert_eq!(order, ["1", "2", "3"]);
}
