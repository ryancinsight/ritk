//! Verify an export that holds several series.

use super::{verify_dicom_file, VerifyIssue, VerifyOptions};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// Verify a set of exported instances that may span several series and studies.
///
/// Every instance passes through [`verify_dicom_file`]. Across the set, each
/// SOP Instance UID must be unique and every series must belong to one study;
/// unlike [`verify_dicom_directory`](super::verify_dicom_directory), several series and studies may be
/// present. Issues are returned with the path they were found in, in input
/// order; an empty result means the set passed.
#[must_use]
pub fn verify_dicom_instances(
    paths: &[PathBuf],
    options: &VerifyOptions,
) -> Vec<(PathBuf, VerifyIssue)> {
    let mut issues = Vec::new();
    let mut sop_seen: HashSet<String> = HashSet::with_capacity(paths.len());
    let mut study_of_series: HashMap<String, String> = HashMap::new();
    for path in paths {
        let file = match verify_dicom_file(path, options) {
            Ok(file) => file,
            Err(error) => {
                issues.push((
                    path.clone(),
                    VerifyIssue::ParseFailure(format!("{error:#}")),
                ));
                continue;
            }
        };
        let uid = |name: &str| {
            file.uids
                .iter()
                .find_map(|(key, value)| (key == name).then(|| value.clone()))
        };
        if let Some(sop) = uid("SOPInstanceUID")
            && !sop_seen.insert(sop.clone())
        {
            issues.push((
                path.clone(),
                VerifyIssue::DuplicateSopInstanceUid { value: sop },
            ));
        }
        if let (Some(series), Some(study)) = (uid("SeriesInstanceUID"), uid("StudyInstanceUID")) {
            let expected = study_of_series
                .entry(series)
                .or_insert_with(|| study.clone());
            if *expected != study {
                issues.push((
                    path.clone(),
                    VerifyIssue::UidMismatch {
                        tag: "StudyInstanceUID".to_owned(),
                        expected: expected.clone(),
                        found: study,
                    },
                ));
            }
        }
        issues.extend(file.issues.into_iter().map(|issue| (path.clone(), issue)));
    }
    issues
}
