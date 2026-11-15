//! Scan a parsed object for identifiers that must not survive export.

use super::{FileVerifyReport, VerifyIssue};
use dicom::core::value::Value;
use dicom::core::VR;
use dicom::object::mem::InMemElement;
use dicom::object::{FileDicomObject, InMemDicomObject};

/// How an element's text is compared with a prohibited value.
enum Scanned {
    /// Free text: the value may sit among other words.
    Words(String),
    /// Coded values separated by `\`: one of them may equal the value.
    Values(String),
}

/// Whether `needle` occurs in `text` with no letter or digit adjoining it.
///
/// Every occurrence is tried, overlapping ones included: in `112-12-12` the
/// occurrence of `12-12` at byte 1 adjoins a digit and the one at byte 4 does
/// not.
fn occurs_as_word(text: &str, needle: &str) -> bool {
    let mut from = 0;
    while let Some(offset) = text[from..].find(needle) {
        let start = from + offset;
        let before = text[..start].chars().next_back();
        let after = text[start + needle.len()..].chars().next();
        if !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric) {
            return true;
        }
        let Some(step) = text[start..].chars().next() else {
            return false;
        };
        from = start + step.len_utf8();
    }
    false
}

/// Text of an element that can carry a name, an identifier, a date or a UID.
///
/// Free-text representations and UN bytes are searched for whole words. UIDs,
/// dates, date-times and URIs are compared value by value, because a short
/// identifier is a component of almost every UID. Numeric and binary
/// representations are not read: a patient ID of `7` equals many row counts.
fn scanned_text(element: &InMemElement) -> Option<Scanned> {
    match element.vr() {
        VR::AE | VR::CS | VR::LO | VR::LT | VR::PN | VR::SH | VR::ST | VR::UC | VR::UT => {
            let text = element.to_str().ok()?;
            Some(Scanned::Words(text.into_owned()))
        }
        VR::UN => {
            let bytes = element.to_bytes().ok()?;
            Some(Scanned::Words(String::from_utf8_lossy(&bytes).into_owned()))
        }
        VR::UI | VR::DA | VR::DT | VR::UR => {
            let text = element.to_str().ok()?;
            Some(Scanned::Values(text.into_owned()))
        }
        _ => None,
    }
}

fn holds(scanned: &Scanned, prohibited: &str) -> bool {
    match scanned {
        Scanned::Words(text) => occurs_as_word(text, prohibited),
        Scanned::Values(text) => text
            .split('\\')
            .any(|value| value.trim_matches(['\0', ' ']) == prohibited),
    }
}

pub(super) fn scan_prohibited_values(
    obj: &FileDicomObject<InMemDicomObject>,
    prohibited: &[String],
    report: &mut FileVerifyReport,
) {
    let mut pending: Vec<&InMemDicomObject> = vec![obj];
    while let Some(dataset) = pending.pop() {
        for element in dataset {
            if let Value::Sequence(sequence) = element.value() {
                pending.extend(sequence.items());
                continue;
            }
            let Some(scanned) = scanned_text(element) else {
                continue;
            };
            if let Some(hit) = prohibited
                .iter()
                .find(|value| !value.is_empty() && holds(&scanned, value))
            {
                report
                    .issues
                    .push(VerifyIssue::PatientLeak { value: hit.clone() });
            }
        }
    }
}
