//! Scan a parsed object for identifiers that must not survive export.

use super::{FileVerifyReport, VerifyIssue};
use dicom::core::value::Value;
use dicom::core::VR;
use dicom::object::mem::InMemElement;
use dicom::object::{FileDicomObject, InMemDicomObject};

/// Whether `needle` occurs in `text` with no letter or digit adjoining it.
fn occurs_as_token(text: &str, needle: &str) -> bool {
    text.match_indices(needle).any(|(start, matched)| {
        let before = text[..start].chars().next_back();
        let after = text[start + matched.len()..].chars().next();
        !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric)
    })
}

/// Text held by an element whose value representation can carry a name or an
/// identifier, or by a UN element, whose bytes may encode either.
fn element_text(element: &InMemElement) -> Option<String> {
    match element.vr() {
        VR::AE | VR::CS | VR::LO | VR::LT | VR::PN | VR::SH | VR::ST | VR::UC | VR::UT => {
            element.to_str().ok().map(std::borrow::Cow::into_owned)
        }
        VR::UN => element
            .to_bytes()
            .ok()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned()),
        _ => None,
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
            let Some(text) = element_text(element) else {
                continue;
            };
            if let Some(hit) = prohibited
                .iter()
                .find(|value| !value.is_empty() && occurs_as_token(&text, value))
            {
                report
                    .issues
                    .push(VerifyIssue::PatientLeak { value: hit.clone() });
            }
        }
    }
}
