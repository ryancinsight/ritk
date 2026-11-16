//! The file meta copy of the SOP Instance UID.
//!
//! PS3.10 Table 7.1-1 defines Media Storage SOP Instance UID (0002,0003) as
//! the SOP Instance UID of the data set the file holds. A de-identified data
//! set whose file meta still carries the source UID is inconsistent, and the
//! stale value links the export back to the source instance.

use super::verify::VerifyIssue;
use dicom::core::Tag;
use dicom::object::{FileDicomObject, InMemDicomObject};

const SOP_INSTANCE_UID: Tag = Tag(0x0008, 0x0018);

fn unpadded(uid: &str) -> &str {
    uid.trim_end_matches(['\0', ' '])
}

fn data_set_uid(object: &FileDicomObject<InMemDicomObject>) -> Option<String> {
    let element = object.element(SOP_INSTANCE_UID).ok()?;
    let text = element.to_str().ok()?;
    let uid = unpadded(&text);
    (!uid.is_empty()).then(|| uid.to_owned())
}

/// Copy the data set's SOP Instance UID into the file meta group. A data set
/// without one leaves the file meta group as it is.
pub(super) fn align_media_storage_uid(object: &mut FileDicomObject<InMemDicomObject>) {
    if let Some(uid) = data_set_uid(object) {
        object.update_meta(|meta| meta.media_storage_sop_instance_uid = uid);
    }
}

/// The issue raised when the file meta group names another instance than the
/// data set does.
pub(super) fn media_storage_uid_issue(
    object: &FileDicomObject<InMemDicomObject>,
) -> Option<VerifyIssue> {
    let expected = data_set_uid(object)?;
    let found = unpadded(object.meta().media_storage_sop_instance_uid());
    (found != expected).then(|| VerifyIssue::UidMismatch {
        tag: "MediaStorageSOPInstanceUID".to_owned(),
        expected,
        found: found.to_owned(),
    })
}
