//! The file meta copy of the SOP Instance UID.
//!
//! PS3.10 Table 7.1-1 defines Media Storage SOP Instance UID (0002,0003) as
//! the SOP Instance UID of the data set the file holds. A de-identified data
//! set whose file meta still carries the source UID is inconsistent, and the
//! stale value links the export back to the source instance.

use super::verify::VerifyIssue;
use crate::format::dicom::identity::uid_is_valid;
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

/// Set Media Storage SOP Instance UID (0002,0003) to the data set's
/// SOPInstanceUID and recompute the group length (0002,0000).
///
/// Nothing changes when the file meta group already holds that value, or when
/// the data set's SOPInstanceUID is absent, empty or not a single UID (PS3.5
/// section 9.1); the last because the dicom-rs reader refuses a
/// file whose (0002,0003) is not a UID. [`media_storage_uid_issue`] reports a
/// non-empty value left differing this way, and nothing for an absent or
/// empty one.
///
/// Returns whether the file meta value changed.
pub(in crate::format::dicom) fn align_media_storage_uid(
    object: &mut FileDicomObject<InMemDicomObject>,
) -> bool {
    let Some(uid) = data_set_uid(object).filter(|uid| uid_is_valid(uid)) else {
        return false;
    };
    if object.meta().media_storage_sop_instance_uid() == uid {
        return false;
    }
    object.update_meta(|meta| meta.media_storage_sop_instance_uid = uid);
    true
}

/// The issue raised when the file meta group names another instance than the
/// data set does.
pub(super) fn media_storage_uid_issue(
    object: &FileDicomObject<InMemDicomObject>,
) -> Option<VerifyIssue> {
    let expected = data_set_uid(object)?;
    let found = object.meta().media_storage_sop_instance_uid();
    (found != expected).then(|| VerifyIssue::UidMismatch {
        tag: "MediaStorageSOPInstanceUID".to_owned(),
        expected,
        found: found.to_owned(),
    })
}
