//! Profile actions on attributes present at the current data set level.

use std::collections::HashMap;

use ritk_dicom::dataset::{Dataset, DicomTag, PrimitiveValue, SmallVec, VR};

use super::uid::generate_uid_from_hash;
use super::{AnonymizeOptions, AnonymizeResult, TagAction};

const PATIENT_NAME: DicomTag = DicomTag::new(0x0010, 0x0010);
const PATIENT_ID: DicomTag = DicomTag::new(0x0010, 0x0020);

pub(super) fn apply_action(
    dataset: &mut Dataset,
    tag: DicomTag,
    action: TagAction,
    options: &AnonymizeOptions,
    result: &mut AnonymizeResult,
    uid_map: &mut HashMap<String, String>,
) {
    let Ok(source) = dataset.element(tag) else {
        return;
    };
    match action {
        TagAction::Keep => {}
        TagAction::Remove => {
            dataset.remove_element(tag);
            result.tags_deleted += 1;
        }
        TagAction::Dummy => {
            let (vr, text) = if tag == PATIENT_ID {
                (VR::LO, &options.patient_id)
            } else {
                debug_assert_eq!(tag, PATIENT_NAME);
                (VR::PN, &options.patient_name)
            };
            dataset.put_value(tag, vr, text.as_str());
            result.tags_zeroed += 1;
        }
        TagAction::Empty => {
            dataset.put_value(tag, source.vr(), PrimitiveValue::Empty);
            result.tags_zeroed += 1;
        }
        TagAction::ReplaceUid => {
            let Ok(text) = source.to_str() else {
                return;
            };
            // UI permits multiple values for tags such as Irradiation Event UID.
            // Map every value independently so references to it remain valid.
            let mapped: SmallVec<[String; 2]> = text
                .split('\\')
                .filter(|uid| !uid.is_empty())
                .map(|uid| {
                    uid_map
                        .entry(uid.to_owned())
                        .or_insert_with(|| generate_uid_from_hash(uid, &options.uid_salt))
                        .clone()
                })
                .collect();
            if !mapped.is_empty() {
                dataset.put_value(tag, VR::UI, PrimitiveValue::Strs(mapped));
                result.uids_remapped += 1;
            }
        }
    }
}
