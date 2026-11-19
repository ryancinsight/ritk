//! Enhanced PET Concatenation membership and frame address validation.

use std::collections::HashSet;

use anyhow::{ensure, Context, Result};
use dicom::core::Tag;

use super::frame_source::FrameSource;
use crate::format::dicom::identity::{optional_uid, CONCATENATION_UID};

const CONCATENATION_SOURCE: Tag = Tag(0x0020, 0x0242);
const MEMBER_NUMBER: Tag = Tag(0x0020, 0x9162);
const MEMBER_TOTAL: Tag = Tag(0x0020, 0x9163);
const FRAME_OFFSET: Tag = Tag(0x0020, 0x9228);
const PER_FRAME: Tag = Tag(0x5200, 0x9230);
const SHARED: Tag = Tag(0x5200, 0x9229);
const ORGANIZATIONS: Tag = Tag(0x0020, 0x9221);
const DIMENSIONS: Tag = Tag(0x0020, 0x9222);
const SOP_INSTANCE: Tag = Tag(0x0008, 0x0018);

/// DICOM PS3.3 C.7.6.16.2.2 defines member offsets as the sum of frame counts
/// of preceding instances. This validates the complete known interval before
/// assigning pixel addresses. With the optional total absent, only an
/// unbroken prefix can be established.
pub(super) fn order_members(sources: &[FrameSource<'_>]) -> Result<Vec<usize>> {
    let first = sources
        .first()
        .context("Enhanced PET requires at least one instance")?;
    let Some(concatenation_uid) = optional_uid(first.obj, CONCATENATION_UID)? else {
        ensure!(
            sources.len() == 1,
            "Select one Enhanced PET acquisition or concatenation"
        );
        ensure!(
            [
                CONCATENATION_SOURCE,
                MEMBER_NUMBER,
                MEMBER_TOTAL,
                FRAME_OFFSET
            ]
            .iter()
            .all(|&tag| first.obj.element(tag).is_err()),
            "Enhanced PET concatenation attributes require Concatenation UID"
        );
        return Ok(vec![0]);
    };

    let source_uid = optional_uid(first.obj, CONCATENATION_SOURCE)?
        .context("Enhanced PET concatenation is missing the source SOP Instance UID")?;
    let series_uid = optional_uid(first.obj, Tag(0x0020, 0x000e))?
        .context("Enhanced PET concatenation is missing Series Instance UID")?;
    let identity_tags = [Tag(0x0020, 0x000d), Tag(0x0020, 0x0052)];
    let [study_uid, frame_of_reference] = identity_tags.map(|tag| optional_uid(first.obj, tag));
    let expected_identity = [study_uid?, frame_of_reference?];
    let reference = first.obj;
    let reference_layout = layout(first);
    let instance_count = sources.len();
    let mut instance_uids = HashSet::with_capacity(instance_count);

    for member in sources.iter() {
        let obj = member.obj;
        let path = member.path;
        let [study_uid, frame_of_reference] = identity_tags.map(|tag| optional_uid(obj, tag));
        ensure!(
            optional_uid(obj, CONCATENATION_UID)? == Some(concatenation_uid)
                && optional_uid(obj, CONCATENATION_SOURCE)? == Some(source_uid)
                && optional_uid(obj, Tag(0x0020, 0x000e))? == Some(series_uid)
                && [study_uid?, frame_of_reference?] == expected_identity,
            "Enhanced PET concatenation identity differs in {path:?}"
        );
        let uid = optional_uid(obj, SOP_INSTANCE)?
            .context("Concatenation member is missing SOP Instance UID")?;
        ensure!(
            uid.as_str()
                == obj
                    .meta()
                    .media_storage_sop_instance_uid
                    .trim_end_matches('\0')
                && uid != source_uid
                && instance_uids.insert(uid),
            "Invalid or duplicate Enhanced PET concatenation member SOP Instance UID in {path:?}"
        );
        ensure!(
            layout(member) == reference_layout,
            "Enhanced PET concatenation pixel layout differs in {path:?}"
        );
        for tag in [SHARED, ORGANIZATIONS, DIMENSIONS] {
            let actual = obj.element_opt(tag)?;
            let expected = reference.element_opt(tag)?;
            ensure!(
                actual.map(|e| e.items().context("Enhanced PET concatenation attribute must be a sequence"))
                    .transpose()?
                    == expected.map(|e| e.items().context("Enhanced PET concatenation attribute must be a sequence"))
                        .transpose()?,
                "Enhanced PET concatenation shared groups or dimension definitions differ at {tag:?} in {path:?}"
            );
        }
        ensure!(
            obj.element(PER_FRAME)?.items().context("Expected per-frame functional group sequence")?.len()
                == member.info.n_frames,
            "Enhanced PET concatenation per-frame group count must equal NumberOfFrames in {path:?}"
        );
    }

    // Sort indices with the already validated offset. The number of members is
    // bounded by the scan/decode budget before parsing.
    let mut order = sources
        .iter()
        .enumerate()
        .map(|(index, member)| {
            let offset = member.obj.element(FRAME_OFFSET)?.to_int::<u32>()?;
            let number = member.obj.element(MEMBER_NUMBER)?.to_int::<u16>()?;
            Ok((offset, number, index))
        })
        .collect::<Result<Vec<_>>>()?;
    order.sort_unstable();

    let mut next_offset = 0usize;
    for (rank, (offset, number, index)) in order.iter().copied().enumerate() {
        let member = &sources[index];
        ensure!(
            offset as usize == next_offset && usize::from(number) == rank + 1,
            "Enhanced PET concatenation has a gap, overlap, or incorrect member number at frame offset {offset} in {:?}",
            member.path
        );
        if let Some(total) = member.obj.element_opt(MEMBER_TOTAL)? {
            ensure!(
                instance_count > 1 && usize::from(total.to_int::<u16>()?) == instance_count,
                "Enhanced PET concatenation member count differs from In-concatenation Total Number"
            );
        }
        next_offset = next_offset
            .checked_add(member.info.n_frames)
            .context("Enhanced PET concatenation frame count overflows usize")?;
    }

    Ok(order.into_iter().map(|(_, _, source)| source).collect())
}

fn layout(source: &FrameSource<'_>) -> (usize, usize, u16, u16, ritk_dicom::PixelSignedness) {
    let info = &source.info;
    (
        info.rows,
        info.cols,
        info.bits_allocated,
        info.bits_stored,
        info.pixel_representation,
    )
}
