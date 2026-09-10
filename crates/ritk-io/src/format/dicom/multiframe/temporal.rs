//! Temporal-dimension validation for DICOM multi-frame objects.

use anyhow::{bail, Context, Result};
use dicom::core::{
    value::{PrimitiveValue, Value},
    Tag,
};
use dicom::object::InMemDicomObject;
use std::path::Path;

/// Reject a multi-frame object whose frame dimension is temporal rather than
/// spatial. The viewer's volume carrier has one spatial frame axis; accepting a
/// temporal organization here would present time points as anatomy slices.
pub(super) fn reject_temporal_organization(path: &Path, obj: &InMemDicomObject) -> Result<()> {
    if let Ok(element) = obj.element(Tag(0x0020, 0x0105)) {
        let temporal_positions = element
            .to_int::<u32>()
            .with_context(|| format!("NumberOfTemporalPositions is invalid in {:?}", path))?;
        if temporal_positions == 0 {
            bail!(
                "DICOM multiframe: NumberOfTemporalPositions must be greater than zero in {:?}",
                path
            );
        }
        if temporal_positions > 1 {
            bail!(
                "DICOM multiframe: temporal organization with {} temporal positions is not supported in {:?}; open one temporal position as a spatial volume",
                temporal_positions,
                path
            );
        }
    }

    if dimension_index_contains_temporal(obj, path)? {
        bail!(
            "DICOM multiframe: DimensionIndexSequence identifies TemporalPositionIndex; temporal frame organization is not supported in {:?}",
            path
        );
    }

    let temporal_indices = temporal_position_indices(obj, path)?;
    if temporal_indices.contains(&0) {
        bail!(
            "DICOM multiframe: TemporalPositionIndex must be greater than zero in {:?}",
            path
        );
    }
    if temporal_indices.iter().any(|index| *index > 1)
        || temporal_indices.windows(2).any(|pair| pair[0] != pair[1])
    {
        bail!(
            "DICOM multiframe: per-frame temporal positions are not supported in {:?}; frames must describe one spatial position axis",
            path
        );
    }
    Ok(())
}

/// Return whether the enhanced dimension index sequence names the temporal
/// position index tag `(0020,9128)`.
fn dimension_index_contains_temporal(obj: &InMemDicomObject, path: &Path) -> Result<bool> {
    let Some(element) = obj.element(Tag(0x0020, 0x9222)).ok() else {
        return Ok(false);
    };
    let Value::Sequence(sequence) = element.value() else {
        bail!(
            "DICOM multiframe: DimensionIndexSequence is not a sequence in {:?}",
            path
        );
    };
    for (item_index, item) in sequence.items().iter().enumerate() {
        let pointer = item.element(Tag(0x0020, 0x9165)).with_context(|| {
            format!(
                "DICOM multiframe: DimensionIndexPointer is missing from DimensionIndexSequence item {} in {:?}",
                item_index, path
            )
        })?;
        let Value::Primitive(PrimitiveValue::Tags(tags)) = pointer.value() else {
            bail!(
                "DICOM multiframe: DimensionIndexPointer has an invalid value in item {} in {:?}",
                item_index,
                path
            );
        };
        if tags.contains(&Tag(0x0020, 0x9128)) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Collect temporal position indices from shared and per-frame functional
/// groups. A single value of one is spatially compatible; any variation is not.
fn temporal_position_indices(obj: &InMemDicomObject, path: &Path) -> Result<Vec<u32>> {
    const FRAME_CONTENT_SEQUENCE: Tag = Tag(0x0020, 0x9111);
    const TEMPORAL_POSITION_INDEX: Tag = Tag(0x0020, 0x9128);
    let mut indices = Vec::new();
    for sequence_tag in [Tag(0x5200, 0x9229), Tag(0x5200, 0x9230)] {
        let Some(sequence_element) = obj.element(sequence_tag).ok() else {
            continue;
        };
        let Value::Sequence(sequence) = sequence_element.value() else {
            bail!(
                "DICOM multiframe: functional group tag {:?} is not a sequence in {:?}",
                sequence_tag,
                path
            );
        };
        for (group_index, group) in sequence.items().iter().enumerate() {
            let Some(frame_content_element) = group.element(FRAME_CONTENT_SEQUENCE).ok() else {
                continue;
            };
            let Value::Sequence(frame_content) = frame_content_element.value() else {
                bail!(
                    "DICOM multiframe: FrameContentSequence is not a sequence in group {} in {:?}",
                    group_index,
                    path
                );
            };
            let Some(frame_content_item) = frame_content.items().first() else {
                continue;
            };
            let Some(index_element) = frame_content_item.element(TEMPORAL_POSITION_INDEX).ok()
            else {
                continue;
            };
            indices.push(index_element.to_int::<u32>().with_context(|| {
                format!(
                    "DICOM multiframe: TemporalPositionIndex is invalid in group {} in {:?}",
                    group_index, path
                )
            })?);
        }
    }
    Ok(indices)
}
