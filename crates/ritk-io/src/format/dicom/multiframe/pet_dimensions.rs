//! Group Enhanced PET frames using DICOM Dimension Index Values.

use std::collections::BTreeMap;
use std::path::Path;

use aequitas::systems::si::{
    quantities::{Dimensionless, Time},
    units::Millisecond,
};
use anyhow::{bail, ensure, Context, Result};
use dicom::core::{
    value::{PrimitiveValue, Value},
    Tag,
};
use dicom::dictionary_std::uids::ENHANCED_PET_IMAGE_STORAGE;
use dicom::object::InMemDicomObject;

use super::temporal::reject_temporal_organization;
use super::types::PerFrameInfo;
use crate::format::dicom::reader::{dot, slice_normal_from_iop};

const DIMENSIONS: Tag = Tag(0x0020, 0x9222);
const POINTER: Tag = Tag(0x0020, 0x9165);
const GROUP: Tag = Tag(0x0020, 0x9167);
const PER_FRAME: Tag = Tag(0x5200, 0x9230);
const SHARED: Tag = Tag(0x5200, 0x9229);
const CONTENT: Tag = Tag(0x0020, 0x9111);
const INDICES: Tag = Tag(0x0020, 0x9157);
const IN_STACK: Tag = Tag(0x0020, 0x9057);
const POSITION: Tag = Tag(0x0020, 0x0032);
const TEMPORAL: Tag = Tag(0x0020, 0x9128);

/// A nominal coordinate for a DICOM dimension, when the input supplies it.
#[derive(Debug, Clone, PartialEq)]
pub enum EnhancedPetCoordinate {
    /// Temporal position relative to the organization in the object.
    TemporalPosition(u32),
    /// Nominal cardiac or respiratory phase, expressed as a ratio of a cycle.
    CyclePhase(Dimensionless),
    /// Respiratory amplitude relative to nominal maximum inspiration.
    RespiratoryAmplitude(Dimensionless),
    /// Trigger delay in the cardiac or respiratory cycle.
    TriggerDelay(Time),
    /// Stack identity or a categorical gate label.
    Label(String),
}

/// One nonspatial coordinate in an Enhanced PET volume key.
#[derive(Debug, Clone, PartialEq)]
pub struct EnhancedPetDimension {
    /// The DICOM dimension attribute.
    pub attribute: Tag,
    /// The positive ordinal in Dimension Index Values.
    pub ordinal: u32,
    /// Nominal coordinate when represented in the referenced functional group.
    pub coordinate: Option<EnhancedPetCoordinate>,
}

pub(super) struct PetFrameGroup {
    pub dimensions: Vec<EnhancedPetDimension>,
    pub frames: Vec<usize>,
}

pub(super) fn is_enhanced_pet(uid: &str) -> bool {
    uid.trim_end_matches([' ', '\0']) == ENHANCED_PET_IMAGE_STORAGE
}

fn items(obj: &InMemDicomObject, tag: Tag) -> Option<&[InMemDicomObject]> {
    obj.element(tag).ok()?.items()
}

fn one_item(obj: &InMemDicomObject, tag: Tag) -> Option<&InMemDicomObject> {
    let [item] = items(obj, tag)? else {
        return None;
    };
    Some(item)
}

fn single_tag(obj: &InMemDicomObject, key: Tag) -> Result<Tag> {
    let Value::Primitive(PrimitiveValue::Tags(tags)) = obj.element(key)?.value() else {
        bail!("Enhanced PET dimension pointer {key:?} must contain an attribute tag");
    };
    let [tag] = tags.as_slice() else {
        bail!("Enhanced PET dimension pointer {key:?} must contain exactly one attribute tag");
    };
    Ok(*tag)
}

#[derive(Clone, Copy)]
struct Axis {
    attribute: Tag,
    group: Option<Tag>,
}

impl Axis {
    fn coordinate(
        self,
        frame: &InMemDicomObject,
        shared: Option<&InMemDicomObject>,
    ) -> Option<EnhancedPetCoordinate> {
        let attr = match self.group {
            Some(group) => one_item(frame, group)
                .or_else(|| shared.and_then(|obj| one_item(obj, group)))?
                .element(self.attribute)
                .ok()?,
            None => frame.element(self.attribute).ok()?,
        };
        match self.attribute {
            TEMPORAL => attr
                .to_int::<u32>()
                .ok()
                .filter(|v| *v > 0)
                .map(EnhancedPetCoordinate::TemporalPosition),
            Tag(0x0020, 0x9241 | 0x9245) => {
                let percent = attr.to_float64().ok()?;
                (0.0..=100.0).contains(&percent).then(|| {
                    EnhancedPetCoordinate::CyclePhase(Dimensionless::from_base(percent / 100.0))
                })
            }
            Tag(0x0020, 0x9246 | 0x9248) => {
                let percent = attr.to_float64().ok()?;
                percent.is_finite().then(|| {
                    EnhancedPetCoordinate::RespiratoryAmplitude(Dimensionless::from_base(
                        percent / 100.0,
                    ))
                })
            }
            Tag(0x0020, 0x9153 | 0x9154 | 0x9155 | 0x9252 | 0x9255 | 0x9257) => {
                let delay = attr.to_float64().ok()?;
                delay.is_finite().then(|| {
                    EnhancedPetCoordinate::TriggerDelay(Time::from_unit::<Millisecond>(delay))
                })
            }
            _ => attr
                .to_str()
                .ok()
                .map(|v| v.trim().to_owned())
                .filter(|v| !v.is_empty())
                .map(EnhancedPetCoordinate::Label),
        }
    }
}

/// Group by the declared nonspatial ordinals, then sort the planes within each
/// group by patient position. Pixel decode must use the source frame indices.
pub(super) fn frame_groups(
    obj: &InMemDicomObject,
    path: &Path,
    n_frames: usize,
    geometry: &[PerFrameInfo],
    global_orientation: Option<[f64; 6]>,
) -> Result<Option<Vec<PetFrameGroup>>> {
    let dynamic = obj
        .element(Tag(0x0008, 0x0008))
        .ok()
        .and_then(|element| element.to_str().ok())
        .is_some_and(|value| {
            value
                .split('\\')
                .nth(2)
                .is_some_and(|v| v.trim() == "DYNAMIC")
        });
    let gated = items(obj, PER_FRAME)
        .into_iter()
        .flatten()
        .chain(items(obj, SHARED).into_iter().flatten())
        .any(|group| {
            group.element(Tag(0x0018, 0x9118)).is_ok() || group.element(Tag(0x0020, 0x9253)).is_ok()
        });
    if obj.element(DIMENSIONS).is_err() {
        reject_temporal_organization(path, obj)?;
        ensure!(
            !dynamic && !gated,
            "Dynamic or gated Enhanced PET requires DimensionIndexSequence in {path:?}"
        );
        return Ok(None);
    }
    let axes = items(obj, DIMENSIONS)
        .context("Enhanced PET DimensionIndexSequence must be a sequence")?
        .iter()
        .map(|item| {
            Ok(Axis {
                attribute: single_tag(item, POINTER)?,
                group: item
                    .element(GROUP)
                    .ok()
                    .map(|_| single_tag(item, GROUP))
                    .transpose()?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        !axes.is_empty(),
        "Enhanced PET DimensionIndexSequence is empty"
    );
    ensure!(
        !dynamic || axes.iter().any(|axis| axis.attribute == TEMPORAL),
        "Dynamic Enhanced PET must index Temporal Position"
    );
    ensure!(
        !gated
            || axes.iter().any(|axis| !matches!(
                axis.attribute,
                IN_STACK | POSITION | TEMPORAL | Tag(0x0020, 0x9056)
            )),
        "Gated Enhanced PET has no timing or gate dimension"
    );
    let spatial: Vec<_> = axes
        .iter()
        .enumerate()
        .filter_map(|(i, axis)| matches!(axis.attribute, IN_STACK | POSITION).then_some(i))
        .collect();
    let groups = items(obj, PER_FRAME)
        .context("Enhanced PET dimensions require per-frame functional groups")?;
    ensure!(
        groups.len() == n_frames,
        "Enhanced PET per-frame group count must equal NumberOfFrames"
    );
    ensure!(
        !spatial.is_empty(),
        "Enhanced PET dimensions require an in-stack or patient-position axis"
    );
    let shared = one_item(obj, SHARED);
    let mut partitions: BTreeMap<Vec<u32>, (Vec<EnhancedPetDimension>, BTreeMap<Vec<u32>, usize>)> =
        BTreeMap::new();
    for (index, frame) in groups.iter().enumerate() {
        let content =
            one_item(frame, CONTENT).context("Enhanced PET frame is missing Frame Content")?;
        let ordinals = content.element(INDICES)?.to_multi_int::<u32>()?;
        ensure!(
            ordinals.len() == axes.len() && !ordinals.contains(&0),
            "Enhanced PET frame {index} has invalid Dimension Index Values"
        );
        let key: Vec<_> = ordinals
            .iter()
            .enumerate()
            .filter_map(|(i, &ordinal)| (!spatial.contains(&i)).then_some(ordinal))
            .collect();
        let plane: Vec<_> = spatial.iter().map(|&i| ordinals[i]).collect();
        let (_, planes) = partitions.entry(key).or_insert_with(|| {
            let dimensions = axes
                .iter()
                .enumerate()
                .filter(|(i, _)| !spatial.contains(i))
                .map(|(i, axis)| EnhancedPetDimension {
                    attribute: axis.attribute,
                    ordinal: ordinals[i],
                    coordinate: axis.coordinate(frame, shared),
                })
                .collect();
            (dimensions, BTreeMap::new())
        });
        ensure!(
            planes.insert(plane, index).is_none(),
            "Enhanced PET volume contains duplicate spatial indices"
        );
    }
    let mut results = Vec::with_capacity(partitions.len());
    for (dimensions, planes) in partitions.into_values() {
        let mut frames: Vec<_> = planes.into_values().collect();
        if frames.len() > 1 {
            let first = &geometry[frames[0]];
            let iop = first
                .image_orientation
                .or(global_orientation)
                .context("Enhanced PET volume geometry lacks orientation")?;
            let normal = slice_normal_from_iop(iop).context("Invalid Enhanced PET orientation")?;
            let positions = frames
                .iter()
                .map(|&index| {
                    let frame = &geometry[index];
                    ensure!(
                        frame.image_orientation.or(global_orientation) == Some(iop)
                            && frame.pixel_spacing == first.pixel_spacing,
                        "Enhanced PET geometry changes within one volume"
                    );
                    let position = dot(
                        frame
                            .image_position
                            .context("Enhanced PET slice lacks patient position")?,
                        normal,
                    );
                    ensure!(
                        position.is_finite(),
                        "Enhanced PET slice has a nonfinite position"
                    );
                    Ok((position, index))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut positions = positions;
            positions.sort_by(|a, b| a.0.total_cmp(&b.0));
            ensure!(
                positions.windows(2).all(|pair| pair[0].0 < pair[1].0),
                "Enhanced PET volume contains duplicate patient slice positions"
            );
            frames = positions.into_iter().map(|(_, index)| index).collect();
        }
        results.push(PetFrameGroup { dimensions, frames });
    }
    Ok(Some(results))
}
