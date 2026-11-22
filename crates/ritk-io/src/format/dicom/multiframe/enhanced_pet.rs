//! Quantitative Enhanced PET mapping on stored voxel values before display rescale.

use aequitas::systems::si::{
    quantities::{Activity, ActivityConcentration, Mass, MassDensity, Time},
    units::{BecquerelPerMilliliter, GramPerCubicCentimeter, Megabecquerel, Second},
};
use ritk_dicom::{
    dataset::{Dataset, DicomFile},
    DicomTag as Tag,
};

use super::pet_dimensions::is_enhanced_pet;

use super::super::pet_datetime::Timestamp;
use std::sync::Arc;

/// Quantitative body-weight-normalized uptake aligned with the decoded volume.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyWeightSuvVolume {
    samples: Arc<MassDensity<Vec<f32>>>,
}

impl BodyWeightSuvVolume {
    pub(super) fn new(samples: Vec<f32>) -> Option<Self> {
        samples.iter().any(|value| value.is_finite()).then(|| Self {
            samples: Arc::new(MassDensity::from_base(samples)),
        })
    }

    /// Read the quantitative value at a flat, row-major voxel index.
    #[must_use]
    pub fn at(&self, index: usize) -> Option<MassDensity> {
        let value = f64::from(*self.samples.as_base().get(index)?);
        value.is_finite().then_some(MassDensity::from_base(value))
    }
}

const SHARED: Tag = Tag::new(0x5200, 0x9229);
const PER_FRAME: Tag = Tag::new(0x5200, 0x9230);
const REAL_WORLD: Tag = Tag::new(0x0040, 0x9096);
const AGENT: Tag = Tag::new(0x0018, 0x9729);

struct Mapping {
    lower: f64,
    upper: f64,
    function: Function,
}

enum Function {
    Linear { slope: f64, intercept: f64 },
    Table(Vec<f64>),
}

impl Mapping {
    fn parse(item: &Dataset) -> Option<Self> {
        let lower = number(item, Tag::new(0x0040, 0x9214))
            .or_else(|| number(item, Tag::new(0x0040, 0x9216)))?;
        let upper = number(item, Tag::new(0x0040, 0x9213))
            .or_else(|| number(item, Tag::new(0x0040, 0x9211)))?;
        if lower > upper {
            return None;
        }
        let function = if let Ok(table) = item.element(Tag::new(0x0040, 0x9212)) {
            let values = table.to_numbers::<f64>().ok()?;
            // Integer Pixel Data in Enhanced PET addresses the table by an exact index.
            if lower.fract() != 0.0
                || upper.fract() != 0.0
                || upper - lower + 1.0 != values.len() as f64
                || values.iter().any(|value| !value.is_finite())
            {
                return None;
            }
            Function::Table(values)
        } else {
            Function::Linear {
                slope: number(item, Tag::new(0x0040, 0x9225))?,
                intercept: number(item, Tag::new(0x0040, 0x9224))?,
            }
        };
        Some(Self {
            lower,
            upper,
            function,
        })
    }

    fn evaluate(&self, stored: f64) -> Option<f64> {
        if stored < self.lower || stored > self.upper {
            return None;
        }
        match &self.function {
            Function::Linear { slope, intercept } => Some(slope.mul_add(stored, *intercept)),
            Function::Table(values) => values.get((stored - self.lower) as usize).copied(),
        }
    }
}

pub(super) struct FrameMapping {
    suv: Vec<Mapping>,
    concentration: Vec<Mapping>,
    normalization: Option<Normalization>,
}

struct Normalization {
    mass: Mass,
    activity_at_reference: Activity,
}

impl FrameMapping {
    pub(super) fn read(obj: &DicomFile, index: usize) -> Option<Self> {
        if !is_enhanced_pet(&obj.meta().media_storage_sop_class_uid) {
            return None;
        }
        let shared = single_item(obj, SHARED);
        let frame = sequence(obj, PER_FRAME).and_then(|items| items.get(index));
        let items = group_items(frame, shared, REAL_WORLD)?;
        let mut suv = Vec::new();
        let mut concentration = Vec::new();
        for item in items {
            let Some(units) = single_item(item, Tag::new(0x0040, 0x08ea)) else {
                continue;
            };
            if units
                .element(Tag::new(0x0008, 0x0102))
                .ok()
                .and_then(|code| code.to_str().ok())
                .is_none_or(|code| code.trim() != "UCUM")
            {
                continue;
            }
            let Ok(code) = units.element(Tag::new(0x0008, 0x0100)) else {
                continue;
            };
            let Ok(code) = code.to_str() else { continue };
            let destination = match code.trim() {
                "{SUVbw}g/ml" => &mut suv,
                "Bq/ml" => &mut concentration,
                _ => continue,
            };
            if let Some(mapping) = Mapping::parse(item) {
                destination.push(mapping);
            }
        }
        let normalization = if concentration.is_empty() {
            None
        } else {
            normalization(obj, frame, shared)
        };
        (!suv.is_empty() || normalization.is_some()).then_some(Self {
            suv,
            concentration,
            normalization,
        })
    }

    pub(super) fn sample(&self, stored: f32) -> Option<MassDensity> {
        let stored = f64::from(stored);
        if let Some(value) = unique_value(&self.suv, stored) {
            return Some(MassDensity::from_unit::<GramPerCubicCentimeter>(value));
        }
        let normalization = self.normalization.as_ref()?;
        let raw = unique_value(&self.concentration, stored)?;
        let concentration = ActivityConcentration::from_unit::<BecquerelPerMilliliter>(raw);
        let density = concentration * normalization.mass / normalization.activity_at_reference;
        Some(density)
    }
}

fn unique_value(mappings: &[Mapping], stored: f64) -> Option<f64> {
    let mut values = mappings
        .iter()
        .filter_map(|mapping| mapping.evaluate(stored));
    let first = values.next()?;
    (first.is_finite() && values.all(|other| first == other)).then_some(first)
}

fn normalization(
    obj: &Dataset,
    frame: Option<&Dataset>,
    shared: Option<&Dataset>,
) -> Option<Normalization> {
    let mass = Mass::from_base(positive(obj, Tag::new(0x0010, 0x1030))?);
    let isotopes = sequence(obj, Tag::new(0x0054, 0x0016))?;
    let usage = group_items(frame, shared, Tag::new(0x0018, 0x9737));
    let isotope = match usage {
        Some([item]) => {
            let agent = item.element(AGENT).ok()?.to_num::<u32>().ok()?;
            let mut matching = isotopes.iter().filter(|item| {
                item.element(AGENT)
                    .ok()
                    .and_then(|e| e.to_num::<u32>().ok())
                    == Some(agent)
            });
            let result = matching.next()?;
            if matching.next().is_some() {
                return None;
            }
            result
        }
        None => {
            let [isotope] = isotopes else { return None };
            isotope
        }
        _ => return None,
    };
    let administered =
        Activity::from_unit::<Megabecquerel>(positive(isotope, Tag::new(0x0018, 0x1074))?);
    let half_life = Time::from_unit::<Second>(positive(isotope, Tag::new(0x0018, 0x1075))?);
    let injection = datetime(isotope, Tag::new(0x0018, 0x1078))?;
    let corrected = obj.element(Tag::new(0x0018, 0x9758)).ok()?.to_str().ok()?;
    let reference = match corrected.trim() {
        "YES" => datetime(obj, Tag::new(0x0018, 0x9701))?,
        "NO" => {
            let [content] = group_items(frame, shared, Tag::new(0x0020, 0x9111))? else {
                return None;
            };
            datetime(content, Tag::new(0x0018, 0x9151))?
        }
        _ => return None,
    };
    let elapsed = Time::from_unit::<Second>(reference.seconds_since(injection)?);
    let half_lives = (elapsed / half_life).into_base();
    let activity_at_reference = administered * (-half_lives).exp2();
    let activity = *activity_at_reference.as_base();
    (activity.is_finite() && activity > 0.0).then_some(Normalization {
        mass,
        activity_at_reference,
    })
}

fn group_items<'a>(
    frame: Option<&'a Dataset>,
    shared: Option<&'a Dataset>,
    tag: Tag,
) -> Option<&'a [Dataset]> {
    if let Some(frame) = frame
        && let Ok(element) = frame.element(tag)
    {
        return element.value().items();
    }
    sequence(shared?, tag)
}

fn single_item(obj: &Dataset, tag: Tag) -> Option<&Dataset> {
    let [item] = sequence(obj, tag)? else {
        return None;
    };
    Some(item)
}

fn sequence(obj: &Dataset, tag: Tag) -> Option<&[Dataset]> {
    obj.element(tag).ok()?.value().items()
}

fn datetime(obj: &Dataset, tag: Tag) -> Option<Timestamp> {
    Timestamp::parse(obj.element(tag).ok()?.to_str().ok()?.trim())
}

fn number(obj: &Dataset, tag: Tag) -> Option<f64> {
    let value = obj.element(tag).ok()?.to_num::<f64>().ok()?;
    value.is_finite().then_some(value)
}

fn positive(obj: &Dataset, tag: Tag) -> Option<f64> {
    number(obj, tag).filter(|value| *value > 0.0)
}
