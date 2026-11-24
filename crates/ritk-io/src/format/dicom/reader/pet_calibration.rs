//! Admit a common SUV normalization only when every classic PET image supports it.

use aequitas::systems::si::quantities::{Activity, ActivityConcentration, Mass, MassDensity};
use ritk_dicom::dataset::{Dataset, DicomTag as Tag};

use super::super::pet_datetime::Timestamp;

#[derive(Debug, Clone, Copy, PartialEq)]
enum DecayReference {
    Administration,
    AcquisitionStart,
    Frame,
}

/// A source-qualified normalization shared by the classic PET image series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassicPetCalibration {
    mass: Mass,
    administered: Activity,
    reference: DecayReference,
    activity_at_reference: Activity,
}

impl ClassicPetCalibration {
    /// Normalize activity concentration against activity at its own reference time.
    #[must_use]
    pub fn suvbw(self, concentration: ActivityConcentration) -> MassDensity {
        concentration / (self.activity_at_reference / self.mass)
    }

    pub(super) fn from_image(image: &Dataset) -> Option<Self> {
        let reference = match text(image, Tag::new(0x0054, 0x1102))?.trim_matches(' ') {
            "ADMIN" => DecayReference::Administration,
            "START" => DecayReference::AcquisitionStart,
            "NONE" => DecayReference::Frame,
            _ => return None,
        };
        let mass = Mass::from_base(positive(image, Tag::new(0x0010, 0x1030))?);
        let [isotope] = image
            .element(Tag::new(0x0054, 0x0016))
            .ok()?
            .value()
            .items()?
        else {
            return None;
        };
        // Classic PET (unlike Enhanced PET) encodes this dose in Bq.
        let administered = Activity::from_base(positive(isotope, Tag::new(0x0018, 0x1074))?);
        let activity_at_reference = match reference {
            DecayReference::Administration => administered,
            _ => {
                let half_life = positive(isotope, Tag::new(0x0018, 0x1075))?;
                let elapsed = elapsed_seconds(image, isotope, reference, half_life)?;
                administered * (-elapsed / half_life).exp2()
            }
        };
        let activity = *activity_at_reference.as_base();
        (activity.is_finite() && activity > 0.0).then_some(Self {
            mass,
            administered,
            reference,
            activity_at_reference,
        })
    }
}

#[derive(Default)]
pub(super) enum SeriesPetCalibration {
    #[default]
    Empty,
    Uniform(ClassicPetCalibration),
    Unavailable,
}

impl SeriesPetCalibration {
    pub(super) fn record(&mut self, calibration: Option<ClassicPetCalibration>) {
        *self = match (&*self, calibration) {
            (Self::Empty, Some(calibration)) => Self::Uniform(calibration),
            (Self::Uniform(previous), Some(calibration)) if *previous == calibration => {
                Self::Uniform(calibration)
            }
            _ => Self::Unavailable,
        };
    }

    pub(super) fn common(self) -> Option<ClassicPetCalibration> {
        match self {
            Self::Uniform(calibration) => Some(calibration),
            _ => None,
        }
    }
}

fn elapsed_seconds(
    image: &Dataset,
    isotope: &Dataset,
    reference: DecayReference,
    half_life: f64,
) -> Option<f64> {
    let offset = optional_text(image, Tag::new(0x0008, 0x0201))?;
    let offset = offset.trim_matches(' ');
    let injection_dt = text(isotope, Tag::new(0x0018, 0x1078))?;
    let injection = Timestamp::parse_with_default_offset(&injection_dt, offset)?;
    let injection_tm = optional_text(isotope, Tag::new(0x0018, 0x1072))?;
    if !injection_tm.is_empty() && !Timestamp::matches_clock(&injection_dt, &injection_tm) {
        return None;
    }
    let series = dated_time(
        image,
        Tag::new(0x0008, 0x0021),
        Tag::new(0x0008, 0x0031),
        offset,
    )?;
    let acquisition = dated_time(
        image,
        Tag::new(0x0008, 0x0022),
        Tag::new(0x0008, 0x0032),
        offset,
    )?;
    let acquisition_s = acquisition.seconds_since(injection)?;
    let frame_offset_s = number(image, Tag::new(0x0054, 0x1300))? / 1_000.0;
    let series_s = series.signed_seconds_since(injection)?;
    let frame_s = series_s + frame_offset_s;
    // The classic reader has one SUV scale for the volume. Use only explicitly
    // located references and let the series accumulator reject unequal scales.
    let elapsed = match reference {
        DecayReference::AcquisitionStart => {
            // PS3.3 C.8.9.1.1.5. Undo the image's Decay Factor to locate the
            // time to which the scanner corrected the pixel activity.
            let factor = positive(image, Tag::new(0x0054, 0x1321))?;
            // Subtract offsets before adding the series origin. Equal offsets
            // cancel exactly without losing precision at the acquisition boundary.
            let start = series_s + (frame_offset_s - half_life * factor.log2());
            (factor >= 1.0 && start <= acquisition_s && acquisition_s <= frame_s)
                .then_some(start)?
        }
        DecayReference::Frame => {
            let duration_ms = text(image, Tag::new(0x0018, 0x1242))?
                .trim_matches(' ')
                .parse::<i32>()
                .ok()?;
            let duration_s = f64::from(duration_ms) / 1_000.0;
            (duration_ms > 0 && acquisition_s <= frame_s && frame_s <= acquisition_s + duration_s)
                .then_some(frame_s)?
        }
        DecayReference::Administration => return None,
    };
    (elapsed.is_finite() && elapsed >= 0.0).then_some(elapsed)
}

fn dated_time(image: &Dataset, date: Tag, time: Tag, offset: &str) -> Option<Timestamp> {
    let date = text(image, date)?;
    let time = text(image, time)?;
    if date.len() != 8 || time.split('.').next()?.len() != 6 {
        return None;
    }
    Timestamp::parse_with_default_offset(&format!("{date}{time}"), offset)
}

fn number(dataset: &Dataset, tag: Tag) -> Option<f64> {
    // These attributes have VM 1. Parse the whole value so a valid first
    // component cannot hide an additional contradictory component.
    let element = dataset.element(tag).ok()?;
    let number = element
        .to_str()
        .ok()?
        .trim_matches(' ')
        .parse::<f64>()
        .ok()?;
    number.is_finite().then_some(number)
}

fn positive(dataset: &Dataset, tag: Tag) -> Option<f64> {
    number(dataset, tag).filter(|number| *number > 0.0)
}

fn text(dataset: &Dataset, tag: Tag) -> Option<String> {
    let element = dataset.element(tag).ok()?;
    Some(element.to_str().ok()?.trim_end_matches(' ').to_owned())
}

fn optional_text(dataset: &Dataset, tag: Tag) -> Option<String> {
    match dataset.element(tag) {
        Ok(element) => Some(element.to_str().ok()?.trim_end_matches(' ').to_owned()),
        Err(_) => Some(String::new()),
    }
}
