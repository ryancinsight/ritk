//! Units of DICOM scalar values after the modality transform.

use super::sop_class::{classify_sop_class, SopClassKind};
use aequitas::systems::si::{
    quantities::{ActivityConcentration, Dimensionless, MassDensity},
    units::{BecquerelPerMilliliter, GramPerCubicCentimeter},
};
use arrayvec::ArrayString;
use ritk_dicom::dataset::{Dataset, DicomTag};

/// Unit of a decoded scalar DICOM image after rescale.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DicomValueUnit {
    /// No quantitative unit is declared or established by the storage class.
    #[default]
    Unspecified,
    /// Hounsfield units, a dimensionless CT attenuation scale with water at 0.
    Hounsfield,
    /// Radionuclide activity concentration in Bq/mL at the DICOM boundary.
    ActivityConcentration,
    /// Mass concentration in mg/mL, numerically equal to kg/m³.
    MassConcentration,
    /// Standardized uptake value in grams per millilitre, with its normalization.
    StandardizedUptake(SuvNormalization),
    /// An explicitly declared DICOM unit without a supported physical mapping.
    Declared(ArrayString<16>),
}

/// Normalization of a standardized uptake value encoded as mass per volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuvNormalization {
    /// Normalize by body weight.
    BodyWeight,
    /// Another DICOM SUV Type, preserved without relabelling.
    Other(ArrayString<16>),
}

impl DicomValueUnit {
    /// Interpret one value in the decoded image's DICOM units.
    #[must_use]
    pub fn sample(self, value: f64) -> DicomModalityValue {
        match self {
            Self::Hounsfield => DicomModalityValue::Hounsfield(Dimensionless::from_base(value)),
            Self::ActivityConcentration => {
                DicomModalityValue::ActivityConcentration(ActivityConcentration::from_unit::<
                    BecquerelPerMilliliter,
                >(value))
            }
            Self::MassConcentration => {
                DicomModalityValue::MassConcentration(MassDensity::from_base(value))
            }
            Self::StandardizedUptake(normalization) => DicomModalityValue::StandardizedUptake {
                normalization,
                value: MassDensity::from_unit::<GramPerCubicCentimeter>(value),
            },
            Self::Unspecified | Self::Declared(_) => {
                DicomModalityValue::Unmapped { unit: self, value }
            }
        }
    }
}

/// A decoded scalar, rescale offset, or window value in its modality units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DicomModalityValue {
    /// The Hounsfield scale is dimensionless and has a CT-specific meaning.
    Hounsfield(Dimensionless),
    /// Radionuclide concentration.
    ActivityConcentration(ActivityConcentration),
    /// Mass per unit volume.
    MassConcentration(MassDensity),
    /// Already normalized PET uptake.
    StandardizedUptake {
        /// Normalization declared by the source.
        normalization: SuvNormalization,
        /// Standardized uptake in mass per volume.
        value: MassDensity,
    },
    /// Scalar with its original unit declaration when available.
    Unmapped {
        /// Unit declaration.
        unit: DicomValueUnit,
        /// Scalar in DICOM units.
        value: f64,
    },
}

impl DicomModalityValue {
    /// Retrieve the scalar in the decoded DICOM image's units.
    #[must_use]
    pub fn dicom_value(self) -> f64 {
        match self {
            Self::Hounsfield(value) => value.into_base(),
            Self::ActivityConcentration(value) => value.in_unit::<BecquerelPerMilliliter>(),
            Self::MassConcentration(value) => value.into_base(),
            Self::StandardizedUptake { value, .. } => value.in_unit::<GramPerCubicCentimeter>(),
            Self::Unmapped { value, .. } => value,
        }
    }
}

fn read_text(obj: &Dataset, group: u16, element: u16) -> Option<std::borrow::Cow<'_, str>> {
    obj.element(DicomTag::new(group, element))
        .ok()?
        .to_str()
        .ok()
}

pub(super) fn declared_rescale_unit(obj: &Dataset) -> Option<DicomValueUnit> {
    let text = read_text(obj, 0x0028, 0x1054)?;
    let code = text.trim();
    match code {
        "" => None,
        "HU" => Some(DicomValueUnit::Hounsfield),
        "MGML" => Some(DicomValueUnit::MassConcentration),
        code => Some(
            ArrayString::from(code).map_or(DicomValueUnit::Unspecified, DicomValueUnit::Declared),
        ),
    }
}

/// Resolve classic PET Units and the CT Image Module's implicit HU condition.
/// An explicit modality transform unit takes precedence over default CT HU.
pub(super) fn resolve_value_unit(obj: &Dataset, class: Option<&str>) -> DicomValueUnit {
    let kind = class.map(classify_sop_class);
    if kind == Some(SopClassKind::PetImageStorage)
        && let Some(units) = read_text(obj, 0x0054, 0x1001)
    {
        return match units.trim() {
            "BQML" => DicomValueUnit::ActivityConcentration,
            // PS3.3 C.8.9.1 supplies BW when GML carries no SUV Type.
            "GML" => match read_text(obj, 0x0054, 0x1006).as_deref().map(str::trim) {
                None | Some("" | "BW") => {
                    DicomValueUnit::StandardizedUptake(SuvNormalization::BodyWeight)
                }
                Some(code) => ArrayString::from(code).map_or(DicomValueUnit::Unspecified, |code| {
                    DicomValueUnit::StandardizedUptake(SuvNormalization::Other(code))
                }),
            },
            "" => DicomValueUnit::Unspecified,
            code => ArrayString::from(code)
                .map_or(DicomValueUnit::Unspecified, DicomValueUnit::Declared),
        };
    }
    if let Some(unit) = declared_rescale_unit(obj) {
        return unit;
    }
    if kind == Some(SopClassKind::CtImageStorage)
        && read_text(obj, 0x0008, 0x0008).is_some_and(|image_type| {
            let mut values = image_type.split('\\').map(str::trim);
            values.next() == Some("ORIGINAL")
                && values.nth(1).is_some_and(|value| value != "LOCALIZER")
        })
        && read_text(obj, 0x0018, 0x9361).is_none_or(|value| value.trim() == "NO")
    {
        return DicomValueUnit::Hounsfield;
    }
    DicomValueUnit::Unspecified
}
