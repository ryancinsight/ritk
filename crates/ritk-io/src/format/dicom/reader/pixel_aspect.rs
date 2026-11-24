//! Dimensionless pixel-aspect parsing from retained DICOM elements.

use crate::format::dicom::object_model::{DicomObjectNode, DicomValue};
use ritk_dicom::DicomTag;

const PIXEL_SPACING: DicomTag = DicomTag::new(0x0028, 0x0030);
const IMAGER_PIXEL_SPACING: DicomTag = DicomTag::new(0x0018, 0x1164);
const NOMINAL_SCANNED_PIXEL_SPACING: DicomTag = DicomTag::new(0x0018, 0x2010);
const PIXEL_ASPECT_RATIO: DicomTag = DicomTag::new(0x0028, 0x0034);

/// DICOM attribute from which a dimensionless pixel aspect was derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DicomPixelAspectSource {
    /// Patient-space Pixel Spacing (0028,0030).
    PatientPixelSpacing,
    /// Detector-plane Imager Pixel Spacing (0018,1164).
    ImagerPixelSpacing,
    /// Scanner-plane Nominal Scanned Pixel Spacing (0018,2010).
    NominalScannedPixelSpacing,
    /// Dimensionless Pixel Aspect Ratio (0028,0034).
    PixelAspectRatio,
}

impl DicomPixelAspectSource {
    /// Return the attribute tag represented by this source.
    #[must_use]
    pub const fn tag(self) -> DicomTag {
        match self {
            Self::PatientPixelSpacing => PIXEL_SPACING,
            Self::ImagerPixelSpacing => IMAGER_PIXEL_SPACING,
            Self::NominalScannedPixelSpacing => NOMINAL_SCANNED_PIXEL_SPACING,
            Self::PixelAspectRatio => PIXEL_ASPECT_RATIO,
        }
    }

    const fn expected_vr(self) -> &'static str {
        match self {
            Self::PixelAspectRatio => "IS",
            Self::PatientPixelSpacing
            | Self::ImagerPixelSpacing
            | Self::NominalScannedPixelSpacing => "DS",
        }
    }

    fn from_tag(tag: DicomTag) -> Option<Self> {
        match tag {
            PIXEL_SPACING => Some(Self::PatientPixelSpacing),
            IMAGER_PIXEL_SPACING => Some(Self::ImagerPixelSpacing),
            NOMINAL_SCANNED_PIXEL_SPACING => Some(Self::NominalScannedPixelSpacing),
            PIXEL_ASPECT_RATIO => Some(Self::PixelAspectRatio),
            _ => None,
        }
    }
}

/// A positive, finite row-to-column pixel-size ratio.
///
/// This value is dimensionless. Its source distinguishes patient, detector,
/// scanner, and explicit-aspect metadata; it does not promote detector or
/// scanner spacing to patient-space distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DicomPixelAspect {
    row_over_column: f64,
    source: DicomPixelAspectSource,
}

impl DicomPixelAspect {
    /// Return the vertical pixel extent divided by the horizontal extent.
    #[must_use]
    pub const fn row_over_column(self) -> f64 {
        self.row_over_column
    }

    /// Return the retained attribute that supplied the ratio.
    #[must_use]
    pub const fn source(self) -> DicomPixelAspectSource {
        self.source
    }
}

/// Why a retained DICOM element cannot supply a pixel aspect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DicomPixelAspectError {
    /// The element tag does not define pixel spacing or pixel aspect.
    #[error("{0:?} does not define a pixel aspect")]
    UnsupportedTag(DicomTag),
    /// The element uses a VR other than the one assigned by PS3.6.
    #[error("{0:?} does not use its assigned value representation")]
    UnexpectedValueRepresentation(DicomPixelAspectSource),
    /// The element value is not retained as text.
    #[error("{0:?} is not available as readable text")]
    UnreadableValue(DicomPixelAspectSource),
    /// The element does not encode two valid positive components and ratio.
    #[error("{0:?} does not encode a valid positive pixel aspect")]
    MalformedValue(DicomPixelAspectSource),
}

impl TryFrom<&DicomObjectNode> for DicomPixelAspect {
    type Error = DicomPixelAspectError;

    /// Parse a recognized spacing or aspect element without selecting between
    /// competing attributes or resolving frame-level geometry.
    ///
    /// DICOM PS3.3 defines spacing components in row-then-column order and
    /// Pixel Aspect Ratio in vertical-then-horizontal order. PS3.5 permits only
    /// ASCII SPACE padding for the DS and IS value representations.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_io::{
    ///     DicomObjectNode, DicomPixelAspect, DicomPixelAspectSource, DicomTag,
    /// };
    ///
    /// let node = DicomObjectNode::text(
    ///     DicomTag::new(0x0028, 0x0034),
    ///     "IS",
    ///     "6\\5",
    /// );
    /// let aspect = DicomPixelAspect::try_from(&node)?;
    /// assert_eq!(aspect.row_over_column(), 1.2);
    /// assert_eq!(aspect.source(), DicomPixelAspectSource::PixelAspectRatio);
    /// # Ok::<(), ritk_io::DicomPixelAspectError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the tag or value representation is not recognized,
    /// the value is not retained as text, or either component is invalid.
    fn try_from(node: &DicomObjectNode) -> Result<Self, Self::Error> {
        let source = DicomPixelAspectSource::from_tag(node.tag)
            .ok_or(DicomPixelAspectError::UnsupportedTag(node.tag))?;
        if node.vr.as_deref() != Some(source.expected_vr()) {
            return Err(DicomPixelAspectError::UnexpectedValueRepresentation(source));
        }
        let DicomValue::Text(text) = &node.value else {
            return Err(DicomPixelAspectError::UnreadableValue(source));
        };
        parse_aspect(text, source)
    }
}

fn parse_aspect(
    text: &str,
    source: DicomPixelAspectSource,
) -> Result<DicomPixelAspect, DicomPixelAspectError> {
    // A DICOM value field has even byte length. When present, its final SPACE
    // pad is not part of the last component's PS3.5 length limit.
    let text = if text.len().is_multiple_of(2) {
        text.strip_suffix(' ').unwrap_or(text)
    } else {
        text
    };
    let mut components = text.split('\\');
    let Some(row) = components.next() else {
        return Err(DicomPixelAspectError::MalformedValue(source));
    };
    let Some(column) = components.next() else {
        return Err(DicomPixelAspectError::MalformedValue(source));
    };
    if components.next().is_some() {
        return Err(DicomPixelAspectError::MalformedValue(source));
    }

    let row = parse_component(row, source)?;
    let column = parse_component(column, source)?;
    let row_over_column = row / column;
    if !row_over_column.is_finite() || row_over_column <= 0.0 {
        return Err(DicomPixelAspectError::MalformedValue(source));
    }
    Ok(DicomPixelAspect {
        row_over_column,
        source,
    })
}

fn parse_component(
    component: &str,
    source: DicomPixelAspectSource,
) -> Result<f64, DicomPixelAspectError> {
    let limit = if source == DicomPixelAspectSource::PixelAspectRatio {
        12
    } else {
        16
    };
    if component.len() > limit {
        return Err(DicomPixelAspectError::MalformedValue(source));
    }
    let component = component.trim_matches(' ');
    let value = if source == DicomPixelAspectSource::PixelAspectRatio {
        f64::from(
            component
                .parse::<i32>()
                .map_err(|_| DicomPixelAspectError::MalformedValue(source))?,
        )
    } else {
        component
            .parse::<f64>()
            .map_err(|_| DicomPixelAspectError::MalformedValue(source))?
    };
    if !value.is_finite() || value <= 0.0 {
        return Err(DicomPixelAspectError::MalformedValue(source));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(source: DicomPixelAspectSource, value: &str) -> DicomObjectNode {
        DicomObjectNode::text(source.tag(), source.expected_vr(), value)
    }

    #[test]
    fn recognized_sources_preserve_provenance_and_component_order() {
        for (source, value, expected) in [
            (
                DicomPixelAspectSource::PatientPixelSpacing,
                "0.3\\0.25",
                1.2,
            ),
            (DicomPixelAspectSource::ImagerPixelSpacing, "0.6\\0.3", 2.0),
            (
                DicomPixelAspectSource::NominalScannedPixelSpacing,
                "0.4\\0.8",
                0.5,
            ),
            (DicomPixelAspectSource::PixelAspectRatio, "6\\5", 1.2),
        ] {
            let aspect = DicomPixelAspect::try_from(&node(source, value))
                .expect("conformant pixel aspect parses");
            assert_eq!(aspect.source(), source);
            assert_eq!(aspect.row_over_column(), expected);
            assert_eq!(source.tag(), node(source, value).tag);
        }
    }

    #[test]
    fn positive_integer_components_preserve_their_ratio() {
        for row in 1..=32 {
            for column in 1..=32 {
                let value = format!("{row}\\{column}");
                let aspect = DicomPixelAspect::try_from(&node(
                    DicomPixelAspectSource::PixelAspectRatio,
                    &value,
                ))
                .expect("positive IS components parse");
                assert_eq!(aspect.row_over_column(), f64::from(row) / f64::from(column));
            }
        }
    }

    #[test]
    fn tag_value_representation_and_storage_failures_are_distinct() {
        let unsupported = DicomObjectNode::text(DicomTag::new(0x0010, 0x0010), "PN", "A");
        assert_eq!(
            DicomPixelAspect::try_from(&unsupported),
            Err(DicomPixelAspectError::UnsupportedTag(unsupported.tag))
        );

        let wrong_vr = DicomObjectNode::text(PIXEL_ASPECT_RATIO, "DS", "6\\5");
        assert_eq!(
            DicomPixelAspect::try_from(&wrong_vr),
            Err(DicomPixelAspectError::UnexpectedValueRepresentation(
                DicomPixelAspectSource::PixelAspectRatio,
            ))
        );

        let unreadable = DicomObjectNode::bytes(IMAGER_PIXEL_SPACING, "DS", vec![1, 2]);
        assert_eq!(
            DicomPixelAspect::try_from(&unreadable),
            Err(DicomPixelAspectError::UnreadableValue(
                DicomPixelAspectSource::ImagerPixelSpacing,
            ))
        );
    }

    #[test]
    fn malformed_components_are_rejected() {
        for (source, value) in [
            (DicomPixelAspectSource::PixelAspectRatio, "6"),
            (DicomPixelAspectSource::PixelAspectRatio, "6\\5\\4"),
            (DicomPixelAspectSource::PixelAspectRatio, "6.0\\5"),
            (DicomPixelAspectSource::ImagerPixelSpacing, "NaN\\1"),
            (DicomPixelAspectSource::ImagerPixelSpacing, "1E+308\\1E-308"),
            (DicomPixelAspectSource::PixelAspectRatio, "0\\1"),
            (
                DicomPixelAspectSource::ImagerPixelSpacing,
                "00000000000000001\\1",
            ),
            (DicomPixelAspectSource::PixelAspectRatio, "\t6\\5"),
            (DicomPixelAspectSource::PixelAspectRatio, "\n6\\5"),
            (DicomPixelAspectSource::PixelAspectRatio, "\u{a0}6\\5"),
        ] {
            assert_eq!(
                DicomPixelAspect::try_from(&node(source, value)),
                Err(DicomPixelAspectError::MalformedValue(source))
            );
        }
    }

    #[test]
    fn final_wire_padding_does_not_consume_the_component_limit() {
        for (source, value, expected) in [
            (
                DicomPixelAspectSource::PixelAspectRatio,
                "12\\000000000001 ",
                12.0,
            ),
            (
                DicomPixelAspectSource::ImagerPixelSpacing,
                "12\\0000000000000001 ",
                12.0,
            ),
        ] {
            let aspect = DicomPixelAspect::try_from(&node(source, value))
                .expect("final wire padding is excluded from the component limit");
            assert_eq!(aspect.row_over_column(), expected);
        }
    }

    #[test]
    fn integer_string_uses_the_signed_32_bit_domain() {
        let maximum = DicomPixelAspect::try_from(&node(
            DicomPixelAspectSource::PixelAspectRatio,
            "2147483647\\1",
        ))
        .expect("largest positive IS parses");
        assert_eq!(maximum.row_over_column(), f64::from(i32::MAX));

        for value in ["2147483648\\1", "-2147483648\\1"] {
            assert_eq!(
                DicomPixelAspect::try_from(&node(DicomPixelAspectSource::PixelAspectRatio, value,)),
                Err(DicomPixelAspectError::MalformedValue(
                    DicomPixelAspectSource::PixelAspectRatio,
                ))
            );
        }
    }
}
