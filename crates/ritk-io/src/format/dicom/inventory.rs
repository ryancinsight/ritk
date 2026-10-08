//! Source-owned DICOM metadata inventory → conversion-loss projection.
//!
//! The DICOM reader accounts for every parsed element: it either interprets it,
//! retains it opaquely, or records a [`DicomRetentionLoss`]. This module is the
//! single place that turns those records into the shared
//! [`FormatMetadataLoss`] vocabulary that `ritk_image_io::prepare_conversion`
//! consumes, so a DICOM conversion preflight rejects a source whose metadata
//! could not be retained instead of importing a volume with missing fields.
//!
//! The inventory is **source-owned**: it travels on the reader's own metadata
//! (`DicomReadMetadata::preservation`) and is never derived from the
//! destination. Keeping the projection here — rather than in the object model —
//! keeps [`DicomPreservationSet`] a pure data model with no conversion
//! dependency (SRP). Tag rendering uses [`DicomTag`]'s own `Display`, so the
//! `(GGGG,EEEE)` form has one definition.

use ritk_image_io::{ConversionLocation, FormatMetadataLoss};

use super::object_model::{DicomPreservationSet, DicomRetentionLoss};

/// The scoped field name reported for one unretained element.
fn loss_field(loss: &DicomRetentionLoss) -> Box<str> {
    format!("{} {}", loss.tag, loss.reason.describe()).into_boxed_str()
}

/// Project the inventory's unretained elements into shared conversion losses.
///
/// `location` scopes every reported loss to the series, volume, or frame the
/// caller is inspecting. An empty result means the parser retained or
/// interpreted every element and the source is loss-free.
///
/// Each unretained element becomes
/// [`FormatMetadataLoss::UnknownSemantics`]: the adapter could not read the
/// value, so it cannot claim to know the field's semantics.
#[must_use]
pub fn dicom_metadata_losses(
    preservation: &DicomPreservationSet,
    location: ConversionLocation,
) -> Box<[FormatMetadataLoss]> {
    preservation
        .losses
        .iter()
        .map(|loss| FormatMetadataLoss::UnknownSemantics {
            location,
            field: loss_field(loss),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::object_model::{DicomPreservedElement, DicomRetentionReason, DicomTag};
    use super::*;

    #[test]
    fn retained_source_projects_no_losses() {
        let mut preservation = DicomPreservationSet::new();
        preservation.preserve(DicomPreservedElement::new(
            DicomTag::new(0x0009, 0x1001),
            None,
            vec![1, 2, 3],
        ));

        assert!(
            dicom_metadata_losses(&preservation, ConversionLocation::Series).is_empty(),
            "an opaquely retained element is not a loss"
        );
    }

    #[test]
    fn every_recorded_loss_projects_to_a_scoped_unknown_semantics_entry() {
        let mut preservation = DicomPreservationSet::new();
        preservation.record_loss(
            DicomTag::new(0x0009, 0x1001),
            DicomRetentionReason::ValueBytesUnavailable,
        );
        preservation.record_loss(
            DicomTag::new(0x0040, 0xA730),
            DicomRetentionReason::NestingDepthExceeded,
        );

        let location = ConversionLocation::Frame {
            volume_index: 2,
            frame_index: 7,
        };
        let losses = dicom_metadata_losses(&preservation, location);
        assert_eq!(losses.len(), 2);

        let FormatMetadataLoss::UnknownSemantics { location: l, field } = &losses[0] else {
            panic!("a retention loss must project to UnknownSemantics");
        };
        assert_eq!(*l, location, "the loss keeps the caller's scope");
        assert_eq!(
            &**field,
            "(0009,1001) value could not be re-encoded as bytes"
        );

        let FormatMetadataLoss::UnknownSemantics { field, .. } = &losses[1] else {
            panic!("a retention loss must project to UnknownSemantics");
        };
        assert_eq!(
            &**field,
            "(0040,A730) sequence nesting exceeded the retention limit"
        );
    }
}
