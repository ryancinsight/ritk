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
use super::reader::DicomReadMetadata;

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

/// Project a whole series' inventory into shared conversion losses.
///
/// This is the entry point a conversion preflight uses, and it is deliberately
/// not the same as reading `DicomReadMetadata::preservation`: the reader
/// records every loss while parsing a single instance, so the losses live on
/// `DicomSliceMetadata::preservation`. The series-level set carries only
/// series-scope losses, which is none today. A preflight that read the series
/// set alone would see an empty inventory and import a volume with missing
/// metadata.
///
/// Series-scope losses are reported at [`ConversionLocation::Series`]; each
/// slice's losses are reported at that slice's [`ConversionLocation::Frame`],
/// so a loss names the exact instance it came from. `volume_index` is the
/// zero-based volume the slices belong to.
#[must_use]
pub fn dicom_series_metadata_losses(
    metadata: &DicomReadMetadata,
    volume_index: usize,
) -> Box<[FormatMetadataLoss]> {
    let mut losses: Vec<FormatMetadataLoss> =
        dicom_metadata_losses(&metadata.preservation, ConversionLocation::Series).into_vec();
    for (frame_index, slice) in metadata.slices.iter().enumerate() {
        losses.extend(dicom_metadata_losses(
            &slice.preservation,
            ConversionLocation::Frame {
                volume_index,
                frame_index,
            },
        ));
    }
    losses.into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::super::object_model::{DicomPreservedElement, DicomRetentionReason, DicomTag};
    use super::super::reader::DicomSliceMetadata;
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

    /// A slice with a loss, so the series fixtures below have something to find.
    fn slice_with_loss(tag: DicomTag, reason: DicomRetentionReason) -> DicomSliceMetadata {
        let mut preservation = DicomPreservationSet::new();
        preservation.record_loss(tag, reason);
        DicomSliceMetadata {
            preservation,
            ..DicomSliceMetadata::default()
        }
    }

    /// The series entry point finds losses the series-level set does not carry.
    ///
    /// This is the defect the entry point exists for: the reader records every
    /// loss while parsing one instance, so they live on the slices. Reading
    /// `DicomReadMetadata::preservation` alone reports a fully retained source
    /// for a series that dropped metadata, which is exactly the silent loss
    /// ADR 0054 forbids.
    #[test]
    fn the_series_set_alone_is_empty_when_only_slices_dropped_metadata() {
        let metadata = DicomReadMetadata {
            slices: vec![slice_with_loss(
                DicomTag::new(0x0009, 0x1001),
                DicomRetentionReason::ValueBytesUnavailable,
            )],
            ..DicomReadMetadata::default()
        };

        assert!(
            dicom_metadata_losses(&metadata.preservation, ConversionLocation::Series).is_empty(),
            "the series-level set carries no per-slice loss"
        );
        assert_eq!(
            dicom_series_metadata_losses(&metadata, 0).len(),
            1,
            "the series projection must still see the slice's loss"
        );
    }

    /// Each loss names its own frame, not a compacted counter.
    #[test]
    fn a_series_projection_scopes_each_slice_loss_to_its_own_frame() {
        let metadata = DicomReadMetadata {
            slices: vec![
                slice_with_loss(
                    DicomTag::new(0x0009, 0x1001),
                    DicomRetentionReason::ValueBytesUnavailable,
                ),
                DicomSliceMetadata::default(),
                slice_with_loss(
                    DicomTag::new(0x0040, 0xA730),
                    DicomRetentionReason::NestingDepthExceeded,
                ),
            ],
            ..DicomReadMetadata::default()
        };

        let losses = dicom_series_metadata_losses(&metadata, 1);
        assert_eq!(losses.len(), 2, "only the two dropping slices contribute");

        let FormatMetadataLoss::UnknownSemantics { location, field } = &losses[0] else {
            panic!("a retention loss must project to UnknownSemantics");
        };
        assert_eq!(
            *location,
            ConversionLocation::Frame {
                volume_index: 1,
                frame_index: 0,
            },
            "the first slice's loss names slice 0"
        );
        assert_eq!(
            &**field,
            "(0009,1001) value could not be re-encoded as bytes"
        );

        let FormatMetadataLoss::UnknownSemantics { location, .. } = &losses[1] else {
            panic!("a retention loss must project to UnknownSemantics");
        };
        assert_eq!(
            *location,
            ConversionLocation::Frame {
                volume_index: 1,
                frame_index: 2,
            },
            "the third slice's loss names slice 2, not a compacted counter"
        );
    }
}
