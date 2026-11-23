//! Destructive clearing of native, top-level DICOM Pixel Data.

use ritk_dicom::dataset::{Dataset, DicomTag, Value};

const PIXEL_DATA: DicomTag = DicomTag::new(0x7FE0, 0x0010);

/// Present Pixel Data could not be cleared as requested.
///
/// This is a distinct typed cause in the existing [`anyhow::Error`] channel.
/// It does not add variants to the existing exhaustive header error enum.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PixelCleaningError {
    /// Encapsulated frames cannot be replaced with a flat native byte buffer.
    #[error("cannot clear encapsulated Pixel Data (7FE0,0010)")]
    Encapsulated,
    /// Pixel Data was represented as an ordinary sequence, not native bytes.
    #[error("cannot clear sequence-valued Pixel Data (7FE0,0010)")]
    Sequence,
    /// An existing element has no readable bytes to clear.
    #[error("cannot clear empty Pixel Data (7FE0,0010)")]
    Empty,
}

pub(super) fn clear_top_level_pixel_data(dataset: &mut Dataset) -> Result<(), PixelCleaningError> {
    let Some(pixel) = dataset.get(PIXEL_DATA) else {
        return Ok(());
    };
    let len = match pixel.value() {
        Value::PixelSequence(_) => return Err(PixelCleaningError::Encapsulated),
        Value::Sequence(_) => return Err(PixelCleaningError::Sequence),
        Value::Primitive(value) => value.to_bytes().len(),
    };
    if len == 0 {
        return Err(PixelCleaningError::Empty);
    }

    let vr = pixel.vr();
    dataset.put_value(PIXEL_DATA, vr, vec![0_u8; len]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{clear_top_level_pixel_data, PixelCleaningError, PIXEL_DATA};
    use crate::format::dicom::anonymize::tests_verify::build_object;
    use crate::format::dicom::anonymize::{anonymize_object, AnonymizeOptions, CleaningPolicy};
    use ritk_dicom::dataset::{DataElement, Dataset, PixelFragments, PrimitiveValue, Sequence, VR};

    #[test]
    fn typed_error_reaches_the_existing_object_error_channel() {
        let mut input = build_object("2.25.1", "2.25.2", "2.25.3");
        input.update_meta(|meta| meta.transfer_syntax = "1.2.840.10008.1.2.5".to_owned());
        input.put(DataElement::new(
            PIXEL_DATA,
            VR::OB,
            PixelFragments::from_fragments([vec![1, 2]]),
        ));

        let (preserved, _) = anonymize_object(input.clone(), &AnonymizeOptions::default())
            .expect("default header export preserves Pixel Data");
        assert_eq!(preserved.get(PIXEL_DATA), input.get(PIXEL_DATA));

        let options = AnonymizeOptions {
            clean_pixel_data: CleaningPolicy::Clean,
            ..AnonymizeOptions::default()
        };
        let failure = anonymize_object(input, &options)
            .expect_err("encapsulated Pixel Data cannot be cleared");
        assert_eq!(
            failure.downcast_ref::<PixelCleaningError>(),
            Some(&PixelCleaningError::Encapsulated),
        );
    }

    #[test]
    fn distinguishes_an_absent_element_from_empty_and_sequence_values() {
        let mut input = Dataset::new();
        assert!(clear_top_level_pixel_data(&mut input).is_ok());
        assert!(input.get(PIXEL_DATA).is_none());

        input.put_value(PIXEL_DATA, VR::OB, PrimitiveValue::Empty);
        assert_eq!(
            clear_top_level_pixel_data(&mut input),
            Err(PixelCleaningError::Empty)
        );

        input.put(DataElement::new(
            PIXEL_DATA,
            VR::SQ,
            Sequence::from(vec![Dataset::new()]),
        ));
        assert_eq!(
            clear_top_level_pixel_data(&mut input),
            Err(PixelCleaningError::Sequence)
        );
    }
}
