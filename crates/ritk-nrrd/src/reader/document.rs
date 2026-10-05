//! NRRD document reads that retain format metadata outside the volume model.

use ritk_codecs::{ByteOrder, SampleType};
use ritk_image_io::ImageReadBudget;
use std::path::Path;

use super::header::NrrdHeader;
use super::stored::NrrdStoredReadError;
use super::volume::{NrrdPayloadPlan, NrrdReadPurpose};

/// A decoded NRRD payload with its complete parsed header and file-axis sizes.
///
/// Unlike [`ritk_image_io::StoredVolume`], this type does not interpret or
/// discard spatial, calibration, acquisition, or modality fields. The payload
/// remains in NRRD file-axis order and in its declared fixed-width sample type.
///
/// # Example
///
/// ```no_run
/// use ritk_image_io::ImageReadBudget;
/// use ritk_nrrd::read_nrrd_document;
///
/// let document = read_nrrd_document("input.nrrd", ImageReadBudget::DEFAULT)?;
/// assert_eq!(document.sizes().len(), 3);
/// assert_eq!(document.sample_count(), document.sample_bytes().len()
///     / document.sample_type().byte_width());
/// let _source_version = document.header().format_version();
/// let _byte_order = document.byte_order();
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct NrrdDocument {
    header: NrrdHeader,
    sizes: Vec<usize>,
    sample_type: SampleType,
    byte_order: ByteOrder,
    sample_count: usize,
    samples: Vec<u8>,
}

impl NrrdDocument {
    /// Returns the parsed source header, including comments and repeated
    /// custom key/value records.
    #[must_use]
    pub const fn header(&self) -> &NrrdHeader {
        &self.header
    }

    /// Returns array sizes in NRRD file-axis order, fastest axis first.
    #[must_use]
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// Returns the fixed-width sample representation declared by the file.
    #[must_use]
    pub const fn sample_type(&self) -> SampleType {
        self.sample_type
    }

    /// Returns the byte order used by the retained sample bytes.
    #[must_use]
    pub const fn byte_order(&self) -> ByteOrder {
        self.byte_order
    }

    /// Returns the number of retained samples.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns decoded sample bytes without changing their fixed-width bits.
    ///
    /// Binary payloads retain the declared byte order. ASCII samples are
    /// encoded little-endian because the source text has no byte order.
    #[must_use]
    pub fn sample_bytes(&self) -> &[u8] {
        &self.samples
    }
}

/// Reads a complete NRRD document without projecting its metadata into a volume.
///
/// This path retains metadata such as sample units, thicknesses, measurement
/// frames, comments, repeated custom fields, and acquisition-axis semantics
/// even when [`crate::reader::read_nrrd_stored`] cannot represent them.
/// Payload bytes are decoded from raw, ASCII, gzip, or a supported detached
/// source and remain in the declared fixed-width sample representation.
///
/// # Errors
///
/// Returns a typed NRRD parsing, payload, allocation, or resource-budget error.
///
/// # Example
///
/// ```no_run
/// use ritk_image_io::ImageReadBudget;
/// use ritk_nrrd::read_nrrd_document;
///
/// let document = read_nrrd_document("input.nrrd", ImageReadBudget::DEFAULT)?;
/// assert_eq!(document.header().format_version(), 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn read_nrrd_document<P: AsRef<Path>>(
    path: P,
    budget: ImageReadBudget,
) -> Result<NrrdDocument, NrrdStoredReadError> {
    let mut plan = NrrdPayloadPlan::open(path, NrrdReadPurpose::StoredDocument)?;
    let byte_order = plan.byte_order()?;
    let sample_count = plan.sample_count()?;
    let samples = plan.read_payload(budget)?;
    let (header, sizes, sample_type, sample_count) = plan.into_document_parts(sample_count);
    Ok(NrrdDocument {
        header,
        sizes,
        sample_type,
        byte_order,
        sample_count,
        samples,
    })
}
