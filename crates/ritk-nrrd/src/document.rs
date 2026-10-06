//! Validated NRRD documents for loss-aware format conversion.
use crate::reader::NrrdHeaderError;
use crate::writer::{
    write_nrrd_header_with_metadata, write_nrrd_series_header_with_metadata, HeaderBuffer,
    SeriesLayout,
};
use crate::{read_nrrd_header, read_nrrd_stored_series, NrrdStoredReadError, NrrdStoredWriteError};
use ritk_image_io::{validate_physical_geometry, ImageReadBudget, SeriesAxis, StoredSeries};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use thiserror::Error;
/// A complete NRRD document with stored samples and retained header metadata.
#[derive(Debug)]
pub struct NrrdDocument {
    series: StoredSeries,
    comments: Vec<String>,
    pub(crate) records: Vec<(String, String)>,
}
/// A document construction or serialization failure.
#[derive(Debug, Error)]
pub enum NrrdDocumentError {
    /// The stored source could not be read.
    #[error(transparent)]
    Read(#[from] NrrdStoredReadError),
    /// Header syntax is invalid.
    #[error(transparent)]
    Header(#[from] NrrdHeaderError),
    /// A retained field or record would override generated structure.
    #[error("NRRD metadata conflicts with generated field {name:?}")]
    ConflictingMetadata { name: String },
    /// A standard field cannot be retained by this document model.
    #[error("NRRD standard field {field:?} cannot be retained")]
    UnsupportedField { field: String },
    /// Series validation rejected the document before output.
    #[error(transparent)]
    Write(#[from] NrrdStoredWriteError),
    /// Output creation or flushing failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
impl NrrdDocument {
    /// Constructs a document without writing an intermediate file.
    pub fn new(
        series: StoredSeries,
        comments: Vec<String>,
        records: Vec<(String, String)>,
    ) -> Result<Self, NrrdDocumentError> {
        if comments.iter().any(|comment| {
            !comment.is_ascii() || !comment.starts_with('#') || comment.contains(['\r', '\n'])
        }) {
            return Err(NrrdDocumentError::UnsupportedField {
                field: "metadata".to_owned(),
            });
        }
        if records.iter().any(|(key, value)| {
            key.is_empty() || !key.is_ascii() || !value.is_ascii() || value.contains('\r')
        }) {
            return Err(NrrdDocumentError::UnsupportedField {
                field: "metadata".to_owned(),
            });
        }
        Ok(Self {
            series,
            comments,
            records,
        })
    }
    /// Returns the stored volumes in acquisition order.
    pub fn series(&self) -> &StoredSeries {
        &self.series
    }
    /// Returns retained comments in source order.
    pub fn comments(&self) -> &[String] {
        &self.comments
    }
    /// Returns retained custom records in source order.
    pub fn records(&self) -> &[(String, String)] {
        &self.records
    }
    fn write_to<P: AsRef<Path>>(&self, path: P) -> Result<(), NrrdDocumentError> {
        let first = self
            .series
            .volumes()
            .first()
            .ok_or(NrrdStoredWriteError::EmptySeries)?;
        let entries = 32usize
            .saturating_add(self.series.volumes().len())
            .saturating_add(self.comments.len())
            .saturating_add(self.records.len());
        if entries > crate::reader::MAX_HEADER_ENTRIES {
            return Err(NrrdDocumentError::Header(NrrdHeaderError::TooManyEntries {
                maximum_entries: crate::reader::MAX_HEADER_ENTRIES,
            }));
        }
        for (name, _) in self.records.iter() {
            if generated_metadata_name(name, matches!(self.series.axis(), SeriesAxis::Diffusion(_)))
            {
                return Err(NrrdDocumentError::ConflictingMetadata { name: name.clone() });
            }
        }
        let mut header = HeaderBuffer::new();
        let result = if matches!(self.series.axis(), SeriesAxis::SingleVolume) {
            write_nrrd_header_with_metadata(
                &mut header,
                first.shape(),
                first.metadata().spacing(),
                first.metadata().origin(),
                first.metadata().direction(),
                crate::writer::nrrd_type_name(first.samples().sample_type())?,
                first.coordinate_map(),
                &self.comments,
                &self.records,
            )
        } else {
            write_nrrd_series_header_with_metadata(
                &mut header,
                first.shape(),
                self.series.volumes().len(),
                first.metadata().spacing(),
                first.metadata().origin(),
                first.metadata().direction(),
                crate::writer::nrrd_type_name(first.samples().sample_type())?,
                first.coordinate_map(),
                SeriesLayout::AcquisitionSlowest,
                self.series.axis(),
                &self.comments,
                &self.records,
            )
        };
        if header.exceeded_limit() {
            return Err(NrrdDocumentError::Write(
                NrrdStoredWriteError::HeaderTooLarge {
                    header_bytes: crate::reader::MAX_HEADER_BYTES.saturating_add(1),
                    maximum_bytes: crate::reader::MAX_HEADER_BYTES,
                },
            ));
        }
        result?;
        let Some((first, rest)) = self.series.volumes().split_first() else {
            return Err(NrrdStoredWriteError::EmptySeries.into());
        };
        crate::writer::validate_series_axis(self.series.axis())?;
        crate::writer::validate_series_header_entries(self.series.axis(), first.coordinate_map())?;
        crate::writer::validate_calibration(first)?;
        validate_physical_geometry(first.metadata())
            .map_err(|source| NrrdStoredWriteError::PhysicalGeometry { source })?;
        let sample_type = first.samples().sample_type();
        for (offset, volume) in rest.iter().enumerate() {
            let index = offset + 1;
            crate::writer::validate_calibration(volume)?;
            if volume.shape() != first.shape() {
                return Err(NrrdStoredWriteError::ShapeMismatch {
                    index,
                    expected: first.shape(),
                    actual: volume.shape(),
                }
                .into());
            }
            if volume.metadata() != first.metadata() {
                return Err(NrrdStoredWriteError::GeometryMismatch { index }.into());
            }
            if volume.coordinate_map() != first.coordinate_map() {
                return Err(NrrdStoredWriteError::CoordinateMapMismatch { index }.into());
            }
            if volume.samples().sample_type() != sample_type {
                return Err(NrrdStoredWriteError::SampleTypeMismatch { index }.into());
            }
        }
        let file = File::create(path)?;
        let mut output = BufWriter::new(file);
        output.write_all(header.bytes())?;
        for volume in self.series.volumes() {
            crate::writer::write_sample_payload(volume, &mut output)?;
        }
        output.flush()?;
        Ok(())
    }
}
/// Reads one-volume documents while retaining their exact metadata.
pub fn read_nrrd_document<P: AsRef<Path>>(
    path: P,
    budget: ImageReadBudget,
) -> Result<NrrdDocument, NrrdDocumentError> {
    let header = read_nrrd_header(path.as_ref())?;
    let series = read_nrrd_stored_series(path, budget)?;
    let comments = header.comments().to_vec();
    let records = header
        .key_value_records()
        .iter()
        .map(|record| (record.key().to_owned(), record.value().to_owned()))
        .collect();
    let supported = "type|dimension|space|space units|sizes|space directions|kinds|endian|encoding|space origin|measurement frame";
    if header
        .fields()
        .keys()
        .any(|key| !supported.split('|').any(|field| field == key))
    {
        return Err(NrrdDocumentError::UnsupportedField {
            field: "standard header field".to_owned(),
        });
    }
    NrrdDocument::new(series, comments, records)
}
fn generated_metadata_name(name: &str, diffusion: bool) -> bool {
    let lower = name.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "type"
            | "dimension"
            | "space"
            | "space units"
            | "sizes"
            | "space directions"
            | "kinds"
            | "endian"
            | "encoding"
            | "space origin"
            | "measurement frame"
            | "ritk:coordinate-map"
            | "modality"
            | "dwmri_b-value"
    ) || (diffusion && lower.starts_with("dwmri_gradient_"))
}
/// Writes a validated NRRD document without an intermediate conversion file.
pub fn write_nrrd_document<P: AsRef<Path>>(
    path: P,
    document: &NrrdDocument,
) -> Result<(), NrrdDocumentError> {
    document.write_to(path)
}
