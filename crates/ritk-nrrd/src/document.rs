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
#[derive(Debug)]
pub struct NrrdDocument {
    series: StoredSeries,
    comments: Vec<String>,
    pub(crate) records: Vec<(String, String)>,
}
#[derive(Debug, Error)]
pub enum NrrdDocumentError {
    #[error(transparent)]
    Read(#[from] NrrdStoredReadError),
    #[error(transparent)]
    Header(#[from] NrrdHeaderError),
    #[error("NRRD metadata conflicts with generated field {name:?}")]
    ConflictingMetadata { name: String },
    #[error("NRRD standard field {field:?} cannot be retained")]
    UnsupportedField { field: String },
    #[error(transparent)]
    Write(#[from] NrrdStoredWriteError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
impl NrrdDocument {
    pub fn new(
        series: StoredSeries,
        comments: Vec<String>,
        records: Vec<(String, String)>,
    ) -> Result<Self, NrrdDocumentError> {
        if comments.iter().any(|comment| {
            !comment.is_ascii()
                || comment.len() < 2
                || !comment.starts_with('#')
                || comment.contains(['\r', '\n'])
                || comment.chars().all(|character| character == '#')
                || comment == GENERATED_COMMENT
        }) || records.iter().any(|(key, value)| {
            key.is_empty()
                || !key.is_ascii()
                || key.starts_with('#')
                || key.contains(":=")
                || key.contains(['\r', '\n'])
                || !value.is_ascii()
                || value.contains(['\r', '\n'])
                || unsupported_dwmri_name(key)
                || unsupported_modality(key, value)
        }) {
            return Err(NrrdDocumentError::UnsupportedField {
                field: "metadata".into(),
            });
        }
        Ok(Self {
            series,
            comments,
            records,
        })
    }
    pub fn series(&self) -> &StoredSeries {
        &self.series
    }
    pub fn comments(&self) -> &[String] {
        &self.comments
    }
    pub fn records(&self) -> &[(String, String)] {
        &self.records
    }
    fn write_to<P: AsRef<Path>>(&self, path: P) -> Result<(), NrrdDocumentError> {
        let Some(first) = self.series.volumes().first() else {
            return Err(NrrdStoredWriteError::EmptySeries.into());
        };
        for (name, value) in self.records.iter() {
            if generated_metadata_name(name, value) {
                return Err(NrrdDocumentError::ConflictingMetadata { name: name.clone() });
            }
            if unsupported_dwmri_name(name) || unsupported_modality(name, value) {
                return Err(NrrdDocumentError::UnsupportedField {
                    field: name.clone(),
                });
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
        let entries = header
            .bytes()
            .split(|byte| *byte == b'\n')
            .skip(1)
            .take_while(|line| !line.is_empty())
            .count();
        if entries > crate::reader::MAX_HEADER_ENTRIES {
            return Err(NrrdDocumentError::Header(NrrdHeaderError::TooManyEntries {
                maximum_entries: crate::reader::MAX_HEADER_ENTRIES,
            }));
        }
        crate::writer::validate_series_axis(self.series.axis())?;
        crate::writer::validate_calibration(first)?;
        validate_physical_geometry(first.metadata())
            .map_err(|source| NrrdStoredWriteError::PhysicalGeometry { source })?;
        let sample_type = first.samples().sample_type();
        for (offset, volume) in self.series.volumes().iter().skip(1).enumerate() {
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
pub fn read_nrrd_document<P: AsRef<Path>>(
    path: P,
    budget: ImageReadBudget,
) -> Result<NrrdDocument, NrrdDocumentError> {
    let header = read_nrrd_header(path.as_ref())?;
    let series = read_nrrd_stored_series(path, budget)?;
    let comments = header
        .comments()
        .iter()
        .filter(|c| c.as_str() != GENERATED_COMMENT)
        .cloned()
        .collect();
    let records = header
        .key_value_records()
        .iter()
        .filter(|record| !generated_metadata_name(record.key(), record.value()))
        .map(|record| (record.key().to_owned(), record.value().to_owned()))
        .collect();
    if header
        .fields()
        .keys()
        .any(|key| !SUPPORTED_FIELDS.split('|').any(|field| field == key))
    {
        return Err(NrrdDocumentError::UnsupportedField {
            field: "standard header field".to_owned(),
        });
    }
    NrrdDocument::new(series, comments, records)
}
const SUPPORTED_FIELDS: &str = "type|dimension|space|space units|sizes|space directions|kinds|endian|encoding|space origin|measurement frame";
const GENERATED_COMMENT: &str = "# Complete NRRD file written by ritk";

fn generated_metadata_name(name: &str, value: &str) -> bool {
    matches!(
        name,
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
            | "ritk_coordinate_map"
            | "DWMRI_b-value"
    ) || name.starts_with("DWMRI_gradient_")
        || (name == "modality" && value == "DWMRI")
}

fn unsupported_dwmri_name(name: &str) -> bool {
    name.get(..6)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("DWMRI_"))
        && !generated_metadata_name(name, "DWMRI")
}

fn unsupported_modality(name: &str, value: &str) -> bool {
    name.eq_ignore_ascii_case("modality")
        && (name != "modality" || value.eq_ignore_ascii_case("DWMRI"))
}
pub fn write_nrrd_document<P: AsRef<Path>>(
    path: P,
    document: &NrrdDocument,
) -> Result<(), NrrdDocumentError> {
    document.write_to(path)
}
