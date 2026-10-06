use crate::reader::{NrrdHeader, NrrdHeaderError, NrrdReadPlan, NrrdReadSession};
use crate::writer::{
    write_nrrd_header_with_metadata, write_nrrd_series_header_with_metadata, HeaderBuffer,
    SeriesLayout,
};
use crate::{NrrdStoredReadError, NrrdStoredWriteError};
use ritk_image_io::{validate_physical_geometry, ImageReadBudget, SeriesAxis, StoredSeries};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use thiserror::Error;

/// In-memory NRRD samples with validated, round-trippable metadata.
#[derive(Debug)]
pub struct NrrdDocument {
    series: StoredSeries,
    comments: Vec<String>,
    pub(crate) records: Vec<(String, String)>,
}
/// Typed construction, parsing, and serialization failure.
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
    /// Constructs a document without an intermediate file.
    ///
    /// Unsupported standard fields, generated records, and parser-dropped comment forms return a typed error.
    pub fn new(
        series: StoredSeries,
        comments: Vec<String>,
        records: Vec<(String, String)>,
    ) -> Result<Self, NrrdDocumentError> {
        let document = Self {
            series,
            comments,
            records,
        };
        document
            .validate_for_write()
            .map_err(map_conflicting_metadata)?;
        Ok(document)
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
    fn validate_for_write(&self) -> Result<HeaderBuffer, NrrdDocumentError> {
        let Some(first) = self.series.volumes().first() else {
            return Err(NrrdStoredWriteError::EmptySeries.into());
        };
        validate_document_metadata(
            &self.comments,
            &self.records,
            matches!(self.series.axis(), SeriesAxis::Diffusion(_)),
        )?;
        let header = build_document_header(
            first.shape(),
            self.series.volumes().len(),
            first.metadata().spacing(),
            first.metadata().origin(),
            first.metadata().direction(),
            crate::writer::nrrd_type_name(first.samples().sample_type())?,
            first.coordinate_map(),
            self.series.axis(),
            &self.comments,
            &self.records,
        )?;
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
        Ok(header)
    }
    fn write_to<P: AsRef<Path>>(&self, path: P) -> Result<(), NrrdDocumentError> {
        let header = self.validate_for_write()?;
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

fn build_document_header(
    shape: [usize; 3],
    volume_count: usize,
    spacing: &ritk_spatial::Spacing<3>,
    origin: &ritk_spatial::Point<3>,
    direction: &ritk_spatial::Direction<3>,
    element_type: &str,
    coordinate_map: &ritk_spatial::CoordinateMap,
    axis: &SeriesAxis,
    comments: &[String],
    records: &[(String, String)],
) -> Result<HeaderBuffer, NrrdDocumentError> {
    let mut header = HeaderBuffer::new();
    let result = if matches!(axis, SeriesAxis::SingleVolume) {
        write_nrrd_header_with_metadata(
            &mut header,
            shape,
            spacing,
            origin,
            direction,
            element_type,
            coordinate_map,
            comments,
            records,
        )
    } else {
        write_nrrd_series_header_with_metadata(
            &mut header,
            shape,
            volume_count,
            spacing,
            origin,
            direction,
            element_type,
            coordinate_map,
            SeriesLayout::AcquisitionSlowest,
            axis,
            comments,
            records,
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
    Ok(header)
}

/// Reads a document and rejects metadata the typed model cannot retain.
pub fn read_nrrd_document<P: AsRef<Path>>(
    path: P,
    budget: ImageReadBudget,
) -> Result<NrrdDocument, NrrdDocumentError> {
    let session = NrrdReadSession::open(path.as_ref())?;
    read_nrrd_document_from_session(session, budget)
}

/// Builds a document from the parsed header and the same open payload source.
pub(crate) fn read_nrrd_document_from_session(
    session: NrrdReadSession,
    budget: ImageReadBudget,
) -> Result<NrrdDocument, NrrdDocumentError> {
    let metadata = validate_document_header(session.header())?;
    let prepared = session.prepare_stored_series(budget)?;
    validate_document_header_limits(prepared.plan(), &metadata)?;
    let series = prepared.into_stored_series(budget)?;
    NrrdDocument::new(series, metadata.comments, metadata.records)
}

struct NrrdDocumentMetadata {
    comments: Vec<String>,
    records: Vec<(String, String)>,
}

fn validate_document_header_limits(
    plan: &NrrdReadPlan,
    metadata: &NrrdDocumentMetadata,
) -> Result<(), NrrdDocumentError> {
    let Some(axis) = plan.series_axis.as_ref() else {
        return Err(NrrdDocumentError::UnsupportedField {
            field: "series axis".into(),
        });
    };
    let header = build_document_header(
        plan.dims,
        plan.volumes,
        &plan.spacing,
        &plan.origin,
        &plan.direction,
        crate::writer::nrrd_type_name(plan.sample_type)?,
        &plan.coordinate_map,
        axis,
        &metadata.comments,
        &metadata.records,
    )?;
    drop(header);
    Ok(())
}

fn validate_document_header(
    header: &NrrdHeader,
) -> Result<NrrdDocumentMetadata, NrrdDocumentError> {
    let records = header.key_value_records();
    for record in records {
        if record.key().eq_ignore_ascii_case("DWMRI_b-value")
            && (record.key() != "DWMRI_b-value"
                || records
                    .iter()
                    .filter(|candidate| candidate.key() == "DWMRI_b-value")
                    .count()
                    != 1)
        {
            return Err(NrrdDocumentError::UnsupportedField {
                field: record.key().to_owned(),
            });
        }
        if record
            .key()
            .eq_ignore_ascii_case(crate::coordinate_map::COORDINATE_MAP_KEY)
            && (record.key() != crate::coordinate_map::COORDINATE_MAP_KEY
                || records
                    .iter()
                    .filter(|candidate| {
                        candidate.key() == crate::coordinate_map::COORDINATE_MAP_KEY
                    })
                    .count()
                    != 1
                || crate::coordinate_map::decode(record.value()).is_err())
        {
            return Err(NrrdDocumentError::UnsupportedField {
                field: record.key().to_owned(),
            });
        }
        if record.key().eq_ignore_ascii_case("modality")
            && record.value().eq_ignore_ascii_case("DWMRI")
            && (record.key() != "modality"
                || record.value() != "DWMRI"
                || records
                    .iter()
                    .any(|candidate| candidate.key() == "modality" && candidate.value() != "DWMRI")
                || records
                    .iter()
                    .filter(|candidate| {
                        candidate.key() == "modality" && candidate.value() == "DWMRI"
                    })
                    .count()
                    != 1)
        {
            return Err(NrrdDocumentError::UnsupportedField {
                field: record.key().to_owned(),
            });
        }
        if record
            .key()
            .get(..15)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("DWMRI_gradient_"))
            && (record.key().get(..15) != Some("DWMRI_gradient_")
                || records
                    .iter()
                    .filter(|candidate| candidate.key() == record.key())
                    .count()
                    != 1)
        {
            return Err(NrrdDocumentError::UnsupportedField {
                field: record.key().to_owned(),
            });
        }
    }
    if header
        .fields()
        .keys()
        .any(|key| !generated_standard_field(key))
        || records
            .iter()
            .any(|record| standard_metadata_name(record.key()))
    {
        return Err(NrrdDocumentError::UnsupportedField {
            field: "standard metadata".to_owned(),
        });
    }
    let comments = header
        .comments()
        .iter()
        .filter(|comment| comment.as_str() != GENERATED_COMMENT)
        .cloned()
        .collect::<Vec<_>>();
    let retained_records = records
        .iter()
        .filter(|record| !generated_metadata_name(record.key(), record.value()))
        .map(|record| (record.key().to_owned(), record.value().to_owned()))
        .collect::<Vec<_>>();
    validate_document_metadata(
        &comments,
        &retained_records,
        crate::reader::has_diffusion_metadata(header),
    )
    .map_err(map_conflicting_metadata)?;
    Ok(NrrdDocumentMetadata {
        comments,
        records: retained_records,
    })
}

fn map_conflicting_metadata(error: NrrdDocumentError) -> NrrdDocumentError {
    match error {
        NrrdDocumentError::ConflictingMetadata { .. } => NrrdDocumentError::UnsupportedField {
            field: "metadata".into(),
        },
        other => other,
    }
}

fn validate_document_metadata(
    comments: &[String],
    records: &[(String, String)],
    is_diffusion: bool,
) -> Result<(), NrrdDocumentError> {
    if comments.iter().any(|comment| {
        !comment.is_ascii()
            || comment.len() < 2
            || !comment.starts_with('#')
            || comment.contains(['\r', '\n'])
            || comment.trim_start_matches(['#', ' ']).is_empty()
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
            || generated_metadata_name(key, value)
    }) {
        return Err(NrrdDocumentError::UnsupportedField {
            field: "metadata".into(),
        });
    }
    for (name, value) in records {
        if generated_metadata_name(name, value)
            || standard_metadata_name(name)
            || (is_diffusion && name == "modality")
        {
            return Err(NrrdDocumentError::ConflictingMetadata { name: name.clone() });
        }
        if unsupported_dwmri_name(name) || unsupported_modality(name, value) {
            return Err(NrrdDocumentError::UnsupportedField {
                field: name.clone(),
            });
        }
    }
    Ok(())
}
const GENERATED_COMMENT: &str = "# Complete NRRD file written by ritk";
const STANDARD_FIELDS: &str = "type|dimension|space|space units|sizes|space directions|kinds|endian|encoding|space origin|measurement frame|content|labels|data file|line skip|byte skip|spacings|thicknesses|axis mins|axis maxs|centers|block size|old min|old max|sample units|space dimension";
const GENERATED_STANDARD_FIELDS: &str = "type|dimension|space|space units|sizes|space directions|kinds|endian|encoding|space origin|measurement frame";

fn generated_metadata_name(name: &str, value: &str) -> bool {
    name.eq_ignore_ascii_case("ritk_coordinate_map")
        || name.eq_ignore_ascii_case("DWMRI_b-value")
        || name.starts_with("DWMRI_gradient_")
        || (name == "modality" && value == "DWMRI")
}
fn standard_metadata_name(name: &str) -> bool {
    STANDARD_FIELDS.split('|').any(|field| field == name)
}
fn generated_standard_field(name: &str) -> bool {
    GENERATED_STANDARD_FIELDS
        .split('|')
        .any(|field| field == name)
}
fn unsupported_dwmri_name(name: &str) -> bool {
    name.get(..6)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("DWMRI_"))
        && !generated_metadata_name(name, "DWMRI")
}
fn unsupported_modality(name: &str, value: &str) -> bool {
    name.eq_ignore_ascii_case("modality")
        && (name != "modality" || value.eq_ignore_ascii_case("DWMRI"))
        || standard_metadata_name(name)
}
/// Writes a validated document without opening invalid destinations.
pub fn write_nrrd_document<P: AsRef<Path>>(
    path: P,
    document: &NrrdDocument,
) -> Result<(), NrrdDocumentError> {
    document.write_to(path)
}
