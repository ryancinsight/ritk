//! Prepared NRRD document serialization.

use ritk_codecs::ByteOrder;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use thiserror::Error;

use super::HeaderBuffer;
use crate::reader::{NrrdDocument, MAX_HEADER_ENTRIES};

/// A prepared NRRD document cannot be serialized or atomically written.
///
/// # Example
///
/// ```no_run
/// use ritk_image_io::ImageReadBudget;
/// use ritk_nrrd::{prepare_nrrd_document, read_nrrd_document};
///
/// let document = read_nrrd_document("input.nrrd", ImageReadBudget::DEFAULT)?;
/// let prepared = prepare_nrrd_document(&document)?;
/// assert!(!prepared.header_bytes().is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NrrdDocumentWriteError {
    /// The retained sample byte count does not match its dimensions and type.
    #[error("NRRD document has {actual_bytes} sample bytes; expected {expected_bytes}")]
    PayloadSizeMismatch {
        /// Required byte length derived from sizes and sample type.
        expected_bytes: usize,
        /// Retained payload byte length.
        actual_bytes: usize,
    },
    /// The product of the declared axis sizes overflows `usize`.
    #[error("NRRD document sample count overflows usize")]
    SampleCountOverflow,
    /// The retained sample count disagrees with the declared axis sizes.
    #[error("NRRD document retains {actual_samples} samples; sizes require {expected_samples}")]
    SampleCountMismatch {
        /// Sample count derived from the axis sizes.
        expected_samples: usize,
        /// Sample count retained by the document.
        actual_samples: usize,
    },
    /// The sample byte count overflows `usize`.
    #[error("NRRD document sample byte count overflows usize")]
    SampleByteCountOverflow,
    /// The number of output metadata records exceeds the reader's bound.
    #[error("NRRD output has {entries} metadata records; the limit is {maximum_entries}")]
    HeaderTooManyEntries {
        /// Number of standard fields and custom key/value records.
        entries: usize,
        /// Maximum accepted record count.
        maximum_entries: usize,
    },
    /// The serialized header exceeds the reader's byte bound.
    #[error("NRRD output header exceeds the {maximum_bytes}-byte limit")]
    HeaderTooLarge {
        /// Maximum accepted NRRD header size.
        maximum_bytes: usize,
    },
    /// An output allocation could not be reserved.
    #[error("cannot reserve memory for NRRD document output: {source}")]
    Allocation {
        /// Reservation failure.
        #[source]
        source: std::collections::TryReserveError,
    },
    /// Header or sample output failed.
    #[error("NRRD document output failed: {source}")]
    Io {
        /// Filesystem or writer failure.
        #[source]
        source: io::Error,
    },
}

/// A validated NRRD header and borrowed, fixed-width sample payload.
///
/// Preparation completes all structural checks and serializes the bounded
/// header before a destination is opened. The output uses inline raw payloads;
/// source packaging fields such as `data file`, `line skip`, and `byte skip`
/// are normalized, while comments, semantic fields, custom records, sample
/// bits, and sample byte order remain represented. Standard field order follows
/// the [NRRD format specification, section 1.3](https://teem.sourceforge.net/nrrd/format.html).
///
/// # Example
///
/// ```no_run
/// use ritk_image_io::ImageReadBudget;
/// use ritk_nrrd::{prepare_nrrd_document, read_nrrd_document};
///
/// let document = read_nrrd_document("input.nrrd", ImageReadBudget::DEFAULT)?;
/// let prepared = prepare_nrrd_document(&document)?;
/// let mut output = Vec::new();
/// prepared.write_to(&mut output)?;
/// assert!(output.ends_with(prepared.sample_bytes()));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PreparedNrrdDocument<'a> {
    header: Vec<u8>,
    document: &'a NrrdDocument,
}

impl PreparedNrrdDocument<'_> {
    /// Writes the prepared header and sample bytes to a caller-owned stream.
    ///
    /// # Errors
    ///
    /// Returns [`NrrdDocumentWriteError::Io`] if the stream rejects a write.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), NrrdDocumentWriteError> {
        writer
            .write_all(&self.header)
            .and_then(|()| writer.write_all(self.document.sample_bytes()))
            .map_err(|source| NrrdDocumentWriteError::Io { source })
    }

    /// Returns the preflighted NRRD header bytes.
    #[must_use]
    pub fn header_bytes(&self) -> &[u8] {
        &self.header
    }

    /// Returns the retained sample bytes.
    #[must_use]
    pub fn sample_bytes(&self) -> &[u8] {
        self.document.sample_bytes()
    }
}

/// Validates and serializes the header without opening an output destination.
///
/// The prepared writer preserves all parsed NRRD semantic fields and custom
/// key/value records. It normalizes storage details to one inline raw payload
/// and writes the retained samples in their current byte order.
///
/// # Errors
///
/// Returns a typed payload-size, metadata-count, header-size, allocation, or
/// header-serialization error. No output path is opened during this function.
///
/// # Example
///
/// ```no_run
/// use ritk_image_io::ImageReadBudget;
/// use ritk_nrrd::{prepare_nrrd_document, read_nrrd_document};
///
/// let document = read_nrrd_document("input.nrrd", ImageReadBudget::DEFAULT)?;
/// let prepared = prepare_nrrd_document(&document)?;
/// assert!(!prepared.header_bytes().is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn prepare_nrrd_document(
    document: &NrrdDocument,
) -> Result<PreparedNrrdDocument<'_>, NrrdDocumentWriteError> {
    let expected_samples = document
        .sizes()
        .iter()
        .try_fold(1_usize, |count, size| count.checked_mul(*size))
        .ok_or(NrrdDocumentWriteError::SampleCountOverflow)?;
    if expected_samples != document.sample_count() {
        return Err(NrrdDocumentWriteError::SampleCountMismatch {
            expected_samples,
            actual_samples: document.sample_count(),
        });
    }
    let expected_bytes = expected_samples
        .checked_mul(document.sample_type().byte_width())
        .ok_or(NrrdDocumentWriteError::SampleByteCountOverflow)?;
    if document.sample_bytes().len() != expected_bytes {
        return Err(NrrdDocumentWriteError::PayloadSizeMismatch {
            expected_bytes,
            actual_bytes: document.sample_bytes().len(),
        });
    }

    let fields = document.header().fields();
    let output_endian = document.sample_type().byte_width() > 1 || fields.contains_key("endian");
    let retained_fields = fields
        .keys()
        .filter(|name| {
            !matches!(
                name.as_str(),
                "data file" | "line skip" | "byte skip" | "encoding" | "endian"
            )
        })
        .count();
    let record_count = retained_fields
        .checked_add(1)
        .and_then(|count| count.checked_add(usize::from(output_endian)))
        .and_then(|count| count.checked_add(document.header().key_value_records().len()))
        .ok_or(NrrdDocumentWriteError::HeaderTooManyEntries {
            entries: usize::MAX,
            maximum_entries: MAX_HEADER_ENTRIES,
        })?;
    if record_count > MAX_HEADER_ENTRIES {
        return Err(NrrdDocumentWriteError::HeaderTooManyEntries {
            entries: record_count,
            maximum_entries: MAX_HEADER_ENTRIES,
        });
    }

    let mut header = HeaderBuffer::new();
    let header_result = write_document_header(&mut header, document, output_endian);
    if header.exceeded_limit() {
        return Err(NrrdDocumentWriteError::HeaderTooLarge {
            maximum_bytes: crate::reader::MAX_HEADER_BYTES,
        });
    }
    header_result.map_err(|source| NrrdDocumentWriteError::Io { source })?;
    Ok(PreparedNrrdDocument {
        header: header.bytes,
        document,
    })
}

/// Writes a prepared NRRD document through a same-directory temporary file.
///
/// Preparation completes before the temporary output is created. The final
/// rename atomically replaces an existing destination on supported local
/// filesystems; failure before that point leaves the destination unchanged.
///
/// # Errors
///
/// Returns the preparation error or a typed temporary-file, write, flush, or
/// replacement error.
///
/// # Example
///
/// ```no_run
/// use ritk_image_io::ImageReadBudget;
/// use ritk_nrrd::{read_nrrd_document, write_nrrd_document};
///
/// let document = read_nrrd_document("input.nrrd", ImageReadBudget::DEFAULT)?;
/// write_nrrd_document("output.nrrd", &document)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn write_nrrd_document<P: AsRef<Path>>(
    path: P,
    document: &NrrdDocument,
) -> Result<(), NrrdDocumentWriteError> {
    let path = path.as_ref();
    let prepared = prepare_nrrd_document(document)?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .map_err(|source| NrrdDocumentWriteError::Io { source })?;
    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        prepared.write_to(&mut writer)?;
        writer
            .flush()
            .map_err(|source| NrrdDocumentWriteError::Io { source })?;
    }
    temporary
        .persist(path)
        .map_err(|failure| NrrdDocumentWriteError::Io {
            source: failure.error,
        })?;
    Ok(())
}

fn write_document_header(
    writer: &mut impl Write,
    document: &NrrdDocument,
    output_endian: bool,
) -> io::Result<()> {
    writeln!(writer, "NRRD000{}", document.header().format_version())?;
    for comment in document.header().comments() {
        writer.write_all(comment.as_bytes())?;
        writer.write_all(b"\n")?;
    }

    let fields = document.header().fields();
    let mut names = Vec::new();
    names
        .try_reserve_exact(fields.len())
        .map_err(io::Error::other)?;
    names.extend(fields.keys());
    names.sort_unstable_by(|left, right| {
        field_group(left)
            .cmp(&field_group(right))
            .then_with(|| left.cmp(right))
    });
    for group in 0..=4 {
        for name in names
            .iter()
            .copied()
            .filter(|name| field_group(name) == group)
        {
            if matches!(
                name.as_str(),
                "data file" | "line skip" | "byte skip" | "encoding" | "endian"
            ) {
                continue;
            }
            let value = fields
                .get(name.as_str())
                .expect("invariant: field name came from this map");
            writeln!(writer, "{name}: {value}")?;
        }
        if group == 0 {
            writeln!(writer, "encoding: raw")?;
            if output_endian {
                writeln!(writer, "endian: {}", endian_name(document.byte_order()))?;
            }
        }
    }

    for (key, value) in document.header().key_value_records() {
        write_escaped(writer, key)?;
        writer.write_all(b":=")?;
        write_escaped(writer, value)?;
        writer.write_all(b"\n")?;
    }
    writer.write_all(b"\n")
}

fn field_group(name: &str) -> usize {
    match name {
        "dimension" => 1,
        "space" | "space dimension" => 2,
        "space origin" | "space units" | "measurement frame" => 3,
        "sizes" | "spacings" | "thicknesses" | "axis mins" | "axis maxs" | "centers" | "labels"
        | "units" | "kinds" | "space directions" => 4,
        _ => 0,
    }
}

fn endian_name(byte_order: ByteOrder) -> &'static str {
    match byte_order {
        ByteOrder::MostSignificantByteFirst => "big",
        ByteOrder::LeastSignificantByteFirst => "little",
    }
}

fn write_escaped(writer: &mut impl Write, value: &str) -> io::Result<()> {
    let bytes = value.as_bytes();
    let mut segment_start = 0;
    for (index, byte) in bytes.iter().copied().enumerate() {
        let escape = match byte {
            b'\\' => Some(&b"\\\\"[..]),
            b'\n' => Some(&b"\\n"[..]),
            _ => None,
        };
        if let Some(escape) = escape {
            writer.write_all(&bytes[segment_start..index])?;
            writer.write_all(escape)?;
            segment_start = index + 1;
        }
    }
    writer.write_all(&bytes[segment_start..])
}
