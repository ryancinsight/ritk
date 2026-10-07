use std::collections::HashMap;
use std::collections::TryReserveError;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use thiserror::Error;

mod parsed;
mod records;
pub use parsed::{NrrdHeader, NrrdKeyValueRecord};
use records::insert_key_value;

/// Maximum bytes retained while parsing one NRRD header.
///
/// This bounds both the current line and the cumulative header, including
/// comments and the required blank separator.
pub(crate) const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;

/// Maximum number of standard fields, comments, and key/value records retained.
///
/// The entry cap bounds collection and string-header overhead independently
/// of the byte budget. Repeated key/value keys count as separate records.
pub(crate) const MAX_HEADER_ENTRIES: usize = 65_536;

/// A failure while reading or parsing a NRRD header.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NrrdHeaderError {
    /// The input ended before the NRRD magic line.
    #[error("NRRD header is missing its magic line")]
    MissingMagicLine,
    /// The first header line does not identify a NRRD file.
    #[error("Not a supported NRRD file: invalid or unsupported magic line")]
    InvalidMagic,
    /// NRRD 1.0 does not permit key/value metadata pairs.
    #[error("NRRD key/value metadata requires format version 2 or later (line {line_number})")]
    KeyValueBeforeVersionTwo {
        /// One-based line number in the header.
        line_number: usize,
    },
    /// A standard field is used before the NRRD version that introduced it.
    #[error(
        "NRRD field {field:?} requires format version {minimum_version}, got {actual_version}"
    )]
    FieldRequiresVersion {
        /// Header field whose version requirement was violated.
        field: String,
        /// First format version that defines the field.
        minimum_version: u8,
        /// Version declared by the magic line.
        actual_version: u8,
    },
    /// The header did not contain its required empty separator line.
    #[error("NRRD header ended before its required blank separator")]
    MissingSeparator,
    /// Header content exceeded the bounded working-set limit.
    #[error("NRRD header exceeds the {maximum_bytes}-byte limit")]
    HeaderTooLarge {
        /// Maximum accepted cumulative header size.
        maximum_bytes: usize,
    },
    /// The header contains too many retained fields, comments, and records.
    #[error("NRRD header exceeds the {maximum_entries}-entry limit")]
    TooManyEntries {
        /// Maximum accepted count of retained fields, comments, and records.
        maximum_entries: usize,
    },
    /// A header line contains bytes outside the NRRD ASCII header encoding.
    #[error("NRRD header line {line_number} is not ASCII")]
    NonAsciiLine {
        /// One-based line number in the header.
        line_number: usize,
    },
    /// A non-comment header line is neither a field nor a key/value pair.
    #[error("NRRD header line {line_number} is malformed")]
    MalformedLine {
        /// One-based line number in the header.
        line_number: usize,
    },
    /// A field specification occurs more than once.
    #[error("NRRD field {field:?} occurs more than once")]
    DuplicateField {
        /// Case-folded field identifier.
        field: String,
    },
    /// A standard field and custom key/value pair have the same lowercase name.
    #[error("NRRD field and key/value pair collide at {key:?}")]
    FieldKeyValueCollision {
        /// Lowercase name shared by the two namespaces.
        key: String,
    },
    /// A key/value pair has an empty key.
    #[error("NRRD key/value pair on line {line_number} has an empty key")]
    EmptyKey {
        /// One-based line number in the header.
        line_number: usize,
    },
    /// A key/value value uses an unsupported or incomplete escape.
    #[error("NRRD key/value escape on line {line_number} is invalid")]
    InvalidKeyValueEscape {
        /// One-based line number in the header.
        line_number: usize,
    },
    /// A header line could not be read.
    #[error("cannot read NRRD header: {source}")]
    Read {
        /// Input read failure.
        #[source]
        source: io::Error,
    },
    /// Memory for a bounded header allocation could not be reserved.
    #[error("cannot reserve memory for NRRD header {operation}: {source}")]
    Allocation {
        /// Header allocation being requested.
        operation: &'static str,
        /// Reservation failure.
        #[source]
        source: TryReserveError,
    },
    /// A NRRD header file could not be opened.
    #[error("cannot open NRRD header {path:?}: {source}")]
    Open {
        /// Header path supplied by the caller.
        path: PathBuf,
        /// Filesystem failure.
        #[source]
        source: io::Error,
    },
}

/// Parse NRRD fields and key/value pairs into one lookup map without decoding
/// the payload.
///
/// Names are returned in lowercase for map lookup. The separate namespaces
/// merge only when their names do not collide; a collision returns a typed
/// error instead of replacing a structural field. Header parsing is limited to
/// 16 MiB and 65,536 standard fields, comments, and key/value records to bound
/// memory use on untrusted input.
///
/// # Errors
///
/// Returns a typed error when the file cannot be opened, the header is
/// malformed, a resource limit is exceeded, or reading fails.
pub fn read_nrrd_header_map<P: AsRef<Path>>(path: P) -> anyhow::Result<HashMap<String, String>> {
    let NrrdHeader {
        mut fields,
        key_values,
        ..
    } = read_nrrd_header(path)?;
    for (key, value) in key_values {
        let mut key = copy_string(&key, "combined map key")?;
        key.make_ascii_lowercase();
        if fields.contains_key(&key) {
            return Err(NrrdHeaderError::FieldKeyValueCollision { key }.into());
        }
        fields
            .try_reserve(1)
            .map_err(|source| NrrdHeaderError::Allocation {
                operation: "combined field map",
                source,
            })?;
        fields.insert(key, value);
    }
    Ok(fields)
}

/// Reads and parses a NRRD header without decoding its sample payload.
///
/// The result retains canonical standard fields, comments, the effective
/// custom key/value map, every decoded key/value record in source order, and
/// the format version. Repeated custom keys follow NRRD's last-value lookup
/// rule while remaining available as individual records.
///
/// Header size and retained-entry count are bounded to limit memory use on
/// untrusted input.
///
/// # Errors
///
/// Returns a typed error if the file cannot be opened or read, its header is
/// malformed, or either resource limit is exceeded.
///
/// # Examples
///
/// ```no_run
/// use ritk_nrrd::read_nrrd_header;
///
/// let header = read_nrrd_header("input.nrrd")?;
/// assert_eq!(header.format_version(), 4);
/// assert!(header.fields().contains_key("dimension"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn read_nrrd_header<P: AsRef<Path>>(path: P) -> Result<NrrdHeader, NrrdHeaderError> {
    let (_, header) = open_nrrd_header_reader(path.as_ref())?;
    Ok(header)
}

pub(super) fn open_nrrd_header_reader(
    path: &Path,
) -> Result<(BufReader<std::fs::File>, NrrdHeader), NrrdHeaderError> {
    let file = std::fs::File::open(path).map_err(|source| NrrdHeaderError::Open {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let header = parse_nrrd_header_from_reader(&mut reader)?;
    Ok((reader, header))
}

pub(super) fn parse_nrrd_header_from_reader<R: BufRead>(
    reader: &mut R,
) -> Result<NrrdHeader, NrrdHeaderError> {
    let mut total_bytes = 0_usize;
    let magic =
        read_header_line(reader, &mut total_bytes)?.ok_or(NrrdHeaderError::MissingMagicLine)?;
    let magic = line_text(&magic, 1)?;
    let format_version = match magic {
        "NRRD0001" | "NRRD00.01" => 1,
        "NRRD0002" => 2,
        "NRRD0003" => 3,
        "NRRD0004" => 4,
        "NRRD0005" => 5,
        _ => return Err(NrrdHeaderError::InvalidMagic),
    };

    let mut header = NrrdHeader {
        fields: HashMap::new(),
        key_values: HashMap::new(),
        key_value_records: Vec::new(),
        comments: Vec::new(),
        format_version,
    };
    let mut line_number = 1_usize;
    loop {
        let Some(line) = read_header_line(reader, &mut total_bytes)? else {
            if header.fields.contains_key("data file") {
                return Ok(header);
            }
            return Err(NrrdHeaderError::MissingSeparator);
        };
        line_number = line_number
            .checked_add(1)
            .ok_or(NrrdHeaderError::HeaderTooLarge {
                maximum_bytes: MAX_HEADER_BYTES,
            })?;
        let text = line_text(&line, line_number)?;
        if text.is_empty() {
            return Ok(header);
        }
        if text.starts_with('#') {
            if !text.trim_start_matches(['#', ' ']).is_empty() {
                reserve_entry(
                    header.fields.len(),
                    header.key_value_records.len(),
                    header.comments.len(),
                )?;
                header
                    .comments
                    .try_reserve(1)
                    .map_err(|source| NrrdHeaderError::Allocation {
                        operation: "comment table",
                        source,
                    })?;
                header.comments.push(copy_string(text, "comment line")?);
            }
            continue;
        }
        let field_separator = text.find(": ");
        let key_value_separator = text.find(":=");
        let (separator, is_field) = match (field_separator, key_value_separator) {
            (Some(field), Some(_)) if is_standard_field_name(&text[..field]) => (field, true),
            (Some(_), Some(key_value)) => (key_value, false),
            (Some(field), None) => (field, true),
            (None, Some(key_value)) => (key_value, false),
            (None, None) => return Err(NrrdHeaderError::MalformedLine { line_number }),
        };
        let (key, remainder) = text.split_at(separator);
        if is_field {
            let value = remainder
                .strip_prefix(": ")
                .ok_or(NrrdHeaderError::MalformedLine { line_number })?;
            insert_field(
                &mut header,
                key,
                value.trim_end(),
                line_number,
                format_version,
            )?;
        } else {
            let value = remainder
                .strip_prefix(":=")
                .ok_or(NrrdHeaderError::MalformedLine { line_number })?;
            if format_version < 2 {
                return Err(NrrdHeaderError::KeyValueBeforeVersionTwo { line_number });
            }
            insert_key_value(&mut header, key, value, line_number)?;
        }
    }
}

fn read_header_line<R: BufRead>(
    reader: &mut R,
    total_bytes: &mut usize,
) -> Result<Option<Vec<u8>>, NrrdHeaderError> {
    let mut line = Vec::new();
    loop {
        let available = reader
            .fill_buf()
            .map_err(|source| NrrdHeaderError::Read { source })?;
        if available.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Ok(Some(line))
            };
        }
        let (count, complete) = match available.iter().position(|byte| *byte == b'\n') {
            Some(newline) => (newline + 1, true),
            None => (available.len(), false),
        };
        let next_total = total_bytes
            .checked_add(count)
            .filter(|next| *next <= MAX_HEADER_BYTES)
            .ok_or(NrrdHeaderError::HeaderTooLarge {
                maximum_bytes: MAX_HEADER_BYTES,
            })?;
        line.try_reserve_exact(count)
            .map_err(|source| NrrdHeaderError::Allocation {
                operation: "line buffer",
                source,
            })?;
        line.extend_from_slice(&available[..count]);
        reader.consume(count);
        *total_bytes = next_total;
        if complete {
            return Ok(Some(line));
        }
    }
}

fn line_text(line: &[u8], line_number: usize) -> Result<&str, NrrdHeaderError> {
    let without_newline = line.strip_suffix(b"\n").unwrap_or(line);
    let without_terminator = without_newline
        .strip_suffix(b"\r")
        .unwrap_or(without_newline);
    if !without_terminator.is_ascii() {
        return Err(NrrdHeaderError::NonAsciiLine { line_number });
    }
    std::str::from_utf8(without_terminator)
        .map_err(|_| NrrdHeaderError::NonAsciiLine { line_number })
}

fn insert_field(
    header: &mut NrrdHeader,
    field: &str,
    value: &str,
    line_number: usize,
    format_version: u8,
) -> Result<(), NrrdHeaderError> {
    if field.is_empty() || field.trim() != field {
        return Err(NrrdHeaderError::MalformedLine { line_number });
    }
    let field = canonical_field_name(field);
    reserve_entry(
        header.fields.len(),
        header.key_value_records.len(),
        header.comments.len(),
    )?;
    if let Some(minimum_version) = minimum_field_version(field)
        && format_version < minimum_version
    {
        return Err(NrrdHeaderError::FieldRequiresVersion {
            field: field.to_owned(),
            minimum_version,
            actual_version: format_version,
        });
    }
    let mut key = copy_string(field, "field name")?;
    key.make_ascii_lowercase();
    if header.fields.contains_key(&key) {
        return Err(NrrdHeaderError::DuplicateField { field: key });
    }
    header
        .fields
        .try_reserve(1)
        .map_err(|source| NrrdHeaderError::Allocation {
            operation: "field table",
            source,
        })?;
    let value = copy_string(value.trim(), "field value")?;
    header.fields.insert(key, value);
    Ok(())
}

fn canonical_field_name(field: &str) -> &str {
    if field.eq_ignore_ascii_case("byteskip") {
        "byte skip"
    } else if field.eq_ignore_ascii_case("lineskip") {
        "line skip"
    } else if field.eq_ignore_ascii_case("datafile") {
        "data file"
    } else if field.eq_ignore_ascii_case("axismins") {
        "axis mins"
    } else if field.eq_ignore_ascii_case("axismaxs") {
        "axis maxs"
    } else if field.eq_ignore_ascii_case("centerings") {
        "centers"
    } else if field.eq_ignore_ascii_case("blocksize") {
        "block size"
    } else if field.eq_ignore_ascii_case("oldmin") {
        "old min"
    } else if field.eq_ignore_ascii_case("oldmax") {
        "old max"
    } else if field.eq_ignore_ascii_case("sampleunits") {
        "sample units"
    } else {
        field
    }
}

const STANDARD_FIELD_NAMES: &str = "type|dimension|space|space units|sizes|space directions|kinds|endian|encoding|space origin|measurement frame|content|labels|data file|line skip|byte skip|spacings|thicknesses|axis mins|axis maxs|centers|block size|old min|old max|sample units|space dimension";

pub(crate) fn is_standard_field_name(field: &str) -> bool {
    let canonical = canonical_field_name(field);
    STANDARD_FIELD_NAMES
        .split('|')
        .any(|name| name.eq_ignore_ascii_case(canonical))
}

fn minimum_field_version(field: &str) -> Option<u8> {
    if field.eq_ignore_ascii_case("kinds") {
        Some(3)
    } else if [
        "thicknesses",
        "sample units",
        "space",
        "space dimension",
        "space directions",
        "space origin",
        "space units",
    ]
    .iter()
    .any(|name| field.eq_ignore_ascii_case(name))
    {
        Some(4)
    } else if field.eq_ignore_ascii_case("measurement frame") {
        Some(5)
    } else {
        None
    }
}

pub(super) fn reserve_entry(
    field_count: usize,
    key_value_count: usize,
    comment_count: usize,
) -> Result<(), NrrdHeaderError> {
    let count = field_count
        .checked_add(key_value_count)
        .and_then(|count| count.checked_add(comment_count))
        .and_then(|count| count.checked_add(1))
        .ok_or(NrrdHeaderError::TooManyEntries {
            maximum_entries: MAX_HEADER_ENTRIES,
        })?;
    if count > MAX_HEADER_ENTRIES {
        return Err(NrrdHeaderError::TooManyEntries {
            maximum_entries: MAX_HEADER_ENTRIES,
        });
    }
    Ok(())
}

pub(super) fn copy_string(value: &str, operation: &'static str) -> Result<String, NrrdHeaderError> {
    let mut copied = String::new();
    copied
        .try_reserve_exact(value.len())
        .map_err(|source| NrrdHeaderError::Allocation { operation, source })?;
    copied.push_str(value);
    Ok(copied)
}

#[cfg(test)]
#[path = "header/tests.rs"]
mod tests;
