//! MRtrix `.mif` text header parser.
//!
//! The `.mif` header is a sequence of `key: value` lines terminated by a
//! bare `END` line.  Line continuation uses a trailing backslash `\`:
//! the next line's content is appended after stripping leading whitespace.
//! Comment lines start with `#` and are ignored.
//!
//! ```text
//! mrtrix image: version 3.0
//! dim: 128 128 60 33
//! vox: 1.7 1.7 2.2 1.0
//! layout: +0,+1,+2,+3
//! datatype: Float32LE
//! transform:
//! 0.999 0.017 -0.005 -12.3
//! -0.017 0.998 -0.020 45.1
//! 0.006 0.020 0.999 -7.8
//! 0.0 0.0 0.0 1.0
//! DW_scheme: 2,4
//! 0,0,0,0
//! 1,0,0,1000
//! file: . 256
//! END
//! ```
//!
//! The `file` offset counts bytes from the start of the file and, for an inline
//! file, lies at or after the end of the `END` line (the example above is 256
//! bytes, so its data starts at byte 256). The parser stops after `END`; the
//! reader locates the data from that position and the `file` offset.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};

use anyhow::{anyhow, Context, Result};
use consus_core::ByteOrder;
use ritk_codecs::sample::SampleType;

/// Parsed `.mif` header key-value map plus the multi-file offset hint.
#[derive(Debug)]
pub(crate) struct MifHeader {
    pub entries: HashMap<String, HeaderValue>,
}

/// A `.mif` header value, which may be a single string or a multi-line block.
#[derive(Debug, Clone)]
pub(crate) enum HeaderValue {
    /// A single-line value (e.g. `dim: 128 128 60`).
    Line(String),
    /// A multi-line block where each line after the key is one row
    /// (e.g. `transform:` followed by four matrix rows).
    Block(Vec<String>),
}

impl HeaderValue {
    /// Single-line value string, panics on Block.
    pub fn as_line(&self) -> &str {
        match self {
            Self::Line(s) => s.as_str(),
            Self::Block(_) => panic!("expected single-line header value, got block"),
        }
    }

    /// Multi-line block rows, panics on Line.
    pub fn as_block(&self) -> &[String] {
        match self {
            Self::Line(_) => panic!("expected block header value, got line"),
            Self::Block(rows) => rows.as_slice(),
        }
    }

    /// True when this is a multi-line block.
    #[cfg(test)]
    pub fn is_block(&self) -> bool {
        matches!(self, Self::Block(_))
    }
}

/// Parse the `.mif` header from a reader, consuming up through the `END` line.
///
/// Returns a map keyed by lowercased key names.  Multi-line values (keys
/// whose first line is empty or whose value is continued across lines via
/// trailing `\`) are collected into `HeaderValue::Block`.
///
/// The reader is left positioned immediately after the `END\n` line so
/// the caller can take its position as the header length. The binary payload
/// starts at the `file` offset, which may lie further on (alignment padding).
pub(crate) fn parse_mif_header<R: BufRead>(reader: &mut R) -> Result<MifHeader> {
    let mut entries: HashMap<String, HeaderValue> = HashMap::new();

    // Collect all physical lines first, handling backslash continuation.
    let logical_lines = collect_logical_lines(reader)?;

    // Track multi-line blocks: when a key's value is empty (e.g.
    // `transform:` followed by data rows with no colons), subsequent
    // non-key lines accumulate as block rows.
    let mut current_block_key: Option<String> = None;

    for line in &logical_lines {
        if let Some((key, rest)) = line.split_once(':') {
            let key = key.trim().to_lowercase();
            let rest = rest.trim().to_string();

            if key == "transform" {
                let mut block_rows = Vec::new();
                if !rest.is_empty() {
                    block_rows.push(rest);
                }
                entries.insert(key.clone(), HeaderValue::Block(block_rows));
                current_block_key = Some(key);
            } else if key == "dw_scheme" {
                // DW_scheme is a single-line key followed by N data rows
                // that also lack colons.  Store the key line and let
                // subsequent rows accumulate.
                let block_rows = vec![rest];
                entries.insert(key.clone(), HeaderValue::Block(block_rows));
                current_block_key = Some(key);
            } else {
                entries.insert(key, HeaderValue::Line(rest));
                current_block_key = None;
            }
        } else if let Some(ref key) = current_block_key {
            // Line without a colon — continuation of the current block.
            if let Some(HeaderValue::Block(rows)) = entries.get_mut(key) {
                rows.push(line.trim().to_string());
            }
        }
        // Lines without colons and with no current block are silently
        // skipped (e.g. blank lines between blocks).
    }

    Ok(MifHeader { entries })
}

/// Read lines from the reader, handling backslash continuation, and stop
/// at the `END` marker.
fn collect_logical_lines<R: BufRead>(reader: &mut R) -> Result<Vec<String>> {
    let mut logical: Vec<String> = Vec::new();
    let mut current = String::new();

    loop {
        let mut physical = String::new();
        let n = reader
            .read_line(&mut physical)
            .context("Failed to read .mif header line")?;
        if n == 0 {
            return Err(anyhow!(".mif header truncated: EOF before END marker"));
        }

        // Detect END marker — case-insensitive, must be alone on its line
        // (possibly with trailing whitespace).
        if physical.trim().eq_ignore_ascii_case("END") {
            // Flush any pending logical line.
            if !current.is_empty() {
                logical.push(current);
            }
            return Ok(logical);
        }

        // Skip comment lines.
        let trimmed_start = physical.trim_start();
        if trimmed_start.starts_with('#') {
            continue;
        }

        // Handle backslash continuation.
        let ends_with_continuation = physical.trim_end().ends_with('\\');
        if ends_with_continuation {
            // Strip the trailing backslash (and any whitespace before it)
            // and append to current logical line without a newline separator.
            let stripped = physical.trim_end_matches(|c: char| c == '\\' || c.is_whitespace());
            current.push_str(stripped);
            // Next physical line continues this logical line.
        } else {
            // Complete logical line.
            current.push_str(physical.trim_end());
            logical.push(std::mem::take(&mut current));
        }
    }
}

/// Parse a `.mif` header from a file path.
///
/// Opens `path`, wraps in a `BufReader`, and delegates to
/// [`parse_mif_header`].  The returned reader consumes the header;
/// the caller locates the data from the stream position and the `file` offset.
pub(crate) fn parse_mif_header_from_path(
    path: &std::path::Path,
) -> Result<(MifHeader, BufReader<std::fs::File>)> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open .mif file {:?}", path))?;
    let mut reader = BufReader::new(file);
    let header = parse_mif_header(&mut reader)?;
    Ok((header, reader))
}

// ── Key extractors used by the reader ────────────────────────────────────

/// Parse a space-separated list of `usize` from a header value.
pub(crate) fn parse_dim(value: &str, expected_count: usize) -> Result<Vec<usize>> {
    let parts: Vec<usize> = value
        .split_whitespace()
        .map(|s| {
            s.parse::<usize>()
                .with_context(|| format!("Invalid dimension component '{}'", s))
        })
        .collect::<Result<_>>()?;
    if parts.len() < expected_count {
        return Err(anyhow!(
            "dim: expected at least {} values, got {} ({:?})",
            expected_count,
            parts.len(),
            parts
        ));
    }
    Ok(parts)
}

/// Parse the `vox` value into the voxel sizes of the spatial axes.
///
/// A fourth component, when present, is the frame spacing and is returned with
/// the rest; callers read the first three.
pub(crate) fn parse_vox(value: &str) -> Result<Vec<f64>> {
    let sizes: Vec<f64> = value
        .split_whitespace()
        .map(|s| {
            s.parse::<f64>()
                .with_context(|| format!("Invalid voxel size '{s}'"))
        })
        .collect::<Result<_>>()?;
    if sizes.len() < 3 {
        return Err(anyhow!(
            ".mif 'vox' expected at least 3 spatial sizes, got {}",
            sizes.len()
        ));
    }
    Ok(sizes)
}

/// Parse the `layout` value into axis strides.
///
/// MRtrix layout encodes data strides.  The common contiguous layout is
/// `+0,+1,+2,+3`.  The format also supports explicit strides like
/// `+0:128,+1:1,+2:128,+3:16384`.  This parser extracts the integer strides.
pub(crate) fn parse_layout(value: &str) -> Result<Vec<isize>> {
    value
        .split(',')
        .map(|s| {
            let s = s.trim();
            // Strip leading sign for parsing, then restore.
            if let Some(rest) = s.strip_prefix('+') {
                rest.parse::<isize>()
                    .with_context(|| format!("Invalid layout stride '{}'", s))
            } else if let Some(rest) = s.strip_prefix('-') {
                let val: isize = rest
                    .parse()
                    .with_context(|| format!("Invalid layout stride '{}'", s))?;
                Ok(-val)
            } else {
                s.parse::<isize>()
                    .with_context(|| format!("Invalid layout stride '{}'", s))
            }
        })
        .collect()
}

/// Parse the `datatype` header value into the stored sample type and its byte
/// order.
///
/// MRtrix spells a type as a case-insensitive name (`Int8`, `UInt8`, `Int16`,
/// `UInt16`, `Int32`, `UInt32`, `Int64`, `UInt64`, `Float32`, `Float64`) and,
/// for types wider than one byte, an `LE` or `BE` suffix. A multi-byte name
/// without a suffix is stored in the byte order of the machine that wrote it,
/// which MRtrix reads as the order of the reading machine; so does this parser.
/// A one-byte type has no byte order, and a suffix on one is rejected as MRtrix
/// rejects it.
///
/// # Sources
///
/// - MRtrix3 documentation, "Image data" page (`getting_started/image_data`),
///   section "Data types": the specifier table lists `Bit`, `Int8`, `UInt8`,
///   and `Int16` through `Float64` and the complex `CFloat32`/`CFloat64`, each
///   with `LE`/`BE` variants for the multi-byte types. It states that
///   specifiers are case-insensitive and that a name without a suffix uses the
///   native endianness.
/// - MRtrix3 `core/datatype.cpp`, `DataType::parse`: the table omits `Int64`
///   and `UInt64`, which `parse` accepts as `int64`, `uint64`, and their
///   `le`/`be` forms, matched on the lower-cased name.
/// - MRtrix3 `core/datatype.h`: `Int16`, `UInt16`, and the other multi-byte
///   names are defined without the `LittleEndian` (`0x40`) or `BigEndian`
///   (`0x80`) bit, and `DataType::set_byte_order_native` adds the bit of the
///   host to every type but `Bit`, `Int8`, and `UInt8`; a bare multi-byte name
///   therefore means native order.
///
/// # Errors
///
/// Returns an error for `Bit`, the complex types `CFloat32` and `CFloat64`
/// (which store two values per voxel), a one-byte type with a byte-order
/// suffix, and any other name.
pub(crate) fn parse_datatype(value: &str) -> Result<(SampleType, ByteOrder)> {
    let name = value.trim();
    let lower = name.to_ascii_lowercase();
    let (base, suffix) = if let Some(base) = lower.strip_suffix("le") {
        (base, Some(ByteOrder::LittleEndian))
    } else if let Some(base) = lower.strip_suffix("be") {
        (base, Some(ByteOrder::BigEndian))
    } else {
        (lower.as_str(), None)
    };
    let sample_type = match base {
        "int8" => SampleType::I8,
        "uint8" => SampleType::U8,
        "int16" => SampleType::I16,
        "uint16" => SampleType::U16,
        "int32" => SampleType::I32,
        "uint32" => SampleType::U32,
        "int64" => SampleType::I64,
        "uint64" => SampleType::U64,
        "float32" => SampleType::F32,
        "float64" => SampleType::F64,
        "bit" | "cfloat32" | "cfloat64" => {
            return Err(anyhow!(
                "Unsupported .mif datatype '{name}': RITK images hold one real scalar per voxel, and Bit and complex voxels are not one"
            ))
        }
        _ => return Err(anyhow!("Unknown .mif datatype '{name}'")),
    };
    match (sample_type.byte_width(), suffix) {
        (1, Some(_)) => Err(anyhow!(
            "Invalid .mif datatype '{name}': a one-byte type has no byte order"
        )),
        (1, None) => Ok((sample_type, ByteOrder::LittleEndian)),
        (_, Some(order)) => Ok((sample_type, order)),
        (_, None) => Ok((sample_type, native_byte_order())),
    }
}

/// The `datatype` header value that stores `sample_type` in `order`: the
/// inverse of [`parse_datatype`], with the `LE` or `BE` suffix on every type
/// wider than one byte.
pub(crate) fn datatype_name(sample_type: SampleType, order: ByteOrder) -> String {
    let base = match sample_type {
        SampleType::I8 => "Int8",
        SampleType::U8 => "UInt8",
        SampleType::I16 => "Int16",
        SampleType::U16 => "UInt16",
        SampleType::I32 => "Int32",
        SampleType::U32 => "UInt32",
        SampleType::I64 => "Int64",
        SampleType::U64 => "UInt64",
        SampleType::F32 => "Float32",
        SampleType::F64 => "Float64",
    };
    match (sample_type.byte_width(), order) {
        (1, _) => base.to_owned(),
        (_, ByteOrder::LittleEndian) => format!("{base}LE"),
        (_, ByteOrder::BigEndian) => format!("{base}BE"),
    }
}

/// The byte order of the machine this code runs on.
const fn native_byte_order() -> ByteOrder {
    if cfg!(target_endian = "big") {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    }
}

/// Parse a 4×4 affine matrix from a `transform` block.
///
/// Returns `[[f64; 4]; 4]` in row-major order (row 0 through row 3).
pub(crate) fn parse_transform(block: &[String]) -> Result<[[f64; 4]; 4]> {
    if block.len() != 4 {
        return Err(anyhow!("transform: expected 4 rows, got {}", block.len()));
    }
    let mut matrix = [[0.0f64; 4]; 4];
    for (i, row_str) in block.iter().enumerate() {
        let values: Vec<f64> = row_str
            .split_whitespace()
            .map(|s| {
                s.parse::<f64>()
                    .with_context(|| format!("transform row {}: invalid float '{}'", i, s))
            })
            .collect::<Result<_>>()?;
        if values.len() != 4 {
            return Err(anyhow!(
                "transform row {}: expected 4 values, got {}",
                i,
                values.len()
            ));
        }
        matrix[i] = [values[0], values[1], values[2], values[3]];
    }
    Ok(matrix)
}

#[cfg(test)]
#[path = "tests_header.rs"]
mod tests;
