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
//! file: . 2
//! END
//! ```
//!
//! After `END`, raw binary voxel data follows directly in the stream.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};

use anyhow::{anyhow, Context, Result};

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
/// the caller can read the binary payload directly.
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
/// the caller can continue reading binary data from it.
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

/// Parse a space-separated list of `f64` from a header value.
pub(crate) fn parse_f64_vec(value: &str) -> Result<Vec<f64>> {
    value
        .split_whitespace()
        .map(|s| {
            s.parse::<f64>()
                .with_context(|| format!("Invalid float component '{}'", s))
        })
        .collect()
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

/// Parse the `datatype` header value into (byte_size, is_signed, is_float).
pub(crate) fn parse_datatype(value: &str) -> Result<(usize, bool, bool)> {
    let lower = value.trim().to_lowercase();
    // Strip endian suffix to match the base type.
    let base = lower
        .strip_suffix("le")
        .or_else(|| lower.strip_suffix("be"))
        .unwrap_or(&lower);
    match base {
        "float32" => Ok((4, true, true)),
        "float64" => Ok((8, true, true)),
        "int32" => Ok((4, true, false)),
        "uint32" => Ok((4, false, false)),
        "int16" => Ok((2, true, false)),
        "uint16" => Ok((2, false, false)),
        "int8" => Ok((1, true, false)),
        "uint8" => Ok((1, false, false)),
        _ => Err(anyhow!("Unsupported .mif datatype '{}'", value.trim())),
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
#[path = "header_tests.rs"]
mod tests;
