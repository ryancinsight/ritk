//! Exact transport of a FreeSurfer volume between raw and gzip framing.
use super::{open_file, read_mgh_header, MAX_MGZ_UNCOMPRESSED_BYTES};
use crate::{is_gzip_path, MghError, HEADER_SIZE};
use flate2::{read::MultiGzDecoder, write::GzEncoder, Compression};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use tempfile::NamedTempFile;
use thiserror::Error;

/// Failure while preserving a complete MGH document.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MghDocumentError {
    /// The fixed header cannot describe a supported volume.
    #[error(transparent)]
    Header(#[from] MghError),
    /// The document ends before all frames declared by the header.
    #[error("MGH payload is {actual} bytes, expected at least {expected}")]
    Truncated {
        /// Minimum number of sample bytes declared by the header.
        expected: usize,
        /// Bytes available after the header.
        actual: u64,
    },
    /// The expanded MGZ document exceeds the MGZ admission limit.
    #[error("MGZ document exceeds the {maximum}-byte expanded limit")]
    ExpandedLimit {
        /// Maximum expanded length including the fixed header.
        maximum: u64,
    },
    /// The source, compression stream or destination failed.
    #[error("MGH document I/O failed: {0}")]
    Io(#[from] io::Error),
}

/// Reframe a supported MGH or MGZ volume without changing its expanded bytes.
///
/// Checks the header and all declared sample bytes while retaining any footer,
/// tags and padding verbatim. Gzip inputs are drained through each trailer.
/// The destination is atomically replaced after validation and successful
/// output finalization. Source and destination may be the same path.
///
/// # Errors
///
/// Returns a typed header, truncation, expanded-size or I/O error. Transfers
/// involving gzip admit at most 1 GiB of expanded data, including metadata.
pub fn transcode_mgh_document(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
) -> Result<(), MghDocumentError> {
    let source = source.as_ref();
    let destination = destination.as_ref();
    let compressed_input = is_gzip_path(source);
    let compressed_output = is_gzip_path(destination);
    let limit = if compressed_input || compressed_output {
        MAX_MGZ_UNCOMPRESSED_BYTES
    } else {
        u64::MAX
    };
    let file = open_file(source)?;
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temporary = NamedTempFile::new_in(parent)?;
    if compressed_input {
        let input = MultiGzDecoder::new(BufReader::new(file));
        write_document(input, temporary.as_file_mut(), compressed_output, limit)?;
    } else {
        write_document(
            BufReader::new(file),
            temporary.as_file_mut(),
            compressed_output,
            limit,
        )?;
    }
    temporary
        .persist(destination)
        .map_err(|failure| failure.error)?;
    Ok(())
}

fn write_document(
    source: impl Read,
    output: &mut std::fs::File,
    gzip: bool,
    limit: u64,
) -> Result<(), MghDocumentError> {
    let mut writer = BufWriter::new(output);
    if gzip {
        let mut encoder = GzEncoder::new(&mut writer, Compression::default());
        copy_document(source, &mut encoder, limit)?;
        encoder.finish()?;
    } else {
        copy_document(source, &mut writer, limit)?;
    }
    writer.flush()?;
    Ok(())
}

fn copy_document(
    mut input: impl Read,
    output: &mut impl Write,
    limit: u64,
) -> Result<(), MghDocumentError> {
    let mut raw = [0; HEADER_SIZE];
    input
        .read_exact(&mut raw)
        .map_err(|source| MghError::Header { source })?;
    let header = read_mgh_header(&mut raw.as_slice())?;
    let remaining = limit
        .checked_sub(HEADER_SIZE as u64)
        .filter(|&available| available >= header.payload_bytes as u64)
        .ok_or(MghDocumentError::ExpandedLimit { maximum: limit })?;
    output.write_all(&raw)?;
    // The extra byte distinguishes a footer ending at the limit from one
    // extending beyond it. The total includes the fixed header.
    let copied = io::copy(&mut input.take(remaining + 1), output)?;
    if copied > remaining {
        return Err(MghDocumentError::ExpandedLimit { maximum: limit });
    }
    if copied < header.payload_bytes as u64 {
        return Err(MghDocumentError::Truncated {
            expected: header.payload_bytes,
            actual: copied,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
