//! Encoded-source preflight for NRRD payload reads.
//!
//! Both helpers run before payload allocation: the encoded-byte budget is
//! checked against the declared or observed source length, and detached
//! `data file` references are validated to stay inside the header's
//! directory. Neither helper retains payload bytes.

use ritk_image_io::{ImageReadBudget, ImageReadResource};
use std::io::ErrorKind;
use std::path::{Component, Path};

use super::super::super::stored::NrrdStoredReadError;

pub(super) fn check_encoded_source(
    file: &std::fs::File,
    data_start: u64,
    expected_payload_bytes: usize,
    byte_skip: i32,
    budget: ImageReadBudget,
) -> Result<(), NrrdStoredReadError> {
    let encoded_bytes = if byte_skip == -1 {
        u64::try_from(expected_payload_bytes).map_err(|_| {
            NrrdStoredReadError::PayloadLengthNotRepresentable {
                expected_bytes: expected_payload_bytes,
            }
        })?
    } else {
        let file_length = file
            .metadata()
            .map_err(|source| NrrdStoredReadError::PayloadIo { source })?
            .len();
        file_length
            .checked_sub(data_start)
            .ok_or_else(|| NrrdStoredReadError::PayloadIo {
                source: std::io::Error::new(
                    ErrorKind::UnexpectedEof,
                    "NRRD data source ended before the header parser position",
                ),
            })?
    };
    budget.check(ImageReadResource::EncodedBytes, encoded_bytes)?;
    Ok(())
}

pub(super) fn resolve_detached_data_path(
    header_path: &Path,
    data_file: &str,
) -> Result<std::path::PathBuf, NrrdStoredReadError> {
    if data_file.contains('%')
        || data_file
            .split_whitespace()
            .next()
            .is_some_and(|token| token.eq_ignore_ascii_case("LIST"))
    {
        return Err(NrrdStoredReadError::UnsupportedDetachedFileSet {
            data_file: data_file.to_owned(),
        });
    }
    let relative_path = Path::new(data_file);
    if relative_path.is_absolute()
        || relative_path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(NrrdStoredReadError::InvalidDetachedPath {
            data_file: data_file.to_owned(),
        });
    }
    Ok(header_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(relative_path))
}
