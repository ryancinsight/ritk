//! Flat, per-file DICOM anonymization and export.

use std::fs;
use std::io::{BufWriter, ErrorKind, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, ensure, Context, Result};
use ritk_dicom::dataset::DicomFile;

use super::{anonymize_object, verify, AnonymizeOptions};
use crate::format::dicom::export_source::{numbered_instance_path, read_source, Source};

/// Cumulative statistics for a `anonymize_dicom_directory` run.
#[derive(Debug, Clone)]
pub struct AnonymizeStats {
    /// Number of files recognised as valid DICOM and processed.
    pub file_count: usize,
    /// Number of DICOM files successfully anonymized and written.
    pub success_count: usize,
    /// Number of DICOM files that failed anonymization or writing.
    pub error_count: usize,
    /// Per-file error messages for failed files.
    pub errors: Vec<(PathBuf, String)>,
}

/// Anonymize the DICOM files directly in `input_dir`, writing to `output_dir`.
///
/// The destination and any missing parents are created as needed. Files are
/// processed in source-path order and exported as `000001.dcm`, `000002.dcm`,
/// and so on, skipping occupied names. New names contain only an ordinal and
/// the DICOM extension. Each export is encoded before its numbered name is
/// published, and publishing never replaces an existing destination entry.
///
/// Files that cannot be opened as DICOM are skipped silently and not counted.
/// A DICOMDIR is counted as a per-file error because this API does not export
/// the media file set. A parsed file that fails anonymization or writing is recorded in
/// [`AnonymizeStats::errors`] and logged at `WARN` level. Those diagnostics
/// include the source path and can contain patient identifiers.
///
/// # Errors
/// Returns an error if a directory cannot be prepared for export or both
/// paths refer to the same directory. Per-file errors are reported in the stats.
pub fn anonymize_dicom_directory(
    input_dir: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    options: &AnonymizeOptions,
) -> Result<AnonymizeStats> {
    let input_dir = input_dir.as_ref();
    let output_dir = output_dir.as_ref();
    let mut entries: Vec<PathBuf> = fs::read_dir(input_dir)
        .with_context(|| format!("Failed to read input directory {input_dir:?}"))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect();
    entries.sort_unstable();

    fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory {output_dir:?}"))?;
    let same_directory = same_file::is_same_file(input_dir, output_dir)
        .context("Failed to compare input and output directories")?;
    ensure!(
        !same_directory,
        "Input and output refer to the same directory"
    );

    let mut stats = AnonymizeStats {
        file_count: 0,
        success_count: 0,
        error_count: 0,
        errors: Vec::new(),
    };
    let mut ordinal = 0;

    for input_path in entries {
        let result = match read_source(&input_path) {
            Source::NotDicom => continue,
            Source::Index => Err(anyhow!(
                "DICOMDIR is unsupported in flat export; pass the directory containing the image instances"
            )),
            Source::Instance(object) => anonymize_object(*object, options).and_then(|(object, _)| {
                write_numbered_instance(&object, output_dir, &mut ordinal)
            }),
        };
        stats.file_count += 1;

        match result {
            Ok(()) => stats.success_count += 1,
            Err(error) => {
                tracing::warn!(path = ?input_path, error = %error, "anonymization failed");
                stats.error_count += 1;
                stats.errors.push((input_path, format!("{error:#}")));
            }
        }
    }

    Ok(stats)
}

/// Encode once, then claim the next available number without a check/write race.
fn write_numbered_instance(
    object: &DicomFile,
    output_dir: &Path,
    ordinal: &mut usize,
) -> Result<()> {
    let mut temporary = tempfile::NamedTempFile::new_in(output_dir)
        .context("Failed to create temporary anonymized DICOM file")?;
    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        object
            .write_all(&mut writer)
            .context("Failed to encode anonymized DICOM")?;
        writer.flush().context("Failed to write anonymized DICOM")?;
    }

    loop {
        *ordinal = ordinal
            .checked_add(1)
            .context("DICOM export numbering exhausted")?;
        let target = numbered_instance_path(output_dir, *ordinal);
        match temporary.persist_noclobber(&target) {
            Ok(_) => return Ok(()),
            Err(error)
                if error.error.kind() == ErrorKind::AlreadyExists
                    || fs::symlink_metadata(&target).is_ok() =>
            {
                temporary = error.file;
            }
            Err(error) => {
                return Err(error.error).with_context(|| format!("Failed to publish {target:?}"))
            }
        }
    }
}

/// Anonymize the files directly in a directory and verify every exported file.
///
/// Runs `anonymize_dicom_directory` and then a cross-file export gate
/// (`verify::ensure_dicom_directory_clean`) that asserts one Study/Series UID
/// set, unique SOPInstanceUIDs, and clean per-file metadata across the output
/// directory.
///
/// # Errors
/// Returns an error when the directory pass fails or verification finds any
/// issue in any exported file.
pub fn anonymize_dicom_directory_verified(
    input_dir: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    options: &AnonymizeOptions,
    verify_options: &verify::VerifyOptions,
) -> Result<AnonymizeStats> {
    let stats = anonymize_dicom_directory(&input_dir, &output_dir, options)?;
    verify::ensure_dicom_directory_clean(&output_dir, verify_options)?;
    Ok(stats)
}
