//! Resource-bounded DICOM parsing.
//!
//! The byte scanner runs before `dicom-rs` constructs an in-memory object. It
//! validates Part 10 framing, declared spans, sequence delimiters, element
//! count, and nesting against the existing Consus [`ParseBudget`].

mod scan;

use std::fs::{self, File, Metadata};
use std::path::Path;

use anyhow::{Context, Result};
use consus_core::ParseBudget;

use super::DicomParseBackend;

pub use scan::{validate_part10, DicomParseSummary};

/// Read one DICOM file through the supplied byte ceiling.
///
/// The requested path is opened once with the platform's final-component
/// policy, the handle identity is checked against its resolved path, and the
/// returned bytes are read from that same handle. This bounds the allocation
/// before any parser sees the input and detects a file replacement between
/// opening and handle inspection. Callers should retain the returned bytes
/// across later decode stages.
///
/// # Errors
///
/// Returns an error when the path cannot be opened or inspected, its length
/// exceeds the byte ceiling, or the bounded read fails.
pub fn read_file_with_budget<P: AsRef<Path>>(path: P, budget: &ParseBudget) -> Result<Vec<u8>> {
    let requested_path = path.as_ref();
    let mut file = open_read_handle(requested_path)?;
    let path = requested_path
        .canonicalize()
        .with_context(|| format!("failed to resolve DICOM file path {:?}", requested_path))?;
    let handle_metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect DICOM file handle {:?}", path))?;
    let path_metadata = fs::metadata(&path)
        .with_context(|| format!("failed to inspect resolved DICOM file {:?}", path))?;
    if !same_file_identity(&handle_metadata, &path_metadata) {
        anyhow::bail!("DICOM file identity changed while opening {:?}", path);
    }
    let file_length =
        usize::try_from(handle_metadata.len()).context("DICOM file length does not fit usize")?;
    budget
        .read_bounded(&mut file, file_length, "DICOM file bytes")
        .map_err(|error| anyhow::anyhow!("DICOM file exceeds parse budget: {error}"))
}

#[cfg(windows)]
fn open_read_handle(path: &Path) -> Result<File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);

    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};

    // CreateFileW's OPEN_REPARSE_POINT flag prevents following a final
    // reparse point. The handle is rejected if the opened object is one.
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    let file = options
        .open(path)
        .with_context(|| format!("failed to open DICOM file {:?}", path))?;
    if file.metadata()?.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(anyhow::anyhow!(
            "DICOM file path resolves to a Windows reparse point: {:?}",
            path
        ));
    }
    Ok(file)
}

#[cfg(not(windows))]
fn open_read_handle(path: &Path) -> Result<File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }

    options
        .open(path)
        .with_context(|| format!("failed to open DICOM file {:?}", path))
}

#[cfg(unix)]
fn same_file_identity(handle: &Metadata, path: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    handle.dev() == path.dev() && handle.ino() == path.ino()
}

#[cfg(windows)]
fn same_file_identity(handle: &Metadata, path: &Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    handle.file_type() == path.file_type()
        && handle.file_size() == path.file_size()
        && handle.creation_time() == path.creation_time()
        && handle.last_write_time() == path.last_write_time()
        && handle.file_attributes() == path.file_attributes()
}

#[cfg(not(any(unix, windows)))]
fn same_file_identity(handle: &Metadata, path: &Metadata) -> bool {
    handle.file_type() == path.file_type()
        && handle.len() == path.len()
        && handle.modified().ok() == path.modified().ok()
}

/// Parsing backend which can enforce a [`ParseBudget`] before materialization.
pub trait BoundedDicomParseBackend: DicomParseBackend {
    /// Parse a file after checking its encoded bytes and DICOM structure.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, the input violates the
    /// Part 10 structure, or any resource ceiling is exceeded.
    fn parse_file_with_budget(path: &Path, budget: &ParseBudget) -> Result<Self::Object>;

    /// Parse bytes after checking their encoded bytes and DICOM structure.
    ///
    /// # Errors
    ///
    /// Returns an error when the input violates the Part 10 structure or any
    /// resource ceiling is exceeded, or when the backend cannot materialize
    /// the validated data set.
    fn parse_bytes_with_budget(data: &[u8], budget: &ParseBudget) -> Result<Self::Object>;
}

/// Parse a file through a budget-aware DICOM backend.
pub fn parse_file_with_budget<B, P>(
    path: P,
    budget: &ParseBudget,
) -> Result<<B as DicomParseBackend>::Object>
where
    B: BoundedDicomParseBackend,
    P: AsRef<Path>,
{
    B::parse_file_with_budget(path.as_ref(), budget)
}

/// Parse bytes through a budget-aware DICOM backend.
pub fn parse_bytes_with_budget<B>(
    data: &[u8],
    budget: &ParseBudget,
) -> Result<<B as DicomParseBackend>::Object>
where
    B: BoundedDicomParseBackend,
{
    B::parse_bytes_with_budget(data, budget)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

    use super::same_file_identity;

    #[cfg(unix)]
    use super::read_file_with_budget;
    #[cfg(unix)]
    use consus_core::ParseBudget;

    #[test]
    fn file_identity_detects_replaced_path() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("slice.dcm");
        std::fs::write(&path, [1_u8]).unwrap();
        let handle = std::fs::File::open(&path).unwrap();
        let handle_metadata = handle.metadata().unwrap();

        std::fs::write(&path, [1_u8, 2_u8]).unwrap();
        let path_metadata = std::fs::metadata(&path).unwrap();

        assert!(
            !same_file_identity(&handle_metadata, &path_metadata),
            "a replaced path must not match the original open handle"
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_final_symlink_before_canonicalization() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("slice.dcm");
        let link = directory.path().join("selected.dcm");
        std::fs::write(&target, [1_u8, 2_u8, 3_u8]).unwrap();
        symlink(&target, &link).unwrap();

        let error = read_file_with_budget(&link, &ParseBudget::DEFAULT).unwrap_err();
        assert!(
            error.to_string().contains("failed to open DICOM file"),
            "final symlink must be rejected by the no-follow open: {error}"
        );
    }
}
