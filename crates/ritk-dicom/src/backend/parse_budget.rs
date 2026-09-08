//! Resource-bounded DICOM parsing.
//!
//! The byte scanner runs before `dicom-rs` constructs an in-memory object. It
//! validates Part 10 framing, declared spans, sequence delimiters, element
//! count, and nesting against the existing Consus [`ParseBudget`].

mod scan;

use std::path::Path;

use anyhow::{Context, Result};
use consus_core::ParseBudget;

use super::DicomParseBackend;

pub use scan::{validate_part10, DicomParseSummary};

/// Read one DICOM file through the supplied byte ceiling.
///
/// The file is opened before metadata is queried and the returned bytes are
/// the only source passed to subsequent parsing. This bounds the allocation
/// before any parser sees the input; identity and replacement checks remain a
/// responsibility of the caller that selected the file-set member.
///
/// # Errors
///
/// Returns an error when the path cannot be opened or inspected, its length
/// exceeds the byte ceiling, or the bounded read fails.
pub fn read_file_with_budget<P: AsRef<Path>>(path: P, budget: &ParseBudget) -> Result<Vec<u8>> {
    use std::fs::File;

    let path = path.as_ref();
    let mut file =
        File::open(path).with_context(|| format!("failed to open DICOM file {:?}", path))?;
    let file_length = usize::try_from(
        file.metadata()
            .with_context(|| format!("failed to inspect DICOM file {:?}", path))?
            .len(),
    )
    .context("DICOM file length does not fit usize")?;
    budget
        .read_bounded(&mut file, file_length, "DICOM file bytes")
        .map_err(|error| anyhow::anyhow!("DICOM file exceeds parse budget: {error}"))
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
