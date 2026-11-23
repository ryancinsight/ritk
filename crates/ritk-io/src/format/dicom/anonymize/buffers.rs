//! Atomic in-memory equivalent of a selected directory export.
use super::{anonymize_object, study::source_identifiers, verify, AnonymizeOptions};
use crate::format::dicom::export_source::numbered_instance_path;
use anyhow::{ensure, Result};
use ritk_dicom::{dataset::DicomFile, ParseBudget};
use std::{collections::BTreeSet, path::Path};

/// Export one admitted selection in source-path order with neutral names.
/// Serialized copies pass the directory export's per-object and cross-instance
/// checks before any bytes are returned. Supply a unique salt per export.
///
/// # Errors
/// Rejects empty selections, malformed instances, encoding and verification failures.
pub fn anonymize_dicom_buffers(
    mut files: Vec<(&Path, &[u8])>,
    options: &AnonymizeOptions,
) -> Result<Vec<(String, Vec<u8>)>> {
    ensure!(!files.is_empty(), "no instances selected");
    files.sort_by_key(|(path, _)| *path);
    let mut copies = Vec::with_capacity(files.len());
    let mut prohibited = BTreeSet::new();
    for (index, (_, bytes)) in files.into_iter().enumerate() {
        let source = DicomFile::parse(bytes.to_vec().into(), &ParseBudget::DEFAULT)?;
        source_identifiers(&source, options, &mut prohibited);
        let (object, _) = anonymize_object(source, options)?;
        let mut output = Vec::new();
        object.write_all(&mut output)?;
        let name = numbered_instance_path(Path::new(""), index + 1);
        copies.push((name.to_string_lossy().into_owned(), output));
    }
    let gate = verify::VerifyOptions {
        prohibited_values: prohibited.into_iter().collect(),
        ..Default::default()
    };
    let reports = copies
        .iter()
        .map(|(name, bytes)| {
            let obj = DicomFile::parse(bytes.clone().into(), &ParseBudget::DEFAULT)?;
            Ok(verify::verify_object(&obj, Path::new(name), &gate))
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        verify::instance_set::verify_reports(reports).is_empty(),
        "exported DICOM instances failed verification"
    );
    Ok(copies)
}
