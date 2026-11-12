use super::EditorFields;
use crate::format::dicom::anonymize::{anonymize_object, AnonymizationProfile, AnonymizeOptions};
use anyhow::{ensure, Context, Result};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{meta::FileMetaTableBuilder, open_file};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};

/// Receipt for a fully written and reparsed export, not an anonymity certificate.
#[derive(Debug, Clone)]
pub struct ExportReport {
    /// Number of source instances exported successfully.
    pub file_count: usize,
    /// Newly created directory containing the output instances.
    pub output_directory: PathBuf,
}

/// Export explicit input files into a new directory using neutral file names.
///
/// Applies the enhanced metadata profile and then the validated replacements.
/// UID remapping is consistent for a caller-supplied batch salt. Use a new,
/// nonempty salt per unrelated export; retaining it permits linking exports.
/// Pixels are preserved, not inspected or cleaned. This does not certify that
/// standard free-text attributes, burned-in text or anatomy are anonymous.
///
/// The directory must not exist. An `INCOMPLETE` marker remains until every
/// output has been reparsed and checked. Errors remove only the files created
/// by this invocation. If cleanup fails, the error explicitly reports incomplete
/// output; callers must not offer it as a successful export. A process crash can
/// leave this marker and partial files. Input files are never changed.
pub fn export_batch(
    inputs: &[PathBuf],
    output_directory: &Path,
    fields: &EditorFields,
    uid_salt: &str,
) -> Result<ExportReport> {
    let replacements = fields.elements()?;
    ensure!(!inputs.is_empty(), "Select at least one DICOM instance");
    ensure!(
        !uid_salt.trim().is_empty(),
        "A nonempty export UID salt is required"
    );
    fs::create_dir(output_directory)
        .context("Output must be a new directory with an existing parent")?;
    let mut created = Vec::new();
    let marker = output_directory.join("INCOMPLETE");
    let result = (|| -> Result<()> {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&marker)?;
        created.push(marker.clone());
        let options = AnonymizeOptions {
            profile: AnonymizationProfile::Enhanced,
            patient_id: fields.patient_id.clone(),
            patient_name: fields.patient_name.clone(),
            uid_salt: uid_salt.to_owned(),
            ..Default::default()
        };
        let mut sop_uids = HashSet::new();
        for (index, input) in inputs.iter().enumerate() {
            let source = open_file(input)
                .with_context(|| format!("Input instance {} is not readable DICOM", index + 1))?;
            let transfer_syntax = source.meta().transfer_syntax.clone();
            let sop_class = source.meta().media_storage_sop_class_uid.clone();
            let original_pixels = source.element(Tag(0x7fe0, 0x0010)).ok().cloned();
            let (mut object, _) = anonymize_object(source, &options)?;
            for (tag, vr, value) in &replacements {
                let value = if value.is_empty() {
                    PrimitiveValue::Empty
                } else {
                    PrimitiveValue::from(value.as_str())
                };
                object.put(DataElement::new(*tag, *vr, value));
            }
            // Explicitly avoid claiming that operator-entered demographics and
            // uninspected pixels meet a complete de-identification profile.
            object.put(DataElement::new(
                Tag(0x0012, 0x0062),
                VR::CS,
                PrimitiveValue::from("NO"),
            ));
            object.put(DataElement::new(
                Tag(0x0012, 0x0063),
                VR::LO,
                PrimitiveValue::from("Metadata pseudonymization; pixels and metadata need review"),
            ));
            let sop_uid = object
                .element(Tag(0x0008, 0x0018))
                .context("SOP Instance UID is required")?
                .to_str()?
                .into_owned();
            ensure!(
                !sop_uid.is_empty() && sop_uids.insert(sop_uid.clone()),
                "Missing or duplicate SOP Instance UID in batch"
            );
            // Rebuild file meta so source application names and private meta
            // information are not copied, and the SOP instance matches the body.
            let object = object.into_inner().with_meta(
                FileMetaTableBuilder::new()
                    .transfer_syntax(transfer_syntax)
                    .media_storage_sop_class_uid(sop_class)
                    .media_storage_sop_instance_uid(sop_uid.clone()),
            )?;
            let output = output_directory.join(format!("instance-{:06}.dcm", index + 1));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&output)?;
            created.push(output.clone());
            object.write_all(&mut file)?;
            file.sync_all()?;
            drop(file);
            let reopened = open_file(&output).context("Written output did not reparse")?;
            ensure!(
                reopened
                    .meta()
                    .media_storage_sop_instance_uid
                    .trim_end_matches('\0')
                    == sop_uid,
                "Output meta SOP instance does not match data set"
            );
            for (tag, _, expected) in &replacements {
                let actual = reopened.element(*tag)?.to_str()?;
                ensure!(
                    actual.trim_end_matches([' ', '\0']) == expected.trim_end_matches(' '),
                    "Written replacement did not round-trip at {tag}"
                );
            }
            ensure!(
                reopened.element(Tag(0x7fe0, 0x0010)).ok().cloned() == original_pixels,
                "Pixel data changed during export"
            );
        }
        fs::remove_file(&marker)?;
        Ok(())
    })();
    if let Err(error) = result {
        let mut cleanup_ok = true;
        for path in created.iter().rev().filter(|path| **path != marker) {
            if let Err(e) = fs::remove_file(path)
                && e.kind() != std::io::ErrorKind::NotFound
            {
                cleanup_ok = false;
            }
        }
        if cleanup_ok
            && created.contains(&marker)
            && let Err(e) = fs::remove_file(&marker)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            cleanup_ok = false;
        }
        if cleanup_ok && fs::remove_dir(output_directory).is_err() {
            cleanup_ok = false;
            // A late cleanup failure must remain visibly incomplete. Never
            // overwrite a concurrently created marker or other file.
            let _ = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&marker);
        }
        return if cleanup_ok {
            Err(error)
        } else {
            Err(error.context("Export failed; incomplete output could not be fully removed. Do not use the output directory"))
        };
    }
    Ok(ExportReport {
        file_count: inputs.len(),
        output_directory: output_directory.to_owned(),
    })
}
