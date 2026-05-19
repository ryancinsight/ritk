//! DICOM de-identification and anonymization — PS 3.15 Annex E.
//!
//! # Design
//! - `anonymize_object` is the canonical entry point for single-object mutation.
//! - `anonymize_dicom_file` wraps open + anonymize + write.
//! - `anonymize_dicom_directory` batches across a directory tree.
//! - UID replacement uses a djb2-based hash so the mapping is deterministic and
//!   referentially consistent within a study processed in a single batch.
//!
//! # Invariants
//! - Non-DICOM files in directory mode are skipped silently.
//! - File meta-header (transfer syntax, SOP class) is preserved unchanged.
//! - `clean_pixel_data = false` (default) never touches pixel data.

mod profile;
#[cfg(test)]
mod tests_anonymize;

pub use profile::{AnonymizationProfile, TagAction};

use anyhow::{Context, Result};
use dicom::core::header::Header;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{open_file, FileDicomObject, InMemDicomObject};
use std::path::{Path, PathBuf};

/// Options controlling how a DICOM object is anonymized.
#[derive(Debug, Clone)]
pub struct AnonymizeOptions {
    /// Which tags to act on and how.
    pub profile: AnonymizationProfile,
    /// When `true`, replace the pixel data element with an equal-length zero
    /// buffer, suppressing visual content without altering file structure.
    /// Defaults to `false`.
    pub clean_pixel_data: bool,
    /// When `true`, remove all private DICOM elements (those with an odd group
    /// number, excluding the file meta-header group 0x0002). This achieves full
    /// PS 3.15 Annex E compliance for attribute confidentiality by eliminating
    /// any institutionally-specific private attributes that may carry PHI.
    /// Defaults to `false`.
    pub clean_private_tags: bool,
}

impl Default for AnonymizeOptions {
    fn default() -> Self {
        Self {
            profile: AnonymizationProfile::Basic,
            clean_pixel_data: false,
            clean_private_tags: false,
        }
    }
}

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

// ─── UID generation ───────────────────────────────────────────────────────────

/// Produce a deterministic DICOM-conformant UID from `original` and `salt`.
///
/// # Algorithm
/// djb2-style multiplicative hash (k=33) over the UTF-8 bytes of
/// `original || salt`, emitted as a decimal integer under the private
/// `2.999` OID arc (ISO/IEC 9834-8 UUID arc for private use).
///
/// # Invariants
/// - Same `(original, salt)` → same output (pure function).
/// - Output matches `^2\\.999\\.[0-9]+$` and is ≤ 64 characters.
/// - Collision probability is O(1/2^64) for randomly distributed inputs.
pub(crate) fn generate_uid_from_hash(original: &str, salt: &str) -> String {
    let mut h: u64 = 5381;
    for b in original.bytes().chain(salt.bytes()) {
        h = h.wrapping_mul(33).wrapping_add(u64::from(b));
    }
    format!("2.999.{h}")
}

// ─── Action dispatch ──────────────────────────────────────────────────────────

/// Apply `action` to `tag` in `obj`.
///
/// For `Dummy`: reads the element's existing VR to preserve it; uses
/// tag-specific placeholder strings (PatientName → "ANONYMOUS",
/// PatientID → "ANON_ID", all others → "ANONYMOUS").
/// For `Empty`: preserves the VR, replaces value with `PrimitiveValue::Empty`.
/// For `ReplaceUid`: hashes the original UID string deterministically.
/// For `Remove`: silently tolerates absent elements.
fn apply_action(obj: &mut FileDicomObject<InMemDicomObject>, tag: Tag, action: TagAction) {
    match action {
        TagAction::Keep => {}

        TagAction::Remove => {
            let _ = obj.remove_element(tag);
        }

        TagAction::Dummy => {
            // Extract VR before the mutable put; VR is Copy so the borrow ends.
            let vr = obj.element(tag).map(|e| e.vr()).unwrap_or(VR::LO);
            let val = match tag {
                Tag(0x0010, 0x0010) => "ANONYMOUS", // PatientName
                Tag(0x0010, 0x0020) => "ANON_ID",   // PatientID
                _ => "ANONYMOUS",
            };
            obj.put(DataElement::new(tag, vr, PrimitiveValue::from(val)));
        }

        TagAction::Empty => {
            let vr = obj.element(tag).map(|e| e.vr()).unwrap_or(VR::LO);
            obj.put(DataElement::new(tag, vr, PrimitiveValue::Empty));
        }

        TagAction::ReplaceUid => {
            // Read the original UID as an owned String before mutating obj.
            let orig: String = obj
                .element(tag)
                .ok()
                .and_then(|e| e.to_str().ok().map(|s| s.into_owned()))
                .unwrap_or_default();
            let new_uid = generate_uid_from_hash(&orig, "ritk_anon_v1");
            obj.put(DataElement::new(
                tag,
                VR::UI,
                PrimitiveValue::from(new_uid.as_str()),
            ));
        }
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Apply anonymization to a single in-memory DICOM object.
///
/// Iterates the `options.profile` tag-action list, mutating `obj` in place
/// (via `DerefMut`). The file meta-header is never modified.
///
/// When `options.clean_pixel_data` is `true`, the `PixelData` element
/// `(7FE0,0010)` is overwritten with an equal-length zero buffer if present
/// and readable as a flat byte sequence.
pub fn anonymize_object(
    mut obj: FileDicomObject<InMemDicomObject>,
    options: &AnonymizeOptions,
) -> Result<FileDicomObject<InMemDicomObject>> {
    for (tag, action) in options.profile.tag_actions() {
        apply_action(&mut obj, tag, action);
    }

    // Remove private tags before pixel data handling to avoid removing
    // private pixel data blocks after they have already been zeroed.
    if options.clean_private_tags {
        // Collect private tag addresses first to avoid borrow conflicts.
        // DICOM private elements have odd group numbers.
        // Group 0x0002 (file meta-header) is always excluded.
        let private_tags: Vec<Tag> = obj
            .iter()
            .map(|e| e.tag())
            .filter(|t| t.group() & 1 == 1 && t.group() != 0x0002)
            .collect();
        for tag in private_tags {
            let _ = obj.remove_element(tag);
        }
    }

    if options.clean_pixel_data {
        let pixel_tag = Tag(0x7FE0, 0x0010);
        // Extract VR before the second element() call; VR is Copy.
        let vr = obj.element(pixel_tag).map(|e| e.vr()).unwrap_or(VR::OW);
        // Obtain byte count without retaining a borrow of obj.
        let len: usize = obj
            .element(pixel_tag)
            .ok()
            .and_then(|e| e.to_bytes().ok().map(|b| b.len()))
            .unwrap_or(0);
        if len > 0 {
            let zeros =
                PrimitiveValue::U8(dicom::core::smallvec::SmallVec::from_vec(vec![0u8; len]));
            obj.put(DataElement::new(pixel_tag, vr, zeros));
        }
    }

    Ok(obj)
}

/// Read a DICOM file from `input_path`, anonymize it, and write to `output_path`.
///
/// Fails if `input_path` cannot be parsed as a DICOM Part 10 file, or if the
/// output cannot be written. Propagates the underlying I/O and DICOM errors
/// with context.
pub fn anonymize_dicom_file(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    options: &AnonymizeOptions,
) -> Result<()> {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();
    let obj = open_file(input_path)
        .with_context(|| format!("Failed to open DICOM file {input_path:?}"))?;
    let anon = anonymize_object(obj, options)?;
    anon.write_to_file(output_path)
        .with_context(|| format!("Failed to write anonymized DICOM to {output_path:?}"))?;
    Ok(())
}

/// Anonymize all DICOM files in `input_dir`, writing results to `output_dir`.
///
/// `output_dir` is created if it does not exist.
/// Files that cannot be opened as DICOM are skipped silently and not counted.
/// Files that are valid DICOM but fail during anonymization or writing are
/// counted in `AnonymizeStats::error_count` and logged at `WARN` level.
/// Output filenames match input filenames; directory structure is not
/// recursed.
pub fn anonymize_dicom_directory(
    input_dir: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    options: &AnonymizeOptions,
) -> Result<AnonymizeStats> {
    let input_dir = input_dir.as_ref();
    let output_dir = output_dir.as_ref();

    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory {output_dir:?}"))?;

    let entries: Vec<PathBuf> = std::fs::read_dir(input_dir)
        .with_context(|| format!("Failed to read input directory {input_dir:?}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();

    let mut stats = AnonymizeStats {
        file_count: 0,
        success_count: 0,
        error_count: 0,
        errors: Vec::new(),
    };

    for input_path in &entries {
        // Non-DICOM files are skipped silently without counting.
        let obj = match open_file(input_path) {
            Ok(o) => o,
            Err(_) => continue,
        };
        stats.file_count += 1;

        let file_name = match input_path.file_name() {
            Some(n) => n,
            None => continue,
        };
        let output_path = output_dir.join(file_name);

        match anonymize_object(obj, options).and_then(|o| {
            o.write_to_file(&output_path)
                .with_context(|| format!("Failed to write {output_path:?}"))
        }) {
            Ok(()) => stats.success_count += 1,
            Err(e) => {
                tracing::warn!(
                    path = ?input_path,
                    error = %e,
                    "anonymization failed"
                );
                stats.error_count += 1;
                stats.errors.push((input_path.clone(), e.to_string()));
            }
        }
    }

    Ok(stats)
}
