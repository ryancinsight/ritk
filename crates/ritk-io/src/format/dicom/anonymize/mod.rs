//! DICOM de-identification and anonymization — PS 3.15 Annex E.
//!
//! # Design
//! - `anonymize_object` is the canonical entry point for single-object mutation.
//! - `anonymize_dicom_file` wraps open + anonymize + write.
//! - `anonymize_dicom_directory` batches across a directory tree.
//! - UID replacement uses SHA-256 deterministic hashing so the mapping is
//!   cryptographically irreversible without the salt and referentially
//!   consistent within a study processed in a single batch.
//!
//! # Invariants
//! - Non-DICOM files in directory mode are skipped silently.
//! - File meta-header (transfer syntax, SOP class) is preserved unchanged.
//! - `clean_pixel_data = false` (default) never touches pixel data.
//! - Same `(original_uid, salt)` always produces the same replacement UID.

mod profile;
#[cfg(test)]
mod tests_anonymize;

pub use profile::{AnonymizationProfile, TagAction};

use anyhow::{Context, Result};
use dicom::core::header::Header;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{open_file, FileDicomObject, InMemDicomObject};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Options controlling how a DICOM object is anonymized.
#[derive(Debug, Clone)]
pub struct AnonymizeOptions {
    /// Which tags to act on and how.
    pub profile: AnonymizationProfile,
    /// Replacement value for PatientName (0010,0010).
    /// Default: `"ANONYMOUS"`.
    pub patient_name: String,
    /// Replacement value for PatientID (0010,0020).
    /// Default: `"ANON001"`.
    pub patient_id: String,
    /// Salt for deterministic UID remapping (SHA-256).
    /// Default: `"ritk-anon-salt"`.
    pub uid_salt: String,
    /// When `true`, replace the pixel data element with an equal-length zero
    /// buffer, suppressing visual content without altering file structure.
    /// Defaults to `false`.
    pub clean_pixel_data: bool,
    /// When `true`, remove all private DICOM elements (those with an odd group
    /// number, excluding the file meta-header group 0x0002). This achieves full
    /// PS 3.15 Annex E compliance for attribute confidentiality by eliminating
    /// any institutionally-specific private attributes that may carry PHI.
    /// Defaults to `false`. The `Enhanced` profile overrides this to `true`.
    pub clean_private_tags: bool,
}

impl Default for AnonymizeOptions {
    fn default() -> Self {
        Self {
            profile: AnonymizationProfile::Basic,
            patient_name: "ANONYMOUS".to_owned(),
            patient_id: "ANON001".to_owned(),
            uid_salt: "ritk-anon-salt".to_owned(),
            clean_pixel_data: false,
            clean_private_tags: false,
        }
    }
}

/// Per-object statistics returned by `anonymize_object`.
#[derive(Debug, Clone, Default)]
pub struct AnonymizeResult {
    /// Number of tags deleted (Remove action applied to a present element).
    pub tags_deleted: usize,
    /// Number of tags zeroed (Empty action applied to a present element).
    pub tags_zeroed: usize,
    /// Number of UIDs remapped (ReplaceUid action applied to a present element).
    pub uids_remapped: usize,
    /// Number of private tags removed.
    pub private_tags_removed: usize,
    /// Map of original UID → replacement UID for cross-reference tracking.
    pub uid_map: HashMap<String, String>,
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
/// SHA-256 over `original || "||" || salt`, the first 19 bytes of the digest
/// are converted to decimal digits (each byte → 2 or 3 decimal digits),
/// producing at most 57 decimal characters. The result is prefixed with
/// `"2.25."` (ISO/IEC 9834-8 UUID arc), yielding a UID ≤ 64 characters.
///
/// # Invariants
/// - Same `(original, salt)` → same output (pure function).
/// - Output matches `^2\.25\.[0-9]+$` and is ≤ 64 characters.
/// - Original UID cannot be recovered without the salt.
/// - Collision probability is O(1/2^152) (19 bytes of SHA-256 entropy).
pub(crate) fn generate_uid_from_hash(original: &str, salt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(original.as_bytes());
    hasher.update(b"||");
    hasher.update(salt.as_bytes());
    let digest = hasher.finalize();

    // Convert first 19 bytes of SHA-256 digest to decimal string.
    // Each byte [0,255] → 2 or 3 decimal digits → max 57 characters.
    // With "2.25." prefix (5 chars), total ≤ 62 characters < 64 max.
    let decimal: String = digest[..19]
        .iter()
        .flat_map(|b| {
            let d = u32::from(*b);
            if d < 10 {
                vec![b'0', b'0', b'0' + (d as u8)]
            } else if d < 100 {
                let tens = (d / 10) as u8;
                let ones = (d % 10) as u8;
                vec![b'0', b'0' + tens, b'0' + ones]
            } else {
                let hundreds = (d / 100) as u8;
                let rem = (d % 100) as u8;
                let tens = rem / 10;
                let ones = rem % 10;
                vec![b'0' + hundreds, b'0' + tens, b'0' + ones]
            }
        })
        .map(|b| b as char)
        .collect();

    // Strip leading zeros from the decimal portion to avoid UID components
    // with leading zeros (DICOM forbids leading-zero components after the root).
    let decimal_stripped = decimal.trim_start_matches('0');
    let uid_body = if decimal_stripped.is_empty() { "0" } else { decimal_stripped };
    let uid = format!("2.25.{uid_body}");

    debug_assert!(
        uid.len() <= 64,
        "Generated UID exceeds 64-char DICOM limit: {} (len={})",
        uid,
        uid.len()
    );
    debug_assert!(
        uid.chars().all(|c| c.is_ascii_digit() || c == '.'),
        "Generated UID contains invalid characters: {uid}"
    );
    debug_assert!(
        !uid.contains(".00"),
        "Generated UID has leading-zero component: {uid}"
    );

    uid
}

// ─── Action dispatch ──────────────────────────────────────────────────────────

/// Apply `action` to `tag` in `obj`, tracking statistics in `result`.
///
/// For `Dummy`: reads the element's existing VR to preserve it; uses
/// tag-specific placeholder strings controlled by `AnonymizeOptions`.
/// For `Empty`: preserves the VR, replaces value with `PrimitiveValue::Empty`.
/// For `ReplaceUid`: hashes the original UID string deterministically.
/// For `Remove`: silently tolerates absent elements.
fn apply_action(
    obj: &mut FileDicomObject<InMemDicomObject>,
    tag: Tag,
    action: TagAction,
    opts: &AnonymizeOptions,
    result: &mut AnonymizeResult,
    uid_map: &mut HashMap<String, String>,
) {
    match action {
        TagAction::Keep => {}
        TagAction::Remove => {
            if obj.remove_element(tag) {
                result.tags_deleted += 1;
            }
        }
        TagAction::Dummy => {
            let was_present = obj.element(tag).is_ok();
            // Extract VR before the mutable put; VR is Copy so the borrow ends.
            let vr = obj.element(tag).map(|e| e.vr()).unwrap_or(VR::LO);
            let val: &str = match tag {
                Tag(0x0010, 0x0010) => &opts.patient_name, // PatientName
                Tag(0x0010, 0x0020) => &opts.patient_id,  // PatientID
                _ => &opts.patient_name,                   // default dummy
            };
            obj.put(DataElement::new(tag, vr, PrimitiveValue::from(val)));
            if was_present {
                // Dummy replaces; count as zeroed (value suppressed).
                result.tags_zeroed += 1;
            }
        }
        TagAction::Empty => {
            let was_present = obj.element(tag).is_ok();
            let vr = obj.element(tag).map(|e| e.vr()).unwrap_or(VR::LO);
            obj.put(DataElement::new(tag, vr, PrimitiveValue::Empty));
            if was_present {
                result.tags_zeroed += 1;
            }
        }
        TagAction::ReplaceUid => {
            // Read the original UID as an owned String before mutating obj.
            let orig: String = obj
                .element(tag)
                .ok()
                .and_then(|e| e.to_str().ok().map(|s| s.into_owned()))
                .unwrap_or_default();

            if orig.is_empty() {
                return;
            }

            // Deterministic UID: check existing map for cross-reference consistency.
            let new_uid = if let Some(existing) = uid_map.get(&orig) {
                existing.clone()
            } else {
                let generated = generate_uid_from_hash(&orig, &opts.uid_salt);
                uid_map.insert(orig.clone(), generated.clone());
                generated
            };

            obj.put(DataElement::new(
                tag,
                VR::UI,
                PrimitiveValue::from(new_uid.as_str()),
            ));
            result.uids_remapped += 1;
        }
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Apply anonymization to a single in-memory DICOM object.
///
/// Iterates the `options.profile` tag-action list, mutating `obj` in place
/// (via `DerefMut`). The file meta-header is never modified. Returns
/// `AnonymizeResult` with per-operation statistics and the UID cross-reference
/// map.
///
/// When `options.clean_pixel_data` is `true`, the `PixelData` element
/// `(7FE0,0010)` is overwritten with an equal-length zero buffer if present
/// and readable as a flat byte sequence.
pub fn anonymize_object(
    mut obj: FileDicomObject<InMemDicomObject>,
    options: &AnonymizeOptions,
) -> Result<(FileDicomObject<InMemDicomObject>, AnonymizeResult)> {
    let mut result = AnonymizeResult::default();
    let mut uid_map: HashMap<String, String> = HashMap::new();

    for (tag, action) in options.profile.tag_actions() {
        apply_action(&mut obj, tag, action, options, &mut result, &mut uid_map);
    }

    // Remove private tags before pixel data handling to avoid removing
    // private pixel data blocks after they have already been zeroed.
    if options.clean_private_tags || options.profile.removes_private_tags() {
        // Collect private tag addresses first to avoid borrow conflicts.
        // DICOM private elements have odd group numbers.
        // Group 0x0002 (file meta-header) is always excluded.
        let private_tags: Vec<Tag> = obj
            .iter()
            .map(|e| e.tag())
            .filter(|t| t.group() & 1 == 1 && t.group() != 0x0002)
            .collect();
        result.private_tags_removed = private_tags.len();
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

    result.uid_map = uid_map;
    Ok((obj, result))
}

/// Read a DICOM file from `input_path`, anonymize it, and write to `output_path`.
///
/// Fails if `input_path` cannot be parsed as a DICOM Part 10 file, or if the
/// output cannot be written. Propagates the underlying I/O and DICOM errors
/// with context. Returns `AnonymizeResult` with per-operation statistics.
pub fn anonymize_dicom_file(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    options: &AnonymizeOptions,
) -> Result<AnonymizeResult> {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();
    let obj = open_file(input_path)
        .with_context(|| format!("Failed to open DICOM file {input_path:?}"))?;
    let (anon, result) = anonymize_object(obj, options)?;
    anon.write_to_file(output_path)
        .with_context(|| format!("Failed to write anonymized DICOM to {output_path:?}"))?;
    Ok(result)
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

        match anonymize_object(obj, options).and_then(|(o, _)| {
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
