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
//! - Tag actions and private-tag removal are applied at every sequence nesting
//!   level, not only to top-level elements.
//! - File meta-header (transfer syntax, SOP class) is preserved unchanged.
//! - `clean_pixel_data = CleaningPolicy::Skip` (default) never touches pixel data.
//! - Same `(original_uid, salt)` always produces the same replacement UID.
//! - Every count in [`AnonymizeResult`] is taken from the mutation that produced
//!   it, never from a prior survey of candidates, so the report can only
//!   under-state what happened, never over-state it.
//! - A step that cannot complete aborts the run with [`AnonymizeError`]; the
//!   object it would have described is never returned as a success.

mod profile;
#[cfg(test)]
mod tests_anonymize;
#[cfg(test)]
#[path = "tests_anonymize_extended.rs"]
mod tests_anonymize_extended;
#[cfg(test)]
#[path = "tests_anonymize_stats.rs"]
mod tests_anonymize_stats;
#[cfg(test)]
#[path = "tests_recursion.rs"]
mod tests_recursion;
#[cfg(test)]
#[path = "tests_verify.rs"]
mod tests_verify;
pub mod verify;

pub use profile::{AnonymizationProfile, TagAction};

/// Controls whether an anonymization step is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CleaningPolicy {
    /// Do not clean this category.
    #[default]
    Skip,
    /// Remove or zero-fill this category.
    Clean,
}

use anyhow::{Context, Result};
use dicom::core::header::Header;
use dicom::core::value::{DataSetSequence, Value};
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
    /// When `CleaningPolicy::Clean`, replace the pixel data element with an
    /// equal-length zero buffer, suppressing visual content without altering
    /// file structure. Defaults to `CleaningPolicy::Skip`.
    pub clean_pixel_data: CleaningPolicy,
    /// When `CleaningPolicy::Clean`, remove all private DICOM elements (those
    /// with an odd group number, excluding the file meta-header group 0x0002).
    /// This achieves full PS 3.15 Annex E compliance for attribute
    /// confidentiality by eliminating institutionally-specific private
    /// attributes that may carry PHI.
    /// Defaults to `CleaningPolicy::Skip`. The `Enhanced` profile sets this
    /// to `CleaningPolicy::Clean` automatically.
    pub clean_private_tags: CleaningPolicy,
}

impl Default for AnonymizeOptions {
    fn default() -> Self {
        Self {
            profile: AnonymizationProfile::Basic,
            patient_name: String::from("ANONYMOUS"),
            patient_id: String::from("ANON001"),
            uid_salt: String::from("ritk-anon-salt"),
            clean_pixel_data: CleaningPolicy::Skip,
            clean_private_tags: CleaningPolicy::Skip,
        }
    }
}

/// Per-object statistics returned by `anonymize_object`.
///
/// Each count is incremented by the mutation that performed the work, not by a
/// preceding survey of candidates, so a figure here is a record of what the
/// object actually lost.
#[derive(Debug, Clone, Default)]
pub struct AnonymizeResult {
    /// Number of tags deleted (Remove action applied to a present element).
    pub tags_deleted: usize,
    /// Number of tags zeroed (Empty action applied to a present element).
    pub tags_zeroed: usize,
    /// Number of UIDs remapped (ReplaceUid action applied to a present element).
    pub uids_remapped: usize,
    /// Number of private elements actually removed from the object.
    pub private_tags_removed: usize,
    /// Map of original UID → replacement UID for cross-reference tracking.
    pub uid_map: HashMap<String, String>,
}

/// A de-identification step that could not be completed.
///
/// [`AnonymizeResult`] is the evidence a caller relies on to certify a data set
/// as de-identified, so a step that leaves identifying data behind aborts the
/// run rather than being folded into a success report: a report that overstates
/// de-identification is worse than no report. Every variant means the object
/// still holds data the profile was asked to destroy, and the object is
/// therefore never returned.
///
/// The variants are surfaced through the `anyhow::Error` returned by
/// [`anonymize_object`] and its file/directory wrappers; recover the structured
/// form with `anyhow::Error::downcast_ref::<AnonymizeError>`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AnonymizeError {
    /// A private element present in the data set was not removed by the
    /// removal pass, so it survives in the output.
    #[error("private element {tag} at sequence depth {depth} survived removal")]
    PrivateTagRetained {
        /// Address of the element that survived.
        tag: Tag,
        /// Nesting level it was found at; `0` is the top-level data set.
        depth: u32,
    },
    /// A sequence element could not be read back as a sequence for traversal,
    /// so the data sets nested inside it were never visited.
    #[error(
        "sequence {tag} at sequence depth {depth} could not be traversed; \
         the data sets nested inside it were not anonymized"
    )]
    SequenceNotTraversed {
        /// Address of the sequence that could not be traversed.
        tag: Tag,
        /// Nesting level it was found at; `0` is the top-level data set.
        depth: u32,
    },
    /// Nesting reached the traversal bound while sequences remained below it,
    /// so those data sets were left untouched.
    #[error(
        "sequence nesting reached the traversal bound at depth {depth}; \
         the data sets below it were not anonymized"
    )]
    SequenceTooDeep {
        /// Depth at which traversal stopped.
        depth: u32,
    },
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
    let uid_body = if decimal_stripped.is_empty() {
        "0"
    } else {
        decimal_stripped
    };
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
    obj: &mut InMemDicomObject,
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
            // Extract VR before the mutable put; VR is Copy so the borrow ends.
            let vr = obj.element(tag).map(|e| e.vr()).unwrap_or(VR::LO);
            let val: &str = match tag {
                Tag(0x0010, 0x0010) => &opts.patient_name, // PatientName
                Tag(0x0010, 0x0020) => &opts.patient_id,   // PatientID
                _ => &opts.patient_name,                   // default dummy
            };
            // `put` yields the element it displaced, so a suppressed value is
            // counted from the replacement itself rather than from a presence
            // check taken beforehand. Dummy replaces; count as zeroed.
            if obj
                .put(DataElement::new(tag, vr, PrimitiveValue::from(val)))
                .is_some()
            {
                result.tags_zeroed += 1;
            }
        }
        TagAction::Empty => {
            let vr = obj.element(tag).map(|e| e.vr()).unwrap_or(VR::LO);
            if obj
                .put(DataElement::new(tag, vr, PrimitiveValue::Empty))
                .is_some()
            {
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

            // Deterministic UID: entry API avoids double-lookup and reduces clones
            // to one key clone + one value clone on extraction.
            let new_uid = uid_map
                .entry(orig.clone())
                .or_insert_with(|| generate_uid_from_hash(&orig, &opts.uid_salt))
                .clone();

            obj.put(DataElement::new(
                tag,
                VR::UI,
                PrimitiveValue::from(new_uid.as_str()),
            ));
            result.uids_remapped += 1;
        }
    }
}

// ─── Dataset traversal ────────────────────────────────────────────────────────

/// Maximum sequence nesting depth traversed by [`anonymize_dataset`].
///
/// A DICOM data set is a tree, so traversal terminates without a depth limit;
/// this bound exists only to keep a pathologically or maliciously nested object
/// from exhausting the stack. Conformant objects nest far below it — structured
/// reports, the deepest common case, are typically under ten levels.
const MAX_SEQUENCE_DEPTH: u32 = 64;

/// Apply `tag_actions` to `dataset` and to every data set nested within it.
///
/// # Why recursion is required
///
/// Sequence attributes (`VR::SQ`) contain complete nested data sets, which
/// contain their own sequences. Identifying attributes occur at every level:
/// `RequestAttributesSequence (0040,0275)` nests accession numbers,
/// `ReferencedImageSequence (0008,1140)` nests `ReferencedSOPInstanceUID`
/// values that must be remapped consistently with the top-level UIDs or the
/// references dangle, `ContentSequence (0040,A730)` nests operator-authored
/// text and person names at arbitrary depth, and
/// `OriginalAttributesSequence (0400,0561)` nests the original values that a
/// previous de-identification replaced.
///
/// Applying the profile only to top-level elements leaves all of that in place
/// while reporting success.
///
/// # Invariants
///
/// - The same `uid_map` is threaded through every level, so one source UID maps
///   to one replacement UID across the whole object regardless of the depth at
///   which it appears.
/// - Private-tag removal, when enabled, applies at every level.
/// - Statistics in `result` accumulate across all levels, each increment taken
///   from the mutation that performed the work.
///
/// # Errors
///
/// Returns [`AnonymizeError`] when a step leaves identifying data in the object:
/// a private element that survives its own removal, a sequence that cannot be
/// read back for traversal, or nesting that reaches [`MAX_SEQUENCE_DEPTH`] with
/// sequences still below it. `result` then describes a partial pass and the
/// caller must discard the object.
fn anonymize_dataset(
    dataset: &mut InMemDicomObject,
    tag_actions: &[(Tag, TagAction)],
    options: &AnonymizeOptions,
    result: &mut AnonymizeResult,
    uid_map: &mut HashMap<String, String>,
    depth: u32,
) -> Result<(), AnonymizeError> {
    for (tag, action) in tag_actions {
        apply_action(dataset, *tag, *action, options, result, uid_map);
    }

    // Remove private tags at this level before descending, mirroring the
    // top-level ordering relative to pixel-data handling.
    if options.clean_private_tags == CleaningPolicy::Clean || options.profile.removes_private_tags()
    {
        // Collect private tag addresses first to avoid borrow conflicts.
        // DICOM private elements have odd group numbers.
        // Group 0x0002 (file meta-header) is always excluded.
        let private_tags: Vec<Tag> = dataset
            .iter()
            .map(|e| e.tag())
            .filter(|t| t.group() & 1 == 1 && t.group() != 0x0002)
            .collect();
        for tag in private_tags {
            // The address was just read from this data set, so a removal that
            // reports nothing removed means the element is still there. Counting
            // the survey instead would credit the report with a removal that did
            // not happen, which is the one direction this report must not err in.
            if !dataset.remove_element(tag) {
                return Err(AnonymizeError::PrivateTagRetained { tag, depth });
            }
            result.private_tags_removed += 1;
        }
    }

    // Sequence addresses are collected before mutation so the data set is not
    // iterated while it is being modified.
    let sequence_tags: Vec<Tag> = dataset
        .iter()
        .filter(|e| matches!(e.value(), Value::Sequence(_)))
        .map(|e| e.tag())
        .collect();

    // The bound is checked only when there is something below it to visit, so an
    // object that merely reaches the bound still completes; one that would need
    // to descend past it fails closed rather than reporting a walk it truncated.
    if !sequence_tags.is_empty() && depth >= MAX_SEQUENCE_DEPTH {
        tracing::warn!(
            depth,
            "sequence nesting reached {MAX_SEQUENCE_DEPTH} levels; deeper data sets \
             were not anonymized"
        );
        return Err(AnonymizeError::SequenceTooDeep { depth });
    }

    for tag in sequence_tags {
        // The address and its sequence-ness were just read from this data set;
        // failing either check means the element moved under the walk, leaving
        // the data sets it holds unvisited.
        let element = dataset
            .element(tag)
            .cloned()
            .map_err(|_| AnonymizeError::SequenceNotTraversed { tag, depth })?;
        let vr = element.vr();
        let Value::Sequence(sequence) = element.into_value() else {
            return Err(AnonymizeError::SequenceNotTraversed { tag, depth });
        };

        let mut items: Vec<InMemDicomObject> = sequence.into_items().into_iter().collect();
        for item in &mut items {
            anonymize_dataset(item, tag_actions, options, result, uid_map, depth + 1)?;
        }

        dataset.put(DataElement::new(
            tag,
            vr,
            Value::Sequence(DataSetSequence::from(items)),
        ));
    }

    Ok(())
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Apply anonymization to a single in-memory DICOM object.
///
/// Iterates the `options.profile` tag-action list, mutating `obj` in place
/// (via `DerefMut`), and descends into every nested sequence so that
/// identifying attributes below the top level are acted on as well. The file
/// meta-header is never modified. Returns `AnonymizeResult` with per-operation
/// statistics and the UID cross-reference map.
///
/// When `options.clean_pixel_data` is `CleaningPolicy::Clean`, the
/// `PixelData` element `(7FE0,0010)` is overwritten with an equal-length
/// zero buffer if present and readable as a flat byte sequence.
///
/// # Errors
///
/// Returns [`AnonymizeError`] (as `anyhow::Error`; recover it with
/// `downcast_ref`) when a step would leave identifying data in the object — a
/// private element surviving removal, an untraversable sequence, or nesting
/// past `MAX_SEQUENCE_DEPTH`. The object is not returned in that case, so a
/// partially de-identified data set can never be mistaken for a clean one.
pub fn anonymize_object(
    mut obj: FileDicomObject<InMemDicomObject>,
    options: &AnonymizeOptions,
) -> Result<(FileDicomObject<InMemDicomObject>, AnonymizeResult)> {
    let mut result = AnonymizeResult::default();
    let tag_actions = options.profile.tag_actions();
    let mut uid_map: HashMap<String, String> = HashMap::with_capacity(tag_actions.len());

    // Tag actions and private-tag removal are applied to the top-level data set
    // and to every data set nested inside a sequence. Private tags are removed
    // before pixel data handling below, so a private pixel data block is not
    // removed after it has already been zeroed.
    anonymize_dataset(
        &mut obj,
        &tag_actions,
        options,
        &mut result,
        &mut uid_map,
        0,
    )?;

    if options.clean_pixel_data == CleaningPolicy::Clean {
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
/// Returns `AnonymizeResult` with per-operation statistics.
///
/// # Errors
///
/// Fails if `input_path` cannot be parsed as a DICOM Part 10 file, if
/// anonymization cannot complete ([`AnonymizeError`]), or if the output cannot
/// be written. Propagates the underlying I/O and DICOM errors with context. An
/// incomplete anonymization aborts before the write, so `output_path` is never
/// left holding a data set the report would have called de-identified.
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
/// Files that are valid DICOM but fail during anonymization ([`AnonymizeError`])
/// or writing are counted in `AnonymizeStats::error_count`, recorded in
/// `AnonymizeStats::errors`, and logged at `WARN` level; no output file is
/// written for them.
/// Output filenames match input filenames; directory structure is not
/// recursed.
///
/// # Errors
///
/// Fails only if `output_dir` cannot be created or `input_dir` cannot be read;
/// per-file failures are reported through `AnonymizeStats` rather than aborting
/// the run.
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

/// Anonymize a DICOM file and verify the exported metadata is intact.
///
/// This is the export-gate entry point recommended for PACS-bound output: it
/// runs `anonymize_dicom_file` and then verifies the written file re-parses,
/// carries conformant and consistent UIDs, keeps geometry metadata coherent
/// with the pixel payload, and contains none of the supplied
/// `verify_options.prohibited_values`.
///
/// # Errors
/// Returns an error when anonymization fails, the export cannot be written, or
/// verification finds any issue.
pub fn anonymize_dicom_file_verified(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    options: &AnonymizeOptions,
    verify_options: &verify::VerifyOptions,
) -> Result<AnonymizeResult> {
    let result = anonymize_dicom_file(&input_path, &output_path, options)?;
    verify::ensure_dicom_file_clean(&output_path, verify_options)?;
    Ok(result)
}

/// Anonymize a directory tree and verify every exported file.
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
