//! DICOM de-identification and anonymization — PS 3.15 Annex E.
//!
//! # Design
//! - `anonymize_object` is the canonical entry point for single-object mutation.
//! - `anonymize_dicom_file` wraps open + anonymize + write.
//! - `anonymize_dicom_directory` batches over the files directly in a directory.
//! - UID replacement uses a deterministic SHA-256-derived UUIDv8 mapping.
//!   Use a secret salt where linkage to known source UIDs must be discouraged.
//!
//! # Invariants
//! - Non-DICOM files in directory mode are skipped silently.
//! - Tag actions and private-tag removal apply to existing attributes at every
//!   sequence nesting level.
//! - The file meta group keeps its transfer syntax, SOP class and
//!   implementation identifiers; its Media Storage SOP Instance UID is set to
//!   the data set's SOPInstanceUID when that is a single conformant UID, and
//!   is otherwise left as it was; its application entity titles and private
//!   information are removed, as is every top-level group 0002 element of
//!   the data set.
//! - `clean_pixel_data = CleaningPolicy::Skip` (default) never touches pixel data.
//! - Same `(original_uid, salt)` always produces the same replacement UID.
//! - Every count in [`AnonymizeResult`] is taken from the mutation that produced
//!   it, never from a prior survey of candidates, so the report can only
//!   under-state what happened, never over-state it.
//! - A step that cannot complete aborts the run with [`AnonymizeError`]; the
//!   object it would have described is never returned as a success.

mod action;
mod directory;
pub(in crate::format::dicom) mod file_meta;
mod profile;
#[cfg(not(target_arch = "wasm32"))]
mod salt;
mod study;
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
mod tests_study;
#[cfg(test)]
#[path = "tests_verify.rs"]
mod tests_verify;
#[cfg(test)]
mod tests_verify_geometry;
mod uid;
pub mod verify;

pub use directory::{
    anonymize_dicom_directory, anonymize_dicom_directory_verified, AnonymizeStats,
};
pub use profile::{AnonymizationProfile, TagAction};
#[cfg(not(target_arch = "wasm32"))]
pub use salt::SaltError;
pub use study::{
    anonymize_dicom_selection, anonymize_dicom_study, StudyAnonymizeError, StudyAnonymizeReport,
};

/// Controls whether an anonymization step is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CleaningPolicy {
    /// Do not clean this category.
    #[default]
    Skip,
    /// Remove or zero-fill this category.
    Clean,
}

use self::action::apply_action;
#[cfg(test)]
use self::uid::generate_uid_from_hash;
use anyhow::{Context, Result};
use ritk_dicom::dataset::Value;
use ritk_dicom::dataset::{DataElement, DicomTag, PrimitiveValue, VR};
use ritk_dicom::dataset::{Dataset, DicomFile};
use std::collections::HashMap;
use std::path::Path;

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
    /// This removes institution-specific private attributes that may carry PHI.
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
    /// Number of present tags emptied or replaced with a dummy value.
    pub tags_zeroed: usize,
    /// Number of attributes with at least one UID remapped.
    pub uids_remapped: usize,
    /// Number of private elements actually removed from the object.
    pub private_tags_removed: usize,
    /// Map of original UID → replacement UID for cross-reference tracking.
    pub uid_map: HashMap<String, String>,
}

/// A de-identification step that could not be completed.
///
/// An incomplete traversal or failed private-tag removal prevents the object
/// from being returned as a successful result.
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
        tag: DicomTag,
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
        tag: DicomTag,
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
    dataset: &mut Dataset,
    tag_actions: &[(DicomTag, TagAction)],
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
        let private_tags: Vec<DicomTag> = dataset
            .iter()
            .map(|e| e.tag())
            .filter(|t| t.group & 1 == 1 && t.group != 0x0002)
            .collect();
        for tag in private_tags {
            // The address was just read from this data set, so a removal that
            // reports nothing removed means the element is still there. Counting
            // the survey instead would credit the report with a removal that did
            // not happen, which is the one direction this report must not err in.
            if dataset.remove_element(tag).is_none() {
                return Err(AnonymizeError::PrivateTagRetained { tag, depth });
            }
            result.private_tags_removed += 1;
        }
    }

    // Sequence addresses are collected before mutation so the data set is not
    // iterated while it is being modified.
    let sequence_tags: Vec<DicomTag> = dataset
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
            .get_mut(tag)
            .ok_or(AnonymizeError::SequenceNotTraversed { tag, depth })?;
        let Value::Sequence(sequence) = element.value_mut() else {
            return Err(AnonymizeError::SequenceNotTraversed { tag, depth });
        };
        for item in sequence.items_mut() {
            anonymize_dataset(item, tag_actions, options, result, uid_map, depth + 1)?;
        }
    }

    Ok(())
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Apply anonymization to a single in-memory DICOM object.
///
/// Iterates the `options.profile` tag-action list, mutating `obj` in place
/// (via `DerefMut`), and descends into every nested sequence so that
/// identifying attributes below the top level are acted on as well. In the file
/// meta group Media Storage SOP Instance UID is set to the data set's
/// SOPInstanceUID when that is a single conformant UID; Source, Sending and
/// Receiving Application Entity Title, Private Information Creator UID and
/// Private Information are removed; the group length is recomputed; no other
/// file meta element changes. Top-level data set elements of group 0002 are
/// removed. Returns
/// `AnonymizeResult` with per-operation statistics and the UID
/// cross-reference map.
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
/// past the configured maximum sequence depth. The object is not returned in
/// that case, so partially de-identified data cannot be mistaken for a clean
/// result.
pub fn anonymize_object(
    mut obj: DicomFile,
    options: &AnonymizeOptions,
) -> Result<(DicomFile, AnonymizeResult)> {
    let mut result = AnonymizeResult::default();
    let tag_actions = options.profile.tag_actions();
    let mut uid_map: HashMap<String, String> = HashMap::with_capacity(tag_actions.len());

    // DicomTag actions and private-tag removal are applied to the top-level data set
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
        let pixel_tag = DicomTag::new(0x7FE0, 0x0010);
        // Extract VR before the second element() call; VR is Copy.
        let vr = obj.element(pixel_tag).map(|e| e.vr()).unwrap_or(VR::OW);
        // Obtain byte count without retaining a borrow of obj.
        let len: usize = obj
            .element(pixel_tag)
            .ok()
            .and_then(|e| e.to_bytes().ok().map(|b| b.len()))
            .unwrap_or(0);
        if len > 0 {
            let zeros = PrimitiveValue::from(vec![0u8; len]);
            obj.put(DataElement::new(pixel_tag, vr, zeros));
        }
    }

    file_meta::align_media_storage_uid(&mut obj);
    file_meta::remove_origin(&mut obj);
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
    let obj = ritk_dicom::dataset::DicomFile::open(input_path)
        .with_context(|| format!("Failed to open DICOM file {input_path:?}"))?;
    let (anon, result) = anonymize_object(obj, options)?;
    anon.write_to_file(output_path)
        .with_context(|| format!("Failed to write anonymized DICOM to {output_path:?}"))?;
    Ok(result)
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
