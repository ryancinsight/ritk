//! Shared post-processing logic for directory and instance DICOM scans.
//!
//! `finalize_scanned_series` implements the common pipeline after per-slice
//! metadata extraction: dimension filtering, series grouping, SOP class
//! policy, geometry assembly, and `DicomSeriesInfo` construction.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{bail, Result};

use super::super::types::{
    uid_to_arraystring, DicomSeriesInfo, DicomSliceMetadata, SeriesFirstSeen, SeriesGeometry,
};

use crate::format::dicom::sop_class::{classify_sop_class, SopClassKind};

use super::build_series_object;
use super::geometry::{
    assemble_direction, assemble_origin, check_iop_consistency, check_pixel_spacing_consistency,
    compute_slice_normal, compute_spacing_z, sort_slices_spatially, synthesize_gantry_tilt,
};

/// Shared post-processing logic for both directory and instance scans.
///
/// After the caller has finished extracting per-slice metadata (the phase
/// unique to each scan mode), this function performs:
///
/// - Canonical-dimension filtering
/// - SeriesInstanceUID grouping
/// - SOP class policy filtering
/// - Slice normal computation
/// - Slice sorting
/// - GantryDetectorTilt synthesis
/// - Cross-slice IOP consistency guard
/// - Cross-slice PixelSpacing consistency guard
/// - Z-spacing computation
/// - Direction/origin assembly
/// - `build_series_object` call
/// - `assemble_metadata` call
/// - `DicomSeriesInfo` construction
///
/// # Parameters
/// - `slices`: already-parsed slice metadata
/// - `part10_bytes_vec`: `Some(bytes)` for SCP instances (attached to slices
///   during filtering/grouping); `None` for directory-based scans
/// - `per_file_dims`: per-slice `(rows, cols)` for canonical-dimension filtering
/// - `per_file_series_uids`: per-slice SeriesInstanceUID for grouping
/// - `first`: accumulated `SeriesFirstSeen` from the parse phase
/// - `series_path`: path to use in the returned `DicomSeriesInfo`
/// - `sort_tiebreaker`: final comparator for slice sorting (filename vs SOP UID)
/// - `empty_error_context`: context string for empty-series error messages
#[allow(clippy::too_many_arguments)]
pub(super) fn finalize_scanned_series(
    mut slices: Vec<DicomSliceMetadata>,
    part10_bytes_vec: Option<Vec<Vec<u8>>>,
    per_file_dims: Vec<(u32, u32)>,
    mut per_file_series_uids: Vec<Option<String>>,
    mut first: SeriesFirstSeen,
    series_path: PathBuf,
    sort_tiebreaker: fn(&DicomSliceMetadata, &DicomSliceMetadata) -> std::cmp::Ordering,
    empty_error_context: &str,
) -> Result<DicomSeriesInfo> {
    let mut part10_bytes_vec = part10_bytes_vec.unwrap_or_default();

    // Canonical-dimension filtering (GAP-R62-02):
    // In DICOMDIR datasets with mixed series (scout + CT), use the plurality
    // (most-frequent) dimensions as canonical; exclude non-matching files.
    {
        let mut dim_freq: HashMap<(u32, u32), usize> = HashMap::with_capacity(per_file_dims.len());
        for &(r, c) in &per_file_dims {
            if r > 0 && c > 0 {
                *dim_freq.entry((r, c)).or_insert(0) += 1;
            }
        }
        if let Some((&(cr, cc), _)) = dim_freq.iter().max_by_key(|(_, &v)| v) {
            let total = per_file_dims
                .iter()
                .filter(|&&(r, c)| r > 0 && c > 0)
                .count();
            let matching = per_file_dims
                .iter()
                .filter(|&&(r, c)| r == cr && c == cc)
                .count();
            if matching < total {
                let excluded = total - matching;
                tracing::warn!(
                    excluded,
                    canonical_rows = cr,
                    canonical_cols = cc,
                    "DICOM series: excluding {} instance(s) with non-canonical image dimensions \
                     (canonical {}x{}); likely a mixed-series dataset",
                    excluded,
                    cr,
                    cc
                );
                let mut new_slices = Vec::with_capacity(slices.len());
                let mut new_uids = Vec::with_capacity(slices.len());
                if part10_bytes_vec.is_empty() {
                    // Directory scan: no bytes to carry.
                    for ((s, u), (r, c)) in slices
                        .into_iter()
                        .zip(per_file_series_uids)
                        .zip(per_file_dims.iter().copied())
                    {
                        if (r == cr && c == cc) || r == 0 {
                            new_slices.push(s);
                            new_uids.push(u);
                        }
                    }
                } else {
                    // In-memory scan: carry bytes alongside slices.
                    let mut new_bytes = Vec::with_capacity(slices.len());
                    for (((s, u), (r, c)), b) in slices
                        .into_iter()
                        .zip(per_file_series_uids)
                        .zip(per_file_dims.iter().copied())
                        .zip(part10_bytes_vec)
                    {
                        if (r == cr && c == cc) || r == 0 {
                            new_slices.push(s);
                            new_uids.push(u);
                            new_bytes.push(b);
                        }
                    }
                    part10_bytes_vec = new_bytes;
                }
                slices = new_slices;
                per_file_series_uids = new_uids;
                first.rows = Some(cr);
                first.cols = Some(cc);
            }
        }
    }

    // SeriesInstanceUID grouping (GAP-R63-04):
    // Select the most-populated series when multiple acquisitions share the same
    // row/col dimensions.  Skip filtering when two series tie.
    {
        let mut uid_counts: HashMap<Option<&str>, usize> =
            HashMap::with_capacity(per_file_series_uids.len());
        for uid in &per_file_series_uids {
            *uid_counts.entry(uid.as_deref()).or_insert(0) += 1;
        }
        let distinct_count = uid_counts.keys().filter(|k| k.is_some()).count();
        if distinct_count > 1 {
            let max_count = uid_counts
                .iter()
                .filter(|(k, _)| k.is_some())
                .map(|(_, &v)| v)
                .max()
                .unwrap_or(0);
            let series_at_max = uid_counts
                .iter()
                .filter(|(k, &v)| k.is_some() && v == max_count)
                .count();
            if series_at_max == 1 {
                let best_uid: Option<String> = uid_counts
                    .iter()
                    .filter(|(k, _)| k.is_some())
                    .max_by(|(ka, &va), (kb, &vb)| va.cmp(&vb).then_with(|| kb.cmp(ka)))
                    .and_then(|(k, _)| k.map(|s| s.to_owned()));
                if let Some(ref uid) = best_uid {
                    let uid_str: &str = uid.as_str();
                    let excluded = slices
                        .iter()
                        .zip(per_file_series_uids.iter())
                        .filter(|(_, u)| u.as_deref() != Some(uid_str))
                        .count();
                    tracing::warn!(
                        excluded,
                        selected_series_uid = uid_str,
                        total_distinct_series = distinct_count,
                        "DICOM series contains {} same-dimension series; selecting \
                         most-populated (SeriesInstanceUID={}); {} instance(s) from other \
                         series excluded",
                        distinct_count,
                        uid_str,
                        excluded
                    );
                    if part10_bytes_vec.is_empty() {
                        // Directory scan: no bytes to carry.
                        let retained: Vec<DicomSliceMetadata> = slices
                            .into_iter()
                            .zip(per_file_series_uids)
                            .filter(|(_, u)| u.as_deref() == Some(uid_str) || u.is_none())
                            .map(|(s, _)| s)
                            .collect();
                        slices = retained;
                    } else {
                        // In-memory scan: carry bytes and attach to slices.
                        let retained: Vec<DicomSliceMetadata> = slices
                            .into_iter()
                            .zip(per_file_series_uids)
                            .zip(part10_bytes_vec)
                            .filter(|((_, u), _)| u.as_deref() == Some(uid_str) || u.is_none())
                            .map(|((s, _), b)| {
                                let mut s = s;
                                s.part10_bytes = Some(b);
                                s
                            })
                            .collect();
                        slices = retained;
                        // Bytes have been consumed into slice metadata; clear the
                        // parallel vec so the post-grouping attachment is a no-op.
                        part10_bytes_vec = Vec::new();
                    }
                }
                if best_uid.is_some() {
                    first.series_instance_uid = best_uid.as_deref().and_then(uid_to_arraystring);
                }
            }
        }
    }

    // Attach part10_bytes to slices that haven't been assigned yet
    // (handles the single-series case where the grouping block is skipped).
    if slices.iter().any(|s| s.part10_bytes.is_none())
        && !part10_bytes_vec.is_empty()
        && part10_bytes_vec.len() == slices.len()
    {
        for (slice, bytes) in slices.iter_mut().zip(part10_bytes_vec) {
            if slice.part10_bytes.is_none() {
                slice.part10_bytes = Some(bytes);
            }
        }
    }

    // SOP class policy: remove non-image-bearing instances.
    // Files without a readable SOP class UID are retained (ambiguous â†’ permissive).
    let original_count = slices.len();
    let mut rejected_uids: Vec<String> = Vec::new();
    slices.retain(|s| match s.sop_class_uid.as_deref() {
        None => true,
        Some(uid) => {
            let kind = classify_sop_class(uid);
            let is_non_image = !kind.is_image_storage() && !matches!(kind, SopClassKind::Other(_));
            if is_non_image {
                tracing::warn!(
                    "skipping non-image DICOM SOP class: {:?} path={} sop_class_uid={}",
                    kind,
                    s.path.display(),
                    uid
                );
                rejected_uids.push(uid.to_string());
                false
            } else {
                true
            }
        }
    });
    if slices.is_empty() && original_count > 0 {
        bail!(
            "{} contain(s) {} instance(s) but none are image-bearing SOP classes;\n\
             rejected SOP class UIDs: [{}]",
            empty_error_context,
            original_count,
            rejected_uids.join(", ")
        );
    }

    // â”€â”€ Geometry pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    let maybe_normal = compute_slice_normal(&slices);
    sort_slices_spatially(&mut slices, maybe_normal, sort_tiebreaker);
    synthesize_gantry_tilt(&mut slices);
    check_iop_consistency(&slices);
    check_pixel_spacing_consistency(&slices);

    let rows = first.rows.unwrap_or(0) as usize;
    let cols = first.cols.unwrap_or(0) as usize;

    let spacing_z = compute_spacing_z(&slices, maybe_normal, first.slice_thickness);
    let in_plane_spacing = first.pixel_spacing.unwrap_or([1.0, 1.0]);
    let spacing: [f64; 3] = [
        spacing_z.abs().max(1e-6),
        in_plane_spacing[0],
        in_plane_spacing[1],
    ];

    let direction = assemble_direction(&slices);
    let origin = assemble_origin(&slices);

    let series_object = build_series_object(&series_path, &slices);
    let metadata = super::super::types::assemble_metadata(
        first,
        slices,
        SeriesGeometry {
            rows,
            cols,
            spacing,
            origin,
            direction,
        },
        series_object,
    );
    Ok(DicomSeriesInfo {
        path: series_path,
        num_slices: metadata.slices.len(),
        metadata,
    })
}
