//! DICOM series directory scanner.
//!
//! `scan_dicom_directory` scans a directory for DICOM files, parses per-slice
//! and series-level metadata, filters to the most-populated series, and
//! returns a `DicomSeriesInfo` with typed metadata and geometry.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use super::dicomdir::try_read_dicomdir;
use super::geometry::{
    analyze_slice_spacing, cross_3d, dot_3d, normalize_3d, slice_normal_from_iop,
};
use super::parse::parse_dicom_file;
use super::types::{DicomSeriesInfo, DicomSliceMetadata, SeriesFirstSeen};
use super::utils::is_likely_dicom_file;
use crate::format::dicom::object_model::{DicomObjectModel, DicomObjectNode, DicomTag};
use crate::format::dicom::sop_class::{classify_sop_class, SopClassKind};

/// Scan a directory for DICOM files, extract metadata, and return a typed series descriptor.
///
/// # Invariants
/// - `path` must be a directory.
/// - At least one DICOM file must be discoverable (directly or via DICOMDIR).
/// - All returned slices belong to the most-populated SeriesInstanceUID when
///   the directory contains multiple series with the same image dimensions.
pub fn scan_dicom_directory<P: AsRef<Path>>(path: P) -> Result<DicomSeriesInfo> {
    let path = path.as_ref();
    if !path.is_dir() {
        bail!("DICOM input path is not a directory");
    }

    // File discovery: prefer DICOMDIR; fall back to flat-folder scan.
    let mut raw_paths: Vec<PathBuf> = Vec::new();

    if let Ok(dicomdir_paths) = try_read_dicomdir(path) {
        raw_paths = dicomdir_paths;
        raw_paths.sort();
        raw_paths.dedup();
    } else {
        for entry in std::fs::read_dir(path).with_context(|| "failed to read DICOM directory")? {
            let entry = entry.with_context(|| "failed to read DICOM directory entry")?;
            let entry_path = entry.path();
            if entry_path.is_file() && is_likely_dicom_file(&entry_path) {
                raw_paths.push(entry_path);
            }
        }
        raw_paths.sort();
        raw_paths.dedup();
    }

    if raw_paths.is_empty() {
        bail!("no DICOM files were discovered in the directory");
    }

    // Parse metadata from each DICOM file.
    let mut slices: Vec<DicomSliceMetadata> = Vec::with_capacity(raw_paths.len());
    let mut first = SeriesFirstSeen::default();
    let mut per_file_dims: Vec<(u32, u32)> = Vec::with_capacity(raw_paths.len());
    let mut per_file_series_uids: Vec<Option<String>> = Vec::with_capacity(raw_paths.len());

    for file_path in &raw_paths {
        if let Some((slice_meta, file_dim, file_series_uid)) =
            parse_dicom_file(file_path, &mut first)
        {
            per_file_dims.push(file_dim);
            per_file_series_uids.push(file_series_uid);
            slices.push(slice_meta);
        }
    }

    // Canonical-dimension filtering (GAP-R62-02):
    // In DICOMDIR datasets with mixed series (scout + CT), use the plurality
    // (most-frequent) dimensions as canonical; exclude non-matching files.
    {
        let mut dim_freq: HashMap<(u32, u32), usize> = HashMap::new();
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
                    "DICOM series: excluding {} file(s) with non-canonical image dimensions \
                     (canonical {}x{}); likely a mixed-series DICOMDIR dataset",
                    excluded,
                    cr,
                    cc
                );
                let (new_slices, new_uids): (Vec<_>, Vec<_>) = slices
                    .into_iter()
                    .zip(per_file_series_uids)
                    .zip(per_file_dims.iter().copied())
                    .filter(|((_, _), (r, c))| (*r == cr && *c == cc) || *r == 0)
                    .map(|((s, u), _)| (s, u))
                    .unzip();
                slices = new_slices;
                per_file_series_uids = new_uids;
                first.rows = Some(cr);
                first.cols = Some(cc);
            }
        }
    }

    // SeriesInstanceUID grouping (GAP-R63-04):
    // Select the most-populated series when multiple acquisitions share the same
    // row/col dimensions. Skip filtering when two series tie.
    {
        let mut uid_counts: HashMap<Option<String>, usize> = HashMap::new();
        for uid in &per_file_series_uids {
            *uid_counts.entry(uid.clone()).or_insert(0) += 1;
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
                    .and_then(|(k, _)| k.clone());
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
                        "DICOM directory contains {} same-dimension series; selecting \
                         most-populated (SeriesInstanceUID={}); {} file(s) from other \
                         series excluded",
                        distinct_count,
                        uid_str,
                        excluded
                    );
                    let retained: Vec<DicomSliceMetadata> = slices
                        .into_iter()
                        .zip(per_file_series_uids)
                        .filter(|(_, u)| u.as_deref() == Some(uid_str) || u.is_none())
                        .map(|(s, _)| s)
                        .collect();
                    slices = retained;
                }
                if best_uid.is_some() {
                    first.series_instance_uid = best_uid;
                }
            }
        }
    }

    // SOP class policy: remove non-image-bearing files.
    // Files without a readable SOP class UID are retained (ambiguous → permissive).
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
            "DICOM directory {:?} contains {} file(s) but none are image-bearing SOP classes;\n\
             rejected SOP class UIDs: [{}]",
            path,
            original_count,
            rejected_uids.join(", ")
        );
    }

    // Compute slice normal for position projection.
    let maybe_normal: Option<[f64; 3]> = slices
        .iter()
        .find_map(|s| s.image_orientation_patient)
        .and_then(slice_normal_from_iop);

    // Sort slices by projection of IPP onto slice normal, then instance number, then filename.
    slices.sort_by(|a, b| {
        let pos_a = match (a.image_position_patient, maybe_normal) {
            (Some(ipp), Some(n)) => dot_3d(ipp, n),
            (Some(ipp), None) => ipp[2],
            (None, _) => f64::MAX,
        };
        let pos_b = match (b.image_position_patient, maybe_normal) {
            (Some(ipp), Some(n)) => dot_3d(ipp, n),
            (Some(ipp), None) => ipp[2],
            (None, _) => f64::MAX,
        };
        pos_a
            .partial_cmp(&pos_b)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.instance_number
                    .unwrap_or(i32::MAX)
                    .cmp(&b.instance_number.unwrap_or(i32::MAX))
            })
            .then_with(|| a.path.file_name().cmp(&b.path.file_name()))
    });

    // GantryDetectorTilt synthesis (GAP-R62-01):
    // When IOP is absent or effectively axial and |tilt| > 0.01°, synthesize oblique IOP.
    const AXIAL_IOP_THRESHOLD: f64 = 1e-4;
    const GANTRY_TILT_MIN_DEGREES: f64 = 0.01;
    {
        let ref_iop = slices.first().and_then(|s| s.image_orientation_patient);
        let is_effectively_axial = ref_iop.is_none_or(|iop| {
            let axial = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0];
            iop.iter()
                .zip(axial.iter())
                .all(|(a, e)| (a - e).abs() < AXIAL_IOP_THRESHOLD)
        });
        if is_effectively_axial {
            if let Some(tilt_deg) = slices.first().and_then(|s| s.gantry_tilt) {
                if tilt_deg.abs() > GANTRY_TILT_MIN_DEGREES {
                    let theta = tilt_deg.to_radians();
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();
                    let synthesized_iop = [1.0_f64, 0.0, 0.0, 0.0, cos_t, -sin_t];
                    tracing::info!(
                        tilt_deg,
                        cos_t,
                        sin_t,
                        "GantryDetectorTilt: synthesizing oblique IOP from tilt angle"
                    );
                    for slice in &mut slices {
                        if slice.image_orientation_patient.is_none()
                            || slice.image_orientation_patient.is_some_and(|iop| {
                                let axial = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0];
                                iop.iter()
                                    .zip(axial.iter())
                                    .all(|(a, e)| (a - e).abs() < AXIAL_IOP_THRESHOLD)
                            })
                        {
                            slice.image_orientation_patient = Some(synthesized_iop);
                        }
                    }
                }
            }
        }
    }

    let rows = first.rows.unwrap_or(0) as usize;
    let cols = first.cols.unwrap_or(0) as usize;
    // Cross-slice IOP consistency guard (DICOM PS3.3 C.7.6.1.1.1).
    const IOP_CONSISTENCY_THRESHOLD: f64 = 1e-4;
    if let Some(ref_iop) = slices.first().and_then(|s| s.image_orientation_patient) {
        for (i, s) in slices.iter().enumerate().skip(1) {
            if let Some(iop) = s.image_orientation_patient {
                let max_dev = iop
                    .iter()
                    .zip(ref_iop.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                if max_dev > IOP_CONSISTENCY_THRESHOLD {
                    tracing::warn!(
                        slice_index = i,
                        max_iop_deviation = max_dev,
                        "DICOM series has inconsistent ImageOrientationPatient across slices; \
                         using first slice IOP as canonical"
                    );
                }
            }
        }
    }

    // Cross-slice PixelSpacing consistency guard.
    const PIXEL_SPACING_CONSISTENCY_THRESHOLD: f64 = 1e-4;
    if let Some(ref_ps) = slices.first().and_then(|s| s.pixel_spacing) {
        for (i, s) in slices.iter().enumerate().skip(1) {
            if let Some(ps) = s.pixel_spacing {
                let max_dev = [(ps[0] - ref_ps[0]).abs(), (ps[1] - ref_ps[1]).abs()]
                    .into_iter()
                    .fold(0.0_f64, f64::max);
                if max_dev > PIXEL_SPACING_CONSISTENCY_THRESHOLD {
                    tracing::warn!(
                        slice_index = i,
                        max_spacing_deviation = max_dev,
                        "DICOM series has inconsistent PixelSpacing across slices; \
                         using first slice PixelSpacing as canonical"
                    );
                }
            }
        }
    }

    // Compute z-spacing using median of adjacent-pair projected positions.
    let spacing_z = {
        let positions: Vec<f64> = if let Some(n) = maybe_normal {
            slices
                .iter()
                .filter_map(|s| s.image_position_patient.map(|ipp| dot_3d(ipp, n)))
                .collect()
        } else {
            slices
                .iter()
                .filter_map(|s| s.image_position_patient.map(|p| p[2]))
                .collect()
        };
        if positions.len() >= 2 {
            analyze_slice_spacing(&positions).nominal_spacing
        } else {
            first.slice_thickness.unwrap_or(1.0)
        }
    };

    let in_plane_spacing = first.pixel_spacing.unwrap_or([1.0, 1.0]);
    let spacing: [f64; 3] = [
        spacing_z.abs().max(1e-6),
        in_plane_spacing[0],
        in_plane_spacing[1],
    ];

    // Direction convention: col 0 = N̂, col 1 = F_c, col 2 = F_r.
    let direction = if let Some(ori) = slices.first().and_then(|s| s.image_orientation_patient) {
        let r = [ori[0], ori[1], ori[2]];
        let c = [ori[3], ori[4], ori[5]];
        let n = normalize_3d(cross_3d(r, c)).unwrap_or([0.0, 0.0, 1.0]);
        [
            n[0], n[1], n[2], ori[3], ori[4], ori[5], ori[0], ori[1], ori[2],
        ]
    } else {
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    };

    let origin = slices
        .first()
        .and_then(|s| s.image_position_patient)
        .unwrap_or([0.0, 0.0, 0.0]);

    let series_object = build_series_object(path, &slices);
    let metadata = super::types::assemble_metadata(
        first,
        slices,
        rows,
        cols,
        spacing,
        origin,
        direction,
        series_object,
    );

    Ok(DicomSeriesInfo {
        path: path.to_path_buf(),
        num_slices: metadata.slices.len(),
        metadata,
    })
}

fn build_series_object(path: &Path, slices: &[DicomSliceMetadata]) -> DicomObjectModel {
    let mut series_object = DicomObjectModel::with_source(path.to_path_buf());
    for slice in slices {
        if let Some(uid) = slice.sop_instance_uid.as_ref() {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0008, 0x0018),
                "UI",
                uid.clone(),
            ));
        }
        if let Some(instance_number) = slice.instance_number {
            series_object.insert(DicomObjectNode::i32(
                DicomTag::new(0x0020, 0x0013),
                "IS",
                instance_number,
            ));
        }
        if let Some(slice_location) = slice.slice_location {
            series_object.insert(DicomObjectNode::f64(
                DicomTag::new(0x0020, 0x1041),
                "DS",
                slice_location,
            ));
        }
        if let Some(position) = slice.image_position_patient {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0020, 0x0032),
                "DS",
                format!("{:.6}\\{:.6}\\{:.6}", position[0], position[1], position[2]),
            ));
        }
        if let Some(orientation) = slice.image_orientation_patient {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0020, 0x0037),
                "DS",
                format!(
                    "{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}",
                    orientation[0],
                    orientation[1],
                    orientation[2],
                    orientation[3],
                    orientation[4],
                    orientation[5]
                ),
            ));
        }
        if let Some(pixel_spacing) = slice.pixel_spacing {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0028, 0x0030),
                "DS",
                format!("{:.6}\\{:.6}", pixel_spacing[0], pixel_spacing[1]),
            ));
        }
        if let Some(slice_thickness) = slice.slice_thickness {
            series_object.insert(DicomObjectNode::f64(
                DicomTag::new(0x0018, 0x0050),
                "DS",
                slice_thickness,
            ));
        }
        if let Some(sop_class_uid) = slice.sop_class_uid.as_ref() {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0008, 0x0016),
                "UI",
                sop_class_uid.clone(),
            ));
        }
    }
    series_object
}
