//! DICOM series directory scanner.
//!
//! `scan_dicom_directory` scans a directory for DICOM files, parses per-slice
//! and series-level metadata, filters to the most-populated series, and
//! returns a `DicomSeriesInfo` with typed metadata and geometry.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use super::dicomdir::try_read_dicomdir;
use super::parse::{parse_dicom_bytes, parse_dicom_file, parse_dicom_file_bytes};
use super::types::{DicomSeriesInfo, DicomSliceMetadata, SeriesFirstSeen};
use super::utils::is_likely_dicom_file;
use crate::format::dicom::networking::scp::StoredInstance;
use crate::format::dicom::object_model::{DicomObjectModel, DicomObjectNode, DicomTag};

mod finalize;
mod geometry;
mod thresholds;

use finalize::finalize_scanned_series;

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

    finalize_scanned_series(
        slices,
        None,
        per_file_dims,
        per_file_series_uids,
        first,
        path.to_path_buf(),
        |a, b| a.path.file_name().cmp(&b.path.file_name()),
        &format!("{:?}", path),
    )
}

/// Scan in-memory SCP-received [`StoredInstance`] values, extract metadata, and
/// return a typed series descriptor.
///
/// This is the zero-disk counterpart to `scan_dicom_directory`: instead of
/// scanning a filesystem directory, it parses Part 10 bytes constructed from
/// each `StoredInstance`, applies the same canonical-dimension filtering and
/// SeriesInstanceUID grouping, sorts slices spatially, and assembles geometry.
///
/// Each slice's `part10_bytes` field is populated so that pixel decoding can
/// proceed without re-opening a file.
///
/// # Invariants
/// - `instances` must be non-empty.
/// - All returned slices belong to the most-populated SeriesInstanceUID when
///   the input contains multiple series with the same image dimensions.
pub fn scan_dicom_instances(instances: &[StoredInstance]) -> Result<DicomSeriesInfo> {
    if instances.is_empty() {
        bail!("no DICOM instances provided for scanning");
    }

    // Parse metadata from each instance's Part 10 bytes.
    let mut slices: Vec<DicomSliceMetadata> = Vec::with_capacity(instances.len());
    let mut first = SeriesFirstSeen::default();
    let mut per_file_dims: Vec<(u32, u32)> = Vec::with_capacity(instances.len());
    let mut per_file_series_uids: Vec<Option<String>> = Vec::with_capacity(instances.len());
    let mut part10_bytes_vec: Vec<Vec<u8>> = Vec::with_capacity(instances.len());
    for inst in instances {
        let part10_bytes = inst.make_part10_bytes();
        if let Some((slice_meta, file_dim, file_series_uid)) =
            parse_dicom_bytes(&part10_bytes, &inst.sop_instance_uid, &mut first)
        {
            per_file_dims.push(file_dim);
            per_file_series_uids.push(file_series_uid);
            part10_bytes_vec.push(part10_bytes);
            slices.push(slice_meta);
        }
    }
    if slices.is_empty() {
        bail!("no DICOM instances could be parsed from the provided stored instances");
    }

    finalize_scanned_series(
        slices,
        Some(part10_bytes_vec),
        per_file_dims,
        per_file_series_uids,
        first,
        PathBuf::from("scp://series"),
        |a, b| a.sop_instance_uid.cmp(&b.sop_instance_uid),
        "SCP instances",
    )
}

/// Scan in-memory DICOM Part 10 byte payloads (e.g. from drag-and-drop),
/// extract metadata, and return a typed series descriptor.
///
/// This is the zero-disk counterpart to both `scan_dicom_directory` and
/// [`scan_dicom_instances`]: instead of scanning a filesystem directory or
/// constructing Part 10 bytes from `StoredInstance` values, it accepts
/// pre-existing Part 10 DICOM byte payloads directly.
///
/// Each slice's `part10_bytes` field is populated so that pixel decoding can
/// proceed without writing to disk.
///
/// # Invariants
/// - `files` must be non-empty.
/// - Each `(&str, &[u8])` pair is a `(name_hint, part10_bytes)` — the name
///   is used for diagnostics and as a synthetic path; the bytes must be a
///   valid DICOM Part 10 file (128-byte preamble + DICM magic + FMI + dataset).
/// - All returned slices belong to the most-populated SeriesInstanceUID when
///   the input contains multiple series with the same image dimensions.
pub fn scan_dicom_part10_bytes(files: &[(&str, &[u8])]) -> Result<DicomSeriesInfo> {
    if files.is_empty() {
        bail!("no DICOM byte payloads provided for scanning");
    }

    // Parse metadata from each Part 10 byte payload.
    let mut slices: Vec<DicomSliceMetadata> = Vec::with_capacity(files.len());
    let mut first = SeriesFirstSeen::default();
    let mut per_file_dims: Vec<(u32, u32)> = Vec::with_capacity(files.len());
    let mut per_file_series_uids: Vec<Option<String>> = Vec::with_capacity(files.len());
    let mut part10_bytes_vec: Vec<Vec<u8>> = Vec::with_capacity(files.len());
    for (name, bytes) in files {
        // Use a synthetic path based on the name hint.
        let synthetic_path = PathBuf::from(format!("dropped://{}", name));
        if let Some((slice_meta, file_dim, file_series_uid)) =
            parse_dicom_file_bytes(bytes, &synthetic_path, &mut first)
        {
            per_file_dims.push(file_dim);
            per_file_series_uids.push(file_series_uid);
            part10_bytes_vec.push(bytes.to_vec());
            slices.push(slice_meta);
        }
    }
    if slices.is_empty() {
        bail!("no DICOM byte payloads could be parsed from the provided files");
    }

    finalize_scanned_series(
        slices,
        Some(part10_bytes_vec),
        per_file_dims,
        per_file_series_uids,
        first,
        PathBuf::from("dropped://series"),
        |a, b| a.sop_instance_uid.cmp(&b.sop_instance_uid),
        "dropped byte payloads",
    )
}

/// Build a `DicomObjectModel` from the slice metadata for a series.
///
/// This constructs a lightweight object model populated with the key
/// per-instance tags (SOP Instance UID, Instance Number, Slice Location,
/// Image Position/Orientation, Pixel Spacing, Slice Thickness, SOP Class UID)
/// so downstream consumers can inspect series-level DICOM attributes without
/// re-parsing the original files.
fn build_series_object(path: &Path, slices: &[DicomSliceMetadata]) -> DicomObjectModel {
    let mut series_object = DicomObjectModel::with_source(path.to_path_buf());
    for slice in slices {
        if let Some(uid) = slice.sop_instance_uid.as_ref() {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0008, 0x0018),
                "UI",
                uid.as_str().to_string(),
            ));
        }
        if let Some(instance_number) = slice.instance_number {
            series_object.insert(DicomObjectNode::with_value(
                DicomTag::new(0x0020, 0x0013),
                "IS",
                instance_number,
            ));
        }
        if let Some(slice_location) = slice.slice_location {
            series_object.insert(DicomObjectNode::with_value(
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
            series_object.insert(DicomObjectNode::with_value(
                DicomTag::new(0x0018, 0x0050),
                "DS",
                slice_thickness,
            ));
        }
        if let Some(sop_class_uid) = slice.sop_class_uid.as_ref() {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0008, 0x0016),
                "UI",
                sop_class_uid.as_str().to_string(),
            ));
        }
    }
    series_object
}
