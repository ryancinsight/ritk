//! Shared post-processing logic for directory and instance DICOM scans.
//!
//! `finalize_scanned_series` implements the common pipeline after per-slice
//! metadata extraction:
//! geometry assembly and `DicomSeriesInfo` construction.

use std::path::PathBuf;

use super::super::types::{DicomSeriesInfo, DicomSliceMetadata, SeriesFirstSeen, SeriesGeometry};

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
/// - Input is one validated series with consistent image dimensions
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
/// - `first`: accumulated `SeriesFirstSeen` from the parse phase
/// - `series_path`: path to use in the returned `DicomSeriesInfo`
/// - `sort_tiebreaker`: final comparator for slice sorting (filename vs SOP UID)
pub(super) fn finalize_scanned_series(
    mut slices: Vec<DicomSliceMetadata>,

    first: SeriesFirstSeen,
    series_path: PathBuf,
    sort_tiebreaker: fn(&DicomSliceMetadata, &DicomSliceMetadata) -> std::cmp::Ordering,
) -> DicomSeriesInfo {
    // ── Geometry pipeline ──────────────────────────────────────────────

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
    DicomSeriesInfo {
        path: series_path,
        num_slices: metadata.slices.len(),
        metadata,
    }
}
