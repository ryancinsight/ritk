//! Geometry helpers for DICOM series post-processing.
//!
//! IOP/spacing consistency checks, gantry tilt synthesis, slice normal
//! computation, spatial sorting, and direction/origin/spacing assembly.

use super::super::geometry::{analyze_slice_spacing, cross, dot, normalize, slice_normal_from_iop};
use super::super::types::DicomSliceMetadata;
use super::thresholds::{
    AXIAL_IOP_THRESHOLD, GANTRY_TILT_MIN_DEGREES, IOP_CONSISTENCY_THRESHOLD,
    PIXEL_SPACING_CONSISTENCY_THRESHOLD,
};

/// Compute the slice normal from the first available IOP in the slice list.
pub(super) fn compute_slice_normal(slices: &[DicomSliceMetadata]) -> Option<[f64; 3]> {
    slices
        .iter()
        .find_map(|s| s.image_orientation_patient)
        .and_then(slice_normal_from_iop)
}

/// Sort slices by projection of IPP onto the slice normal, then instance
/// number, then the caller-provided tiebreaker.
pub(super) fn sort_slices_spatially(
    slices: &mut [DicomSliceMetadata],
    maybe_normal: Option<[f64; 3]>,
    tiebreaker: fn(&DicomSliceMetadata, &DicomSliceMetadata) -> std::cmp::Ordering,
) {
    slices.sort_by(|a, b| {
        let pos_a = match (a.image_position_patient, maybe_normal) {
            (Some(ipp), Some(n)) => dot(ipp, n),
            (Some(ipp), None) => ipp[2],
            (None, _) => f64::MAX,
        };
        let pos_b = match (b.image_position_patient, maybe_normal) {
            (Some(ipp), Some(n)) => dot(ipp, n),
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
            .then_with(|| tiebreaker(a, b))
    });
}

/// GantryDetectorTilt synthesis (GAP-R62-01):
///
/// When IOP is absent or effectively axial and `|tilt| > 0.01°`, synthesize
/// an oblique IOP from the tilt angle.
pub(super) fn synthesize_gantry_tilt(slices: &mut [DicomSliceMetadata]) {
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
                for slice in slices {
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

/// Cross-slice IOP consistency guard (DICOM PS3.3 C.7.6.1.1.1).
///
/// Emits a warning when any slice's IOP deviates from the first slice's IOP
/// by more than [`IOP_CONSISTENCY_THRESHOLD`].
pub(super) fn check_iop_consistency(slices: &[DicomSliceMetadata]) {
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
}

/// Cross-slice PixelSpacing consistency guard.
///
/// Emits a warning when any slice's pixel spacing deviates from the first
/// slice's spacing by more than [`PIXEL_SPACING_CONSISTENCY_THRESHOLD`].
pub(super) fn check_pixel_spacing_consistency(slices: &[DicomSliceMetadata]) {
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
}

/// Compute z-spacing using the median of adjacent-pair projected positions.
pub(super) fn compute_spacing_z(
    slices: &[DicomSliceMetadata],
    maybe_normal: Option<[f64; 3]>,
    fallback_thickness: Option<f64>,
) -> f64 {
    let positions: Vec<f64> = if let Some(n) = maybe_normal {
        slices
            .iter()
            .filter_map(|s| s.image_position_patient.map(|ipp| dot(ipp, n)))
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
        fallback_thickness.unwrap_or(1.0)
    }
}

/// Assemble the 3×3 direction cosine matrix (row-major `[f64; 9]`).
///
/// Convention: col 0 = NÌ‚, col 1 = F_c, col 2 = F_r.
pub(super) fn assemble_direction(slices: &[DicomSliceMetadata]) -> [f64; 9] {
    if let Some(ori) = slices.first().and_then(|s| s.image_orientation_patient) {
        let r = [ori[0], ori[1], ori[2]];
        let c = [ori[3], ori[4], ori[5]];
        let n = normalize(cross(r, c)).unwrap_or([0.0, 0.0, 1.0]);
        [
            n[0], n[1], n[2], ori[3], ori[4], ori[5], ori[0], ori[1], ori[2],
        ]
    } else {
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    }
}

/// Extract the origin from the first slice's ImagePositionPatient.
pub(super) fn assemble_origin(slices: &[DicomSliceMetadata]) -> [f64; 3] {
    slices
        .first()
        .and_then(|s| s.image_position_patient)
        .unwrap_or([0.0, 0.0, 0.0])
}
