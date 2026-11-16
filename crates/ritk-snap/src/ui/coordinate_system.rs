//! Coordinate-system utilities for medical image display.
//!
//! This module provides SSOT helpers for:
//! - Anatomical frame conversion (`LPS` <-> `RAS`)
//! - DICOM patient position parsing (`(0018,5100)`)

// Re-export PatientPosition from ritk-io (SSOT — eliminates duplicate enum).
pub use ritk_io::PatientPosition;

/// Anatomical coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnatomicalFrame {
    /// Left-Posterior-Superior patient frame (DICOM/ITK convention).
    Lps,
    /// Right-Anterior-Superior patient frame (NIfTI/FSL convention).
    Ras,
}

impl AnatomicalFrame {
    /// Short display label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Lps => "LPS",
            Self::Ras => "RAS",
        }
    }
}

/// Convert a point from LPS to RAS.
///
/// Relation:
/// - `R = -L`
/// - `A = -P`
/// - `S = S`
#[inline]
pub fn lps_to_ras(lps: [f64; 3]) -> [f64; 3] {
    [-lps[0], -lps[1], lps[2]]
}

/// Convert a point from RAS to LPS.
///
/// This is the same sign flip as [`lps_to_ras`].
#[inline]
pub fn ras_to_lps(ras: [f64; 3]) -> [f64; 3] {
    [-ras[0], -ras[1], ras[2]]
}

/// Format a physical point in the requested anatomical frame.
pub fn format_point_mm(point: [f64; 3], frame: AnatomicalFrame) -> String {
    format!(
        "{} ({:.2}, {:.2}, {:.2}) mm",
        frame.label(),
        point[0],
        point[1],
        point[2]
    )
}

// ── Orientation labels ──────────────────────────────────────────────────────
//
// Both viewer shells derive edge labels from the direction cosine matrix
// columns. Convention per axis:
//
// | Axis | Fixed index | Left edge | Right edge | Top edge | Bottom edge |
// |------|-------------|-----------|------------|----------|-------------|
// | 0    | depth d     | R or L    | L or R     | A or P   | P or A      |
// | 1    | row r       | R or L    | L or R     | S or I   | I or S      |
// | 2    | col c       | A or P    | P or A     | S or I   | I or S      |
//
// Labels are determined by the dominant component of the relevant direction
// cosine column.

/// Patient-orientation letters for the four viewport edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OrientationLabels {
    pub(crate) left: &'static str,
    pub(crate) right: &'static str,
    pub(crate) top: &'static str,
    pub(crate) bottom: &'static str,
}

/// Edge labels for `axis` from the volume direction cosine columns.
pub(crate) fn orientation_labels(axis: usize, direction: &[f64; 9]) -> OrientationLabels {
    let col = |j: usize| -> [f64; 3] { [direction[j], direction[3 + j], direction[6 + j]] };

    let depth_axis = col(0);
    let row_axis = col(1);
    let col_axis = col(2);

    let (horiz, vert) = match axis {
        0 => (col_axis, row_axis),
        1 => (col_axis, depth_axis),
        _ => (row_axis, depth_axis),
    };

    OrientationLabels {
        left: lps_label(horiz, false),
        right: lps_label(horiz, true),
        top: lps_label(vert, false),
        bottom: lps_label(vert, true),
    }
}

/// Single orientation letter for a direction column.
///
/// The dominant absolute component selects the anatomical axis; the sign
/// selects the facing edge. `positive` chooses the high-side edge.
pub(crate) fn lps_label(v: [f64; 3], positive: bool) -> &'static str {
    let v = if positive { v } else { [-v[0], -v[1], -v[2]] };
    let abs = [v[0].abs(), v[1].abs(), v[2].abs()];
    let max_idx = if abs[0] >= abs[1] && abs[0] >= abs[2] {
        0
    } else if abs[1] >= abs[2] {
        1
    } else {
        2
    };
    match (max_idx, v[max_idx] >= 0.0) {
        (0, true) => "L",
        (0, false) => "R",
        (1, true) => "P",
        (1, false) => "A",
        (2, true) => "S",
        _ => "I",
    }
}

/// Edge slot for a mapped source-edge point: 0 left, 1 right, 2 top, 3 bottom.
///
/// Shared by the eframe and native shells so a transformed orientation label
/// lands on the same edge in both. The comparison runs in `f64`, which
/// represents every panel size below 2^53 pixels exactly; larger sizes
/// saturate, which real panels never reach.
pub(crate) fn edge_slot(output: [f64; 2], output_size: [usize; 2]) -> usize {
    let width = f64::from(u32::try_from(output_size[0]).unwrap_or(u32::MAX));
    if output[0] <= 0.5 {
        0
    } else if output[0] >= width - 0.5 {
        1
    } else if output[1] <= 0.5 {
        2
    } else {
        3
    }
}

/// Reassign labels to edges by mapped slot, keeping the unmapped label when a
/// slot stays empty. `slots` holds the [`edge_slot`] of the source left,
/// right, top, then bottom edge in order.
pub(crate) fn remap_labels_by_slots(
    labels: OrientationLabels,
    slots: [usize; 4],
) -> OrientationLabels {
    let source = [labels.left, labels.right, labels.top, labels.bottom];
    let mut mapped: [Option<&'static str>; 4] = [None; 4];
    for (label, slot) in source.into_iter().zip(slots) {
        if slot < mapped.len() {
            mapped[slot] = Some(label);
        }
    }
    OrientationLabels {
        left: mapped[0].unwrap_or(labels.left),
        right: mapped[1].unwrap_or(labels.right),
        top: mapped[2].unwrap_or(labels.top),
        bottom: mapped[3].unwrap_or(labels.bottom),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lps_to_ras_is_sign_flip_xy_only() {
        let lps = [10.0, -20.0, 30.0];
        assert_eq!(lps_to_ras(lps), [-10.0, 20.0, 30.0]);
    }

    #[test]
    fn ras_to_lps_is_inverse_of_lps_to_ras() {
        let lps = [12.5, 3.0, -7.25];
        let ras = lps_to_ras(lps);
        assert_eq!(ras_to_lps(ras), lps);
    }

    #[test]
    fn patient_position_parser_maps_standard_codes() {
        assert_eq!(
            PatientPosition::from_dicom_code("hfs"),
            PatientPosition::HeadFirstSupine
        );
        assert_eq!(
            PatientPosition::from_dicom_code("FFP"),
            PatientPosition::FeetFirstProne
        );
        assert_eq!(
            PatientPosition::from_dicom_code("HFDL"),
            PatientPosition::HeadFirstDecubitusLeft
        );
    }

    #[test]
    fn patient_position_parser_preserves_unknown_code() {
        use ritk_io::literal_arraystring;
        assert_eq!(
            PatientPosition::from_dicom_code("XYZ"),
            PatientPosition::Unknown(literal_arraystring::<4>("XYZ"))
        );
    }
}
