//! Anatomical edge labels and intensity readout strings for viewport overlays.
//!
//! Labels follow the DICOM LPS convention: the positive axis directions are
//! left, posterior and superior.

use crate::ui::{axis_slice_dimensions, ViewTransform};

/// Anatomical label displayed at each viewport edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OrientationLabels {
    /// Label for the left viewport edge.
    pub(crate) left: &'static str,
    /// Label for the right viewport edge.
    pub(crate) right: &'static str,
    /// Label for the top viewport edge.
    pub(crate) top: &'static str,
    /// Label for the bottom viewport edge.
    pub(crate) bottom: &'static str,
}

/// Labels for an untransformed slice of `axis` under the 3x3 `direction` matrix.
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

/// Labels after `transform` moves the source edges onto the displayed edges.
pub(crate) fn orientation_labels_for_transform(
    axis: usize,
    direction: &[f64; 9],
    volume_shape: [usize; 3],
    transform: ViewTransform,
) -> OrientationLabels {
    let labels = orientation_labels(axis, direction);
    let Some((width, height)) = axis_slice_dimensions(volume_shape, axis) else {
        return labels;
    };
    if width == 0 || height == 0 || transform.is_identity() {
        return labels;
    }
    let source_size = [width, height];
    let output_size = transform.output_size(source_size);
    let coordinate = |extent: usize| u32::try_from(extent).map(f64::from);
    let (Ok(source_width), Ok(source_height), Ok(output_width)) = (
        coordinate(width),
        coordinate(height),
        coordinate(output_size[0]),
    ) else {
        return labels;
    };
    let source_edges = [
        (labels.left, [0.0, source_height * 0.5]),
        (labels.right, [source_width, source_height * 0.5]),
        (labels.top, [source_width * 0.5, 0.0]),
        (labels.bottom, [source_width * 0.5, source_height]),
    ];
    let mut mapped = [None; 4];
    for (label, point) in source_edges {
        let [x, y] = transform.source_to_output_coordinates(point, source_size);
        let slot = if x <= 0.5 {
            0
        } else if x >= output_width - 0.5 {
            1
        } else if y <= 0.5 {
            2
        } else {
            3
        };
        mapped[slot] = Some(label);
    }
    OrientationLabels {
        left: mapped[0].unwrap_or(labels.left),
        right: mapped[1].unwrap_or(labels.right),
        top: mapped[2].unwrap_or(labels.top),
        bottom: mapped[3].unwrap_or(labels.bottom),
    }
}

fn lps_label(v: [f64; 3], positive: bool) -> &'static str {
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

/// Format the pointer-position intensity label.
///
/// Returns `"Pointer SUV: {:.2}"` when `pointer_suv` is `Some`,
/// `"Pointer value: {:.0}"` when `pointer_intensity != 0.0`, or `""` otherwise.
pub(crate) fn format_pointer_str(pointer_intensity: f32, pointer_suv: Option<f32>) -> String {
    match pointer_suv {
        Some(s) => format!("Pointer SUV: {:.2}", s),
        None if pointer_intensity != 0.0 => format!("Pointer value: {:.0}", pointer_intensity),
        _ => String::new(),
    }
}

/// Format the cursor-position intensity label.
///
/// Returns `"Cursor SUV: {:.2}"` when `cursor_suv` is `Some`,
/// `"Cursor value: {:.0}"` when `cursor_value` is `Some`, or `""` otherwise.
pub(crate) fn format_cursor_str(cursor_value: Option<f32>, cursor_suv: Option<f32>) -> String {
    match (cursor_suv, cursor_value) {
        (Some(s), _) => format!("Cursor SUV: {:.2}", s),
        (None, Some(v)) => format!("Cursor value: {:.0}", v),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests;
