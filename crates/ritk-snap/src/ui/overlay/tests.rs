//! Orientation label and intensity readout tests.

use super::*;
use crate::ui::{RotationSteps, ViewTransform};

#[test]
fn test_lps_label_selects_dominant_signed_axis() {
    assert_eq!(lps_label([0.9, 0.1, 0.0], true), "L");
    assert_eq!(lps_label([0.0, -2.0, 0.5], true), "A");
    assert_eq!(lps_label([0.0, 0.2, -3.0], true), "I");
}

#[test]
fn test_orientation_labels_axial_standard_axes() {
    let direction = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let labels = orientation_labels(0, &direction);
    assert_eq!(labels.left, "R");
    assert_eq!(labels.right, "L");
    assert_eq!(labels.top, "A");
    assert_eq!(labels.bottom, "P");
}

#[test]
fn test_orientation_labels_coronal_standard_axes() {
    let direction = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let labels = orientation_labels(1, &direction);
    assert_eq!(labels.left, "R");
    assert_eq!(labels.right, "L");
    assert_eq!(labels.top, "I");
    assert_eq!(labels.bottom, "S");
}

#[test]
fn test_orientation_labels_sagittal_standard_axes() {
    let direction = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let labels = orientation_labels(2, &direction);
    assert_eq!(labels.left, "A");
    assert_eq!(labels.right, "P");
    assert_eq!(labels.top, "I");
    assert_eq!(labels.bottom, "S");
}

#[test]
fn transformed_orientation_labels_follow_displayed_edges() {
    let direction = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let flipped = orientation_labels_for_transform(
        0,
        &direction,
        [4, 6, 8],
        ViewTransform {
            flip_h: true,
            flip_v: false,
            rotation: RotationSteps::Zero,
        },
    );
    assert_eq!(
        flipped,
        OrientationLabels {
            left: "L",
            right: "R",
            top: "A",
            bottom: "P",
        }
    );

    let rotated = orientation_labels_for_transform(
        0,
        &direction,
        [4, 6, 8],
        ViewTransform {
            flip_h: false,
            flip_v: false,
            rotation: RotationSteps::Ninety,
        },
    );
    assert_eq!(
        rotated,
        OrientationLabels {
            left: "P",
            right: "A",
            top: "R",
            bottom: "L",
        }
    );
}

// ── format_pointer_str ────────────────────────────────────────────────────────

#[test]
fn format_pointer_str_zero_intensity_no_suv_returns_empty() {
    assert!(format_pointer_str(0.0, None).is_empty());
}

#[test]
fn format_pointer_str_nonzero_intensity_no_suv_shows_value() {
    assert_eq!(format_pointer_str(512.0, None), "Pointer value: 512");
}

#[test]
fn format_pointer_str_with_suv_shows_suv_label() {
    assert_eq!(
        format_pointer_str(5000.0, Some(1.89_f32)),
        "Pointer SUV: 1.89"
    );
}

#[test]
fn format_pointer_str_zero_intensity_with_suv_still_shows_suv() {
    assert_eq!(format_pointer_str(0.0, Some(2.5_f32)), "Pointer SUV: 2.50");
}

// ── format_cursor_str ─────────────────────────────────────────────────────────

#[test]
fn format_cursor_str_none_cursor_none_suv_returns_empty() {
    assert!(format_cursor_str(None, None).is_empty());
}

#[test]
fn format_cursor_str_cursor_only_shows_value() {
    assert_eq!(format_cursor_str(Some(100.0), None), "Cursor value: 100");
}

#[test]
fn format_cursor_str_suv_takes_priority_over_cursor_value() {
    assert_eq!(
        format_cursor_str(Some(5000.0), Some(1.89_f32)),
        "Cursor SUV: 1.89"
    );
}
