//! Overlay renderer unit tests.
//!
//! egui painter methods require a GPU context, so only pure-computation
//! helpers are tested here.

use super::*;

/// `anchor_pos` for LEFT_TOP must return (min.x + MARGIN, min.y + MARGIN).
#[test]
fn test_anchor_pos_left_top() {
    let rect = Rect::from_min_max(Pos2::new(10.0, 20.0), Pos2::new(110.0, 120.0));
    let pos = OverlayRenderer::anchor_pos(rect, Align2::LEFT_TOP);
    assert!(
        (pos.x - (10.0 + MARGIN)).abs() < 1e-4,
        "LEFT_TOP x must be rect.min.x + MARGIN"
    );
    assert!(
        (pos.y - (20.0 + MARGIN)).abs() < 1e-4,
        "LEFT_TOP y must be rect.min.y + MARGIN"
    );
}

/// `anchor_pos` for RIGHT_BOTTOM must return (max.x − MARGIN, max.y − MARGIN).
#[test]
fn test_anchor_pos_right_bottom() {
    let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(200.0, 100.0));
    let pos = OverlayRenderer::anchor_pos(rect, Align2::RIGHT_BOTTOM);
    assert!(
        (pos.x - (200.0 - MARGIN)).abs() < 1e-4,
        "RIGHT_BOTTOM x must be rect.max.x - MARGIN"
    );
    assert!(
        (pos.y - (100.0 - MARGIN)).abs() < 1e-4,
        "RIGHT_BOTTOM y must be rect.max.y - MARGIN"
    );
}

/// `anchor_pos` for CENTER_CENTER must return the rect centre exactly.
#[test]
fn test_anchor_pos_center_center() {
    let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(100.0, 80.0));
    let pos = OverlayRenderer::anchor_pos(rect, Align2::CENTER_CENTER);
    assert!(
        (pos.x - 50.0).abs() < 1e-4,
        "CENTER_CENTER x must be rect centre x = 50"
    );
    assert!(
        (pos.y - 40.0).abs() < 1e-4,
        "CENTER_CENTER y must be rect centre y = 40"
    );
}

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

/// `format_suv_string` produces `"Cursor SUV: 3.50"` for `Some(3.5)`
/// and `"Pointer SUV: 2.10"` for `Some(2.1)`, verifying the {:.2}
/// format contract.
#[test]
fn test_overlay_suv_display_format() {
    assert_eq!(
        OverlayRenderer::format_suv_string("Cursor SUV", Some(3.5)),
        "Cursor SUV: 3.50"
    );
    assert_eq!(
        OverlayRenderer::format_suv_string("Pointer SUV", Some(2.1)),
        "Pointer SUV: 2.10"
    );
}

/// `format_suv_string` returns an empty string for `None`, ensuring
/// non-PET volumes produce no SUV overlay text.
#[test]
fn test_overlay_suv_none_produces_empty() {
    assert_eq!(OverlayRenderer::format_suv_string("Cursor SUV", None), "");
    assert_eq!(OverlayRenderer::format_suv_string("Pointer SUV", None), "");
}

/// `format_suv_string` returns an empty string for NaN and ±Inf,
/// preventing non-physical display values in the overlay.
#[test]
fn test_overlay_suv_nonfinite_produces_empty() {
    assert_eq!(
        OverlayRenderer::format_suv_string("Cursor SUV", Some(f32::NAN)),
        ""
    );
    assert_eq!(
        OverlayRenderer::format_suv_string("Cursor SUV", Some(f32::INFINITY)),
        ""
    );
    assert_eq!(
        OverlayRenderer::format_suv_string("Cursor SUV", Some(f32::NEG_INFINITY)),
        ""
    );
}
