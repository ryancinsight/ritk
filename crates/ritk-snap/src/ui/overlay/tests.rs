//! Overlay renderer unit tests.
//!
//! Painter shapes are inspected without a GPU; native capture verifies raster output.

use super::*;
use crate::ui::{RotationSteps, ViewTransform};

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

#[test]
fn overlay_fits_or_discloses_complete_metadata_at_small_sizes() {
    use crate::dicom::loader::{load_volume_from_path, tests::fixtures};

    let root = tempfile::tempdir().expect("study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write study");
    let mut volume = load_volume_from_path(root.path()).expect("load study");
    for (width, height, patient, series, overflow) in [
        (176.0, 88.0, "Example Patient", "Acquisition", true),
        (176.0, 264.0, "Example Patient", "Acquisition", false),
        (352.0, 264.0, "Example Patient", "Acquisition", false),
        (
            352.0,
            264.0,
            &"Patient component ".repeat(32),
            &"Series component ".repeat(32),
            true,
        ),
        (8.0, 8.0, "Example Patient", "Acquisition", true),
    ] {
        volume.patient_name = Some(patient.to_owned());
        volume.series_description = Some(series.to_owned());
        let context = egui::Context::default();
        let rect = Rect::from_min_size(Pos2::new(20.0, 20.0), egui::vec2(width, height));
        let mut disclosure = None;
        let output = context.run(egui::RawInput::default(), |context| {
            egui::CentralPanel::default().show(context, |ui| {
                let painter = ui.painter().clone();
                disclosure = OverlayRenderer::draw(
                    &painter,
                    rect,
                    &volume,
                    OverlayContext {
                        axis: 0,
                        slice_index: 1,
                        wl: WindowLevel::new(60.0, 400.0),
                        zoom: 1.0,
                        cursor_value: Some(260.0),
                        pointer_intensity: 40.0,
                        pointer_suv: None,
                        cursor_suv: None,
                        view_transform: ViewTransform::default(),
                    },
                );
                if let Some(details) = &disclosure {
                    OverlayRenderer::show_details(ui, rect, details);
                }
            });
        });
        let texts: Vec<_> = output
            .shapes
            .iter()
            .filter_map(|shape| match &shape.shape {
                egui::Shape::Text(text) => Some(text),
                _ => None,
            })
            .collect();
        if overflow {
            let details = disclosure.expect("overflow must disclose metadata");
            for expected in [
                patient,
                series,
                "Spacing:",
                "Dims:",
                "W:400 C:60",
                "Zoom: 100%",
                "Cursor value: 260",
                "Pointer value: 40",
                "Orientation: left",
            ] {
                assert!(details.contains(expected), "missing disclosure {expected}");
            }
            assert_eq!(texts.len(), 1);
            assert_eq!(texts[0].galley.text(), "Details");
            let bounds = Rect::from_min_size(texts[0].pos, texts[0].galley.size());
            if width > 8.0 {
                assert!(rect.contains_rect(bounds));
            } else {
                assert!(
                    !rect.contains_rect(bounds),
                    "tiny images use surrounding UI control"
                );
            }
        } else {
            assert_eq!(disclosure, None);
            assert_eq!(texts.len(), 8);
            assert_eq!(texts[0].galley.text(), patient);
            assert!(texts[1].galley.text().contains(series));
            let backings: Vec<_> = output
                .shapes
                .iter()
                .filter_map(|shape| match &shape.shape {
                    egui::Shape::Rect(backing) if backing.fill == Color32::BLACK => {
                        Some(backing.rect)
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(backings.len(), 8);
            for (index, (text, backing)) in texts.iter().zip(&backings).enumerate() {
                let bounds = Rect::from_min_size(text.pos, text.galley.size());
                assert!(backing.contains_rect(bounds));
                assert!(rect.contains_rect(*backing));
                assert!(text.galley.job.sections.iter().all(|section| {
                    section.format.color == OVERLAY_TEXT_COLOR
                        || section.format.color == ORIENT_LABEL_COLOR
                }));
                for other in &backings[index + 1..] {
                    assert!(!backing.intersects(*other));
                }
            }
        }
    }
}

mod details;
