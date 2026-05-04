//! Annotation history panel SSOT — per-entry delete and CSV export.
//!
//! # Responsibilities
//!
//! - Render the completed-annotation list inside a vertical scroll area.
//! - Provide a "✕" delete button per row so the caller can remove exactly one
//!   annotation without rebuilding the whole list.
//! - Provide a "Clear All" button that signals full list removal.
//! - Provide an "Export CSV" button that returns a fully-formed CSV string the
//!   caller can write to disk or the clipboard.
//!
//! # Formal specification
//!
//! ```text
//! draw_annotation_panel : [Annotation] × Ui → AnnotationPanelAction
//!
//! AnnotationPanelAction ∈ { None, Delete(i), ClearAll, ExportCsv(s) }
//!
//! Invariants:
//!   Delete(i) is only produced when i < annotations.len()   (index validity)
//!   ExportCsv(s) where s = csv_for(annotations)             (deterministic)
//!   Only one non-None action is returned per call            (single event)
//! ```
//!
//! # CSV format
//!
//! ```text
//! index,type,primary_value,unit,extra
//! 0,Length,12.30,mm,
//! 1,Angle,45.00,deg,
//! 2,ROI Rect,100.00,HU,σ=10.00 area=50.00mm²
//! 3,ROI Ellipse,80.00,HU,σ=5.00 area=25.00mm²
//! 4,HU Point,200.00,HU,
//! ```

use egui::Ui;

use crate::tools::interaction::Annotation;

// ── Action returned to the caller ────────────────────────────────────────────

/// Action produced by [`draw_annotation_panel`].
///
/// At most one non-[`None`][AnnotationPanelAction::None] variant is returned
/// per [`draw_annotation_panel`] call (egui single-frame guarantee).
#[derive(Debug, Clone, PartialEq)]
pub enum AnnotationPanelAction {
    /// No user action this frame.
    None,
    /// The user clicked the delete button on row `i`.  `i < annotations.len()`
    /// is guaranteed at the call site inside [`draw_annotation_panel`].
    Delete(usize),
    /// The user clicked "Clear All".
    ClearAll,
    /// The user clicked "Export CSV".  The payload is the fully-formed CSV
    /// string (header + one row per annotation).
    ExportCsv(String),
}

// ── CSV serialisation ─────────────────────────────────────────────────────────

/// Build a CSV string for `annotations`.
///
/// Schema: `index,type,primary_value,unit,extra`
///
/// | Variant      | primary_value      | unit | extra                          |
/// |--------------|--------------------|------|-------------------------------|
/// | Length       | `length_mm`        | mm   | _(empty)_                     |
/// | Angle        | `angle_deg`        | deg  | _(empty)_                     |
/// | ROI Rect     | `mean`             | HU   | `σ=N area=Nmm²`               |
/// | ROI Ellipse  | `mean`             | HU   | `σ=N area=Nmm²`               |
/// | HU Point     | `value`            | HU   | _(empty)_                     |
///
/// The empty extra field keeps the column count constant at 5 across all
/// variants so spreadsheet applications can parse the file without special
/// handling.
pub fn csv_for(annotations: &[Annotation]) -> String {
    let mut out = String::from("index,type,primary_value,unit,extra\n");
    for (i, ann) in annotations.iter().enumerate() {
        let row = match ann {
            Annotation::Length { length_mm, .. } => {
                format!("{i},Length,{length_mm:.2},mm,\n")
            }
            Annotation::Angle { angle_deg, .. } => {
                format!("{i},Angle,{angle_deg:.2},deg,\n")
            }
            Annotation::RoiRect {
                mean,
                std_dev,
                area_mm2,
                ..
            } => {
                format!("{i},ROI Rect,{mean:.2},HU,σ={std_dev:.2} area={area_mm2:.2}mm²\n")
            }
            Annotation::RoiEllipse {
                mean,
                std_dev,
                area_mm2,
                ..
            } => {
                format!("{i},ROI Ellipse,{mean:.2},HU,σ={std_dev:.2} area={area_mm2:.2}mm²\n")
            }
            Annotation::HuPoint { value, .. } => {
                format!("{i},HU Point,{value:.2},HU,\n")
            }
        };
        out.push_str(&row);
    }
    out
}

// ── Row label ─────────────────────────────────────────────────────────────────

/// Format a short human-readable label for one annotation row in the panel.
fn annotation_label(i: usize, ann: &Annotation) -> String {
    match ann {
        Annotation::Length { length_mm, .. } => {
            format!("#{i}  Length: {length_mm:.1} mm")
        }
        Annotation::Angle { angle_deg, .. } => {
            format!("#{i}  Angle: {angle_deg:.2}°")
        }
        Annotation::RoiRect {
            mean,
            std_dev,
            min,
            max,
            area_mm2,
            ..
        } => {
            format!(
                "#{i}  ROI Rect  μ={mean:.1} σ={std_dev:.1} [{min:.0},{max:.0}] {area_mm2:.1}mm²"
            )
        }
        Annotation::RoiEllipse {
            mean,
            std_dev,
            min,
            max,
            area_mm2,
            ..
        } => {
            format!(
                "#{i}  ROI Ellipse  μ={mean:.1} σ={std_dev:.1} [{min:.0},{max:.0}] {area_mm2:.1}mm²"
            )
        }
        Annotation::HuPoint { value, pos } => {
            format!("#{i}  HU ({:.0},{:.0}): {value:.0}", pos[1], pos[0])
        }
    }
}

// ── Public render function ────────────────────────────────────────────────────

/// Render the annotation history panel inside `ui`.
///
/// Returns at most one [`AnnotationPanelAction`] per call.  The caller is
/// responsible for applying the returned action to the annotation list.
///
/// # Guarantees
///
/// - [`AnnotationPanelAction::Delete`]`(i)` satisfies `i < annotations.len()`.
/// - [`AnnotationPanelAction::ExportCsv`] contains the full CSV produced by
///   [`csv_for`] applied to the current `annotations` slice.
/// - When `annotations` is empty the function renders a placeholder and
///   returns [`AnnotationPanelAction::None`].
pub fn draw_annotation_panel(annotations: &[Annotation], ui: &mut Ui) -> AnnotationPanelAction {
    if annotations.is_empty() {
        ui.label("None.");
        return AnnotationPanelAction::None;
    }

    let mut action = AnnotationPanelAction::None;

    egui::ScrollArea::vertical()
        .id_source("annotation_list_scroll")
        .max_height(200.0)
        .show(ui, |ui| {
            for (i, ann) in annotations.iter().enumerate() {
                ui.horizontal(|ui| {
                    // Delete button: small "✕" on the left; index bound is
                    // guaranteed because i is produced by enumerate over the
                    // slice whose length equals annotations.len().
                    if ui.small_button("✕").clicked() {
                        action = AnnotationPanelAction::Delete(i);
                    }
                    ui.label(annotation_label(i, ann));
                });
            }
        });

    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("Clear All").clicked() {
            action = AnnotationPanelAction::ClearAll;
        }
        if ui.button("Export CSV").clicked() {
            action = AnnotationPanelAction::ExportCsv(csv_for(annotations));
        }
    });

    action
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::interaction::Annotation;

    // ── CSV determinism ───────────────────────────────────────────────────────

    #[test]
    fn csv_empty_annotations_returns_header_only() {
        let csv = csv_for(&[]);
        assert_eq!(
            csv,
            "index,type,primary_value,unit,extra\n",
            "empty annotation list must produce only the header row"
        );
    }

    #[test]
    fn csv_header_row_is_canonical() {
        let csv = csv_for(&[]);
        let first_line = csv.lines().next().expect("header line must exist");
        assert_eq!(
            first_line, "index,type,primary_value,unit,extra",
            "CSV header must match canonical schema"
        );
    }

    #[test]
    fn csv_row_count_equals_annotation_count_plus_header() {
        let annotations = vec![
            Annotation::Length {
                p1: [0.0, 0.0],
                p2: [3.0, 4.0],
                length_mm: 5.0,
            },
            Annotation::HuPoint {
                pos: [10.0, 20.0],
                value: 42.0,
            },
        ];
        let csv = csv_for(&annotations);
        let line_count = csv.lines().count();
        assert_eq!(
            line_count,
            annotations.len() + 1,
            "CSV must have one header row + one data row per annotation"
        );
    }

    #[test]
    fn csv_length_row_analytical() {
        // Analytical ground truth: length_mm = 5.0, index = 0
        // Expected: "0,Length,5.00,mm,"
        let ann = Annotation::Length {
            p1: [0.0, 0.0],
            p2: [3.0, 4.0],
            length_mm: 5.0,
        };
        let csv = csv_for(&[ann]);
        let data_row = csv.lines().nth(1).expect("data row must exist");
        assert_eq!(
            data_row, "0,Length,5.00,mm,",
            "Length annotation CSV row must match analytical format"
        );
    }

    #[test]
    fn csv_angle_row_analytical() {
        // Analytical: angle_deg = 90.0, index = 0
        // Expected: "0,Angle,90.00,deg,"
        let ann = Annotation::Angle {
            p1: [0.0, 0.0],
            p2: [0.0, 1.0],
            p3: [1.0, 1.0],
            angle_deg: 90.0,
        };
        let csv = csv_for(&[ann]);
        let data_row = csv.lines().nth(1).expect("data row must exist");
        assert_eq!(
            data_row, "0,Angle,90.00,deg,",
            "Angle annotation CSV row must match analytical format"
        );
    }

    #[test]
    fn csv_roi_rect_row_analytical() {
        // Analytical: mean=100.0, σ=10.0, area=50.0, index=0
        // Expected: "0,ROI Rect,100.00,HU,σ=10.00 area=50.00mm²"
        let ann = Annotation::RoiRect {
            top_left: [0.0, 0.0],
            bottom_right: [5.0, 10.0],
            mean: 100.0,
            std_dev: 10.0,
            min: 80.0,
            max: 120.0,
            area_mm2: 50.0,
        };
        let csv = csv_for(&[ann]);
        let data_row = csv.lines().nth(1).expect("data row must exist");
        assert_eq!(
            data_row, "0,ROI Rect,100.00,HU,σ=10.00 area=50.00mm²",
            "ROI Rect CSV row must match analytical format"
        );
    }

    #[test]
    fn csv_roi_ellipse_row_analytical() {
        // Analytical: mean=80.0, σ=5.0, area=25.0, index=0
        // Expected: "0,ROI Ellipse,80.00,HU,σ=5.00 area=25.00mm²"
        let ann = Annotation::RoiEllipse {
            center: [5.0, 5.0],
            radii: [3.0, 2.0],
            mean: 80.0,
            std_dev: 5.0,
            min: 60.0,
            max: 100.0,
            area_mm2: 25.0,
        };
        let csv = csv_for(&[ann]);
        let data_row = csv.lines().nth(1).expect("data row must exist");
        assert_eq!(
            data_row, "0,ROI Ellipse,80.00,HU,σ=5.00 area=25.00mm²",
            "ROI Ellipse CSV row must match analytical format"
        );
    }

    #[test]
    fn csv_hu_point_row_analytical() {
        // Analytical: value=200.0, index=0
        // Expected: "0,HU Point,200.00,HU,"
        let ann = Annotation::HuPoint {
            pos: [10.0, 20.0],
            value: 200.0,
        };
        let csv = csv_for(&[ann]);
        let data_row = csv.lines().nth(1).expect("data row must exist");
        assert_eq!(
            data_row, "0,HU Point,200.00,HU,",
            "HU Point CSV row must match analytical format"
        );
    }

    #[test]
    fn csv_multi_annotation_indices_are_sequential() {
        // Analytical: three annotations → data rows must carry indices 0, 1, 2
        let annotations = vec![
            Annotation::Length {
                p1: [0.0, 0.0],
                p2: [1.0, 0.0],
                length_mm: 1.0,
            },
            Annotation::HuPoint {
                pos: [0.0, 0.0],
                value: 0.0,
            },
            Annotation::Angle {
                p1: [0.0, 0.0],
                p2: [1.0, 0.0],
                p3: [1.0, 1.0],
                angle_deg: 90.0,
            },
        ];
        let csv = csv_for(&annotations);
        let rows: Vec<&str> = csv.lines().skip(1).collect();
        assert_eq!(rows.len(), 3, "expected 3 data rows");
        assert!(rows[0].starts_with("0,"), "first data row must have index 0");
        assert!(rows[1].starts_with("1,"), "second data row must have index 1");
        assert!(rows[2].starts_with("2,"), "third data row must have index 2");
    }

    // ── AnnotationPanelAction equality ────────────────────────────────────────

    #[test]
    fn action_none_equals_none() {
        assert_eq!(AnnotationPanelAction::None, AnnotationPanelAction::None);
    }

    #[test]
    fn action_delete_carries_index() {
        let action = AnnotationPanelAction::Delete(3);
        match action {
            AnnotationPanelAction::Delete(i) => assert_eq!(i, 3, "Delete must carry the exact index"),
            _ => panic!("expected Delete variant"),
        }
    }

    #[test]
    fn action_clear_all_is_distinct_from_delete() {
        assert_ne!(
            AnnotationPanelAction::ClearAll,
            AnnotationPanelAction::Delete(0),
            "ClearAll and Delete(0) must be distinct actions"
        );
    }

    #[test]
    fn action_export_csv_payload_matches_csv_for() {
        let annotations = vec![Annotation::Length {
            p1: [0.0, 0.0],
            p2: [3.0, 4.0],
            length_mm: 5.0,
        }];
        let expected_csv = csv_for(&annotations);
        let action = AnnotationPanelAction::ExportCsv(expected_csv.clone());
        match action {
            AnnotationPanelAction::ExportCsv(payload) => {
                assert_eq!(
                    payload, expected_csv,
                    "ExportCsv payload must equal csv_for output"
                );
            }
            _ => panic!("expected ExportCsv variant"),
        }
    }

    // ── annotation_label determinism ──────────────────────────────────────────

    #[test]
    fn label_length_format() {
        let ann = Annotation::Length {
            p1: [0.0, 0.0],
            p2: [3.0, 4.0],
            length_mm: 5.0,
        };
        let label = annotation_label(0, &ann);
        assert!(
            label.contains("5.0 mm"),
            "length label must contain formatted length: got '{label}'"
        );
        assert!(label.starts_with("#0"), "label must begin with index");
    }

    #[test]
    fn label_angle_format() {
        let ann = Annotation::Angle {
            p1: [0.0, 0.0],
            p2: [1.0, 0.0],
            p3: [1.0, 1.0],
            angle_deg: 90.0,
        };
        let label = annotation_label(1, &ann);
        assert!(
            label.contains("90.00°"),
            "angle label must contain formatted angle: got '{label}'"
        );
        assert!(label.starts_with("#1"), "label must begin with index");
    }

    #[test]
    fn label_hu_point_format() {
        let ann = Annotation::HuPoint {
            pos: [10.0, 20.0],
            value: 150.0,
        };
        let label = annotation_label(2, &ann);
        assert!(
            label.contains("150"),
            "HU point label must contain intensity value: got '{label}'"
        );
        assert!(label.starts_with("#2"), "label must begin with index");
    }
}
