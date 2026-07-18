//! Annotation history panel SSOT â€” per-entry delete and CSV export.
//!
//! # Responsibilities
//!
//! - Render the completed-annotation list inside a vertical scroll area.
//! - Provide a "âœ•" delete button per row so the caller can remove exactly one
//!   annotation without rebuilding the whole list.
//! - Provide a "Clear All" button that signals full list removal.
//! - Provide an "Export CSV" button that returns a fully-formed CSV string the
//!   caller can write to disk or the clipboard.
//!
//! # Formal specification
//!
//! ```text
//! draw_annotation_panel : [Annotation] Ã— Ui â†’ AnnotationPanelAction
//!
//! AnnotationPanelAction âˆˆ { None, Delete(i), ClearAll, ExportCsv(s) }
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
//! 2,ROI Rect,100.00,HU,Ïƒ=10.00 area=50.00mmÂ²
//! 3,ROI Ellipse,80.00,HU,Ïƒ=5.00 area=25.00mmÂ²
//! 4,HU Point,200.00,HU,
//! ```

use egui::Ui;

use crate::tools::interaction::Annotation;

// â”€â”€ Action returned to the caller â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    ExportCsv(String) }

// â”€â”€ CSV serialisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Build a CSV string for `annotations`.
///
/// Schema: `index,type,primary_value,unit,extra`
///
/// | Variant      | primary_value      | unit | extra                          |
/// |--------------|--------------------|------|-------------------------------|
/// | Length       | `length_mm`        | mm   | _(empty)_                     |
/// | Angle        | `angle_deg`        | deg  | _(empty)_                     |
/// | ROI Rect     | `mean`             | HU   | `Ïƒ=N area=NmmÂ²`               |
/// | ROI Ellipse  | `mean`             | HU   | `Ïƒ=N area=NmmÂ²`               |
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
                format!("{i},ROI Rect,{mean:.2},HU,Ïƒ={std_dev:.2} area={area_mm2:.2}mmÂ²\n")
            }
            Annotation::RoiEllipse {
                mean,
                std_dev,
                area_mm2,
                ..
            } => {
                format!("{i},ROI Ellipse,{mean:.2},HU,Ïƒ={std_dev:.2} area={area_mm2:.2}mmÂ²\n")
            }
            Annotation::HuPoint { value, .. } => {
                format!("{i},HU Point,{value:.2},HU,\n")
            }
        };
        out.push_str(&row);
    }
    out
}

// â”€â”€ Row label â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Format a short human-readable label for one annotation row in the panel.
fn annotation_label(i: usize, ann: &Annotation) -> String {
    match ann {
        Annotation::Length { length_mm, .. } => {
            format!("#{i}  Length: {length_mm:.1} mm")
        }
        Annotation::Angle { angle_deg, .. } => {
            format!("#{i}  Angle: {angle_deg:.2}Â°")
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
                "#{i}  ROI Rect  Î¼={mean:.1} Ïƒ={std_dev:.1} [{min:.0},{max:.0}] {area_mm2:.1}mmÂ²"
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
                "#{i}  ROI Ellipse  Î¼={mean:.1} Ïƒ={std_dev:.1} [{min:.0},{max:.0}] {area_mm2:.1}mmÂ²"
            )
        }
        Annotation::HuPoint { value, pos } => {
            format!("#{i}  HU ({:.0},{:.0}): {value:.0}", pos[1], pos[0])
        }
    }
}

// â”€â”€ Public render function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
                    // Delete button: small "âœ•" on the left; index bound is
                    // guaranteed because i is produced by enumerate over the
                    // slice whose length equals annotations.len().
                    if ui.small_button("âœ•").clicked() {
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

#[cfg(test)]
#[path = "annotation_panel/tests.rs"]
mod tests;
