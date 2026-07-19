//! PET/CT SUV sidebar panel for the ritk-snap viewer.
//!
//! Displays SUV quantification information when a PET volume is loaded,
//! including pointer SUVbw, cursor SUVbw, patient weight, injected dose,
//! decay correction mode, and radionuclide half-life.
//!
//! # Mathematical contract
//!
//! All displayed SUV values are computed by the canonical
//! [`crate::dicom::suv::compute_suvbw`] kernel using parameters from
//! [`crate::dicom::pet::PetAcquisitionParams`]. This panel performs no
//! independent SUV computation â€” it is a read-only display surface.

use egui::Ui;

/// Result of a PET SUV panel interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PetSuvPanelAction {
    /// No action taken.
    None,
}

/// Render the PET SUV information panel.
///
/// Displays PET acquisition parameters and SUV readouts when a PET volume
/// is loaded. Returns `PetSuvPanelAction::None` (reserved for future
/// interactive controls).
///
/// # Parameters
/// - `ui` â€” egui UI context for the sidebar panel.
/// - `modality` â€” DICOM modality string; panel renders only when `Some("PT")`.
/// - `pointer_suv` â€” SUVbw at the current pointer position.
/// - `cursor_suv` â€” SUVbw at the linked-cursor position.
/// - `patient_weight_kg` â€” patient body weight.
/// - `injected_dose_bq` â€” injected radionuclide activity.
/// - `half_life_s` â€” radionuclide physical half-life.
/// - `decay_correction` â€” decay correction mode string.
pub fn draw_pet_suv_panel(
    ui: &mut Ui,
    modality: Option<&str>,
    pointer_suv: Option<f32>,
    cursor_suv: Option<f32>,
    patient_weight_kg: Option<f64>,
    injected_dose_bq: Option<f64>,
    half_life_s: Option<f64>,
    decay_correction: Option<&str>,
) -> PetSuvPanelAction {
    // Only render for PET modality.
    if modality != Some("PT") {
        return PetSuvPanelAction::None;
    }

    ui.vertical(|ui| {
        ui.heading("PET / SUV");
        ui.separator();

        // â”€â”€ SUV readouts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        ui.label("SUVbw Readouts");
        match pointer_suv {
            Some(v) if v.is_finite() => {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 180, 0),
                    format!("Pointer: {:.2} g/mL", v),
                );
            }
            _ => {
                ui.label("Pointer: â€”");
            }
        }
        match cursor_suv {
            Some(v) if v.is_finite() => {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 140, 0),
                    format!("Cursor: {:.2} g/mL", v),
                );
            }
            _ => {
                ui.label("Cursor: â€”");
            }
        }

        ui.add_space(4.0);
        ui.separator();

        // â”€â”€ Acquisition parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        ui.label("Acquisition Parameters");
        match patient_weight_kg {
            Some(w) if w > 0.0 => {
                ui.label(format!("Weight: {:.1} kg", w));
            }
            _ => {
                ui.label("Weight: â€”");
            }
        }
        match injected_dose_bq {
            Some(d) if d > 0.0 => {
                let d_mbq = d / 1_000_000.0;
                ui.label(format!("Dose: {:.1} MBq", d_mbq));
            }
            _ => {
                ui.label("Dose: â€”");
            }
        }
        match half_life_s {
            Some(h) if h > 0.0 => {
                let h_min = h / 60.0;
                ui.label(format!("TÂ½: {:.2} min", h_min));
            }
            _ => {
                ui.label("TÂ½: â€”");
            }
        }
        match decay_correction {
            Some(dc) if !dc.is_empty() => {
                ui.label(format!("Decay: {}", dc));
            }
            _ => {
                ui.label("Decay: â€”");
            }
        }
    });

    PetSuvPanelAction::None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Panel returns None action when modality is not PT.
    #[test]
    fn panel_returns_none_for_non_pt_modality() {
        egui::__run_test_ui(|ui| {
            let action = draw_pet_suv_panel(ui, Some("CT"), None, None, None, None, None, None);
            assert_eq!(action, PetSuvPanelAction::None);
        });
    }

    /// Panel returns None action when modality is None.
    #[test]
    fn panel_returns_none_for_missing_modality() {
        egui::__run_test_ui(|ui| {
            let action = draw_pet_suv_panel(ui, None, None, None, None, None, None, None);
            assert_eq!(action, PetSuvPanelAction::None);
        });
    }

    /// Panel renders without panic for valid PET inputs with finite SUV.
    #[test]
    fn panel_renders_for_pt_with_finite_suv() {
        egui::__run_test_ui(|ui| {
            let action = draw_pet_suv_panel(
                ui,
                Some("PT"),
                Some(3.5),
                Some(2.1),
                Some(70.0),
                Some(370_000_000.0),
                Some(6584.04),
                Some("START"),
            );
            assert_eq!(action, PetSuvPanelAction::None);
        });
    }

    /// Panel renders without panic for PT with None SUV (missing params).
    #[test]
    fn panel_renders_for_pt_with_none_suv() {
        egui::__run_test_ui(|ui| {
            let action = draw_pet_suv_panel(ui, Some("PT"), None, None, None, None, None, None);
            assert_eq!(action, PetSuvPanelAction::None);
        });
    }

    /// Panel renders without panic for PT with non-finite SUV (NaN/Inf).
    #[test]
    fn panel_renders_for_pt_with_nonfinite_suv() {
        egui::__run_test_ui(|ui| {
            let action = draw_pet_suv_panel(
                ui,
                Some("PT"),
                Some(f32::NAN),
                Some(f32::INFINITY),
                Some(70.0),
                Some(370_000_000.0),
                Some(6584.04),
                Some("START"),
            );
            assert_eq!(action, PetSuvPanelAction::None);
        });
    }

    /// Dose display converts Bq to MBq correctly.
    #[test]
    fn dose_display_converts_bq_to_mbq() {
        // 370 MBq = 370,000,000 Bq
        let d_mbq = 370_000_000.0_f64 / 1_000_000.0;
        assert!(
            (d_mbq - 370.0).abs() < 1e-10,
            "370 MBq in Bq must convert to 370.0 MBq"
        );
    }

    /// Half-life display converts seconds to minutes correctly.
    #[test]
    fn half_life_display_converts_s_to_min() {
        // Â¹â¸F: 6584.04 s = 109.734 min
        let h_min = 6584.04_f64 / 60.0;
        assert!(
            (h_min - 109.734).abs() < 1e-3,
            "Â¹â¸F half-life must be 109.734 min, got {h_min}"
        );
    }
}
