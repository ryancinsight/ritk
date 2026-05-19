//! Image processing filter selection panel.
//!
//! Exposes a compact egui widget that lets the user choose a [`FilterKind`]
//! and configure its parameters. The caller is responsible for wiring the
//! "Apply" confirmation into the application state.
//!
//! # Design contract
//!
//! - [`show_filter_panel`] is a pure egui widget: it modifies `active_filter`
//!   in-place and returns `true` exactly when the user clicks "Apply".
//! - All numeric controls are clamped to analytically valid ranges:
//!   - Gaussian σ ∈ [0.1, 20.0] mm
//!   - Median radius ∈ [0, 10] voxels
//!   - CLAHE tile grid ∈ [1, 32] per axis; clip limit ∈ [1.0, 200.0]
//!   - HistEq bins ∈ [2, 1024]
//!   - UnsharpMask σ ∈ [0.1, 10.0] mm; amount ∈ [0.0, 5.0]; threshold ∈ [0.0, 100.0]
//!   - GradientAnisotropicDiffusion iterations ∈ [1, 50]; time_step ∈ [0.01, 0.1667]; conductance ∈ [0.1, 100.0]
//!   - ConnectedComponents background_value (any f32); connectivity_26 boolean
//!   - RelabelComponents minimum_object_size ∈ [0, MAX_u32] voxels
//!   - MultiOtsuThreshold num_classes ∈ [2, 8]
//!   - BinaryErode radius ∈ [0, 10]; foreground_value ∈ any f32
//!   - BinaryDilate radius ∈ [0, 10]; foreground_value ∈ any f32
//!   - BinaryClosing radius ∈ [0, 10]; foreground_value ∈ any f32
//!   - BinaryOpening radius ∈ [0, 10]; foreground_value ∈ any f32
//!   - GrayscaleClosing radius ∈ [0, 10] voxels
//!   - GrayscaleOpening radius ∈ [0, 10] voxels
//!   - GrayscaleFillhole (no parameters)
//! - The widget does not mutate the image; it only modifies the
//!   `FilterKind` selector held by the caller.

use crate::FilterKind;

mod controls;
mod controls_cpr;
mod controls_geom;
mod controls_morph;
mod controls_pointwise;
mod selector;

#[cfg(test)]
mod tests_smoothing;

#[cfg(test)]
mod tests_integrity;

/// Display the filter selection panel inside `ui`.
///
/// Returns `true` exactly when the user clicks the **Apply** button.
///
/// # Parameters
/// - `ui`: mutable reference to the egui [`Ui`] context.
/// - `active_filter`: mutable reference to the currently configured
///   [`FilterKind`]. Updated in-place as the user changes controls.
pub fn show_filter_panel(ui: &mut egui::Ui, active_filter: &mut FilterKind) -> bool {
    let mut apply = false;
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.label(egui::RichText::new("Image Processing").strong());
        ui.separator();

        // ── Filter selector ────────────────────────────────────────────────
        selector::show_selector(ui, active_filter);

        ui.add_space(4.0);

        // ── Per-filter parameter controls ──────────────────────────────────
        if !controls::show_controls(ui, active_filter)
            && !controls_morph::show_controls(ui, active_filter)
            && !controls_geom::show_controls(ui, active_filter)
            && !controls_cpr::show_controls(ui, active_filter)
        {
            controls_pointwise::show_controls(ui, active_filter);
        }

        ui.add_space(6.0);

        // ── Apply button ───────────────────────────────────────────────────
        if ui
            .add(egui::Button::new("Apply").min_size(egui::vec2(80.0, 22.0)))
            .clicked()
        {
            apply = true;
        }
    });
    apply
}
