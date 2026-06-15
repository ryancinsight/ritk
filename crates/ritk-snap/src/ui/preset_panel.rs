//! Window/level preset quick-select button strip.
//!
//! # Purpose
//!
//! This module is the **canonical SSOT** for rendering a row of clinical
//! window/level preset buttons in the W/L sidebar panel, providing
//! ITK-SNAP-parity one-click preset application.
//!
//! # Mathematical specification
//!
//! [`draw_preset_buttons`] is a **pure render function**:
//!
//! ```text
//! draw_preset_buttons : ([WindowPreset], &mut Ui) → Option<WindowPreset>
//!
//! Post-condition:
//!   result = Some(p)  ⟺  exactly one button for preset p was clicked this frame
//!   result = None     ⟺  no button was clicked this frame
//!
//! Purity: the function does not mutate any state visible to the caller;
//! all state transitions are the caller's responsibility upon receiving Some(p).
//! ```
//!
//! The returned `WindowPreset` carries the full `(center, width)` pair already
//! validated by `WindowPreset::ct_presets()` / `WindowPreset::mr_presets()`:
//! `width > 0` is guaranteed by those constructors.
//!
//! # Layout contract
//!
//! Buttons are rendered in a [`egui::ScrollArea`] horizontal strip using
//! [`egui::Ui::small_button`] so they fit the compact sidebar width without
//! truncation. Button labels are the preset `name` field verbatim. The scroll
//! area prevents overflow when the preset list is longer than the available
//! panel width.

use egui::Ui;

use crate::ui::window_presets::WindowPreset;

// ── draw_preset_buttons ───────────────────────────────────────────────────────

/// Render a horizontal scrollable strip of window/level preset buttons.
///
/// # Parameters
/// - `presets` — ordered slice of [`WindowPreset`] values; empty slices render
///   nothing and return `None`.
/// - `ui` — the egui [`Ui`] context for this frame.
///
/// # Returns
/// `Some(preset)` when exactly one button was clicked this frame; `None`
/// when no button was activated.
///
/// # Formal contract
/// - Output depends on `presets` and the egui frame input state; not on any
///   mutable global or module-level variable.
/// - The returned preset's `(center, width)` values are identical to the
///   corresponding entry in `presets` — no transformation is applied.
/// - If `presets` is empty, returns `None` without rendering any widget.
pub fn draw_preset_buttons(presets: &[WindowPreset], ui: &mut Ui) -> Option<WindowPreset> {
    if presets.is_empty() {
        return None;
    }
    let mut selected: Option<WindowPreset> = None;
    egui::ScrollArea::horizontal()
        .id_source("preset_scroll")
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for &preset in presets {
                    if ui.small_button(preset.name).clicked() {
                        selected = Some(preset);
                    }
                }
            });
        });
    selected
}

#[cfg(test)]
#[path = "tests_preset_panel.rs"]
mod tests;
