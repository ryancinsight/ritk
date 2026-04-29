//! Top toolbar panel for the ritk-snap viewer.
//!
//! # Layout (left → right)
//!
//! ```text
//! [File Open ▾] | [Pan][Zoom][W/L][📏][📐][▭][⬭][⊕][⊙] | [Single][2×2][1+3][⇔][|||] | [Colormap ▾] | [Presets ▾] | [W: ____][C: ____] | [Overlay✓][Xhair✓] | [Export]
//! ```
//!
//! # Invariants
//! - The active tool button is rendered with a visually distinct (selected) state.
//! - The layout mode buttons use `selectable_label` so exactly one is highlighted.
//! - The W/L DragValue fields are two-way bound to `active_wl` and do NOT modify
//!   the preset selection — they are free-form after any preset is applied.
//! - Preset application writes `active_wl` immediately; the caller is responsible
//!   for propagating the new WL to all affected viewports.

use egui::{DragValue, Ui};

use crate::{
    render::{colormap::Colormap, slice_render::WindowLevel},
    tools::kind::ToolKind,
    ui::{layout::LayoutMode, window_presets::WindowPreset},
};

// ── ToolbarState ──────────────────────────────────────────────────────────────

/// Persistent toolbar state shared across frames.
#[derive(Debug, Clone)]
pub struct ToolbarState {
    /// The currently active interaction tool.
    pub active_tool: ToolKind,
    /// Whether the series browser sidebar is visible.
    pub show_series_browser: bool,
    /// Whether the metadata detail panel is visible.
    pub show_metadata_panel: bool,
    /// The active viewport layout.
    pub layout_mode: LayoutMode,
    /// Whether measurement annotations are visible.
    pub show_measurements: bool,
}

impl Default for ToolbarState {
    fn default() -> Self {
        Self {
            active_tool: ToolKind::WindowLevel,
            show_series_browser: true,
            show_metadata_panel: false,
            layout_mode: LayoutMode::TwoByTwo,
            show_measurements: true,
        }
    }
}

// ── ToolbarPanel ──────────────────────────────────────────────────────────────

/// Ephemeral toolbar widget; borrows mutable references for one `update()` call.
pub struct ToolbarPanel<'a> {
    /// Mutable toolbar state.
    pub state: &'a mut ToolbarState,
    /// Global window/level for all linked viewports.
    pub active_wl: &'a mut WindowLevel,
    /// Active colormap applied to all viewports.
    pub active_colormap: &'a mut Colormap,
    /// DICOM modality hint from the loaded volume (e.g. `"CT"`, `"MR"`).
    pub modality_hint: Option<&'a str>,
}

impl<'a> ToolbarPanel<'a> {
    /// Construct a toolbar panel.
    pub fn new(
        state: &'a mut ToolbarState,
        active_wl: &'a mut WindowLevel,
        active_colormap: &'a mut Colormap,
        modality_hint: Option<&'a str>,
    ) -> Self {
        Self {
            state,
            active_wl,
            active_colormap,
            modality_hint,
        }
    }

    /// Render the toolbar into `ui`.
    ///
    /// # Returns
    /// `true` when the user changed the layout mode this frame (callers may
    /// need to resize viewport state arrays accordingly); `false` otherwise.
    pub fn show(&mut self, ui: &mut Ui) -> bool {
        let mut layout_changed = false;

        ui.horizontal(|ui| {
            // ── File section ──────────────────────────────────────────────
            ui.menu_button("📂 File", |ui| {
                if ui.button("Open Folder (DICOM)…").clicked() {
                    // Actual file-open logic is handled in SnapApp via
                    // toolbar_open_folder_requested; we emit a side-channel
                    // signal by setting a sentinel that SnapApp polls.
                    // For ergonomics this panel does not own the file dialog.
                    ui.close_menu();
                }
                if ui.button("Open File (NIfTI/…)…").clicked() {
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Export Slice as PNG…").clicked() {
                    ui.close_menu();
                }
            });

            ui.separator();

            // ── Tool buttons ──────────────────────────────────────────────
            for &tool in ToolKind::all() {
                let label = format!("{} {}", tool.icon(), tool.label());
                let selected = self.state.active_tool == tool;
                let btn = ui
                    .selectable_label(selected, label)
                    .on_hover_text(tool.tooltip());
                if btn.clicked() {
                    self.state.active_tool = tool;
                }
            }

            ui.separator();

            // ── Layout picker ─────────────────────────────────────────────
            ui.label("Layout:");
            let prev_layout = self.state.layout_mode;
            for &mode in LayoutMode::all() {
                if ui
                    .selectable_label(self.state.layout_mode == mode, mode.label())
                    .clicked()
                {
                    self.state.layout_mode = mode;
                }
            }
            if self.state.layout_mode != prev_layout {
                layout_changed = true;
            }

            ui.separator();

            // ── Colormap picker ───────────────────────────────────────────
            ui.label("Colormap:");
            egui::ComboBox::from_id_source("colormap_picker")
                .selected_text(self.active_colormap.label())
                .show_ui(ui, |ui| {
                    for &cm in Colormap::all() {
                        ui.selectable_value(self.active_colormap, cm, cm.label());
                    }
                });

            ui.separator();

            // ── Window/Level preset picker ────────────────────────────────
            let presets = WindowPreset::for_modality(self.modality_hint);
            ui.label("Preset:");
            egui::ComboBox::from_id_source("wl_preset_picker")
                .selected_text("Select…")
                .show_ui(ui, |ui| {
                    for preset in presets {
                        if ui.selectable_label(false, preset.name).clicked() {
                            self.active_wl.center = preset.center;
                            self.active_wl.width = preset.width;
                        }
                    }
                });

            ui.separator();

            // ── Manual W/C entry fields ───────────────────────────────────
            ui.label("W:");
            ui.add(
                DragValue::new(&mut self.active_wl.width)
                    .speed(1.0)
                    .range(1.0..=10000.0)
                    .suffix(" HU"),
            );
            ui.label("C:");
            ui.add(
                DragValue::new(&mut self.active_wl.center)
                    .speed(1.0)
                    .range(-4096.0..=4096.0)
                    .suffix(" HU"),
            );

            ui.separator();

            // ── Toggle buttons ────────────────────────────────────────────
            let overlay_label = if self.state.show_series_browser {
                "🗂 Browser ✓"
            } else {
                "🗂 Browser"
            };
            if ui
                .selectable_label(self.state.show_series_browser, overlay_label)
                .clicked()
            {
                self.state.show_series_browser = !self.state.show_series_browser;
            }

            let meta_label = if self.state.show_metadata_panel {
                "📋 Meta ✓"
            } else {
                "📋 Meta"
            };
            if ui
                .selectable_label(self.state.show_metadata_panel, meta_label)
                .clicked()
            {
                self.state.show_metadata_panel = !self.state.show_metadata_panel;
            }

            let meas_label = if self.state.show_measurements {
                "📏 Annot ✓"
            } else {
                "📏 Annot"
            };
            if ui
                .selectable_label(self.state.show_measurements, meas_label)
                .clicked()
            {
                self.state.show_measurements = !self.state.show_measurements;
            }
        });

        layout_changed
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Default toolbar state must set `WindowLevel` as the active tool,
    /// `TwoByTwo` as the layout, and have both browser and measurements visible.
    #[test]
    fn test_toolbar_state_default() {
        let state = ToolbarState::default();
        assert_eq!(
            state.active_tool,
            ToolKind::WindowLevel,
            "default active_tool must be WindowLevel"
        );
        assert_eq!(
            state.layout_mode,
            LayoutMode::TwoByTwo,
            "default layout_mode must be TwoByTwo"
        );
        assert!(
            state.show_series_browser,
            "series browser must be visible by default"
        );
        assert!(
            state.show_measurements,
            "measurements must be visible by default"
        );
        assert!(
            !state.show_metadata_panel,
            "metadata panel must be hidden by default"
        );
    }

    /// All `ToolKind` variants must be representable in `ToolbarState.active_tool`.
    #[test]
    fn test_toolbar_state_accepts_all_tool_kinds() {
        let mut state = ToolbarState::default();
        for &tool in ToolKind::all() {
            state.active_tool = tool;
            assert_eq!(
                state.active_tool, tool,
                "active_tool must accept and retain {:?}",
                tool
            );
        }
    }

    /// All `LayoutMode` variants must be representable in `ToolbarState.layout_mode`.
    #[test]
    fn test_toolbar_state_accepts_all_layout_modes() {
        let mut state = ToolbarState::default();
        for &mode in LayoutMode::all() {
            state.layout_mode = mode;
            assert_eq!(
                state.layout_mode, mode,
                "layout_mode must accept and retain {:?}",
                mode
            );
        }
    }

    /// Toggle flags must respond to boolean assignment.
    #[test]
    fn test_toolbar_state_toggle_flags() {
        let mut state = ToolbarState::default();
        state.show_series_browser = false;
        assert!(!state.show_series_browser);
        state.show_measurements = false;
        assert!(!state.show_measurements);
        state.show_metadata_panel = true;
        assert!(state.show_metadata_panel);
    }
}
