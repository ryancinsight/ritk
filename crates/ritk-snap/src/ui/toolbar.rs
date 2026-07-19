//! Menu-based toolbar for the ritk-snap viewer.
//!
//! # Layout
//!
//! ```text
//! [File â–¾] | [Image â–¾] | [Tools â–¾] | [View â–¾] | [Help â–¾]
//! ```
//!
//! **File Menu:**
//! - Open DICOM Folder…
//! - Open File (NIfTI/MetaImage/…)…
//! - Open Recent
//! - Close Study
//! - ─────────────────
//! - Save Segmentation…
//! - Export Surface (VTK)…
//! - Export Slices (PNG)…
//! - ─────────────────
//! - Exit
//!
//! **Image Menu:**
//! - Window/Level Presets → (Brain, Lung, Bone, Soft Tissue, Custom)
//! - Colormap → (Grayscale, Hot, Jet, Plasma, Viridis, Turbo, Phase, Seismic)
//! - Manual W/C DragValues
//!
//! **Tools Menu:**
//! - Window/Level (Ctrl+1)
//! - Pan (Ctrl+2)
//! - Zoom (Ctrl+3)
//! - Measure Length (Ctrl+4)
//! - Measure Angle (Ctrl+5)
//! - Draw ROI Rect (Ctrl+6)
//! - Draw ROI Ellipse (Ctrl+7)
//! - Paint Label (Ctrl+8)
//! - Erase Label (Ctrl+9)
//! - Query HU (Ctrl+0)
//!
//! **View Menu:**
//! - Layout → (Single, 2×2, 1+3, 3+1, Side-by-Side)
//! - ─────────────────
//! - Show Series Browser (Ctrl+B)
//! - Show Metadata Panel (Ctrl+M)
//! - Show Measurements (Ctrl+A)
//! - Show Crosshair
//! - Show Orientation Labels
//!
//! **Help Menu:**
//! - Keyboard Shortcuts
//! - About ritk-snap
//!
//! # Invariants
//! - The active tool is visually distinct in the Tools menu.
//! - The active layout mode is visually distinct in the View menu.
//! - W/L DragValues are two-way bound and do NOT modify preset selection.
//! - Preset application writes W/L immediately; caller propagates to viewports.

use egui::Ui;

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

    /// Render the menu bar into `ui`.
    ///
    /// # Returns
    /// `true` when the user changed the layout mode this frame (callers may
    /// need to resize viewport state arrays accordingly); `false` otherwise.
    pub fn show(&mut self, ui: &mut Ui) -> bool {
        let mut layout_changed = false;

        ui.horizontal(|ui| {
            // ── File Menu ─────────────────────────────────────────────────
            ui.menu_button("📂 File", |ui| {
                if ui.button("Open DICOM Folder…").clicked() {
                    ui.close_menu();
                }
                if ui.button("Open File (NIfTI/MetaImage/…)…").clicked() {
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Close Study").clicked() {
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Save Segmentation…").clicked() {
                    ui.close_menu();
                }
                if ui
                    .menu_button("Export", |ui| {
                        if ui.button("Surface as VTK…").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Slices as PNG…").clicked() {
                            ui.close_menu();
                        }
                    })
                    .inner
                    .is_none()
                {
                    // Submenu not opened
                }
                ui.separator();
                if ui.button("Exit").clicked() {
                    ui.close_menu();
                }
            });

            ui.separator();

            // ── Image Menu (Presets, Colormap, W/L) ───────────────────────
            ui.menu_button("🎨 Image", |ui| {
                ui.label("Window/Level Presets:");
                let presets = WindowPreset::for_modality(self.modality_hint);
                for preset in presets {
                    if ui.button(preset.name).clicked() {
                        self.active_wl.center = preset.center;
                        self.active_wl.width = preset.width;
                        ui.close_menu();
                    }
                }
                ui.separator();

                ui.label("Colormap:");
                for &cm in Colormap::all() {
                    if ui
                        .selectable_label(*self.active_colormap == cm, cm.label())
                        .clicked()
                    {
                        *self.active_colormap = cm;
                        ui.close_menu();
                    }
                }
                ui.separator();

                ui.label("Manual Window/Level:");
                ui.add(
                    egui::DragValue::new(&mut self.active_wl.width)
                        .speed(1.0)
                        .range(1.0..=10000.0)
                        .prefix("Width: ")
                        .suffix(" HU"),
                );
                ui.add(
                    egui::DragValue::new(&mut self.active_wl.center)
                        .speed(1.0)
                        .range(-4096.0..=4096.0)
                        .prefix("Center: ")
                        .suffix(" HU"),
                );
            });

            ui.separator();

            // ── Tools Menu ────────────────────────────────────────────────
            ui.menu_button("🔨 Tools", |ui| {
                for &tool in ToolKind::all() {
                    let is_active = self.state.active_tool == tool;
                    let label = format!("{} {}", tool.icon(), tool.label());
                    if ui.selectable_label(is_active, &label).clicked() {
                        self.state.active_tool = tool;
                        ui.close_menu();
                    }
                }
            });

            ui.separator();

            // ── View Menu (Layout, Panels) ────────────────────────────────
            ui.menu_button("👁 View", |ui| {
                ui.label("Layout Mode:");
                for &mode in LayoutMode::all() {
                    if ui
                        .selectable_label(self.state.layout_mode == mode, mode.label())
                        .clicked()
                    {
                        self.state.layout_mode = mode;
                        layout_changed = true;
                        ui.close_menu();
                    }
                }
                ui.separator();

                let browser_label = if self.state.show_series_browser {
                    "✓ Series Browser (Ctrl+B)"
                } else {
                    "  Series Browser (Ctrl+B)"
                };
                if ui.button(browser_label).clicked() {
                    self.state.show_series_browser = !self.state.show_series_browser;
                    ui.close_menu();
                }

                let metadata_label = if self.state.show_metadata_panel {
                    "✓ Metadata Panel (Ctrl+M)"
                } else {
                    "  Metadata Panel (Ctrl+M)"
                };
                if ui.button(metadata_label).clicked() {
                    self.state.show_metadata_panel = !self.state.show_metadata_panel;
                    ui.close_menu();
                }

                let meas_label = if self.state.show_measurements {
                    "✓ Measurements (Ctrl+A)"
                } else {
                    "  Measurements (Ctrl+A)"
                };
                if ui.button(meas_label).clicked() {
                    self.state.show_measurements = !self.state.show_measurements;
                    ui.close_menu();
                }
            });

            ui.separator();

            // ── Help Menu ─────────────────────────────────────────────────
            ui.menu_button("❓ Help", |ui| {
                if ui.button("Keyboard Shortcuts").clicked() {
                    ui.close_menu();
                }
                if ui.button("About ritk-snap").clicked() {
                    ui.close_menu();
                }
            });
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
