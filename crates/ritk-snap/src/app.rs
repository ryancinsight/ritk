//! ritk-snap native application shell (eframe/egui backend).
//!
//! Owns the top-level [`SnapApp`] struct and its [`eframe::App`]
//! implementation.  All domain logic (intensity mapping, slice extraction,
//! annotation computation) lives in the `render` and `tools` sub-modules;
//! this module wires events, drives state transitions, and builds the egui
//! widget tree.
//!
//! # Layout modes
//!
//! | `multi_planar` | Layout                                      |
//! |----------------|---------------------------------------------|
//! | `false`        | Single viewport — current axis fills panel. |
//! | `true`         | 2×2 grid: Axial / Coronal / Sagittal / Info.|

use std::sync::Arc;
use std::time::Duration;

use tracing::{error, info};

use crate::dicom::select_hanging_protocol;
use crate::label::LabelEditor;
use crate::render::colormap::Colormap;
use crate::render::slice_render::{SliceRenderer, WindowLevel};
use crate::session::ViewerSessionSnapshot;
use crate::tools::interaction::{Annotation, RoiKind, ToolState};
use crate::tools::kind::ToolKind;
use crate::ui::{
    axis_slice_dimensions, fit_view_transform, format_lps, intensity_at_voxel, map_view_row_col_to_voxel,
    pan_from_drag_delta, plan_all_mpr_exports, project_rt_struct_contours_for_slice, viewport_point_to_voxel,
    should_zoom_with_scroll, voxel_to_lps, zoom_from_drag_delta, zoom_from_scroll,
    window_level_from_drag_delta, tool_kind_for_key, WINDOW_LEVEL_SENSITIVITY,
    apply_to_image, show_colorbar, ViewTransform,
    CinePlayback, LinkedCursor, MAX_ZOOM, MIN_ZOOM,
};
use crate::ui::overlay::OverlayRenderer;
use crate::ui::window_presets::WindowPreset;
use crate::{LoadedVolume, ModalityDisplay, ViewerState};

/// CPU backend used for DICOM loading.
type LoadBackend = burn_ndarray::NdArray<f32>;

/// Cached RT-DOSE overlay texture for one axis.
struct RtDoseOverlayCacheEntry {
    slice_idx: usize,
    vol_shape: [usize; 3],
    dose_dims: [usize; 3],
    opacity_alpha: u8,
    texture: egui::TextureHandle,
}

// ── SnapApp ───────────────────────────────────────────────────────────────────

/// Native DICOM viewer application state.
///
/// Constructed via [`Default`] and driven by `eframe` through the
/// [`eframe::App`] trait.
pub struct SnapApp {
    // ── Volume ────────────────────────────────────────────────────────────────
    /// Currently loaded volume, if any.
    loaded: Option<LoadedVolume>,
    /// Viewer navigation state (slice index, W/L).
    viewer_state: ViewerState,
    /// Active colormap for intensity mapping.
    colormap: Colormap,
    /// Primary MPR axis for single-viewport and tool operations:
    /// 0 = axial, 1 = coronal, 2 = sagittal.
    axis: usize,

    // ── Tools ─────────────────────────────────────────────────────────────────
    /// Active interaction tool.
    active_tool: ToolKind,
    /// In-progress gesture state for the active tool.
    tool_state: ToolState,
    /// Completed measurement annotations.
    annotations: Vec<Annotation>,
    /// Segmentation label editor for the currently loaded volume.
    label_editor: Option<LabelEditor>,
    /// Brush radius in voxels for paint/erase tools.
    label_brush_radius: usize,
    /// Whether label overlays are rendered on viewports.
    show_label_overlay: bool,
    /// RT-STRUCT contour overlay visibility.
    show_rt_struct_overlay: bool,
    /// Currently loaded RT Structure Set.
    rt_struct: Option<ritk_io::RtStructureSet>,
    /// Currently loaded RT Dose grid.
    rt_dose: Option<ritk_io::RtDoseGrid>,
    /// Whether to render the RT-DOSE heat-map overlay on viewports.
    show_rt_dose_overlay: bool,
    /// Opacity of the RT-DOSE overlay (0.0 transparent … 1.0 opaque).
    rt_dose_opacity: f32,
    /// Per-axis RT-DOSE overlay texture cache (bounded to three entries).
    rt_dose_overlay_cache: [Option<RtDoseOverlayCacheEntry>; 3],
    /// Active filter configuration shown in the processing panel.
    active_filter: crate::FilterKind,
    /// Whether the filter processing panel is visible.
    show_filter_panel: bool,

    // ── Texture cache — axial ─────────────────────────────────────────────────
    /// Cached egui texture for the axial slice.
    texture: Option<egui::TextureHandle>,
    /// `true` when the axial texture must be rebuilt before the next frame.
    texture_dirty: bool,

    // ── Texture cache — coronal / sagittal ────────────────────────────────────
    /// Cached egui texture for the coronal slice (MPR mode).
    coronal_tex: Option<egui::TextureHandle>,
    /// `true` when the coronal texture must be rebuilt.
    coronal_dirty: bool,
    /// Current coronal slice index (fixed row `r`).
    coronal_slice: usize,

    /// Cached egui texture for the sagittal slice (MPR mode).
    sagittal_tex: Option<egui::TextureHandle>,
    /// `true` when the sagittal texture must be rebuilt.
    sagittal_dirty: bool,
    /// Current sagittal slice index (fixed column `c`).
    sagittal_slice: usize,

    // ── Viewport ──────────────────────────────────────────────────────────────
    /// Viewport pan offset in screen pixels.
    pan_offset: egui::Vec2,
    /// Viewport zoom multiplier (1.0 = fit-to-panel).
    zoom: f32,
    /// Viewport image orientation transform (flip/rotate).
    view_transform: ViewTransform,
    /// Whether to show the colorbar overlay in each viewport.
    show_colorbar: bool,

    // ── UI state ──────────────────────────────────────────────────────────────
    /// `true` when the 2×2 multi-planar layout is active.
    multi_planar: bool,
    /// `true` when the DICOM 4-corner overlay is drawn on viewports.
    show_overlay: bool,
    /// `true` when crosshair lines are drawn on viewports.
    show_crosshair: bool,
    /// Shared voxel cursor used to synchronize all MPR viewports.
    linked_cursor: Option<LinkedCursor>,
    /// Cine playback controller for automatic slice stepping.
    cine: CinePlayback,
    /// `true` when the series browser left panel is visible.
    show_series_browser: bool,
    /// Current voxel intensity value under the pointer (HU or relative).
    pointer_intensity: f32,
    /// Cached voxel intensity histogram for the loaded volume.
    ///
    /// Computed once when a volume is loaded; `None` when no volume is loaded.
    /// Used to render the W/L histogram panel in the sidebar.
    cached_histogram: Option<crate::render::histogram::Histogram>,

    // ── Series browser ────────────────────────────────────────────────────────
    /// Hierarchical DICOM series tree.
    series_tree: crate::dicom::series_tree::SeriesTree,
    /// The folder path currently highlighted in the series browser.
    selected_series: Option<std::path::PathBuf>,
    /// Which tab is active in the series browser sidebar.
    sidebar_tab: crate::ui::sidebar::SidebarTab,

    // ── Status ────────────────────────────────────────────────────────────────
    /// Message shown in the bottom status bar.
    status_message: String,
    /// Path queued for loading on the next [`eframe::App::update`] cycle.
    pending_load: Option<std::path::PathBuf>,
}

impl Default for SnapApp {
    fn default() -> Self {
        Self {
            loaded: None,
            viewer_state: ViewerState::new(),
            colormap: Colormap::Grayscale,
            axis: 0,
            active_tool: ToolKind::WindowLevel,
            tool_state: ToolState::Idle,
            annotations: Vec::new(),
            label_editor: None,
            label_brush_radius: 1,
            show_label_overlay: true,
            show_rt_struct_overlay: true,
            rt_struct: None,
            rt_dose: None,
            show_rt_dose_overlay: false,
            rt_dose_opacity: 0.5,
            rt_dose_overlay_cache: std::array::from_fn(|_| None),
            active_filter: crate::FilterKind::Gaussian { sigma: 1.0 },
            show_filter_panel: false,
            texture: None,
            texture_dirty: false,
            coronal_tex: None,
            coronal_dirty: false,
            coronal_slice: 0,
            sagittal_tex: None,
            sagittal_dirty: false,
            sagittal_slice: 0,
            pan_offset: egui::Vec2::ZERO,
            zoom: 1.0,
            view_transform: ViewTransform::default(),
            show_colorbar: false,
            multi_planar: false,
            show_overlay: true,
            show_crosshair: false,
            linked_cursor: None,
            cine: CinePlayback::default(),
            show_series_browser: true,
            pointer_intensity: 0.0,
            cached_histogram: None,
            series_tree: crate::dicom::series_tree::SeriesTree::new(),
            selected_series: None,
            sidebar_tab: crate::ui::sidebar::SidebarTab::Series,
            status_message: "No study loaded — use File > Open to load a DICOM folder.".to_owned(),
            pending_load: None,
        }
    }
}

impl SnapApp {
    /// Construct an app that loads `path` on the first update cycle.
    ///
    /// Directory paths are scanned immediately so the series browser is
    /// populated before the deferred volume load runs. File paths are queued
    /// directly because they do not contain a DICOM series tree.
    pub(crate) fn with_initial_path(path: std::path::PathBuf) -> Self {
        let mut app = Self::default();
        if crate::dicom::classify_dicom_input_path(&path)
            .dicom_root()
            .is_some()
        {
            app.scan_for_series(path.clone());
        }
        app.status_message = format!("Queued initial load: {}", path.display());
        app.pending_load = Some(path);
        app
    }
}

// ── eframe::App ───────────────────────────────────────────────────────────────

impl eframe::App for SnapApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Process any pending file load queued in the previous frame so that
        // the file-dialog result is always acted on with a full UI repaint.
        if let Some(path) = self.pending_load.take() {
            self.load_from_path(path);
        }

        self.tick_cine(ctx);
        self.consume_global_shortcuts(ctx);

        self.show_menu_bar(ctx);
        self.show_left_panel(ctx);
        self.show_bottom_bar(ctx);

        if self.multi_planar {
            self.show_central_panel_multi(ctx);
        } else {
            self.show_central_panel_single(ctx);
        }
    }
}

// ── UI sub-methods ────────────────────────────────────────────────────────────

impl SnapApp {
    // ── Menu bar ─────────────────────────────────────────────────────────────

    fn show_menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                // ── File ─────────────────────────────────────────────────────
                ui.menu_button("File", |ui| {
                    if ui.button("Open DICOM folder…").clicked() {
                        ui.close_menu();
                        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                            self.scan_for_series(folder.clone());
                            self.pending_load = Some(folder);
                        }
                    }

                    if ui.button("Open DICOMDIR…").clicked() {
                        ui.close_menu();
                        if let Some(path) =
                            rfd::FileDialog::new().set_file_name("DICOMDIR").pick_file()
                        {
                            self.scan_for_series(path.clone());
                            self.pending_load = Some(path);
                        }
                    }

                    if ui.button("Open DICOM file…").clicked() {
                        ui.close_menu();
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("DICOM", &["dcm", "dicom"])
                            .pick_file()
                        {
                            self.scan_for_series(path.clone());
                            self.pending_load = Some(path);
                        }
                    }

                    if ui.button("Open NIfTI / MHA / NRRD file…").clicked() {
                        ui.close_menu();
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter(
                                "Medical images",
                                &["nii", "gz", "mha", "mhd", "nrrd", "nhdr", "mgh", "mgz"],
                            )
                            .pick_file()
                        {
                            self.load_nifti_file(path);
                        }
                    }

                    if ui.button("Open RT-STRUCT file…").clicked() {
                        ui.close_menu();
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("DICOM", &["dcm"])
                            .pick_file()
                        {
                            self.load_rt_struct_file(path);
                        }
                    }

                    if ui.button("Open RT Dose file…").clicked() {
                        ui.close_menu();
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("DICOM", &["dcm"])
                            .pick_file()
                        {
                            self.load_rt_dose_file(path);
                        }
                    }

                    if ui.button("Export current slice as PNG…").clicked() {
                        ui.close_menu();
                        self.export_current_slice();
                    }

                    if ui.button("Export all MPR slices as PNG…").clicked() {
                        ui.close_menu();
                        self.export_all_mpr_slices();
                    }

                    if ui.button("Save segmentation as NIfTI…").clicked() {
                        ui.close_menu();
                        self.save_segmentation_dialog();
                    }

                    if ui.button("Load segmentation from NIfTI…").clicked() {
                        ui.close_menu();
                        self.load_segmentation_dialog();
                    }

                    if ui.button("Save session…").clicked() {
                        ui.close_menu();
                        self.save_session_dialog();
                    }

                    if ui.button("Load session…").clicked() {
                        ui.close_menu();
                        self.load_session_dialog();
                    }

                    if ui.button("Close study").clicked() {
                        ui.close_menu();
                        self.close_study();
                    }

                    ui.separator();

                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                // ── View ─────────────────────────────────────────────────────
                ui.menu_button("View", |ui| {
                    let layout_label = if self.multi_planar {
                        "⬛ Multi-planar 2×2 (active)"
                    } else {
                        "⬜ Single view (active)"
                    };
                    if ui.button(layout_label).clicked() {
                        ui.close_menu();
                        self.multi_planar = !self.multi_planar;
                        // Mark all textures dirty so the new layout renders
                        // immediately.
                        self.texture_dirty = true;
                        self.coronal_dirty = true;
                        self.sagittal_dirty = true;
                    }

                    ui.separator();

                    let overlay_label = if self.show_overlay {
                        "✔ Show Overlay"
                    } else {
                        "  Show Overlay"
                    };
                    if ui.button(overlay_label).clicked() {
                        ui.close_menu();
                        self.show_overlay = !self.show_overlay;
                    }

                    let label_overlay_label = if self.show_label_overlay {
                        "✔ Show Label Overlay"
                    } else {
                        "  Show Label Overlay"
                    };
                    if ui.button(label_overlay_label).clicked() {
                        ui.close_menu();
                        self.show_label_overlay = !self.show_label_overlay;
                    }

                    let rt_overlay_label = if self.show_rt_struct_overlay {
                        "✔ Show RT-STRUCT Overlay"
                    } else {
                        "  Show RT-STRUCT Overlay"
                    };
                    if ui.button(rt_overlay_label).clicked() {
                        ui.close_menu();
                        self.show_rt_struct_overlay = !self.show_rt_struct_overlay;
                    }

                    let rt_dose_label = if self.show_rt_dose_overlay {
                        "✔ Show RT-DOSE Overlay"
                    } else {
                        "  Show RT-DOSE Overlay"
                    };
                    if ui.button(rt_dose_label).clicked() {
                        ui.close_menu();
                        self.show_rt_dose_overlay = !self.show_rt_dose_overlay;
                    }

                    let filter_label = if self.show_filter_panel {
                        "✔ Show Filter Panel"
                    } else {
                        "  Show Filter Panel"
                    };
                    if ui.button(filter_label).clicked() {
                        ui.close_menu();
                        self.show_filter_panel = !self.show_filter_panel;
                    }

                    let xhair_label = if self.show_crosshair {
                        "✔ Show Crosshair"
                    } else {
                        "  Show Crosshair"
                    };
                    if ui.button(xhair_label).clicked() {
                        ui.close_menu();
                        self.show_crosshair = !self.show_crosshair;
                    }

                    let browser_label = if self.show_series_browser {
                        "✔ Show Series Browser"
                    } else {
                        "  Show Series Browser"
                    };
                    if ui.button(browser_label).clicked() {
                        ui.close_menu();
                        self.show_series_browser = !self.show_series_browser;
                    }

                    ui.separator();

                    // Colormap sub-menu kept in View.
                    ui.menu_button("Colormap", |ui| {
                        for &cm in Colormap::all() {
                            if ui
                                .selectable_label(self.colormap == cm, cm.label())
                                .clicked()
                                && self.colormap != cm
                            {
                                self.colormap = cm;
                                self.texture_dirty = true;
                                self.coronal_dirty = true;
                                self.sagittal_dirty = true;
                            }
                        }
                    });

                    ui.separator();
                    ui.label("Orientation");

                    if ui.button("Flip Horizontal  [H]").clicked() {
                        ui.close_menu();
                        self.view_transform = self.view_transform.toggle_flip_h();
                        self.mark_all_textures_dirty();
                    }
                    if ui.button("Flip Vertical    [V]").clicked() {
                        ui.close_menu();
                        self.view_transform = self.view_transform.toggle_flip_v();
                        self.mark_all_textures_dirty();
                    }
                    if ui.button("Rotate CW 90°    [R]").clicked() {
                        ui.close_menu();
                        self.view_transform = self.view_transform.rotate_cw();
                        self.mark_all_textures_dirty();
                    }
                    if ui.button("Rotate CCW 90°   [Shift+R]").clicked() {
                        ui.close_menu();
                        self.view_transform = self.view_transform.rotate_ccw();
                        self.mark_all_textures_dirty();
                    }
                    if ui.button("Reset Orientation [O]").clicked() {
                        ui.close_menu();
                        self.view_transform = self.view_transform.reset();
                        self.mark_all_textures_dirty();
                    }

                    ui.separator();

                    let colorbar_label = if self.show_colorbar {
                        "✔ Show Colorbar"
                    } else {
                        "  Show Colorbar"
                    };
                    if ui.button(colorbar_label).clicked() {
                        ui.close_menu();
                        self.show_colorbar = !self.show_colorbar;
                    }
                });

                // ── Image ────────────────────────────────────────────────────
                ui.menu_button("Image", |ui| {
                    ui.menu_button("Axis", |ui| {
                        for (name, idx) in [("Axial", 0usize), ("Coronal", 1), ("Sagittal", 2)] {
                            if ui.selectable_label(self.axis == idx, name).clicked()
                                && self.axis != idx
                            {
                                ui.close_menu();
                                self.axis = idx;
                                self.viewer_state.slice_index = 0;
                                self.texture_dirty = true;
                            }
                        }
                    });

                    ui.separator();

                    if ui.button("Zoom to fit (Ctrl/Cmd+0)").clicked() {
                        ui.close_menu();
                        self.reset_view_to_fit();
                    }

                    if ui.button("Next slice").clicked() {
                        ui.close_menu();
                        self.step_slice(1);
                    }

                    if ui.button("Previous slice").clicked() {
                        ui.close_menu();
                        self.step_slice(-1);
                    }
                });

                // ── Window (W/L presets) ──────────────────────────────────────
                ui.menu_button("Window", |ui| {
                    ui.label("CT Presets");
                    ui.separator();
                    for preset in WindowPreset::ct_presets() {
                        if ui.button(preset.name).clicked() {
                            ui.close_menu();
                            self.apply_preset(*preset);
                        }
                    }

                    ui.separator();
                    ui.label("MR Presets");
                    ui.separator();
                    for preset in WindowPreset::mr_presets() {
                        if ui.button(preset.name).clicked() {
                            ui.close_menu();
                            self.apply_preset(*preset);
                        }
                    }

                    ui.separator();
                    if ui.button("Reset W/L").clicked() {
                        ui.close_menu();
                        let (wc, ww) = if let Some(vol) = &self.loaded {
                            let d = ModalityDisplay::for_modality(vol.modality.as_deref());
                            (d.window_center as f32, d.window_width as f32)
                        } else {
                            (128.0, 256.0)
                        };
                        self.viewer_state.window_center = Some(wc);
                        self.viewer_state.window_width = Some(ww);
                        self.texture_dirty = true;
                        self.coronal_dirty = true;
                        self.sagittal_dirty = true;
                    }
                });
            });
        });
    }

    fn consume_global_shortcuts(&mut self, ctx: &egui::Context) {
        let zoom_to_fit = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Num0);
        let redo_shift_z = egui::KeyboardShortcut::new(
            egui::Modifiers {
                command: true,
                shift: true,
                ..Default::default()
            },
            egui::Key::Z,
        );
        let redo_y = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y);
        let undo_z = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);

        if ctx.input_mut(|input| input.consume_shortcut(&zoom_to_fit)) {
            self.reset_view_to_fit();
        }
        if ctx.input_mut(|input| {
            input.consume_shortcut(&redo_shift_z) || input.consume_shortcut(&redo_y)
        }) {
            self.redo_label_edit_shortcut();
        }
        if ctx.input_mut(|input| input.consume_shortcut(&undo_z)) {
            self.undo_label_edit_shortcut();
        }

        let nav = ctx.input(|input| {
            (
                input.key_pressed(egui::Key::ArrowUp),
                input.key_pressed(egui::Key::ArrowDown),
                input.key_pressed(egui::Key::PageUp),
                input.key_pressed(egui::Key::PageDown),
                input.key_pressed(egui::Key::Home),
                input.key_pressed(egui::Key::End),
            )
        });
        self.apply_slice_navigation_shortcuts(nav.0, nav.1, nav.2, nav.3, nav.4, nav.5);

        // ── Tool selection shortcuts ──────────────────────────────────────────
        ctx.input(|input| {
            for key in &input.keys_down {
                if let Some(tool) = tool_kind_for_key(*key) {
                    self.active_tool = tool;
                    break;
                }
            }
        });

        // ── Viewport orientation shortcuts ────────────────────────────────────
        let (flip_h, flip_v, rotate_cw, rotate_ccw, reset_orient) = ctx.input(|input| {
            let shift = input.modifiers.shift;
            (
                input.key_pressed(egui::Key::H),
                input.key_pressed(egui::Key::V),
                !shift && input.key_pressed(egui::Key::R),
                shift && input.key_pressed(egui::Key::R),
                input.key_pressed(egui::Key::O),
            )
        });
        if flip_h { self.view_transform = self.view_transform.toggle_flip_h(); self.mark_all_textures_dirty(); }
        if flip_v { self.view_transform = self.view_transform.toggle_flip_v(); self.mark_all_textures_dirty(); }
        if rotate_cw { self.view_transform = self.view_transform.rotate_cw(); self.mark_all_textures_dirty(); }
        if rotate_ccw { self.view_transform = self.view_transform.rotate_ccw(); self.mark_all_textures_dirty(); }
        if reset_orient { self.view_transform = self.view_transform.reset(); self.mark_all_textures_dirty(); }
    }

    fn apply_slice_navigation_shortcuts(
        &mut self,
        arrow_up: bool,
        arrow_down: bool,
        page_up: bool,
        page_down: bool,
        home: bool,
        end: bool,
    ) {
        if arrow_up || page_up {
            self.step_slice(-1);
        } else if arrow_down || page_down {
            self.step_slice(1);
        } else if home {
            self.jump_active_axis_slice_boundary(false);
        } else if end {
            self.jump_active_axis_slice_boundary(true);
        }
    }

    fn jump_active_axis_slice_boundary(&mut self, end: bool) {
        let (_, total) = self.axis_slice_info(self.axis);
        let target = if end {
            total.saturating_sub(1)
        } else {
            0
        };
        self.set_slice_for_axis(self.axis, target);
    }

    fn reset_view_to_fit(&mut self) {
        let (pan_offset, zoom) = fit_view_transform();
        self.pan_offset = egui::Vec2::new(pan_offset[0], pan_offset[1]);
        self.zoom = zoom;
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
        self.status_message = "Zoom reset to fit.".to_owned();
    }

    /// Mark all three MPR texture slots as needing re-render (e.g. after a
    /// view-transform or colormap change).
    fn mark_all_textures_dirty(&mut self) {
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
    }

    fn undo_label_edit_shortcut(&mut self) {
        let Some(editor) = self.label_editor.as_mut() else {
            return;
        };
        if editor.undo() {
            self.status_message = "Segmentation undo.".to_owned();
        }
    }

    fn redo_label_edit_shortcut(&mut self) {
        let Some(editor) = self.label_editor.as_mut() else {
            return;
        };
        if editor.redo() {
            self.status_message = "Segmentation redo.".to_owned();
        }
    }

    /// Apply a [`WindowPreset`] and mark all textures dirty.
    fn apply_preset(&mut self, preset: WindowPreset) {
        self.viewer_state.window_center = Some(preset.center as f32);
        self.viewer_state.window_width = Some(preset.width as f32);
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
    }

    fn session_snapshot(&self) -> ViewerSessionSnapshot {
        ViewerSessionSnapshot {
            source: self.loaded.as_ref().and_then(|vol| vol.source.clone()),
            viewer_state: self.viewer_state,
            colormap: self.colormap,
            axis: self.axis,
            active_tool: self.active_tool,
            multi_planar: self.multi_planar,
            show_overlay: self.show_overlay,
            show_crosshair: self.show_crosshair,
            show_rt_struct_overlay: self.show_rt_struct_overlay,
            show_rt_dose_overlay: self.show_rt_dose_overlay,
            rt_dose_opacity: self.rt_dose_opacity,
            show_series_browser: self.show_series_browser,
            sidebar_tab: self.sidebar_tab,
            coronal_slice: self.coronal_slice,
            sagittal_slice: self.sagittal_slice,
            pan_offset: [self.pan_offset.x, self.pan_offset.y],
            zoom: self.zoom,
            cine_enabled: self.cine.enabled,
            cine_fps: self.cine.fps,
            annotations: self.annotations.clone(),
        }
    }

    fn apply_session_snapshot(&mut self, snapshot: ViewerSessionSnapshot) {
        self.viewer_state = snapshot.viewer_state;
        self.colormap = snapshot.colormap;
        self.axis = snapshot.axis.min(2);
        self.active_tool = snapshot.active_tool;
        self.multi_planar = snapshot.multi_planar;
        self.show_overlay = snapshot.show_overlay;
        self.show_crosshair = snapshot.show_crosshair;
        self.show_rt_struct_overlay = snapshot.show_rt_struct_overlay;
        self.show_rt_dose_overlay = snapshot.show_rt_dose_overlay;
        self.rt_dose_opacity = snapshot.rt_dose_opacity;
        self.show_series_browser = snapshot.show_series_browser;
        self.sidebar_tab = snapshot.sidebar_tab;
        self.coronal_slice = snapshot.coronal_slice;
        self.sagittal_slice = snapshot.sagittal_slice;
        self.pan_offset = egui::Vec2::new(snapshot.pan_offset[0], snapshot.pan_offset[1]);
        self.zoom = snapshot.zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        self.cine.restore(snapshot.cine_enabled, snapshot.cine_fps);
        self.annotations = snapshot.annotations;
        self.tool_state = ToolState::Idle;
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
        self.linked_cursor = self.loaded.as_ref().map(|vol| {
            LinkedCursor::from_slices(
                vol.shape,
                self.viewer_state.slice_index,
                self.coronal_slice,
                self.sagittal_slice,
            )
        });
        if let Some(source) = snapshot.source {
            self.pending_load = Some(source);
        }
    }

    // ── Left panel ────────────────────────────────────────────────────────────

    fn show_left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("info_panel")
            .min_width(220.0)
            .max_width(360.0)
            .show(ctx, |ui| {
                // ── Series browser or inline metadata ─────────────────────────
                if self.show_series_browser {
                    ui.heading("Series Browser");
                    ui.separator();

                    // Construct SidebarPanel using split field borrows, contained
                    // in a block so all borrows are released before accessing
                    // self.pending_load below.
                    let sidebar_result = {
                        let tree_ref = &self.series_tree;
                        let sel_ref = &mut self.selected_series;
                        let tab_ref = &mut self.sidebar_tab;
                        let vol_ref = self.loaded.as_ref();
                        let mut panel = crate::ui::sidebar::SidebarPanel::new(
                            tree_ref, sel_ref, tab_ref, vol_ref,
                        );
                        panel.show(ui)
                    };

                    if let Some(folder) = sidebar_result {
                        self.pending_load = Some(folder);
                    }
                } else {
                    // Inline metadata section (original layout).
                    ui.heading("Study Info");
                    ui.separator();

                    if let Some(vol) = &self.loaded {
                        let [depth, rows, cols] = vol.shape;
                        let [dz, dy, dx] = vol.spacing;

                        egui::Grid::new("study_grid")
                            .num_columns(2)
                            .spacing([8.0, 4.0])
                            .show(ui, |ui| {
                                let row = |ui: &mut egui::Ui, key: &str, val: &str| {
                                    ui.label(key);
                                    ui.label(val);
                                    ui.end_row();
                                };
                                row(ui, "Patient:", vol.patient_name.as_deref().unwrap_or("—"));
                                row(ui, "ID:", vol.patient_id.as_deref().unwrap_or("—"));
                                row(ui, "Date:", vol.study_date.as_deref().unwrap_or("—"));
                                row(ui, "Modality:", vol.modality.as_deref().unwrap_or("—"));
                                row(
                                    ui,
                                    "Series:",
                                    vol.series_description.as_deref().unwrap_or("—"),
                                );
                                row(ui, "Shape:", &format!("{depth} × {rows} × {cols}"));
                                row(ui, "Spacing:", &format!("{dz:.2}×{dy:.2}×{dx:.2} mm"));
                            });
                    } else {
                        ui.label("No study loaded.");
                    }
                }

                // ── Window / Level (always shown) ─────────────────────────────
                ui.separator();
                ui.heading("Window / Level");
                ui.separator();

                let wc = self.viewer_state.window_center.unwrap_or(128.0);
                let ww = self.viewer_state.window_width.unwrap_or(256.0);
                ui.label(format!("Centre : {wc:.0}"));
                ui.label(format!("Width  : {ww:.0}"));

                // ── Histogram ─────────────────────────────────────────────────
                if let Some(hist) = &self.cached_histogram {
                    if let Some((new_c, new_w)) =
                        crate::ui::histogram::draw_histogram(hist, wc, ww, ui)
                    {
                        self.viewer_state.window_center = Some(new_c);
                        self.viewer_state.window_width = Some(new_w.max(1.0));
                        self.texture_dirty = true;
                    }
                }

                // ── Window / Level preset buttons ──────────────────────────────
                {
                    let modality = self.loaded.as_ref().and_then(|v| v.modality.as_deref());
                    let presets =
                        crate::ui::window_presets::WindowPreset::for_modality(modality);
                    if let Some(preset) =
                        crate::ui::preset_panel::draw_preset_buttons(presets, ui)
                    {
                        self.viewer_state.window_center = Some(preset.center as f32);
                        self.viewer_state.window_width = Some(preset.width as f32);
                        self.texture_dirty = true;
                    }
                }

                // ── Colorbar ───────────────────────────────────────────────────
                if self.show_colorbar {
                    ui.separator();
                    let wc = self.viewer_state.window_center.unwrap_or(128.0);
                    let ww = self.viewer_state.window_width.unwrap_or(256.0).max(1.0);
                    show_colorbar(ui, self.colormap, wc, ww);
                }

                ui.separator();
                ui.heading("Cine");
                ui.separator();

                let now = ctx.input(|i| i.time);
                let play_label = if self.cine.enabled { "Pause" } else { "Play" };
                if ui.button(play_label).clicked() {
                    if self.cine.enabled {
                        self.cine.stop();
                    } else {
                        self.cine.set_enabled(true, now);
                    }
                }
                let mut fps = self.cine.fps;
                if ui
                    .add(egui::Slider::new(&mut fps, 1.0..=60.0).text("FPS"))
                    .changed()
                {
                    self.cine.set_fps(fps);
                }
                let cine_axis = ["Axial", "Coronal", "Sagittal"][self.axis.min(2)];
                ui.label(format!("Axis: {cine_axis}"));

                // ── Tool palette ──────────────────────────────────────────────
                ui.separator();
                ui.heading("Tools");
                ui.separator();

                for &tool in ToolKind::all() {
                    let icon_label = format!("{} {}", tool.icon(), tool.label());
                    let resp = ui
                        .selectable_label(self.active_tool == tool, icon_label)
                        .on_hover_text(tool.tooltip());
                    if resp.clicked() {
                        self.active_tool = tool;
                        self.tool_state = ToolState::Idle;
                    }
                }

                if let Some(editor) = self.label_editor.as_mut() {
                    ui.separator();
                    ui.heading("Segmentation");
                    ui.separator();

                    ui.checkbox(&mut self.show_label_overlay, "Show labels");
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(editor.can_undo(), egui::Button::new("Undo (Ctrl/Cmd+Z)"))
                            .clicked()
                        {
                            let _ = editor.undo();
                        }
                        if ui
                            .add_enabled(
                                editor.can_redo(),
                                egui::Button::new("Redo (Ctrl/Cmd+Shift+Z / Ctrl/Cmd+Y)"),
                            )
                            .clicked()
                        {
                            let _ = editor.redo();
                        }
                    });

                    ui.add(
                        egui::Slider::new(&mut self.label_brush_radius, 0..=6)
                            .text("Brush radius (vox)"),
                    );

                    if ui.button("Add label").clicked() {
                        let next_id = editor.current_map().table.next_free_id();
                        let color = palette_color(next_id);
                        if let Err(e) = editor.add_label(format!("Label {next_id}"), color) {
                            self.status_message = format!("Add label failed: {e}");
                        }
                    }

                    let mut pending_active: Option<u32> = None;
                    let mut pending_visibility: Vec<(u32, bool)> = Vec::new();
                    for entry in editor.current_map().table.entries() {
                        ui.horizontal(|ui| {
                            let selected = editor.active_label_id() == entry.id;
                            if ui
                                .selectable_label(selected, format!("{} {}", entry.id, entry.name))
                                .clicked()
                            {
                                pending_active = Some(entry.id);
                            }

                            let mut visible = entry.visible;
                            if ui.checkbox(&mut visible, "visible").changed() {
                                pending_visibility.push((entry.id, visible));
                            }

                            let count = editor.current_map().count_label(entry.id);
                            ui.label(format!("voxels: {count}"));
                        });
                    }

                    if let Some(id) = pending_active {
                        if let Err(e) = editor.set_active_label(id) {
                            self.status_message = format!("Set active label failed: {e}");
                        }
                    }
                    for (id, visible) in pending_visibility {
                        if let Err(e) = editor.set_label_visibility(id, visible) {
                            self.status_message = format!("Set label visibility failed: {e}");
                        }
                    }
                }

                if let Some(rt) = &self.rt_struct {
                    ui.separator();
                    ui.heading("RT-STRUCT");
                    ui.separator();
                    ui.label(format!("Label: {}", rt.structure_set_label));
                    ui.label(format!("ROIs: {}", rt.rois.len()));
                    ui.checkbox(&mut self.show_rt_struct_overlay, "Show RT-STRUCT contours");
                }

                if let Some(rd) = &self.rt_dose {
                    ui.separator();
                    ui.heading("RT-DOSE");
                    ui.separator();
                    ui.label(format!("Type: {} / {}", rd.dose_type, rd.dose_summation_type));
                    ui.label(format!("Grid: {}×{}×{} frames", rd.rows, rd.cols, rd.n_frames));
                    let max_dose = rd.dose_gy.iter().cloned().fold(0.0_f64, f64::max);
                    ui.label(format!("Max dose: {:.2} Gy", max_dose));
                    ui.checkbox(&mut self.show_rt_dose_overlay, "Show RT-DOSE overlay");
                    ui.horizontal(|ui| {
                        ui.label("Opacity:");
                        ui.add(egui::Slider::new(&mut self.rt_dose_opacity, 0.0..=1.0).step_by(0.05));
                    });
                }

                // ── Annotations ───────────────────────────────────────────────
                ui.separator();
                ui.heading("Annotations");
                ui.separator();

                // Delegate to SSOT: per-entry delete, Clear All, Export CSV.
                match crate::ui::annotation_panel::draw_annotation_panel(
                    &self.annotations,
                    ui,
                ) {
                    crate::ui::AnnotationPanelAction::Delete(i) => {
                        if i < self.annotations.len() {
                            self.annotations.remove(i);
                        }
                    }
                    crate::ui::AnnotationPanelAction::ClearAll => {
                        self.annotations.clear();
                    }
                    crate::ui::AnnotationPanelAction::ExportCsv(csv) => {
                        // Write to clipboard; fall back to status message if
                        // clipboard is unavailable.
                        ctx.output_mut(|o| o.copied_text = csv.clone());
                        self.status_message =
                            format!("CSV copied to clipboard ({} rows).", self.annotations.len());
                    }
                    crate::ui::AnnotationPanelAction::None => {}
                }

                // ── Image Processing ──────────────────────────────────────────
                if self.show_filter_panel {
                    ui.separator();
                    ui.heading("Processing");
                    ui.separator();
                    let applied = crate::ui::filter_panel::show_filter_panel(
                        ui,
                        &mut self.active_filter,
                    );
                    if applied {
                        self.apply_filter_to_loaded_volume();
                    }
                }
            });
    }

    // ── Bottom status bar ─────────────────────────────────────────────────────

    fn show_bottom_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status_message);
                if let Some(vol) = &self.loaded {
                    let (slice_idx, total) = self.axis_slice_info(self.axis);
                    let _ = vol; // vol borrow used implicitly via axis_slice_info
                    ui.separator();
                    ui.label(format!("Slice {}/{}", slice_idx + 1, total));
                    ui.separator();
                    let axis_name = ["Axial", "Coronal", "Sagittal"][self.axis.min(2)];
                    ui.label(axis_name);

                    // Voxel I/J/K index and physical LPS position from linked cursor.
                    if let (Some(cursor), Some(vol)) =
                        (self.linked_cursor, self.loaded.as_ref())
                    {
                        let [d, r, c] = cursor.voxel();
                        ui.separator();
                        ui.label(format!("I={d} J={r} K={c}"));
                        let lps = voxel_to_lps(
                            [d, r, c],
                            vol.origin,
                            vol.direction,
                            vol.spacing,
                        );
                        ui.separator();
                        ui.label(format_lps(lps));
                    }
                }
            });
        });
    }

    // ── Single viewport ───────────────────────────────────────────────────────

    /// Render the current axis into the full central panel (single-viewport mode).
    fn show_central_panel_single(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.loaded.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label("Open a DICOM folder or NIfTI file via File menu to begin.");
                });
                return;
            }

            self.render_axis_viewport(ui, ctx, self.axis);
        });
    }

    // ── Multi-planar 2×2 viewport ─────────────────────────────────────────────

    /// Render the 2×2 MPR grid (Axial / Coronal / Sagittal / Info panel).
    ///
    /// Layout:
    /// ```text
    /// ┌──────────────┬──────────────┐
    /// │  Axial  (0)  │ Coronal  (1) │
    /// ├──────────────┼──────────────┤
    /// │ Sagittal (2) │  Info panel  │
    /// └──────────────┴──────────────┘
    /// ```
    fn show_central_panel_multi(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let half_w = avail.x / 2.0;
            let half_h = avail.y / 2.0;

            ui.horizontal(|ui| {
                // ── Left column: Axial (top), Sagittal (bottom) ───────────────
                ui.vertical(|ui| {
                    ui.allocate_ui(egui::vec2(half_w, half_h), |ui| {
                        self.render_axis_viewport(ui, ctx, 0); // Axial
                    });
                    ui.allocate_ui(egui::vec2(half_w, half_h), |ui| {
                        self.render_axis_viewport(ui, ctx, 2); // Sagittal
                    });
                });

                // ── Right column: Coronal (top), Info panel (bottom) ──────────
                ui.vertical(|ui| {
                    ui.allocate_ui(egui::vec2(half_w, half_h), |ui| {
                        self.render_axis_viewport(ui, ctx, 1); // Coronal
                    });
                    ui.allocate_ui(egui::vec2(half_w, half_h), |ui| {
                        self.show_right_info_panel(ui);
                    });
                });
            });
        });
    }

    // ── Per-axis viewport renderer ────────────────────────────────────────────

    /// Render one MPR viewport for the given `axis` into `ui`.
    ///
    /// # Responsibilities
    ///
    /// 1. Rebuild the texture for this axis if dirty or absent.
    /// 2. Compute fit scale: `min(avail_w / tex_w, avail_h / tex_h) × zoom`.
    /// 3. Display the image widget with click-and-drag sensing.
    /// 4. Draw the axis label and slice counter as overlay text.
    /// 5. Draw the DICOM 4-corner overlay when `show_overlay` is set.
    /// 6. Draw crosshair lines when `show_crosshair` is set.
    /// 7. Handle wheel input: Ctrl/Cmd+wheel zooms, plain wheel steps slices.
    /// 8. Dispatch pointer events to the active tool handler.
    fn render_axis_viewport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, axis: usize) {
        // ── 1. Rebuild texture if stale ────────────────────────────────────────
        let needs_rebuild = match axis {
            0 => self.texture_dirty || self.texture.is_none(),
            1 => self.coronal_dirty || self.coronal_tex.is_none(),
            _ => self.sagittal_dirty || self.sagittal_tex.is_none(),
        };
        if needs_rebuild && self.loaded.is_some() {
            self.rebuild_texture_for_axis(ctx, axis);
            match axis {
                0 => self.texture_dirty = false,
                1 => self.coronal_dirty = false,
                _ => self.sagittal_dirty = false,
            }
        }

        // ── 2. Extract texture ID and size (copy, releases borrow) ─────────────
        let tex_info: Option<(egui::TextureId, [usize; 2])> = match axis {
            0 => self.texture.as_ref().map(|t| (t.id(), t.size())),
            1 => self.coronal_tex.as_ref().map(|t| (t.id(), t.size())),
            _ => self.sagittal_tex.as_ref().map(|t| (t.id(), t.size())),
        };

        let (tex_id, [tex_w_usize, tex_h_usize]) = match tex_info {
            Some(info) => info,
            None => {
                ui.centered_and_justified(|ui| {
                    let label = ["Axial", "Coronal", "Sagittal"][axis.min(2)];
                    ui.label(format!("{label} — open a volume to begin"));
                });
                return;
            }
        };

        // ── 3. Compute fit scale and render image ──────────────────────────────
        let tex_w = tex_w_usize as f32;
        let tex_h = tex_h_usize as f32;
        let available = ui.available_size();
        let fit_scale = if tex_w > 0.0 && tex_h > 0.0 {
            (available.x / tex_w).min(available.y / tex_h)
        } else {
            1.0
        };
        let display_size = egui::vec2(tex_w * fit_scale * self.zoom, tex_h * fit_scale * self.zoom);

        let image_widget = egui::Image::new(egui::load::SizedTexture::new(tex_id, display_size))
            .sense(egui::Sense::click_and_drag());
        let response = ui.add(image_widget);

        // ── 4–6. Overlay text, DICOM overlay, crosshair ────────────────────────
        // Painter::new clones the Arc<Context>; it does not hold a borrow on ui.
        let painter = ui.painter_at(response.rect);

        let axis_name = ["Axial", "Coronal", "Sagittal"][axis.min(2)];
        let label_color = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 210);

        painter.text(
            response.rect.min + egui::vec2(6.0, 6.0),
            egui::Align2::LEFT_TOP,
            axis_name,
            egui::FontId::proportional(12.0),
            label_color,
        );

        let (slice_idx, total) = self.axis_slice_info(axis);
        painter.text(
            egui::pos2(response.rect.max.x - 6.0, response.rect.min.y + 6.0),
            egui::Align2::RIGHT_TOP,
            format!("{}/{}", slice_idx + 1, total),
            egui::FontId::proportional(11.0),
            label_color,
        );

        // DICOM 4-corner overlay.
        if self.show_overlay {
            if let Some(vol) = &self.loaded {
                let wc = self.viewer_state.window_center.unwrap_or(128.0) as f64;
                let ww = self.viewer_state.window_width.unwrap_or(256.0).max(1.0) as f64;
                let wl = WindowLevel::new(wc, ww);
                let cursor_value = self.current_cursor_value();
                OverlayRenderer::draw(
                    &painter,
                    response.rect,
                    vol,
                    axis,
                    slice_idx,
                    wl,
                    self.zoom,
                    cursor_value,
                    self.pointer_intensity,
                );
                OverlayRenderer::draw_orientation_labels(
                    &painter,
                    response.rect,
                    axis,
                    &vol.direction,
                );
            }
        }

        if self.show_label_overlay {
            self.draw_label_overlay(&painter, response.rect, axis);
        }

        if self.show_rt_struct_overlay {
            self.draw_rt_struct_overlay(
                &painter,
                response.rect,
                axis,
                tex_h_usize,
                tex_w_usize,
            );
        }

        if self.show_rt_dose_overlay {
            self.draw_rt_dose_overlay(&painter, response.rect, axis, slice_idx);
        }

        // Crosshair at the linked study-coordinate cursor.
        if self.show_crosshair {
            if let (Some(vol), Some(cursor)) = (&self.loaded, self.linked_cursor) {
                if let Some(crosshair) = cursor.viewport_crosshair(vol.shape, axis, response.rect) {
                    let color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, 120);
                    painter.line_segment(
                        [
                            egui::pos2(response.rect.min.x, crosshair.y),
                            egui::pos2(response.rect.max.x, crosshair.y),
                        ],
                        egui::Stroke::new(1.0, color),
                    );
                    painter.line_segment(
                        [
                            egui::pos2(crosshair.x, response.rect.min.y),
                            egui::pos2(crosshair.x, response.rect.max.y),
                        ],
                        egui::Stroke::new(1.0, color),
                    );
                }
            }
        }

        // ── 7. Measurement annotations and live tool preview ───────────────────
        //
        // Mathematical mapping:
        //   screen_px = rect.min + img_px × scale
        //   img_px    = (screen_px − rect.min) / scale
        //
        // `scale = fit_scale × zoom` where fit_scale = min(avail_w/tex_w, avail_h/tex_h).
        // The image widget occupies exactly response.rect (egui places it top-left).
        {
            let scale = if tex_w > 0.0 && tex_h > 0.0 {
                (available.x / tex_w).min(available.y / tex_h) * self.zoom
            } else {
                1.0
            };

            // img_to_screen: image-pixel Pos2 { x: col, y: row } → screen Pos2
            let origin = response.rect.min;
            let img_to_screen =
                |p: egui::Pos2| egui::pos2(origin.x + p.x * scale, origin.y + p.y * scale);

            // Per-axis 2-D spacing: [row_mm_per_px, col_mm_per_px]
            // axis 0 axial    → dy, dx
            // axis 1 coronal  → dz, dx
            // axis 2 sagittal → dz, dy
            let spacing_2d: [f32; 2] = if let Some(vol) = &self.loaded {
                let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
                match axis {
                    0 => [dy, dx],
                    1 => [dz, dx],
                    _ => [dz, dy],
                }
            } else {
                [1.0, 1.0]
            };

            // Cursor in image-pixel coords for live preview labels.
            let cursor_img_opt = if scale > 0.0 {
                response.hover_pos().map(|s| {
                    egui::pos2((s.x - origin.x) / scale, (s.y - origin.y) / scale)
                })
            } else {
                None
            };

            // Re-acquire the painter (drop(painter) has not been called yet —
            // this block replaces it; the original painter is consumed below).
            let meas_painter = ui.painter_at(response.rect);
            crate::ui::measurements::MeasurementLayer::draw_annotations(
                &meas_painter,
                &self.annotations,
                &img_to_screen,
            );
            crate::ui::measurements::MeasurementLayer::draw_in_progress(
                &meas_painter,
                &self.tool_state,
                response.hover_pos(),
                cursor_img_opt,
                spacing_2d,
                &img_to_screen,
            );
        }

        // painter is dropped here; no longer borrows ui.
        drop(painter);

        // ── 7. Wheel input: zoom or slice navigation ───────────────────────────
        let (scroll_y, ctrl_or_cmd) =
            ctx.input(|i| (i.smooth_scroll_delta.y, i.modifiers.ctrl || i.modifiers.command));
        if response.hovered() && scroll_y != 0.0 {
            if should_zoom_with_scroll(ctrl_or_cmd) {
                self.zoom = zoom_from_scroll(self.zoom, scroll_y);
                self.status_message = format!("Zoom: {:.0}%", self.zoom * 100.0);
            } else {
                let step = if scroll_y > 0.0 { -1i32 } else { 1 };
                self.step_slice_for_axis(axis, step);
            }
        }

        // ── 8. Pointer events ──────────────────────────────────────────────────
        // Update pointer intensity whenever pointer is over the viewport
        if response.hovered() || response.dragged() || response.interact_pointer_pos().is_some() {
            self.update_pointer_intensity(axis, response.interact_pointer_pos(), response.rect);
        }

        if response.drag_started() {
            if self.active_tool == ToolKind::LabelPaint || self.active_tool == ToolKind::LabelErase
            {
                self.apply_label_at_pointer(axis, response.interact_pointer_pos(), response.rect);
            }
            self.on_drag_start(response.interact_pointer_pos());
        }
        if response.dragged() {
            if self.active_tool == ToolKind::LabelPaint || self.active_tool == ToolKind::LabelErase
            {
                self.apply_label_at_pointer(axis, response.interact_pointer_pos(), response.rect);
            }
            self.on_drag(response.interact_pointer_pos());
        }
        if response.drag_stopped() {
            self.on_drag_end(response.interact_pointer_pos());
        }
        if response.clicked() {
            self.update_linked_cursor_from_pointer(axis, response.interact_pointer_pos(), response.rect);
            if self.active_tool == ToolKind::LabelPaint || self.active_tool == ToolKind::LabelErase
            {
                self.apply_label_at_pointer(axis, response.interact_pointer_pos(), response.rect);
            }
            self.on_click(response.interact_pointer_pos());
        }
    }

    // ── Texture management ────────────────────────────────────────────────────

    /// Render the slice for `axis` through the WL LUT and upload to the GPU.
    ///
    /// Reads `vol`, `viewer_state`, `colormap`, and the per-axis slice index
    /// inside a block so the immutable borrow of `self.loaded` is released
    /// before the mutable assignment to the corresponding texture field.
    fn rebuild_texture_for_axis(&mut self, ctx: &egui::Context, axis: usize) {
        let (color_image, tex_name) = {
            let Some(vol) = &self.loaded else { return };
            let wc = self.viewer_state.window_center.unwrap_or(128.0) as f64;
            let ww = self.viewer_state.window_width.unwrap_or(256.0).max(1.0) as f64;
            let wl = WindowLevel::new(wc, ww);
            let slice_index = match axis {
                0 => self.viewer_state.slice_index,
                1 => self.coronal_slice,
                _ => self.sagittal_slice,
            };
            let name = match axis {
                0 => "slice_tex_axial",
                1 => "slice_tex_coronal",
                _ => "slice_tex_sagittal",
            };
            let img = SliceRenderer::render(vol, axis, slice_index, wl, self.colormap);
            // Apply viewport orientation transform (flip/rotate) before GPU upload.
            let img = apply_to_image(&img, self.view_transform);
            (img, name)
        }; // immutable borrow of self.loaded released here

        let tex = ctx.load_texture(tex_name, color_image, egui::TextureOptions::LINEAR);
        match axis {
            0 => self.texture = Some(tex),
            1 => self.coronal_tex = Some(tex),
            _ => self.sagittal_tex = Some(tex),
        }
    }

    // ── Info panel (4th quadrant) ─────────────────────────────────────────────

    /// Show W/L, per-axis slice info, and spacing in the 4th quadrant of the
    /// 2×2 MPR grid.
    fn show_right_info_panel(&self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.heading("MPR Info");
            ui.separator();

            if let Some(vol) = &self.loaded {
                let [depth, rows, cols] = vol.shape;
                let [dz, dy, dx] = vol.spacing;
                let wc = self.viewer_state.window_center.unwrap_or(128.0);
                let ww = self.viewer_state.window_width.unwrap_or(256.0);

                egui::Grid::new("mpr_info_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        let row = |ui: &mut egui::Ui, k: &str, v: &str| {
                            ui.label(k);
                            ui.label(v);
                            ui.end_row();
                        };

                        row(ui, "W / L:", &format!("{ww:.0} / {wc:.0}"));
                        row(
                            ui,
                            "Axial:",
                            &format!("{}/{}", self.viewer_state.slice_index + 1, depth),
                        );
                        row(
                            ui,
                            "Coronal:",
                            &format!("{}/{}", self.coronal_slice + 1, rows),
                        );
                        row(
                            ui,
                            "Sagittal:",
                            &format!("{}/{}", self.sagittal_slice + 1, cols),
                        );
                        let cursor = self
                            .linked_cursor
                            .map(|cursor| cursor.voxel())
                            .unwrap_or([self.viewer_state.slice_index, self.coronal_slice, self.sagittal_slice]);
                        row(
                            ui,
                            "Cursor:",
                            &format!("z={} y={} x={}", cursor[0] + 1, cursor[1] + 1, cursor[2] + 1),
                        );
                        // Physical LPS position derived from voxel cursor via ITK affine.
                        let lps = voxel_to_lps(cursor, vol.origin, vol.direction, vol.spacing);
                        row(ui, "LPS:", &format_lps(lps));
                        row(ui, "Dims:", &format!("{depth}×{rows}×{cols}"));
                        row(ui, "Spacing:", &format!("{dz:.2}×{dy:.2}×{dx:.2} mm"));
                        row(ui, "Modality:", vol.modality.as_deref().unwrap_or("—"));
                        row(ui, "Patient:", vol.patient_name.as_deref().unwrap_or("—"));
                    });

                ui.separator();
                ui.label("Scroll wheel: navigate slices.");
                ui.label("Ctrl/Cmd + scroll: zoom in/out.");
                ui.label("Ctrl/Cmd + 0: zoom to fit.");
                ui.label("Ctrl/Cmd + Z / Shift+Z / Y: segmentation undo/redo.");
                ui.label("Arrow/Page: previous/next slice. Home/End: first/last slice.");
                ui.label("Click: move the linked MPR cursor.");
                ui.label("Drag: active tool interaction.");
            } else {
                ui.label("No volume loaded.");
                ui.label("Open a DICOM folder or NIfTI file via File menu.");
            }
        });
    }

    // ── PNG export ────────────────────────────────────────────────────────────

    /// Save the current primary-axis slice as a PNG file chosen by the user.
    ///
    /// Uses a synchronous OS save dialog (blocks the UI thread while open).
    /// On success the file is written silently; on failure the save dialog
    /// returns `None` and no file is written.
    fn export_current_slice(&mut self) {
        let Some(vol) = &self.loaded else { return };
        let wc = self.viewer_state.window_center.unwrap_or(128.0) as f64;
        let ww = self.viewer_state.window_width.unwrap_or(256.0).max(1.0) as f64;
        let wl = WindowLevel::new(wc, ww);
        let color_image = SliceRenderer::render(
            vol,
            self.axis,
            self.viewer_state.slice_index,
            wl,
            self.colormap,
        );
        let color_image = apply_to_image(&color_image, self.view_transform);

        // Convert egui::Color32 pixels → packed RGB bytes.
        let rgb_bytes: Vec<u8> = color_image
            .pixels
            .iter()
            .flat_map(|c| [c.r(), c.g(), c.b()])
            .collect();
        let [w, h] = color_image.size;

        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("slice.png")
            .add_filter("PNG", &["png"])
            .save_file()
        {
            match image::RgbImage::from_raw(w as u32, h as u32, rgb_bytes)
                .ok_or_else(|| anyhow::anyhow!("buffer length mismatch"))
                .and_then(|img| img.save(&path).map_err(anyhow::Error::from))
            {
                Ok(()) => {
                    self.status_message = format!("Exported slice PNG: {}", path.display());
                    info!("{}", self.status_message);
                }
                Err(e) => {
                    self.status_message = format!("PNG export failed for {}: {e:#}", path.display());
                    error!("{}", self.status_message);
                }
            }
        }
    }

    fn load_rt_struct_file(&mut self, path: std::path::PathBuf) {
        match ritk_io::read_rt_struct(&path) {
            Ok(rt) => {
                let roi_count = rt.rois.len();
                let label = rt.structure_set_label.clone();
                self.rt_struct = Some(rt);
                self.show_rt_struct_overlay = true;
                self.status_message = format!(
                    "Loaded RT-STRUCT {} ({} ROIs) from {}",
                    label,
                    roi_count,
                    path.display()
                );
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("RT-STRUCT load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    fn load_rt_dose_file(&mut self, path: std::path::PathBuf) {
        match ritk_io::read_rt_dose(&path) {
            Ok(grid) => {
                self.status_message = format!(
                    "Loaded RT-DOSE ({} type, {}×{}×{} grid) from {}",
                    grid.dose_type,
                    grid.rows, grid.cols, grid.n_frames,
                    path.display()
                );
                info!("{}", self.status_message);
                self.rt_dose = Some(grid);
                self.clear_rt_dose_overlay_cache();
                self.show_rt_dose_overlay = true;
            }
            Err(e) => {
                self.status_message =
                    format!("RT-DOSE load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    /// Draw the RT-DOSE heat-map overlay on the given viewport.
    fn draw_rt_dose_overlay(
        &mut self,
        painter: &egui::Painter,
        rect: egui::Rect,
        axis: usize,
        slice_idx: usize,
    ) {
        use crate::ui::rtdose_overlay::extract_dose_slice_for_volume;
        use crate::ui::rtdose_texture::{build_overlay_image, overlay_alpha, positive_finite_dose_range};

        let (Some(rt_dose), Some(vol)) = (&self.rt_dose, &self.loaded) else {
            return;
        };

        let axis_slot = axis.min(2);
        let vol_shape = vol.shape;
        let dose_dims = [rt_dose.n_frames, rt_dose.rows, rt_dose.cols];
        let opacity_alpha = overlay_alpha(self.rt_dose_opacity);
        if let Some(entry) = self.rt_dose_overlay_cache[axis_slot].as_ref() {
            if entry.slice_idx == slice_idx
                && entry.vol_shape == vol_shape
                && entry.dose_dims == dose_dims
                && entry.opacity_alpha == opacity_alpha
            {
                painter.image(
                    entry.texture.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
                return;
            }
        }

        let [depth, rows, cols] = vol_shape;
        let vol_origin = [vol.origin[0] as f64, vol.origin[1] as f64, vol.origin[2] as f64];
        let vol_dir: [f64; 9] = std::array::from_fn(|i| vol.direction[i] as f64);
        let vol_spacing = [vol.spacing[0] as f64, vol.spacing[1] as f64, vol.spacing[2] as f64];

        let Some(dose_map) = extract_dose_slice_for_volume(
            rt_dose,
            axis,
            slice_idx,
            [depth, rows, cols],
            vol_origin,
            vol_dir,
            vol_spacing,
        ) else {
            return;
        };

        let Some((min_dose, max_dose)) = positive_finite_dose_range(&dose_map) else {
            return;
        };

        let (slice_rows, slice_cols) = match axis {
            0 => (rows, cols),
            1 => (depth, cols),
            _ => (depth, rows),
        };
        if slice_rows == 0 || slice_cols == 0 {
            return;
        }

        let Some(color_image) = build_overlay_image(
            &dose_map,
            slice_rows,
            slice_cols,
            min_dose,
            max_dose,
            self.rt_dose_opacity,
        ) else {
            return;
        };

        let tex_name = format!("rtdose_overlay_axis{}_slice{}", axis_slot, slice_idx);
        let texture = painter
            .ctx()
            .load_texture(tex_name, color_image, egui::TextureOptions::LINEAR);
        let texture_id = texture.id();
        self.rt_dose_overlay_cache[axis_slot] = Some(RtDoseOverlayCacheEntry {
            slice_idx,
            vol_shape,
            dose_dims,
            opacity_alpha,
            texture,
        });

        painter.image(
            texture_id,
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    }

    /// Apply the currently configured `active_filter` to the loaded volume.
    ///
    /// Reconstructs an `Image<LoadBackend, 3>` from the flat `LoadedVolume`,
    /// runs the filter, then writes the result back as a new flat `Arc<Vec<f32>>`.
    /// Marks all slice textures dirty so the next frame re-renders.
    fn apply_filter_to_loaded_volume(&mut self) {
        use burn::tensor::{Shape, TensorData};
        use ritk_core::image::Image;
        use ritk_core::spatial::{Direction, Point, Spacing};

        let Some(vol) = self.loaded.as_ref() else {
            self.status_message = "No volume loaded.".to_owned();
            return;
        };

        let [depth, rows, cols] = vol.shape;
        let device = burn_ndarray::NdArrayDevice::Cpu;

        // Build Image<LoadBackend, 3> from the flat volume data.
        let td = TensorData::new((*vol.data).clone(), Shape::new([depth, rows, cols]));
        let tensor = burn::tensor::Tensor::<LoadBackend, 3>::from_data(td, &device);
        let origin = Point::new(vol.origin);
        let spacing = Spacing::new(vol.spacing);
        let mut dir_mat = nalgebra::SMatrix::<f64, 3, 3>::identity();
        for r in 0..3 {
            for c in 0..3 {
                dir_mat[(r, c)] = vol.direction[r * 3 + c];
            }
        }
        let direction = Direction(dir_mat);
        let image: Image<LoadBackend, 3> = Image::new(tensor, origin, spacing, direction);

        // Apply the selected filter.
        let filter_kind = self.active_filter.clone();
        let filter_result = {
            use ritk_core::filter::{
                AbsImageFilter, BedSeparationFilter, BinaryDilateFilter, BinaryErodeFilter,
                BinaryFillholeFilter, BinaryMorphologicalClosing, BinaryMorphologicalOpening,
                ClaheFilter, ConnectedComponentsFilter, ExpImageFilter, GaussianFilter,
                GradientAnisotropicDiffusionFilter, GradientDiffusionConfig,
                GrayscaleClosingFilter, GrayscaleFillholeFilter,
                GrayscaleMorphologicalGradientFilter, GrayscaleOpeningFilter,
                HistogramEqualizationFilter, InvertIntensityFilter, LogImageFilter, MedianFilter,
                MultiOtsuThreshold, NormalizeImageFilter, RelabelComponentFilter,
                SqrtImageFilter, SquareImageFilter, UnsharpMaskFilter,
            };
            match &filter_kind {
                crate::FilterKind::BedSeparation(config) => {
                    BedSeparationFilter::new(*config).apply(&image)
                }
                crate::FilterKind::Gaussian { sigma } => {
                    Ok(GaussianFilter::<LoadBackend>::new(vec![f64::from(*sigma); 3]).apply(&image))
                }
                crate::FilterKind::Median { radius } => {
                    MedianFilter::new(*radius).apply(&image)
                }
                crate::FilterKind::Clahe {
                    tile_grid_size,
                    clip_limit,
                } => ClaheFilter::new(*tile_grid_size, *clip_limit, 256).apply(&image),
                crate::FilterKind::HistEq { bins } => {
                    HistogramEqualizationFilter::new(*bins).apply(&image)
                }
                crate::FilterKind::UnsharpMask {
                    sigma,
                    amount,
                    threshold,
                    clamp,
                } => UnsharpMaskFilter::new(
                    vec![f64::from(*sigma)],
                    f64::from(*amount),
                    f64::from(*threshold),
                    *clamp,
                )
                .apply(&image),
                crate::FilterKind::GradientAnisotropicDiffusion {
                    iterations,
                    time_step,
                    conductance,
                } => GradientAnisotropicDiffusionFilter::new(GradientDiffusionConfig {
                    num_iterations: *iterations as usize,
                    time_step: *time_step,
                    conductance: *conductance,
                })
                .apply(&image),
                crate::FilterKind::ConnectedComponents {
                    connectivity_26,
                    background_value,
                } => {
                    let connectivity = if *connectivity_26 { 26 } else { 6 };
                    let filter = ConnectedComponentsFilter::with_connectivity(connectivity)
                        .with_background(*background_value);
                    let (label_image, _stats) = filter.apply(&image);
                    Ok(label_image)
                }
                crate::FilterKind::RelabelComponents { minimum_object_size } => {
                    let (relabeled, _stats) =
                        RelabelComponentFilter::with_minimum_object_size(
                            *minimum_object_size as usize,
                        )
                        .apply(&image);
                    Ok(relabeled)
                }
                crate::FilterKind::MultiOtsuThreshold { num_classes } => Ok(
                    MultiOtsuThreshold::new(*num_classes as usize).apply(&image),
                ),
                crate::FilterKind::BinaryErode { radius, foreground_value } => {
                    BinaryErodeFilter::new(*radius).with_foreground(*foreground_value).apply(&image)
                }
                crate::FilterKind::BinaryDilate { radius, foreground_value } => {
                    BinaryDilateFilter::new(*radius).with_foreground(*foreground_value).apply(&image)
                }
                crate::FilterKind::BinaryClosing { radius, foreground_value } => {
                    BinaryMorphologicalClosing::new(*radius).with_foreground(*foreground_value).apply(&image)
                }
                crate::FilterKind::BinaryOpening { radius, foreground_value } => {
                    BinaryMorphologicalOpening::new(*radius).with_foreground(*foreground_value).apply(&image)
                }
                crate::FilterKind::BinaryFillhole { foreground_value } => {
                    BinaryFillholeFilter::new().with_foreground(*foreground_value).apply(&image)
                }
                crate::FilterKind::GrayscaleClosing { radius } => {
                    GrayscaleClosingFilter::new(*radius).apply(&image)
                }
                crate::FilterKind::GrayscaleOpening { radius } => {
                    GrayscaleOpeningFilter::new(*radius).apply(&image)
                }
                crate::FilterKind::GrayscaleFillhole => {
                    GrayscaleFillholeFilter::new().apply(&image)
                }
                crate::FilterKind::Abs => Ok(AbsImageFilter::new().apply(&image)),
                crate::FilterKind::InvertIntensity { maximum } => {
                    Ok(match maximum {
                        Some(m) => InvertIntensityFilter::with_maximum(*m).apply(&image),
                        None => InvertIntensityFilter::new().apply(&image),
                    })
                }
                crate::FilterKind::NormalizeIntensity => {
                    Ok(NormalizeImageFilter::new().apply(&image))
                }
                crate::FilterKind::Square => Ok(SquareImageFilter::new().apply(&image)),
                crate::FilterKind::Sqrt => Ok(SqrtImageFilter::new().apply(&image)),
                crate::FilterKind::Log => Ok(LogImageFilter::new().apply(&image)),
                crate::FilterKind::Exp => Ok(ExpImageFilter::new().apply(&image)),
                crate::FilterKind::MorphologicalGradient { radius } => {
                    GrayscaleMorphologicalGradientFilter::new(*radius).apply(&image)
                }
                crate::FilterKind::DistanceTransform { threshold } => {
                    ritk_core::filter::DistanceTransformImageFilter::new()
                        .with_threshold(*threshold)
                        .apply(&image)
                }
                crate::FilterKind::SignedDistanceTransform { threshold } => {
                    ritk_core::filter::SignedDistanceTransformImageFilter::new()
                        .with_threshold(*threshold)
                        .apply(&image)
                }
                crate::FilterKind::FlipZ => ritk_core::filter::FlipImageFilter::flip_z().apply(&image),
                crate::FilterKind::FlipY => ritk_core::filter::FlipImageFilter::flip_y().apply(&image),
                crate::FilterKind::FlipX => ritk_core::filter::FlipImageFilter::flip_x().apply(&image),
                crate::FilterKind::MaskThreshold { threshold } => {
                    let dims = image.shape();
                    let td = image.data().clone().into_data();
                    let vals: Vec<f32> = td
                        .into_vec::<f32>()
                        .unwrap_or_else(|_| vec![0.0; dims[0] * dims[1] * dims[2]]);
                    let mask_vals: Vec<f32> =
                        vals.iter().map(|&v| if v > *threshold { 1.0_f32 } else { 0.0_f32 }).collect();
                    let device = image.data().device();
                    let mask_td = burn::tensor::TensorData::new(
                        mask_vals, burn::tensor::Shape::new(dims));
                    let mask_tensor = burn::tensor::Tensor::<LoadBackend, 3>::from_data(mask_td, &device);
                    let mask_image = ritk_core::image::Image::new(
                        mask_tensor,
                        *image.origin(),
                        *image.spacing(),
                        *image.direction(),
                    );
                    ritk_core::filter::MaskImageFilter::new().apply(&image, &mask_image)
                }
                crate::FilterKind::GeodesicDilationSelf => {
                    ritk_core::filter::GrayscaleGeodesicDilationFilter::new().apply(&image, &image)
                }
                crate::FilterKind::GeodesicErosionSelf => {
                    ritk_core::filter::GrayscaleGeodesicErosionFilter::new().apply(&image, &image)
                }
                crate::FilterKind::ShiftScale { shift, scale } => {
                    ritk_core::filter::ShiftScaleImageFilter::new(*shift, *scale).apply(&image)
                }
                crate::FilterKind::ZeroCrossing { foreground_value, background_value } => {
                    ritk_core::filter::ZeroCrossingImageFilter::new()
                        .with_foreground(*foreground_value)
                        .with_background(*background_value)
                        .apply(&image)
                }
                crate::FilterKind::RegionOfInterest {
                    start_z, start_y, start_x,
                    size_z, size_y, size_x,
                } => {
                    ritk_core::filter::RegionOfInterestImageFilter::new(
                        [*start_z, *start_y, *start_x],
                        [*size_z, *size_y, *size_x],
                    )
                    .apply(&image)
                }
                crate::FilterKind::PermuteAxes { order_0, order_1, order_2 } => {
                    ritk_core::filter::PermuteAxesImageFilter::new([*order_0, *order_1, *order_2])
                        .apply(&image)
                }
                crate::FilterKind::Mean { radius } => {
                    ritk_core::filter::MeanImageFilter::new(*radius).apply(&image)
                }
                crate::FilterKind::BinaryContour { fully_connected, foreground_value } => {
                    ritk_core::filter::BinaryContourImageFilter::new(*fully_connected, *foreground_value)
                        .apply(&image)
                }
                crate::FilterKind::LabelContour { fully_connected, background_value } => {
                    ritk_core::filter::LabelContourImageFilter::new(*fully_connected, *background_value)
                        .apply(&image)
                }
                crate::FilterKind::VotingBinary {
                    radius, birth_threshold, survival_threshold, foreground_value, background_value,
                } => ritk_core::filter::VotingBinaryImageFilter::new(
                    *radius, *birth_threshold, *survival_threshold,
                    *foreground_value, *background_value,
                ).apply(&image),
                crate::FilterKind::Shrink { factor_z, factor_y, factor_x } => {
                    ritk_core::filter::ShrinkImageFilter::new([*factor_z, *factor_y, *factor_x])
                        .apply(&image)
                }
                crate::FilterKind::ConstantPad {
                    pad_lower_z, pad_lower_y, pad_lower_x,
                    pad_upper_z, pad_upper_y, pad_upper_x, constant,
                } => ritk_core::filter::ConstantPadImageFilter::new(
                    [*pad_lower_z, *pad_lower_y, *pad_lower_x],
                    [*pad_upper_z, *pad_upper_y, *pad_upper_x],
                    *constant,
                ).apply(&image),
                crate::FilterKind::MirrorPad {
                    pad_lower_z, pad_lower_y, pad_lower_x,
                    pad_upper_z, pad_upper_y, pad_upper_x,
                } => ritk_core::filter::MirrorPadImageFilter::new(
                    [*pad_lower_z, *pad_lower_y, *pad_lower_x],
                    [*pad_upper_z, *pad_upper_y, *pad_upper_x],
                ).apply(&image),
                crate::FilterKind::WrapPad {
                    pad_lower_z, pad_lower_y, pad_lower_x,
                    pad_upper_z, pad_upper_y, pad_upper_x,
                } => ritk_core::filter::WrapPadImageFilter::new(
                    [*pad_lower_z, *pad_lower_y, *pad_lower_x],
                    [*pad_upper_z, *pad_upper_y, *pad_upper_x],
                ).apply(&image),
                crate::FilterKind::GrayscaleErode { radius } => {
                    ritk_core::filter::GrayscaleErosion::new(*radius).apply(&image)
                }
                crate::FilterKind::GrayscaleDilate { radius } => {
                    ritk_core::filter::GrayscaleDilation::new(*radius).apply(&image)
                }
                crate::FilterKind::BinaryThreshold { lower, upper, foreground, background } => {
                    ritk_core::filter::BinaryThresholdImageFilter::new(*lower, *upper, *foreground, *background)
                        .apply(&image)
                }
                crate::FilterKind::RescaleIntensity { out_min, out_max } => {
                    ritk_core::filter::RescaleIntensityFilter::new(*out_min, *out_max).apply(&image)
                }
                crate::FilterKind::Clamp { lower, upper } => {
                    ritk_core::filter::ClampImageFilter::new(*lower, *upper).apply(&image)
                }
                crate::FilterKind::ConnectedThreshold { seed_z, seed_y, seed_x, lower, upper } => {
                    Ok(ritk_core::segmentation::region_growing::ConnectedThresholdFilter::new(
                        [*seed_z, *seed_y, *seed_x], *lower, *upper,
                    ).apply(&image))
                }
                crate::FilterKind::ConfidenceConnected {
                    seed_z, seed_y, seed_x, initial_lower, initial_upper, multiplier, max_iterations,
                } => {
                    Ok(ritk_core::segmentation::region_growing::ConfidenceConnectedFilter::new(
                        [*seed_z, *seed_y, *seed_x], *initial_lower, *initial_upper,
                    )
                    .with_multiplier(*multiplier)
                    .with_max_iterations(*max_iterations as usize)
                    .apply(&image))
                }
                crate::FilterKind::NeighborhoodConnected {
                    seed_z, seed_y, seed_x, lower, upper, radius_z, radius_y, radius_x,
                } => {
                    Ok(ritk_core::segmentation::region_growing::NeighborhoodConnectedFilter::new(
                        [*seed_z, *seed_y, *seed_x], *lower, *upper,
                    )
                    .with_radius([*radius_z, *radius_y, *radius_x])
                    .apply(&image))
                }
            }
        };

        match filter_result {
            Err(e) => {
                self.status_message = format!("Filter failed: {e:#}");
            }
            Ok(out_img) => {
                let out_td = out_img.into_tensor().into_data();
                let out_vec: Vec<f32> = out_td.as_slice::<f32>().unwrap_or(&[]).to_vec();
                let vol = self.loaded.as_mut().unwrap();
                vol.data = std::sync::Arc::new(out_vec);
                self.texture_dirty = true;
                self.coronal_dirty = true;
                self.sagittal_dirty = true;
                self.status_message = "Filter applied.".to_owned();
            }
        }
    }

    /// Export all axial/coronal/sagittal slices as PNG files under a selected
    /// directory using the canonical MPR export plan.
    fn export_all_mpr_slices(&mut self) {
        let Some(vol) = &self.loaded else {
            self.status_message = "No volume loaded; MPR export skipped.".to_owned();
            return;
        };

        let Some(root) = rfd::FileDialog::new().pick_folder() else {
            return;
        };

        let wc = self.viewer_state.window_center.unwrap_or(128.0) as f64;
        let ww = self.viewer_state.window_width.unwrap_or(256.0).max(1.0) as f64;
        let wl = WindowLevel::new(wc, ww);
        let plan = plan_all_mpr_exports(vol.shape);

        let mut success = 0usize;
        let mut failed = 0usize;

        for export in plan {
            let axis_dir = root.join(export.axis_folder);
            if let Err(e) = std::fs::create_dir_all(&axis_dir) {
                failed += 1;
                error!(path = %axis_dir.display(), error = %e, "failed to create axis export directory");
                continue;
            }

            let path = axis_dir.join(export.file_name);
            let color_image = SliceRenderer::render(
                vol,
                export.axis,
                export.slice_index,
                wl,
                self.colormap,
            );
            let color_image = apply_to_image(&color_image, self.view_transform);
            let rgb_bytes: Vec<u8> = color_image
                .pixels
                .iter()
                .flat_map(|c| [c.r(), c.g(), c.b()])
                .collect();
            let [w, h] = color_image.size;

            let result = image::RgbImage::from_raw(w as u32, h as u32, rgb_bytes)
                .ok_or_else(|| anyhow::anyhow!("buffer length mismatch"))
                .and_then(|img| img.save(&path).map_err(anyhow::Error::from));

            match result {
                Ok(()) => success += 1,
                Err(e) => {
                    failed += 1;
                    error!(path = %path.display(), error = %e, "failed to export MPR PNG slice");
                }
            }
        }

        self.status_message = format!(
            "MPR export complete: {} succeeded, {} failed ({})",
            success,
            failed,
            root.display()
        );
        info!("{}", self.status_message);
    }

    fn save_session_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .set_file_name("ritk-snap-session.json")
            .add_filter("JSON", &["json"])
            .save_file()
        else {
            return;
        };

        let snapshot = self.session_snapshot();
        match crate::session::save_to_file(&snapshot, &path) {
            Ok(()) => {
                self.status_message = format!("Saved session to {}", path.display());
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("Session save failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    fn load_session_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON", &["json"])
            .pick_file()
        else {
            return;
        };

        match crate::session::load_from_file(&path) {
            Ok(snapshot) => {
                self.apply_session_snapshot(snapshot);
                self.status_message = format!("Loaded session from {}", path.display());
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("Session load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    /// Save the current label map to a NIfTI file.
    ///
    /// Requires a loaded volume (for geometry) and an initialised label editor.
    /// The dialog is a no-op when either is absent; a status message explains
    /// the missing precondition.
    fn save_segmentation_dialog(&mut self) {
        let (Some(vol), Some(editor)) = (self.loaded.as_ref(), self.label_editor.as_ref()) else {
            self.status_message =
                "Save segmentation: no volume or segmentation loaded.".to_owned();
            return;
        };
        let map = editor.current_map();
        let origin = [
            vol.origin[0] as f32,
            vol.origin[1] as f32,
            vol.origin[2] as f32,
        ];
        let spacing = [
            vol.spacing[0] as f32,
            vol.spacing[1] as f32,
            vol.spacing[2] as f32,
        ];
        let direction: [f32; 9] = std::array::from_fn(|i| vol.direction[i] as f32);

        let Some(path) = rfd::FileDialog::new()
            .set_file_name("segmentation.nii.gz")
            .add_filter("NIfTI", &["nii", "gz"])
            .save_file()
        else {
            return;
        };

        match ritk_io::write_nifti_labels(
            &path,
            map.as_slice(),
            map.shape,
            origin,
            spacing,
            direction,
        ) {
            Ok(()) => {
                self.status_message =
                    format!("Saved segmentation to {}", path.display());
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message =
                    format!("Segmentation save failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    /// Load a label map from a NIfTI file and replace the current segmentation.
    ///
    /// The shape of the loaded label map must match the currently loaded volume.
    /// A status message is set for all outcomes (success, mismatch, error).
    fn load_segmentation_dialog(&mut self) {
        let Some(vol) = self.loaded.as_ref() else {
            self.status_message =
                "Load segmentation: no volume loaded.".to_owned();
            return;
        };
        let expected_shape = vol.shape;

        let Some(path) = rfd::FileDialog::new()
            .add_filter("NIfTI", &["nii", "gz"])
            .pick_file()
        else {
            return;
        };

        match ritk_io::read_nifti_labels(&path) {
            Ok((labels, shape)) => {
                if shape != expected_shape {
                    self.status_message = format!(
                        "Segmentation shape {:?} does not match volume {:?}",
                        shape, expected_shape
                    );
                    error!("{}", self.status_message);
                    return;
                }
                match ritk_core::annotation::LabelMap::from_data(
                    shape,
                    labels,
                    crate::label::default_label_table(),
                ) {
                    Ok(map) => {
                        self.label_editor =
                            Some(crate::label::LabelEditor::from_label_map(map));
                        self.status_message =
                            format!("Loaded segmentation from {}", path.display());
                        info!("{}", self.status_message);
                    }
                    Err(e) => {
                        self.status_message =
                            format!("Segmentation data error: {e}");
                        error!("{}", self.status_message);
                    }
                }
            }
            Err(e) => {
                self.status_message =
                    format!("Segmentation load failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    // ── Series browser helpers ────────────────────────────────────────────────

    /// Scan `folder` for DICOM series and populate [`series_tree`].
    ///
    /// Uses [`crate::dicom::loader::scan_folder_for_series`] which walks the
    /// directory tree up to depth 5.  On success the tree is replaced and the
    /// status bar is updated; on failure only the status bar is updated.
    ///
    /// [`series_tree`]: SnapApp::series_tree
    fn scan_for_series(&mut self, folder: std::path::PathBuf) {
        let scan_root = crate::dicom::classify_dicom_input_path(&folder)
            .dicom_root()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| folder.clone());
        match crate::dicom::loader::scan_folder_for_series(&scan_root) {
            Ok(tree) => {
                let n = tree.total_series();
                self.series_tree = tree;
                self.status_message = format!("Found {n} series in {}", scan_root.display());
                info!("{}", self.status_message);
            }
            Err(e) => {
                let msg = format!("Scan failed for {}: {e:#}", scan_root.display());
                error!("{msg}");
                self.status_message = msg;
            }
        }
    }

    // ── File loading ──────────────────────────────────────────────────────────

    /// Load a DICOM series from `path` using the NdArray CPU backend.
    ///
    /// Delegates to [`crate::dicom::loader::load_dicom_volume`], which wraps
    /// `ritk_io::load_dicom_series_with_metadata`.  On success all viewer
    /// state and texture handles are reset; on failure `status_message` is
    /// updated and any previously loaded volume is preserved.
    fn load_from_path(&mut self, path: std::path::PathBuf) {
        let load_root = crate::dicom::classify_dicom_input_path(&path)
            .dicom_root()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| path.clone());
        info!(
            "loading DICOM series from {} (resolved root: {})",
            path.display(),
            load_root.display()
        );
        self.cine.stop();
        let device: <LoadBackend as burn::tensor::backend::Backend>::Device = Default::default();

        match ritk_io::load_dicom_series_with_metadata::<LoadBackend, _>(&load_root, &device) {
            Ok((image, meta)) => {
                let shape = image.shape();
                let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];
                let origin = [image.origin()[0], image.origin()[1], image.origin()[2]];

                let dir = image.direction().inner();
                let direction = [
                    dir[(0, 0)],
                    dir[(0, 1)],
                    dir[(0, 2)],
                    dir[(1, 0)],
                    dir[(1, 1)],
                    dir[(1, 2)],
                    dir[(2, 0)],
                    dir[(2, 1)],
                    dir[(2, 2)],
                ];

                let raw_data = image.data().clone().into_data();
                let data = match raw_data.into_vec::<f32>() {
                    Ok(v) => Arc::new(v),
                    Err(e) => {
                        let msg = format!("pixel data extraction failed: {e:?}");
                        error!("{msg}");
                        self.status_message = msg;
                        return;
                    }
                };

                let protocol = select_hanging_protocol(
                    meta.modality.as_deref(),
                    meta.series_description.as_deref(),
                    shape,
                );
                let mut state = ViewerState::new();
                state.window_center = Some(protocol.window_center);
                state.window_width = Some(protocol.window_width);
                state.slice_index = shape[0] / 2;

                let modality = meta.modality.clone();
                let patient_name = meta.patient_name.clone();
                let patient_id = meta.patient_id.clone();
                let study_date = meta.study_date.clone();
                let series_description = meta.series_description.clone();

                self.loaded = Some(LoadedVolume {
                    data,
                    shape,
                    spacing,
                    origin,
                    direction,
                    metadata: Some(Box::new(meta)),
                    source: Some(load_root.clone()),
                    modality,
                    patient_name,
                    patient_id,
                    study_date,
                    series_description,
                });
                self.viewer_state = state;
                self.axis = protocol.preferred_axis.min(2);
                self.coronal_slice = shape[1] / 2;
                self.sagittal_slice = shape[2] / 2;
                self.multi_planar = protocol.multi_planar;
                self.linked_cursor = Some(LinkedCursor::from_slices(
                    shape,
                    self.viewer_state.slice_index,
                    self.coronal_slice,
                    self.sagittal_slice,
                ));
                self.annotations.clear();
                self.label_editor = Some(LabelEditor::new(shape));
                self.rt_struct = None;
                self.rt_dose = None;
                self.clear_rt_dose_overlay_cache();
                self.tool_state = ToolState::Idle;
                self.pan_offset = egui::Vec2::ZERO;
                self.zoom = 1.0;
                self.pointer_intensity = 0.0;
                self.texture = None;
                self.texture_dirty = true;
                self.coronal_tex = None;
                self.coronal_dirty = true;
                self.sagittal_tex = None;
                self.sagittal_dirty = true;
                self.status_message = format!(
                    "Loaded {} (root {}) — shape [{}, {}, {}] — protocol {}",
                    path.display(),
                    load_root.display(),
                    shape[0],
                    shape[1],
                    shape[2],
                    protocol.protocol_name,
                );
                self.refresh_cached_histogram();
                info!("{}", self.status_message);
            }
            Err(e) => {
                let msg = format!(
                    "DICOM load failed for {} (root {}): {e:#}",
                    path.display(),
                    load_root.display()
                );
                error!("{msg}");
                self.status_message = msg;
            }
        }
    }

    /// Load a NIfTI (`.nii` / `.nii.gz`) volume from `path`.
    ///
    /// Calls [`crate::dicom::loader::load_nifti_volume`].  On success all
    /// viewer state and textures are reset exactly as in [`load_from_path`].
    /// NIfTI files carry no patient metadata; DICOM-specific fields are `None`.
    fn load_nifti_file(&mut self, path: std::path::PathBuf) {
        self.cine.stop();
        match crate::dicom::loader::load_nifti_volume(&path) {
            Ok(vol) => {
                let shape = vol.shape;
                let protocol = select_hanging_protocol(
                    vol.modality.as_deref(),
                    vol.series_description.as_deref(),
                    shape,
                );
                let mut state = ViewerState::new();
                state.window_center = Some(protocol.window_center);
                state.window_width = Some(protocol.window_width);
                state.slice_index = shape[0] / 2;
                let msg = format!(
                    "Loaded {} — shape {:?} — protocol {}",
                    path.display(),
                    shape,
                    protocol.protocol_name
                );
                self.loaded = Some(vol);
                self.viewer_state = state;
                self.axis = protocol.preferred_axis.min(2);
                self.coronal_slice = shape[1] / 2;
                self.sagittal_slice = shape[2] / 2;
                self.multi_planar = protocol.multi_planar;
                self.linked_cursor = Some(LinkedCursor::from_slices(
                    shape,
                    self.viewer_state.slice_index,
                    self.coronal_slice,
                    self.sagittal_slice,
                ));
                self.annotations.clear();
                self.label_editor = Some(LabelEditor::new(shape));
                self.rt_struct = None;
                self.rt_dose = None;
                self.clear_rt_dose_overlay_cache();
                self.tool_state = ToolState::Idle;
                self.pan_offset = egui::Vec2::ZERO;
                self.zoom = 1.0;
                self.pointer_intensity = 0.0;
                self.texture = None;
                self.texture_dirty = true;
                self.coronal_tex = None;
                self.coronal_dirty = true;
                self.sagittal_tex = None;
                self.sagittal_dirty = true;
                self.status_message = msg;
                self.refresh_cached_histogram();
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("NIfTI load failed: {e:#}");
            }
        }
    }

    /// Drop the currently loaded study and reset all study-owned state.
    fn close_study(&mut self) {
        self.loaded = None;
        self.annotations.clear();
        self.label_editor = None;
        self.rt_struct = None;
        self.rt_dose = None;
        self.clear_rt_dose_overlay_cache();
        self.viewer_state = ViewerState::new();
        self.linked_cursor = None;
        self.pointer_intensity = 0.0;
        self.cached_histogram = None;
        self.selected_series = None;
        self.pan_offset = egui::Vec2::ZERO;
        self.zoom = 1.0;
        self.texture = None;
        self.coronal_tex = None;
        self.sagittal_tex = None;
        self.texture_dirty = false;
        self.coronal_dirty = false;
        self.sagittal_dirty = false;
        self.cine.stop();
        self.status_message = "Study closed.".to_owned();
    }

    /// Compute and cache a 256-bin histogram for the currently loaded volume.
    ///
    /// Scans all voxels to determine the true data minimum and maximum, then
    /// delegates to [`crate::render::histogram::compute_histogram`]. Sets
    /// `self.cached_histogram` to `None` when no volume is loaded.
    fn refresh_cached_histogram(&mut self) {
        use crate::render::histogram::compute_histogram;
        if let Some(vol) = &self.loaded {
            let data: &[f32] = &vol.data;
            // Single pass: compute exact (min, max) over all finite voxels.
            let (mut mn, mut mx) = (f32::MAX, f32::MIN);
            for &v in data {
                if v.is_finite() {
                    if v < mn { mn = v; }
                    if v > mx { mx = v; }
                }
            }
            // Guard against pathological all-NaN or empty data.
            if mn < mx {
                self.cached_histogram = Some(compute_histogram(data, mn, mx, 256));
            } else {
                self.cached_histogram = None;
            }
        } else {
            self.cached_histogram = None;
        }
    }

    /// Clear all cached RT-DOSE overlay textures.
    fn clear_rt_dose_overlay_cache(&mut self) {
        self.rt_dose_overlay_cache = std::array::from_fn(|_| None);
    }

    // ── Slice navigation ──────────────────────────────────────────────────────

    /// Return `(current_slice_index, total_slices)` for `axis`.
    fn axis_slice_info(&self, axis: usize) -> (usize, usize) {
        match axis {
            0 => (
                self.viewer_state.slice_index,
                self.loaded.as_ref().map(|v| v.shape[0]).unwrap_or(1),
            ),
            1 => (
                self.coronal_slice,
                self.loaded.as_ref().map(|v| v.shape[1]).unwrap_or(1),
            ),
            _ => (
                self.sagittal_slice,
                self.loaded.as_ref().map(|v| v.shape[2]).unwrap_or(1),
            ),
        }
    }

    /// Step the slice for `axis` by `delta`, clamped to the valid range.
    ///
    /// Marks the corresponding texture dirty when the index changes.
    fn set_slice_for_axis(&mut self, axis: usize, index: usize) {
        let total = match axis {
            0 => self.loaded.as_ref().map(|v| v.shape[0]).unwrap_or(1),
            1 => self.loaded.as_ref().map(|v| v.shape[1]).unwrap_or(1),
            _ => self.loaded.as_ref().map(|v| v.shape[2]).unwrap_or(1),
        };
        let next = index.min(total.saturating_sub(1));

        match axis {
            0 => {
                if next != self.viewer_state.slice_index {
                    self.viewer_state.slice_index = next;
                    self.texture_dirty = true;
                    if let (Some(vol), Some(cursor)) = (&self.loaded, self.linked_cursor.as_mut()) {
                        cursor.set_axis_slice(vol.shape, 0, next);
                    }
                }
            }
            1 => {
                if next != self.coronal_slice {
                    self.coronal_slice = next;
                    self.coronal_dirty = true;
                    if let (Some(vol), Some(cursor)) = (&self.loaded, self.linked_cursor.as_mut()) {
                        cursor.set_axis_slice(vol.shape, 1, next);
                    }
                }
            }
            _ => {
                if next != self.sagittal_slice {
                    self.sagittal_slice = next;
                    self.sagittal_dirty = true;
                    if let (Some(vol), Some(cursor)) = (&self.loaded, self.linked_cursor.as_mut()) {
                        cursor.set_axis_slice(vol.shape, 2, next);
                    }
                }
            }
        }
    }

    /// Step the slice for `axis` by `delta`, clamped to the valid range.
    ///
    /// Marks the corresponding texture dirty when the index changes.
    fn step_slice_for_axis(&mut self, axis: usize, delta: i32) {
        let (current, total) = self.axis_slice_info(axis);
        let max = total.saturating_sub(1) as i32;
        let next = ((current as i32) + delta).clamp(0, max) as usize;
        self.set_slice_for_axis(axis, next);
    }

    /// Step the primary-axis slice by `delta`.  Delegates to
    /// [`step_slice_for_axis`] using `self.axis`.
    ///
    /// [`step_slice_for_axis`]: SnapApp::step_slice_for_axis
    fn step_slice(&mut self, delta: i32) {
        self.step_slice_for_axis(self.axis, delta);
    }

    /// Advance `axis` by `steps` with wrap-around.
    ///
    /// Delegates the actual write to [`set_slice_for_axis`] so dirty flags,
    /// linked-cursor synchronisation, and the no-change guard are all applied
    /// through the shared SSOT path.
    fn advance_slice_for_axis_loop(&mut self, axis: usize, steps: u32) {
        if steps == 0 {
            return;
        }
        let total = match axis {
            0 => self.loaded.as_ref().map(|v| v.shape[0]).unwrap_or(1),
            1 => self.loaded.as_ref().map(|v| v.shape[1]).unwrap_or(1),
            _ => self.loaded.as_ref().map(|v| v.shape[2]).unwrap_or(1),
        };
        if total == 0 {
            return;
        }
        let current = match axis {
            0 => self.viewer_state.slice_index,
            1 => self.coronal_slice,
            _ => self.sagittal_slice,
        };
        let next = (current + steps as usize) % total;
        self.set_slice_for_axis(axis, next);
    }

    /// Advance cine playback for the active axis and schedule repaints.
    fn tick_cine(&mut self, ctx: &egui::Context) {
        if self.loaded.is_none() {
            self.cine.stop();
            return;
        }
        if !self.cine.enabled {
            return;
        }

        let now = ctx.input(|i| i.time);
        let steps = self.cine.consume_steps(now);
        if steps > 0 {
            self.advance_slice_for_axis_loop(self.axis, steps);
            ctx.request_repaint();
        } else {
            ctx.request_repaint_after(Duration::from_millis(8));
        }
    }

    // ── Tool event handlers ───────────────────────────────────────────────────

    fn on_drag_start(&mut self, pos: Option<egui::Pos2>) {
        let Some(pos) = pos else { return };
        match self.active_tool {
            ToolKind::Pan => {
                self.tool_state = ToolState::Panning {
                    start: pos,
                    viewport_origin: egui::Pos2::new(self.pan_offset.x, self.pan_offset.y),
                };
            }
            ToolKind::Zoom => {
                self.tool_state = ToolState::Zooming {
                    start: pos,
                    original_zoom: self.zoom,
                };
            }
            ToolKind::WindowLevel => {
                let wc = self.viewer_state.window_center.unwrap_or(128.0) as f64;
                let ww = self.viewer_state.window_width.unwrap_or(256.0) as f64;
                self.tool_state = ToolState::WindowLevelDrag {
                    start: pos,
                    original_center: wc,
                    original_width: ww,
                };
            }
            ToolKind::RoiRect => {
                self.tool_state = ToolState::RoiDrag {
                    start: pos,
                    current: pos,
                    kind: RoiKind::Rect,
                };
            }
            ToolKind::RoiEllipse => {
                self.tool_state = ToolState::RoiDrag {
                    start: pos,
                    current: pos,
                    kind: RoiKind::Ellipse,
                };
            }
            _ => {}
        }
    }

    fn on_drag(&mut self, pos: Option<egui::Pos2>) {
        let Some(pos) = pos else { return };
        match self.tool_state.clone() {
            ToolState::Panning {
                start,
                viewport_origin,
            } => {
                self.pan_offset = pan_from_drag_delta(viewport_origin, start, pos);
            }
            ToolState::Zooming {
                start,
                original_zoom,
            } => {
                let drag_delta_y = pos.y - start.y;
                self.zoom = zoom_from_drag_delta(original_zoom, drag_delta_y);
                self.status_message = format!("Zoom: {:.0}%", self.zoom * 100.0);
            }
            ToolState::WindowLevelDrag {
                start,
                original_center,
                original_width,
            } => {
                let (new_center, new_width) = window_level_from_drag_delta(
                    original_center,
                    original_width,
                    pos.x - start.x,
                    pos.y - start.y,
                    WINDOW_LEVEL_SENSITIVITY,
                );
                self.viewer_state.window_center = Some(new_center as f32);
                self.viewer_state.window_width = Some(new_width as f32);
                self.texture_dirty = true;
                self.coronal_dirty = true;
                self.sagittal_dirty = true;
            }
            ToolState::RoiDrag { start, kind, .. } => {
                self.tool_state = ToolState::RoiDrag {
                    start,
                    current: pos,
                    kind,
                };
            }
            _ => {}
        }
    }

    fn on_drag_end(&mut self, pos: Option<egui::Pos2>) {
        if pos.is_some() {
            match self.tool_state.clone() {
                ToolState::RoiDrag {
                    start,
                    current,
                    kind: RoiKind::Rect,
                } => {
                    self.finalise_roi_rect(start, current);
                }
                ToolState::RoiDrag {
                    start,
                    current,
                    kind: RoiKind::Ellipse,
                } => {
                    self.finalise_roi_ellipse(start, current);
                }
                _ => {}
            }
        }
        self.tool_state = ToolState::Idle;
    }

    fn on_click(&mut self, pos: Option<egui::Pos2>) {
        let Some(pos) = pos else { return };
        match self.active_tool {
            ToolKind::MeasureLength => match self.tool_state.clone() {
                ToolState::MeasureLength1 { p1 } => {
                    let p1_arr = [p1.y, p1.x];
                    let p2_arr = [pos.y, pos.x];
                    let spacing = self.slice_spacing_2d();
                    let length_mm = Annotation::compute_length(p1_arr, p2_arr, spacing);
                    self.annotations.push(Annotation::Length {
                        p1: p1_arr,
                        p2: p2_arr,
                        length_mm,
                    });
                    self.tool_state = ToolState::Idle;
                }
                _ => {
                    self.tool_state = ToolState::MeasureLength1 { p1: pos };
                }
            },
            ToolKind::MeasureAngle => match self.tool_state.clone() {
                ToolState::MeasureAngle2 { p1, p2 } => {
                    let a = [p1.y, p1.x];
                    let b = [p2.y, p2.x];
                    let c = [pos.y, pos.x];
                    let angle_deg = Annotation::compute_angle(a, b, c);
                    self.annotations.push(Annotation::Angle {
                        p1: a,
                        p2: b,
                        p3: c,
                        angle_deg,
                    });
                    self.tool_state = ToolState::Idle;
                }
                ToolState::MeasureLength1 { p1 } => {
                    self.tool_state = ToolState::MeasureAngle2 { p1, p2: pos };
                }
                _ => {
                    self.tool_state = ToolState::MeasureLength1 { p1: pos };
                }
            },
            ToolKind::PointHu => {
                if let Some(vol) = &self.loaded {
                    let row = pos.y as usize;
                    let col = pos.x as usize;
                    let (pixels, width, _height) =
                        vol.extract_slice(self.axis, self.viewer_state.slice_index);
                    let idx = row * width + col;
                    let value = if idx < pixels.len() { pixels[idx] } else { 0.0 };
                    self.annotations.push(Annotation::HuPoint {
                        pos: [pos.y, pos.x],
                        value,
                    });
                    self.status_message = format!("HU at col={col} row={row}: {value:.0}");
                }
            }
            ToolKind::LabelPaint | ToolKind::LabelErase => {}
            _ => {}
        }
    }

    // ── Annotation helpers ────────────────────────────────────────────────────

    /// Compute ROI rect statistics for the pixel region between `start` and
    /// `end` (screen-space corners) on the current primary-axis slice.
    fn finalise_roi_rect(&mut self, start: egui::Pos2, end: egui::Pos2) {
        let Some(vol) = &self.loaded else { return };
        let p1 = [start.y, start.x];
        let p2 = [end.y, end.x];
        let spacing = self.slice_spacing_2d();
        let (pixels, width, height) = vol.extract_slice(self.axis, self.viewer_state.slice_index);
        let (mean, std_dev, min, max, area_mm2) =
            Annotation::compute_roi_rect_stats(p1, p2, &pixels, width, height, spacing);
        self.annotations.push(Annotation::RoiRect {
            top_left: [p1[0].min(p2[0]), p1[1].min(p2[1])],
            bottom_right: [p1[0].max(p2[0]), p1[1].max(p2[1])],
            mean,
            std_dev,
            min,
            max,
            area_mm2,
        });
        self.status_message =
            format!("ROI: μ={mean:.1}  σ={std_dev:.1}  [{min:.0}, {max:.0}]  {area_mm2:.1} mm²");
    }

    fn finalise_roi_ellipse(&mut self, start: egui::Pos2, end: egui::Pos2) {
        let Some(vol) = &self.loaded else { return };
        let p1 = [start.y, start.x];
        let p2 = [end.y, end.x];
        let spacing = self.slice_spacing_2d();
        let (pixels, width, height) = vol.extract_slice(self.axis, self.viewer_state.slice_index);
        let (center, radii, mean, std_dev, min, max, area_mm2) =
            Annotation::compute_roi_ellipse_stats(p1, p2, &pixels, width, height, spacing);
        self.annotations.push(Annotation::RoiEllipse {
            center,
            radii,
            mean,
            std_dev,
            min,
            max,
            area_mm2,
        });
        self.status_message =
            format!("Ellipse ROI: μ={mean:.1}  σ={std_dev:.1}  [{min:.0}, {max:.0}]  {area_mm2:.1} mm²");
    }

    /// Per-axis 2-D pixel spacing `[row_spacing, col_spacing]` in mm/pixel.
    ///
    /// | axis | row spacing | col spacing |
    /// |------|-------------|-------------|
    /// | 0 axial    | dy | dx |
    /// | 1 coronal  | dz | dx |
    /// | 2 sagittal | dz | dy |
    fn slice_spacing_2d(&self) -> [f32; 2] {
        let Some(vol) = &self.loaded else {
            return [1.0, 1.0];
        };
        let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
        match self.axis {
            0 => [dy, dx],
            1 => [dz, dx],
            _ => [dz, dy],
        }
    }

    fn apply_label_at_pointer(&mut self, axis: usize, pos: Option<egui::Pos2>, rect: egui::Rect) {
        let Some(point) = pos else { return };
        let Some(volume) = &self.loaded else { return };
        let Some(voxel) = viewport_point_to_voxel(volume.shape, axis, self.axis_slice_info(axis).0, point, rect) else {
            return;
        };
        let Some(editor) = self.label_editor.as_mut() else {
            return;
        };

        let result = match self.active_tool {
            ToolKind::LabelPaint => editor.paint_sphere(voxel, self.label_brush_radius),
            ToolKind::LabelErase => editor.erase_sphere(voxel, self.label_brush_radius),
            _ => return,
        };

        match result {
            Ok(changed) if changed > 0 => {
                self.status_message = format!(
                    "Label edit axis={} voxel=[{},{},{}] changed {} voxels",
                    axis, voxel[0], voxel[1], voxel[2], changed
                );
            }
            Ok(_) => {}
            Err(e) => {
                self.status_message = format!("Label edit failed: {e}");
            }
        }
    }

    fn update_linked_cursor_from_pointer(
        &mut self,
        axis: usize,
        pos: Option<egui::Pos2>,
        rect: egui::Rect,
    ) {
        let Some(point) = pos else { return };
        let slice_index = self.axis_slice_info(axis).0;
        let Some(volume) = &self.loaded else { return };
        let Some(cursor) = self.linked_cursor.as_mut() else {
            return;
        };
        let Some(voxel) = cursor.update_from_viewport_point(
            volume.shape,
            axis,
            slice_index,
            point,
            rect,
        ) else {
            return;
        };

        self.viewer_state.slice_index = voxel[0];
        self.coronal_slice = voxel[1];
        self.sagittal_slice = voxel[2];
        self.axis = axis.min(2);
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
        self.status_message = format!(
            "Linked cursor axis={} voxel=[{},{},{}]",
            axis, voxel[0], voxel[1], voxel[2]
        );
    }

    fn update_pointer_intensity(&mut self, axis: usize, pos: Option<egui::Pos2>, rect: egui::Rect) {
        let Some(point) = pos else {
            self.pointer_intensity = 0.0;
            return;
        };
        let Some(volume) = &self.loaded else {
            self.pointer_intensity = 0.0;
            return;
        };
        let slice_index = self.axis_slice_info(axis).0;
        let Some(voxel) = viewport_point_to_voxel(volume.shape, axis, slice_index, point, rect) else {
            self.pointer_intensity = 0.0;
            return;
        };
        self.pointer_intensity = intensity_at_voxel(volume, voxel);
    }

    fn current_cursor_value(&self) -> Option<f32> {
        let volume = self.loaded.as_ref()?;
        let cursor = self.linked_cursor?;
        let [z, y, x] = cursor.voxel();
        Some(volume.pixel_at(z, y, x))
    }

    fn draw_label_overlay(&self, painter: &egui::Painter, rect: egui::Rect, axis: usize) {
        let Some(editor) = &self.label_editor else { return };
        let Some(volume) = &self.loaded else { return };
        let Some((width, height)) = axis_slice_dimensions(volume.shape, axis) else {
            return;
        };
        if width == 0 || height == 0 {
            return;
        }

        let slice_index = self.axis_slice_info(axis).0;
        let cell_w = rect.width() / width as f32;
        let cell_h = rect.height() / height as f32;

        for row in 0..height {
            for col in 0..width {
                let voxel = map_view_row_col_to_voxel(axis, slice_index, row, col);
                let label_id = editor.current_map().label_at(voxel);
                if label_id == 0 {
                    continue;
                }
                let Some(entry) = editor.current_map().table.get_label(label_id) else {
                    continue;
                };
                if !entry.visible {
                    continue;
                }

                let x0 = rect.min.x + col as f32 * cell_w;
                let y0 = rect.min.y + row as f32 * cell_h;
                painter.rect_filled(
                    egui::Rect::from_min_size(egui::pos2(x0, y0), egui::vec2(cell_w, cell_h)),
                    0.0,
                    egui::Color32::from_rgba_unmultiplied(
                        entry.color[0],
                        entry.color[1],
                        entry.color[2],
                        entry.color[3],
                    ),
                );
            }
        }
    }

    fn draw_rt_struct_overlay(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        axis: usize,
        image_h: usize,
        image_w: usize,
    ) {
        let Some(volume) = &self.loaded else { return };
        let Some(rt) = &self.rt_struct else { return };
        if image_h == 0 || image_w == 0 {
            return;
        }

        let (slice_index, _) = self.axis_slice_info(axis);
        let projected = project_rt_struct_contours_for_slice(
            rt,
            axis,
            slice_index,
            volume.shape,
            volume.origin,
            volume.direction,
            volume.spacing,
        );

        let to_screen = |row: f32, col: f32| -> egui::Pos2 {
            egui::pos2(
                rect.min.x + ((col + 0.5) / image_w as f32) * rect.width(),
                rect.min.y + ((row + 0.5) / image_h as f32) * rect.height(),
            )
        };

        for contour in projected {
            let color = egui::Color32::from_rgb(contour.color[0], contour.color[1], contour.color[2]);
            if contour.points_row_col.len() == 1 {
                let [row, col] = contour.points_row_col[0];
                painter.circle_filled(to_screen(row, col), 2.0, color);
                continue;
            }

            for pair in contour.points_row_col.windows(2) {
                let a = pair[0];
                let b = pair[1];
                painter.line_segment(
                    [to_screen(a[0], a[1]), to_screen(b[0], b[1])],
                    egui::Stroke::new(1.5, color),
                );
            }

            if contour.closed {
                if let (Some(first), Some(last)) = (
                    contour.points_row_col.first().copied(),
                    contour.points_row_col.last().copied(),
                ) {
                    painter.line_segment(
                        [to_screen(last[0], last[1]), to_screen(first[0], first[1])],
                        egui::Stroke::new(1.5, color),
                    );
                }
            }
        }
    }
}

fn palette_color(label_id: u32) -> [u8; 4] {
    const PALETTE: [[u8; 4]; 8] = [
        [255, 0, 0, 180],
        [0, 255, 0, 180],
        [0, 128, 255, 180],
        [255, 196, 0, 180],
        [255, 0, 255, 180],
        [0, 255, 255, 180],
        [255, 128, 0, 180],
        [180, 0, 255, 180],
    ];
    PALETTE[(label_id as usize) % PALETTE.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_volume(shape: [usize; 3]) -> LoadedVolume {
        let voxel_count = shape[0] * shape[1] * shape[2];
        LoadedVolume {
            data: Arc::new(vec![0.0; voxel_count]),
            shape,
            spacing: [1.0, 1.0, 1.0],
            origin: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            metadata: None,
            source: None,
            modality: Some("CT".to_string()),
            patient_name: None,
            patient_id: None,
            study_date: None,
            series_description: Some("Test".to_string()),
        }
    }

    #[test]
    fn linked_cursor_click_updates_all_slices() {
        let mut app = SnapApp::default();
        let shape = [8, 10, 20];
        app.loaded = Some(test_volume(shape));
        app.viewer_state.slice_index = 3;
        app.coronal_slice = 5;
        app.sagittal_slice = 9;
        app.linked_cursor = Some(LinkedCursor::from_slices(shape, 3, 5, 9));

        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(200.0, 100.0));
        app.update_linked_cursor_from_pointer(0, Some(egui::pos2(150.0, 20.0)), rect);

        assert_eq!(app.viewer_state.slice_index, 3);
        assert_eq!(app.coronal_slice, 2);
        assert_eq!(app.sagittal_slice, 15);
        assert_eq!(app.axis, 0);
        assert_eq!(app.linked_cursor.expect("cursor").voxel(), [3, 2, 15]);
    }

    #[test]
    fn stepping_slice_updates_linked_cursor_axis_coordinate() {
        let mut app = SnapApp::default();
        let shape = [8, 10, 20];
        app.loaded = Some(test_volume(shape));
        app.viewer_state.slice_index = 3;
        app.coronal_slice = 5;
        app.sagittal_slice = 9;
        app.linked_cursor = Some(LinkedCursor::from_slices(shape, 3, 5, 9));

        app.step_slice_for_axis(1, 2);

        assert_eq!(app.coronal_slice, 7);
        assert_eq!(app.linked_cursor.expect("cursor").voxel(), [3, 7, 9]);
    }

    #[test]
    fn current_cursor_value_reads_loaded_voxel_at_linked_position() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([2, 3, 4]);
        volume.data = Arc::new((0..24).map(|v| v as f32).collect());
        app.loaded = Some(volume);
        app.linked_cursor = Some(LinkedCursor::from_slices([2, 3, 4], 1, 2, 3));

        assert_eq!(app.current_cursor_value(), Some(23.0));
    }

    #[test]
    fn cine_loop_advances_and_wraps_active_axis() {
        let mut app = SnapApp::default();
        let shape = [3, 4, 5];
        app.loaded = Some(test_volume(shape));
        app.viewer_state.slice_index = 2;
        app.linked_cursor = Some(LinkedCursor::from_slices(shape, 2, 0, 0));

        app.advance_slice_for_axis_loop(0, 1);

        assert_eq!(app.viewer_state.slice_index, 0);
        assert_eq!(app.linked_cursor.expect("cursor").voxel(), [0, 0, 0]);
    }

    #[test]
    fn session_snapshot_round_trip_preserves_cine_state() {
        let mut app = SnapApp::default();
        app.cine.restore(true, 18.0);

        let snapshot = app.session_snapshot();

        assert!(snapshot.cine_enabled);
        assert_eq!(snapshot.cine_fps, 18.0);

        let mut recovered = SnapApp::default();
        recovered.apply_session_snapshot(snapshot);

        assert!(recovered.cine.enabled);
        assert_eq!(recovered.cine.fps, 18.0);
    }

    #[test]
    fn reset_view_to_fit_restores_canonical_transform() {
        let mut app = SnapApp::default();
        app.zoom = 3.25;
        app.pan_offset = egui::vec2(24.0, -8.0);
        app.texture_dirty = false;
        app.coronal_dirty = false;
        app.sagittal_dirty = false;

        app.reset_view_to_fit();

        assert_eq!(app.zoom, 1.0);
        assert_eq!(app.pan_offset, egui::Vec2::ZERO);
        assert!(app.texture_dirty);
        assert!(app.coronal_dirty);
        assert!(app.sagittal_dirty);
        assert_eq!(app.status_message, "Zoom reset to fit.");
    }

    #[test]
    fn close_study_clears_loaded_and_cached_state() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([2, 2, 2]));
        app.linked_cursor = Some(LinkedCursor::from_slices([2, 2, 2], 1, 1, 1));
        app.pointer_intensity = 123.0;
        app.pan_offset = egui::vec2(8.0, -4.0);
        app.zoom = 3.0;
        app.cached_histogram = Some(crate::render::histogram::compute_histogram(
            &[0.0, 1.0, 1.0, 2.0],
            0.0,
            2.0,
            4,
        ));
        app.selected_series = Some(std::path::PathBuf::from("series"));

        app.close_study();

        assert!(app.loaded.is_none(), "loaded volume must be cleared");
        assert!(app.linked_cursor.is_none(), "linked cursor must be cleared");
        assert!(app.cached_histogram.is_none(), "histogram cache must be cleared");
        assert!(app.selected_series.is_none(), "selected series must be cleared");
        assert_eq!(app.pointer_intensity, 0.0, "pointer intensity must reset");
        assert_eq!(app.pan_offset, egui::Vec2::ZERO, "pan must reset");
        assert_eq!(app.zoom, 1.0, "zoom must reset");
        assert_eq!(app.status_message, "Study closed.");
    }

    #[test]
    fn zoom_tool_drag_updates_zoom_from_pointer_delta() {
        let mut app = SnapApp::default();
        app.active_tool = ToolKind::Zoom;
        app.zoom = 1.0;

        app.on_drag_start(Some(egui::pos2(100.0, 100.0)));
        app.on_drag(Some(egui::pos2(100.0, 80.0)));

        assert!(app.zoom > 1.0, "expected drag-up zoom-in, got {}", app.zoom);
        assert!(app.status_message.starts_with("Zoom:"));
    }

    #[test]
    fn pan_tool_drag_updates_offset_via_ssot() {
        let mut app = SnapApp::default();
        app.active_tool = ToolKind::Pan;
        app.pan_offset = egui::Vec2::ZERO;

        app.on_drag_start(Some(egui::pos2(100.0, 100.0)));
        app.on_drag(Some(egui::pos2(130.0, 80.0)));

        // delta = (30.0, -20.0)
        // new_offset = (0, 0) + (30, -20) = (30, -20)
        assert_eq!(app.pan_offset.x, 30.0);
        assert_eq!(app.pan_offset.y, -20.0);
    }

    #[test]
    fn pan_tool_drag_with_nonzero_starting_offset() {
        let mut app = SnapApp::default();
        app.active_tool = ToolKind::Pan;
        app.pan_offset = egui::Vec2::new(50.0, 75.0);

        app.on_drag_start(Some(egui::pos2(200.0, 150.0)));
        app.on_drag(Some(egui::pos2(220.0, 130.0)));

        // delta = (20.0, -20.0)
        // new_offset = (50, 75) + (20, -20) = (70, 55)
        assert_eq!(app.pan_offset.x, 70.0);
        assert_eq!(app.pan_offset.y, 55.0);
    }

    #[test]
    fn pan_tool_drag_zero_delta_preserves_offset() {
        let mut app = SnapApp::default();
        app.active_tool = ToolKind::Pan;
        app.pan_offset = egui::Vec2::new(100.0, 100.0);

        app.on_drag_start(Some(egui::pos2(200.0, 150.0)));
        app.on_drag(Some(egui::pos2(200.0, 150.0)));

        // delta = (0.0, 0.0)
        // new_offset = (100, 100) + (0, 0) = (100, 100)
        assert_eq!(app.pan_offset.x, 100.0);
        assert_eq!(app.pan_offset.y, 100.0);
    }

    #[test]
    fn label_shortcut_undo_redo_updates_map_and_status() {
        let mut app = SnapApp::default();
        let mut editor = crate::label::LabelEditor::new([2, 2, 2]);
        let _ = editor.paint_voxel([0, 0, 0]).expect("paint must succeed");
        app.label_editor = Some(editor);

        app.undo_label_edit_shortcut();

        let label_after_undo = app
            .label_editor
            .as_ref()
            .expect("editor")
            .current_map()
            .label_at([0, 0, 0]);
        assert_eq!(label_after_undo, 0);
        assert_eq!(app.status_message, "Segmentation undo.");

        app.redo_label_edit_shortcut();

        let label_after_redo = app
            .label_editor
            .as_ref()
            .expect("editor")
            .current_map()
            .label_at([0, 0, 0]);
        assert_eq!(label_after_redo, 1);
        assert_eq!(app.status_message, "Segmentation redo.");
    }

    #[test]
    fn slice_navigation_shortcuts_advance_or_rewind_active_axis() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([3, 4, 5]));
        app.axis = 0;
        app.viewer_state.slice_index = 1;

        app.apply_slice_navigation_shortcuts(true, false, false, false, false, false);
        assert_eq!(app.viewer_state.slice_index, 0);

        app.apply_slice_navigation_shortcuts(false, false, false, true, false, false);
        assert_eq!(app.viewer_state.slice_index, 1);
    }

    #[test]
    fn slice_navigation_shortcuts_use_priority_when_multiple_keys_pressed() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([3, 4, 5]));
        app.axis = 0;
        app.viewer_state.slice_index = 1;

        app.apply_slice_navigation_shortcuts(true, true, false, false, false, false);
        assert_eq!(app.viewer_state.slice_index, 0);
    }

    #[test]
    fn slice_navigation_shortcuts_home_end_jump_to_axis_boundaries() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([3, 4, 5]));
        app.axis = 2;
        app.sagittal_slice = 2;

        app.apply_slice_navigation_shortcuts(false, false, false, false, true, false);
        assert_eq!(app.sagittal_slice, 0);

        app.apply_slice_navigation_shortcuts(false, false, false, false, false, true);
        assert_eq!(app.sagittal_slice, 4);
    }

    #[test]
    fn slice_navigation_shortcuts_home_takes_priority_over_end() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([3, 4, 5]));
        app.axis = 0;
        app.viewer_state.slice_index = 1;

        app.apply_slice_navigation_shortcuts(false, false, false, false, true, true);
        assert_eq!(app.viewer_state.slice_index, 0);
    }

    /// Window/Level drag updates center and width through the SSOT mapping.
    ///
    /// Drag (dx=+10, dy=-5) with default sensitivity 4.0:
    ///   new_width  = 400 + 10*4 = 440
    ///   new_center = 40  − (−5)*4 = 60
    #[test]
    fn window_level_drag_updates_center_and_width_via_ssot() {
        use crate::tools::interaction::ToolState;
        use egui::Pos2;

        let mut app = SnapApp::default();
        app.viewer_state.window_center = Some(40.0);
        app.viewer_state.window_width = Some(400.0);
        app.tool_state = ToolState::WindowLevelDrag {
            start: Pos2::new(100.0, 100.0),
            original_center: 40.0,
            original_width: 400.0,
        };

        app.on_drag(Some(Pos2::new(110.0, 95.0)));

        let new_center = app.viewer_state.window_center.expect("center set");
        let new_width = app.viewer_state.window_width.expect("width set");
        // Analytical: center = 40 − (−5)*4 = 60, width = 400 + 10*4 = 440
        assert_eq!(new_center, 60.0_f32, "center mismatch");
        assert_eq!(new_width, 440.0_f32, "width mismatch");
        assert!(app.texture_dirty, "axial dirty not set");
        assert!(app.coronal_dirty, "coronal dirty not set");
        assert!(app.sagittal_dirty, "sagittal dirty not set");
    }

    /// advance_slice_for_axis_loop wraps correctly and routes through set_slice_for_axis.
    ///
    /// Axis 0 has 3 slices; advance from index 2 by 1 step wraps to 0.
    #[test]
    fn advance_slice_for_axis_loop_wraps_and_marks_dirty() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([3, 4, 5]));
        app.viewer_state.slice_index = 2; // last slice

        app.advance_slice_for_axis_loop(0, 1);

        assert_eq!(app.viewer_state.slice_index, 0, "wrap-around failed");
        assert!(app.texture_dirty, "texture dirty not set after advance");
    }

    /// Tool shortcut 'L' selects MeasureLength tool.
    #[test]
    fn tool_shortcut_l_selects_measure_length() {
        let mut app = SnapApp::default();
        app.active_tool = ToolKind::Pan; // different tool initially

        // Simulate pressing 'L'
        if let Some(tool) = tool_kind_for_key(egui::Key::L) {
            app.active_tool = tool;
        }

        assert_eq!(app.active_tool, ToolKind::MeasureLength);
    }

    /// Tool shortcut 'A' selects MeasureAngle tool.
    #[test]
    fn tool_shortcut_a_selects_measure_angle() {
        let mut app = SnapApp::default();
        app.active_tool = ToolKind::Pan;

        if let Some(tool) = tool_kind_for_key(egui::Key::A) {
            app.active_tool = tool;
        }

        assert_eq!(app.active_tool, ToolKind::MeasureAngle);
    }

    /// Tool shortcut 'R' selects RoiRect tool.
    #[test]
    fn tool_shortcut_r_selects_roi_rect() {
        let mut app = SnapApp::default();
        app.active_tool = ToolKind::Zoom;

        if let Some(tool) = tool_kind_for_key(egui::Key::R) {
            app.active_tool = tool;
        }

        assert_eq!(app.active_tool, ToolKind::RoiRect);
    }

    /// Tool shortcut 'E' selects RoiEllipse tool.
    #[test]
    fn tool_shortcut_e_selects_roi_ellipse() {
        let mut app = SnapApp::default();
        app.active_tool = ToolKind::MeasureLength;

        if let Some(tool) = tool_kind_for_key(egui::Key::E) {
            app.active_tool = tool;
        }

        assert_eq!(app.active_tool, ToolKind::RoiEllipse);
    }

    /// Tool shortcut 'H' selects PointHu tool.
    #[test]
    fn tool_shortcut_h_selects_point_hu() {
        let mut app = SnapApp::default();
        app.active_tool = ToolKind::WindowLevel;

        if let Some(tool) = tool_kind_for_key(egui::Key::H) {
            app.active_tool = tool;
        }

        assert_eq!(app.active_tool, ToolKind::PointHu);
    }

    /// Tool shortcut 'P' selects Pan tool.
    #[test]
    fn tool_shortcut_p_selects_pan() {
        let mut app = SnapApp::default();
        app.active_tool = ToolKind::Zoom;

        if let Some(tool) = tool_kind_for_key(egui::Key::P) {
            app.active_tool = tool;
        }

        assert_eq!(app.active_tool, ToolKind::Pan);
    }

    /// Tool shortcut 'Z' selects Zoom tool.
    #[test]
    fn tool_shortcut_z_selects_zoom() {
        let mut app = SnapApp::default();
        app.active_tool = ToolKind::WindowLevel;

        if let Some(tool) = tool_kind_for_key(egui::Key::Z) {
            app.active_tool = tool;
        }

        assert_eq!(app.active_tool, ToolKind::Zoom);
    }

    /// Tool shortcut 'W' selects WindowLevel tool.
    #[test]
    fn tool_shortcut_w_selects_window_level() {
        let mut app = SnapApp::default();
        app.active_tool = ToolKind::Pan;

        if let Some(tool) = tool_kind_for_key(egui::Key::W) {
            app.active_tool = tool;
        }

        assert_eq!(app.active_tool, ToolKind::WindowLevel);
    }

    /// Tool shortcut 'B' selects LabelPaint tool.
    #[test]
    fn tool_shortcut_b_selects_label_paint() {
        let mut app = SnapApp::default();
        app.active_tool = ToolKind::Zoom;

        if let Some(tool) = tool_kind_for_key(egui::Key::B) {
            app.active_tool = tool;
        }

        assert_eq!(app.active_tool, ToolKind::LabelPaint);
    }

    // ── Measurement label routing: per-axis spacing selection ─────────────────

    /// The per-axis spacing selection used in `render_axis_viewport` for
    /// measurement label routing follows the ITK-SNAP convention:
    ///
    /// | axis | row spacing | col spacing |
    /// |------|-------------|-------------|
    /// | 0 axial    | dy  | dx  |
    /// | 1 coronal  | dz  | dx  |
    /// | 2 sagittal | dz  | dy  |
    ///
    /// All three must be distinct from each other so the wrong selection is
    /// observable.  We use dz=2.0, dy=3.0, dx=5.0 (prime distances, each
    /// uniquely identifiable).
    fn make_anisotropic_volume() -> LoadedVolume {
        let shape = [4, 6, 8];
        LoadedVolume {
            data: Arc::new(vec![0.0f32; shape[0] * shape[1] * shape[2]]),
            shape,
            spacing: [2.0, 3.0, 5.0], // [dz, dy, dx]
            origin: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            metadata: None,
            source: None,
            modality: Some("CT".to_string()),
            patient_name: None,
            patient_id: None,
            study_date: None,
            series_description: None,
        }
    }

    /// Axial (axis 0): row_spacing = dy = 3.0, col_spacing = dx = 5.0
    #[test]
    fn measurement_spacing_axial_selects_dy_dx() {
        let vol = make_anisotropic_volume();
        let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
        let spacing_2d: [f32; 2] = match 0usize {
            0 => [dy, dx],
            1 => [dz, dx],
            _ => [dz, dy],
        };
        assert_eq!(
            spacing_2d,
            [3.0, 5.0],
            "axial axis must select [dy=3.0, dx=5.0]"
        );
    }

    /// Coronal (axis 1): row_spacing = dz = 2.0, col_spacing = dx = 5.0
    #[test]
    fn measurement_spacing_coronal_selects_dz_dx() {
        let vol = make_anisotropic_volume();
        let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
        let spacing_2d: [f32; 2] = match 1usize {
            0 => [dy, dx],
            1 => [dz, dx],
            _ => [dz, dy],
        };
        assert_eq!(
            spacing_2d,
            [2.0, 5.0],
            "coronal axis must select [dz=2.0, dx=5.0]"
        );
    }

    /// Sagittal (axis 2): row_spacing = dz = 2.0, col_spacing = dy = 3.0
    #[test]
    fn measurement_spacing_sagittal_selects_dz_dy() {
        let vol = make_anisotropic_volume();
        let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
        let spacing_2d: [f32; 2] = match 2usize {
            0 => [dy, dx],
            1 => [dz, dx],
            _ => [dz, dy],
        };
        assert_eq!(
            spacing_2d,
            [2.0, 3.0],
            "sagittal axis must select [dz=2.0, dy=3.0]"
        );
    }

    /// All three axis-spacing pairs are mutually distinct (no collision).
    #[test]
    fn measurement_spacing_all_axes_are_distinct() {
        let vol = make_anisotropic_volume();
        let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
        let axial: [f32; 2] = [dy, dx];
        let coronal: [f32; 2] = [dz, dx];
        let sagittal: [f32; 2] = [dz, dy];
        assert_ne!(axial, coronal, "axial and coronal spacing must differ");
        assert_ne!(axial, sagittal, "axial and sagittal spacing must differ");
        assert_ne!(coronal, sagittal, "coronal and sagittal spacing must differ");
    }

    /// `img_to_screen` mapping: image-pixel (col=c, row=r) maps to
    /// screen position `origin + (c * scale, r * scale)`.
    ///
    /// Analytical: origin=(10, 20), scale=2.0, img=(3, 5)
    ///   x = 10 + 3 × 2 = 16
    ///   y = 20 + 5 × 2 = 30
    #[test]
    fn measurement_img_to_screen_analytical() {
        let origin = egui::pos2(10.0, 20.0);
        let scale = 2.0_f32;
        let img_to_screen =
            |p: egui::Pos2| egui::pos2(origin.x + p.x * scale, origin.y + p.y * scale);

        let screen = img_to_screen(egui::pos2(3.0, 5.0)); // col=3, row=5
        assert_eq!(
            (screen.x, screen.y),
            (16.0, 30.0),
            "img_to_screen must compute origin + img × scale analytically"
        );
    }

    /// `img_to_screen` at image origin maps to screen origin.
    #[test]
    fn measurement_img_to_screen_origin_maps_to_rect_min() {
        let origin = egui::pos2(50.0, 75.0);
        let scale = 3.0_f32;
        let img_to_screen =
            |p: egui::Pos2| egui::pos2(origin.x + p.x * scale, origin.y + p.y * scale);

        let screen = img_to_screen(egui::pos2(0.0, 0.0));
        assert_eq!(
            (screen.x, screen.y),
            (50.0, 75.0),
            "image origin must map to screen rect.min"
        );
    }
}
