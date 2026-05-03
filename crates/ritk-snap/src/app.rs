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
    axis_slice_dimensions, format_lps, map_view_row_col_to_voxel, viewport_point_to_voxel,
    voxel_to_lps, CinePlayback, LinkedCursor,
};
use crate::ui::overlay::OverlayRenderer;
use crate::ui::window_presets::WindowPreset;
use crate::{LoadedVolume, ModalityDisplay, ViewerState};

/// CPU backend used for DICOM loading.
type LoadBackend = burn_ndarray::NdArray<f32>;

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
            multi_planar: false,
            show_overlay: true,
            show_crosshair: false,
            linked_cursor: None,
            cine: CinePlayback::default(),
            show_series_browser: true,
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

                    if ui.button("Export current slice as PNG…").clicked() {
                        ui.close_menu();
                        self.export_current_slice();
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
                        self.loaded = None;
                        self.texture = None;
                        self.coronal_tex = None;
                        self.sagittal_tex = None;
                        self.annotations.clear();
                        self.label_editor = None;
                        self.viewer_state = ViewerState::new();
                        self.texture_dirty = false;
                        self.coronal_dirty = false;
                        self.sagittal_dirty = false;
                        self.cine.stop();
                        self.status_message = "Study closed.".to_owned();
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

                    if ui.button("Reset view").clicked() {
                        ui.close_menu();
                        self.pan_offset = egui::Vec2::ZERO;
                        self.zoom = 1.0;
                        self.texture_dirty = true;
                        self.coronal_dirty = true;
                        self.sagittal_dirty = true;
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
            show_series_browser: self.show_series_browser,
            sidebar_tab: self.sidebar_tab,
            coronal_slice: self.coronal_slice,
            sagittal_slice: self.sagittal_slice,
            pan_offset: [self.pan_offset.x, self.pan_offset.y],
            zoom: self.zoom,
            cine_enabled: self.cine.enabled,
            cine_fps: self.cine.fps,
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
        self.show_series_browser = snapshot.show_series_browser;
        self.sidebar_tab = snapshot.sidebar_tab;
        self.coronal_slice = snapshot.coronal_slice;
        self.sagittal_slice = snapshot.sagittal_slice;
        self.pan_offset = egui::Vec2::new(snapshot.pan_offset[0], snapshot.pan_offset[1]);
        self.zoom = snapshot.zoom.clamp(0.05, 32.0);
        self.cine.restore(snapshot.cine_enabled, snapshot.cine_fps);
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
                            .add_enabled(editor.can_undo(), egui::Button::new("Undo"))
                            .clicked()
                        {
                            let _ = editor.undo();
                        }
                        if ui
                            .add_enabled(editor.can_redo(), egui::Button::new("Redo"))
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

                // ── Annotations ───────────────────────────────────────────────
                ui.separator();
                ui.heading("Annotations");
                ui.separator();

                if self.annotations.is_empty() {
                    ui.label("None.");
                } else {
                    egui::ScrollArea::vertical()
                        .max_height(200.0)
                        .show(ui, |ui| {
                            for (i, ann) in self.annotations.iter().enumerate() {
                                match ann {
                                    Annotation::Length { length_mm, .. } => {
                                        ui.label(format!("#{i} Length: {length_mm:.1} mm"));
                                    }
                                    Annotation::Angle { angle_deg, .. } => {
                                        ui.label(format!("#{i} Angle: {angle_deg:.2}°"));
                                    }
                                    Annotation::RoiRect {
                                        mean,
                                        std_dev,
                                        min,
                                        max,
                                        area_mm2,
                                        ..
                                    } => {
                                        ui.label(format!(
                                            "#{i} ROI  μ={mean:.1} σ={std_dev:.1} \
                                             [{min:.0},{max:.0}] {area_mm2:.1}mm²"
                                        ));
                                    }
                                    Annotation::HuPoint { value, pos } => {
                                        ui.label(format!(
                                            "#{i} HU ({:.0},{:.0}): {value:.0}",
                                            pos[1], pos[0]
                                        ));
                                    }
                                }
                            }
                        });

                    if ui.button("Clear all").clicked() {
                        self.annotations.clear();
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
            // Keyboard slice navigation.
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                self.step_slice(-1);
            }
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                self.step_slice(1);
            }

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
    /// 7. Handle scroll wheel for slice navigation on this axis.
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

        // painter is dropped here; no longer borrows ui.
        drop(painter);

        // ── 7. Scroll wheel: slice navigation ──────────────────────────────────
        let scroll_y = ctx.input(|i| i.smooth_scroll_delta.y);
        if response.hovered() && scroll_y != 0.0 {
            let step = if scroll_y > 0.0 { -1i32 } else { 1 };
            self.step_slice_for_axis(axis, step);
        }

        // ── 8. Pointer events ──────────────────────────────────────────────────
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
    fn export_current_slice(&self) {
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
            match image::RgbImage::from_raw(w as u32, h as u32, rgb_bytes) {
                Some(img) => {
                    if let Err(e) = img.save(&path) {
                        error!(path = %path.display(), error = %e, "PNG export failed");
                    } else {
                        info!(path = %path.display(), "slice exported as PNG");
                    }
                }
                None => {
                    error!("PNG export: buffer length mismatch — image not saved");
                }
            }
        }
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
        match serde_json::to_string_pretty(&snapshot)
            .map_err(anyhow::Error::from)
            .and_then(|json| {
                std::fs::write(&path, json)
                    .map_err(anyhow::Error::from)
                    .map(|_| ())
            }) {
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

        match std::fs::read_to_string(&path)
            .map_err(anyhow::Error::from)
            .and_then(|json| {
                serde_json::from_str::<ViewerSessionSnapshot>(&json).map_err(anyhow::Error::from)
            }) {
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
        info!("loading DICOM series from {}", path.display());
        self.cine.stop();
        let device: <LoadBackend as burn::tensor::backend::Backend>::Device = Default::default();

        match ritk_io::load_dicom_series_with_metadata::<LoadBackend, _>(&path, &device) {
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
                let data = match raw_data.as_slice::<f32>() {
                    Ok(s) => Arc::new(s.to_vec()),
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
                    source: Some(path.clone()),
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
                self.tool_state = ToolState::Idle;
                self.texture = None;
                self.texture_dirty = true;
                self.coronal_tex = None;
                self.coronal_dirty = true;
                self.sagittal_tex = None;
                self.sagittal_dirty = true;
                self.status_message = format!(
                    "Loaded {} — shape [{}, {}, {}] — protocol {}",
                    path.display(),
                    shape[0],
                    shape[1],
                    shape[2],
                    protocol.protocol_name,
                );
                info!("{}", self.status_message);
            }
            Err(e) => {
                let msg = format!("DICOM load failed for {}: {e:#}", path.display());
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
                self.tool_state = ToolState::Idle;
                self.texture = None;
                self.texture_dirty = true;
                self.coronal_tex = None;
                self.coronal_dirty = true;
                self.sagittal_tex = None;
                self.sagittal_dirty = true;
                self.status_message = msg;
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("NIfTI load failed: {e:#}");
            }
        }
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
    fn step_slice_for_axis(&mut self, axis: usize, delta: i32) {
        let total = match axis {
            0 => self.loaded.as_ref().map(|v| v.shape[0]).unwrap_or(1),
            1 => self.loaded.as_ref().map(|v| v.shape[1]).unwrap_or(1),
            _ => self.loaded.as_ref().map(|v| v.shape[2]).unwrap_or(1),
        } as i32;
        let max = (total - 1).max(0);

        match axis {
            0 => {
                let next = ((self.viewer_state.slice_index as i32) + delta).clamp(0, max) as usize;
                if next != self.viewer_state.slice_index {
                    self.viewer_state.slice_index = next;
                    self.texture_dirty = true;
                    if let (Some(vol), Some(cursor)) = (&self.loaded, self.linked_cursor.as_mut()) {
                        cursor.set_axis_slice(vol.shape, 0, next);
                    }
                }
            }
            1 => {
                let next = ((self.coronal_slice as i32) + delta).clamp(0, max) as usize;
                if next != self.coronal_slice {
                    self.coronal_slice = next;
                    self.coronal_dirty = true;
                    if let (Some(vol), Some(cursor)) = (&self.loaded, self.linked_cursor.as_mut()) {
                        cursor.set_axis_slice(vol.shape, 1, next);
                    }
                }
            }
            _ => {
                let next = ((self.sagittal_slice as i32) + delta).clamp(0, max) as usize;
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

    /// Step the primary-axis slice by `delta`.  Delegates to
    /// [`step_slice_for_axis`] using `self.axis`.
    ///
    /// [`step_slice_for_axis`]: SnapApp::step_slice_for_axis
    fn step_slice(&mut self, delta: i32) {
        self.step_slice_for_axis(self.axis, delta);
    }

    /// Advance `axis` by `steps` with wrap-around.
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

        let next = |current: usize| (current + steps as usize) % total;

        match axis {
            0 => {
                self.viewer_state.slice_index = next(self.viewer_state.slice_index);
                self.texture_dirty = true;
                if let (Some(vol), Some(cursor)) = (&self.loaded, self.linked_cursor.as_mut()) {
                    cursor.set_axis_slice(vol.shape, 0, self.viewer_state.slice_index);
                }
            }
            1 => {
                self.coronal_slice = next(self.coronal_slice);
                self.coronal_dirty = true;
                if let (Some(vol), Some(cursor)) = (&self.loaded, self.linked_cursor.as_mut()) {
                    cursor.set_axis_slice(vol.shape, 1, self.coronal_slice);
                }
            }
            _ => {
                self.sagittal_slice = next(self.sagittal_slice);
                self.sagittal_dirty = true;
                if let (Some(vol), Some(cursor)) = (&self.loaded, self.linked_cursor.as_mut()) {
                    cursor.set_axis_slice(vol.shape, 2, self.sagittal_slice);
                }
            }
        }
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
                let delta = pos - start;
                self.pan_offset =
                    egui::Vec2::new(viewport_origin.x + delta.x, viewport_origin.y + delta.y);
            }
            ToolState::WindowLevelDrag {
                start,
                original_center,
                original_width,
            } => {
                // Horizontal drag → window width (4 HU per screen pixel).
                // Vertical drag   → window centre (4 HU per screen pixel, inverted y).
                let sensitivity = 4.0_f64;
                let new_width = (original_width + (pos.x - start.x) as f64 * sensitivity).max(1.0);
                let new_center = original_center - (pos.y - start.y) as f64 * sensitivity;
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
                    // Ellipse mask stats deferred to a [minor] enhancement;
                    // use rect stats as a conservative approximation.
                    self.finalise_roi_rect(start, current);
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
}
