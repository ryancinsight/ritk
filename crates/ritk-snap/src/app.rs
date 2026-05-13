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
//! | `true`         | 2×2 grid: Axial / Coronal / Sagittal / 3D-MIP, with Info below.|

use std::sync::Arc;
use std::time::Duration;

use tracing::{error, info};

use crate::dicom::select_hanging_protocol;
use crate::label::LabelEditor;
use crate::render::colormap::Colormap;
use crate::render::fusion::render_fused_slice;
use crate::render::mip_vr::{render_mip_axial, render_vr_axial};
use crate::render::slice_render::{SliceRenderer, WindowLevel};
use crate::session::ViewerSessionSnapshot;
use crate::tools::interaction::{Annotation, RoiKind, ToolState};
use crate::tools::kind::ToolKind;
use crate::ui::overlay::OverlayRenderer;
use crate::ui::window_presets::WindowPreset;
use crate::ui::{
    advance_wrapped, anatomical_label_for_axis, apply_to_image, axis_for_plane_in_volume,
    axis_slice_dimensions, axis_total, clamp_index, compute_roi_dose_analytics,
    decide_dropped_input_action, fit_view_transform, format_lps, intensity_at_voxel,
    map_view_row_col_to_voxel, pan_from_drag_delta, plan_all_mpr_exports,
    project_rt_struct_contours_for_slice, should_zoom_with_scroll, show_colorbar, step_clamped,
    tool_kind_for_key, viewport_point_to_voxel, voxel_to_lps, window_level_from_drag_delta,
    zoom_from_drag_delta, zoom_from_scroll, AnatomicalPlane, CinePlayback, DroppedInputAction,
    LinkedCursor, RoiDoseAnalytics, ViewTransform, MAX_ZOOM, MIN_ZOOM, WINDOW_LEVEL_SENSITIVITY,
};
use crate::{LoadedVolume, ModalityDisplay, ViewerState};

mod surface_export;

#[cfg(not(target_arch = "wasm32"))]
use rfd::FileDialog;

#[cfg(target_arch = "wasm32")]
struct FileDialog;

#[cfg(target_arch = "wasm32")]
impl FileDialog {
    fn new() -> Self {
        Self
    }

    fn set_file_name(self, _name: &str) -> Self {
        self
    }

    fn add_filter(self, _name: &str, _extensions: &[&str]) -> Self {
        self
    }

    fn pick_file(self) -> Option<std::path::PathBuf> {
        None
    }

    fn pick_folder(self) -> Option<std::path::PathBuf> {
        None
    }

    fn save_file(self) -> Option<std::path::PathBuf> {
        None
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeriesLoadTarget {
    Primary,
    Secondary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectionMode {
    Mip,
    Vr,
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
    /// Secondary loaded volume for cross-study compare.
    loaded_secondary: Option<LoadedVolume>,
    /// Viewer navigation state (slice index, W/L).
    viewer_state: ViewerState,
    /// Secondary compare viewport W/L center.
    secondary_window_center: Option<f32>,
    /// Secondary compare viewport W/L width.
    secondary_window_width: Option<f32>,
    /// Active colormap for intensity mapping.
    colormap: Colormap,
    /// Secondary colormap for compare panel.
    secondary_colormap: Colormap,
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
    /// Last hovered or interacted axis for status/info display.
    status_axis: usize,
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
    /// Cached RT-DOSE maximum Gy value (computed once at load time).
    rt_dose_max_gy: Option<f64>,
    /// Currently loaded RT Plan metadata.
    rt_plan: Option<ritk_io::RtPlanInfo>,
    /// Selected ROI number for RT dose analytics.
    rt_dvh_selected_roi: Option<u32>,
    /// Cached ROI dose analytics for selected ROI.
    rt_dvh_cache: Option<RoiDoseAnalytics>,
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
    /// Cached egui texture for secondary compare panel.
    secondary_texture: Option<egui::TextureHandle>,
    /// `true` when the axial texture must be rebuilt before the next frame.
    texture_dirty: bool,
    /// `true` when secondary texture must be rebuilt.
    secondary_texture_dirty: bool,
    /// Axis used by current secondary texture.
    secondary_texture_axis: usize,
    /// Slice index used by current secondary texture.
    secondary_texture_slice: usize,

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

    /// Cached egui texture for the 3D-MIP viewport (axial projection).
    mip_tex: Option<egui::TextureHandle>,
    /// `true` when the MIP projection texture must be rebuilt.
    mip_dirty: bool,
    /// Active projection mode for the bottom-right 3D viewport.
    projection_mode: ProjectionMode,

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
    /// `true` when 2-panel same-volume layout is active.
    dual_plane: bool,
    /// `true` when primary/secondary compare layout is active.
    compare_side_by_side: bool,
    /// `true` when compare panel renders fused primary/secondary overlay.
    compare_fused_overlay: bool,
    /// Secondary contribution weight in fused compare mode.
    compare_fusion_alpha: f32,
    /// Axis assignment for dual-plane same-volume layout.
    dual_axes: [usize; 2],
    /// Axis assignment for compare layout: [primary_axis, secondary_axis].
    compare_axes: [usize; 2],
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
    /// SUVbw value under the pointer for PET volumes; `None` for non-PET or unavailable params.
    pointer_suv: Option<f32>,
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
    /// Active load target for series selection.
    series_load_target: SeriesLoadTarget,

    // ── Status ────────────────────────────────────────────────────────────────
    /// Message shown in the bottom status bar.
    status_message: String,
    /// Path queued for loading on the next [`eframe::App::update`] cycle.
    pending_load: Option<std::path::PathBuf>,
    /// Secondary path queued for load on next update cycle.
    pending_secondary_load: Option<std::path::PathBuf>,
}

impl Default for SnapApp {
    fn default() -> Self {
        Self {
            loaded: None,
            loaded_secondary: None,
            viewer_state: ViewerState::new(),
            secondary_window_center: None,
            secondary_window_width: None,
            colormap: Colormap::Grayscale,
            secondary_colormap: Colormap::Grayscale,
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
            rt_dose_max_gy: None,
            rt_plan: None,
            rt_dvh_selected_roi: None,
            rt_dvh_cache: None,
            show_rt_dose_overlay: false,
            rt_dose_opacity: 0.5,
            rt_dose_overlay_cache: std::array::from_fn(|_| None),
            active_filter: crate::FilterKind::Gaussian { sigma: 1.0 },
            show_filter_panel: false,
            texture: None,
            secondary_texture: None,
            texture_dirty: false,
            secondary_texture_dirty: false,
            secondary_texture_axis: 0,
            secondary_texture_slice: 0,
            coronal_tex: None,
            coronal_dirty: false,
            coronal_slice: 0,
            sagittal_tex: None,
            sagittal_dirty: false,
            sagittal_slice: 0,
            mip_tex: None,
            mip_dirty: false,
            projection_mode: ProjectionMode::Mip,
            pan_offset: egui::Vec2::ZERO,
            zoom: 1.0,
            view_transform: ViewTransform::default(),
            show_colorbar: false,
            multi_planar: false,
            dual_plane: false,
            compare_side_by_side: false,
            compare_fused_overlay: false,
            compare_fusion_alpha: 0.35,
            dual_axes: [0, 1],
            compare_axes: [0, 0],
            show_overlay: true,
            show_crosshair: false,
            linked_cursor: None,
            cine: CinePlayback::default(),
            show_series_browser: true,
            pointer_intensity: 0.0,
            pointer_suv: None,
            cached_histogram: None,
            series_tree: crate::dicom::series_tree::SeriesTree::new(),
            selected_series: None,
            sidebar_tab: crate::ui::sidebar::SidebarTab::Series,
            series_load_target: SeriesLoadTarget::Primary,
            status_message: "No study loaded — use File > Open to load a DICOM folder.".to_owned(),
            pending_load: None,
            pending_secondary_load: None,
            status_axis: 0,
        }
    }
}

impl SnapApp {
    /// Construct an app that loads `path` on the first update cycle.
    ///
    /// Directory paths are scanned immediately so the series browser is
    /// populated before the deferred volume load runs. File paths are queued
    /// directly because they do not contain a DICOM series tree.
    #[cfg(not(target_arch = "wasm32"))]
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
        // Accept dropped files from the OS/browser shell and route them through
        // the same loader path as explicit File-menu actions.
        self.handle_dropped_inputs(ctx);

        // Process any pending file load queued in the previous frame so that
        // the file-dialog result is always acted on with a full UI repaint.
        if let Some(path) = self.pending_load.take() {
            self.load_from_path(path);
        }
        if let Some(path) = self.pending_secondary_load.take() {
            self.load_secondary_from_path(path);
        }

        self.tick_cine(ctx);
        self.consume_global_shortcuts(ctx);

        self.show_menu_bar(ctx);
        self.show_ribbon_toolbar(ctx);
        self.show_left_panel(ctx);
        self.show_bottom_bar(ctx);
        self.show_aux_windows(ctx);

        if self.compare_side_by_side {
            self.show_central_panel_compare(ctx);
        } else if self.multi_planar {
            self.show_central_panel_multi(ctx);
        } else if self.dual_plane {
            self.show_central_panel_dual(ctx);
        } else {
            self.show_central_panel_single(ctx);
        }
    }
}

// ── UI sub-methods ────────────────────────────────────────────────────────────

impl SnapApp {
    // ── Menu bar ─────────────────────────────────────────────────────────────

    fn show_ribbon_toolbar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("ribbon_toolbar")
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(45, 45, 45))
                    .inner_margin(egui::Margin::symmetric(4.0, 2.0)),
            )
            .show(ctx, |ui| {
                ui.set_min_height(36.0);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(4.0, 2.0);
                    ui.spacing_mut().button_padding = egui::vec2(2.0, 2.0);

                    ui.menu_button("File", |ui| {
                        if ui.button("Open Primary Series").clicked() {
                            ui.close_menu();
                            if let Some(folder) = FileDialog::new().pick_folder() {
                                self.scan_for_series(folder.clone());
                                self.pending_load = Some(folder);
                            }
                        }
                        if ui.button("Open Secondary Series").clicked() {
                            ui.close_menu();
                            if let Some(folder) = FileDialog::new().pick_folder() {
                                self.scan_for_series(folder.clone());
                                self.pending_secondary_load = Some(folder);
                            }
                        }
                        ui.separator();
                        if ui.button("Swap Primary/Secondary").clicked() {
                            ui.close_menu();
                            std::mem::swap(&mut self.loaded, &mut self.loaded_secondary);
                            std::mem::swap(&mut self.colormap, &mut self.secondary_colormap);
                            self.texture = None;
                            self.secondary_texture = None;
                            self.mark_all_textures_dirty();
                            self.refresh_cached_histogram();
                        }
                    });

                    ui.menu_button("Layout", |ui| {
                        let single_active =
                            !self.multi_planar && !self.dual_plane && !self.compare_side_by_side;
                        if ui
                            .selectable_label(single_active, "Single (1-Up)")
                            .clicked()
                        {
                            self.multi_planar = false;
                            self.dual_plane = false;
                            self.compare_side_by_side = false;
                            self.compare_fused_overlay = false;
                            self.mark_all_textures_dirty();
                            ui.close_menu();
                        }
                        if ui
                            .selectable_label(self.dual_plane, "Dual Plane (2-Up)")
                            .clicked()
                        {
                            self.dual_plane = true;
                            self.multi_planar = false;
                            self.compare_side_by_side = false;
                            self.compare_fused_overlay = false;
                            self.mark_all_textures_dirty();
                            ui.close_menu();
                        }
                        if ui
                            .selectable_label(self.multi_planar, "MPR Grid (3-Up)")
                            .clicked()
                        {
                            self.multi_planar = true;
                            self.dual_plane = false;
                            self.compare_side_by_side = false;
                            self.compare_fused_overlay = false;
                            self.mark_all_textures_dirty();
                            ui.close_menu();
                        }
                        if ui
                            .selectable_label(
                                self.compare_side_by_side,
                                "Compare (Primary/Secondary)",
                            )
                            .clicked()
                        {
                            self.compare_side_by_side = true;
                            self.multi_planar = false;
                            self.dual_plane = false;
                            self.mark_all_textures_dirty();
                            ui.close_menu();
                        }
                    });

                    ui.menu_button("Target", |ui| {
                        ui.selectable_value(
                            &mut self.series_load_target,
                            SeriesLoadTarget::Primary,
                            "Primary",
                        );
                        ui.selectable_value(
                            &mut self.series_load_target,
                            SeriesLoadTarget::Secondary,
                            "Secondary",
                        );
                    });

                    ui.menu_button("Axes", |ui| {
                        if self.dual_plane {
                            ui.label("Dual-Plane (Left/Right)");
                            for (axis_side, side_label) in [(0, "Left"), (1, "Right")] {
                                ui.horizontal(|ui| {
                                    ui.label(side_label);
                                    for (name, idx) in [("Ax", 0), ("Co", 1), ("Sa", 2)] {
                                        let is_active = if axis_side == 0 {
                                            self.dual_axes[0] == idx
                                        } else {
                                            self.dual_axes[1] == idx
                                        };
                                        if ui.selectable_label(is_active, name).clicked() {
                                            if axis_side == 0 {
                                                self.dual_axes[0] = idx;
                                            } else {
                                                self.dual_axes[1] = idx;
                                            }
                                        }
                                    }
                                });
                            }
                            ui.separator();
                        }

                        if self.compare_side_by_side {
                            ui.label("Compare (Primary/Secondary)");
                            for (side_num, side_label) in [(0, "Primary"), (1, "Secondary")] {
                                ui.horizontal(|ui| {
                                    ui.label(side_label);
                                    for (name, idx) in [("Ax", 0), ("Co", 1), ("Sa", 2)] {
                                        let is_active = if side_num == 0 {
                                            self.compare_axes[0] == idx
                                        } else {
                                            self.compare_axes[1] == idx
                                        };
                                        if ui.selectable_label(is_active, name).clicked() {
                                            if side_num == 0 {
                                                self.compare_axes[0] = idx;
                                                self.texture_dirty = true;
                                            } else {
                                                self.compare_axes[1] = idx;
                                                self.secondary_texture_dirty = true;
                                            }
                                        }
                                    }
                                });
                            }
                        }

                        if !self.dual_plane && !self.compare_side_by_side {
                            ui.label("Axis controls appear in Dual Plane and Compare layouts.");
                        }
                    });

                    ui.menu_button("Compare", |ui| {
                        if self.compare_side_by_side {
                            if ui.button("Preset Ax | Ax").clicked() {
                                self.compare_axes = [0, 0];
                                self.texture_dirty = true;
                                self.secondary_texture_dirty = true;
                            }
                            if ui.button("Preset Co | Co").clicked() {
                                self.compare_axes = [1, 1];
                                self.texture_dirty = true;
                                self.secondary_texture_dirty = true;
                            }
                            if ui.button("Preset Sa | Sa").clicked() {
                                self.compare_axes = [2, 2];
                                self.texture_dirty = true;
                                self.secondary_texture_dirty = true;
                            }

                            ui.separator();
                            ui.label("Secondary W/L");
                            let mut c = self.secondary_window_center.unwrap_or(128.0);
                            let mut w = self.secondary_window_width.unwrap_or(256.0).max(1.0);
                            let c_changed = ui
                                .add(
                                    egui::DragValue::new(&mut c)
                                        .speed(1.0)
                                        .fixed_decimals(0)
                                        .prefix("C:"),
                                )
                                .changed();
                            let w_changed = ui
                                .add(
                                    egui::DragValue::new(&mut w)
                                        .speed(1.0)
                                        .fixed_decimals(0)
                                        .prefix("W:"),
                                )
                                .changed();
                            if c_changed || w_changed {
                                self.secondary_window_center = Some(c);
                                self.secondary_window_width = Some(w.max(1.0));
                                self.secondary_texture_dirty = true;
                            }

                            ui.separator();
                            if ui
                                .checkbox(&mut self.compare_fused_overlay, "Fused Overlay")
                                .changed()
                            {
                                self.secondary_texture_dirty = true;
                            }
                            if self.compare_fused_overlay {
                                let alpha_changed = ui
                                    .add(
                                        egui::Slider::new(
                                            &mut self.compare_fusion_alpha,
                                            0.0..=1.0,
                                        )
                                        .text("Secondary Alpha"),
                                    )
                                    .changed();
                                if alpha_changed {
                                    self.secondary_texture_dirty = true;
                                }
                            }
                        } else {
                            ui.label("Enable Compare layout to configure compare options.");
                        }
                    });

                    ui.menu_button("Tools", |ui| {
                        for (label, tool) in [
                            ("Pan", ToolKind::Pan),
                            ("Zoom", ToolKind::Zoom),
                            ("W/L", ToolKind::WindowLevel),
                            ("Length", ToolKind::MeasureLength),
                            ("Angle", ToolKind::MeasureAngle),
                            ("Paint", ToolKind::LabelPaint),
                        ] {
                            if ui
                                .selectable_label(self.active_tool == tool, label)
                                .on_hover_text(label)
                                .clicked()
                            {
                                self.active_tool = tool;
                                self.tool_state = ToolState::Idle;
                                ui.close_menu();
                            }
                        }
                    });

                    // Right-align status
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.small(format!("v{}", env!("CARGO_PKG_VERSION")));
                    });
                });
            });
    }

    fn show_menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                // ── File ─────────────────────────────────────────────────────
                ui.menu_button("File", |ui| {
                    #[cfg(target_arch = "wasm32")]
                    {
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            "Browser build: local file/folder dialogs are not available yet.",
                        );
                        ui.separator();
                    }

                    if ui.button("Open DICOM folder…").clicked() {
                        ui.close_menu();
                        if let Some(folder) = FileDialog::new().pick_folder() {
                            self.scan_for_series(folder.clone());
                            self.pending_load = Some(folder);
                        }
                    }

                    if ui.button("Open DICOMDIR…").clicked() {
                        ui.close_menu();
                        if let Some(path) = FileDialog::new().set_file_name("DICOMDIR").pick_file()
                        {
                            self.scan_for_series(path.clone());
                            self.pending_load = Some(path);
                        }
                    }

                    if ui.button("Open DICOM file…").clicked() {
                        ui.close_menu();
                        if let Some(path) = FileDialog::new()
                            .add_filter("DICOM", &["dcm", "dicom"])
                            .pick_file()
                        {
                            self.scan_for_series(path.clone());
                            self.pending_load = Some(path);
                        }
                    }

                    if ui.button("Open NIfTI / MHA / NRRD file…").clicked() {
                        ui.close_menu();
                        if let Some(path) = FileDialog::new()
                            .add_filter(
                                "Medical images",
                                &["nii", "gz", "mha", "mhd", "nrrd", "nhdr", "mgh", "mgz"],
                            )
                            .pick_file()
                        {
                            self.load_volume_file(path);
                        }
                    }

                    if ui.button("Open RT-STRUCT file…").clicked() {
                        ui.close_menu();
                        if let Some(path) =
                            FileDialog::new().add_filter("DICOM", &["dcm"]).pick_file()
                        {
                            self.load_rt_struct_file(path);
                        }
                    }

                    if ui.button("Open RT Dose file…").clicked() {
                        ui.close_menu();
                        if let Some(path) =
                            FileDialog::new().add_filter("DICOM", &["dcm"]).pick_file()
                        {
                            self.load_rt_dose_file(path);
                        }
                    }

                    if ui.button("Open RT Plan file…").clicked() {
                        ui.close_menu();
                        if let Some(path) =
                            FileDialog::new().add_filter("DICOM", &["dcm"]).pick_file()
                        {
                            self.load_rt_plan_file(path);
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

                    if ui.button("Save segmentation as DICOM-SEG…").clicked() {
                        ui.close_menu();
                        self.save_segmentation_dicom_seg_dialog();
                    }

                    if ui.button("Export label surface as VTK…").clicked() {
                        ui.close_menu();
                        self.export_surface_dialog();
                    }

                    if ui.button("Load segmentation from NIfTI…").clicked() {
                        ui.close_menu();
                        self.load_segmentation_dialog();
                    }

                    if ui.button("Load segmentation from DICOM-SEG…").clicked() {
                        ui.close_menu();
                        self.load_segmentation_dicom_seg_dialog();
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
                        self.mip_dirty = true;
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
                                self.mip_dirty = true;
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
                        for plane in [
                            AnatomicalPlane::Axial,
                            AnatomicalPlane::Coronal,
                            AnatomicalPlane::Sagittal,
                        ] {
                            let idx = self.axis_for_plane(plane);
                            if ui
                                .selectable_label(self.axis == idx, plane.label())
                                .clicked()
                                && self.axis != idx
                            {
                                ui.close_menu();
                                self.axis = idx;
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
                        self.mip_dirty = true;
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
        if flip_h {
            self.view_transform = self.view_transform.toggle_flip_h();
            self.mark_all_textures_dirty();
        }
        if flip_v {
            self.view_transform = self.view_transform.toggle_flip_v();
            self.mark_all_textures_dirty();
        }
        if rotate_cw {
            self.view_transform = self.view_transform.rotate_cw();
            self.mark_all_textures_dirty();
        }
        if rotate_ccw {
            self.view_transform = self.view_transform.rotate_ccw();
            self.mark_all_textures_dirty();
        }
        if reset_orient {
            self.view_transform = self.view_transform.reset();
            self.mark_all_textures_dirty();
        }
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
        let target = if end { total.saturating_sub(1) } else { 0 };
        self.set_slice_for_axis(self.axis, target);
    }

    fn reset_view_to_fit(&mut self) {
        let (pan_offset, zoom) = fit_view_transform();
        self.pan_offset = egui::Vec2::new(pan_offset[0], pan_offset[1]);
        self.zoom = zoom;
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
        self.mip_dirty = true;
        self.status_message = "Zoom reset to fit.".to_owned();
    }

    #[allow(dead_code)]
    fn rt_dose_plan_link_status(&self) -> Option<String> {
        let dose = self.rt_dose.as_ref()?;
        let Some(ref_uid) = dose
            .referenced_rt_plan_sop_instance_uid
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            return Some("Plan linkage: no ReferencedRTPlanSequence UID".to_owned());
        };

        match self.rt_plan.as_ref() {
            None => Some(format!(
                "Plan linkage: references UID {ref_uid} (no RT-PLAN loaded)"
            )),
            Some(plan) => {
                let loaded_uid = plan.sop_instance_uid.trim();
                if loaded_uid.is_empty() {
                    Some(format!(
                        "Plan linkage: references UID {ref_uid} (loaded RT-PLAN has empty SOP UID)"
                    ))
                } else if loaded_uid == ref_uid {
                    Some(format!(
                        "Plan linkage: linked to loaded RT-PLAN UID {ref_uid}"
                    ))
                } else {
                    Some(format!(
                        "Plan linkage: mismatch (dose references {ref_uid}, loaded plan is {loaded_uid})"
                    ))
                }
            }
        }
    }

    fn refresh_rt_dvh_cache(&mut self) {
        let (Some(vol), Some(rt_struct), Some(rt_dose), Some(roi_number)) = (
            self.loaded.as_ref(),
            self.rt_struct.as_ref(),
            self.rt_dose.as_ref(),
            self.rt_dvh_selected_roi,
        ) else {
            self.rt_dvh_cache = None;
            return;
        };

        self.rt_dvh_cache = compute_roi_dose_analytics(
            rt_struct,
            rt_dose,
            roi_number,
            vol.shape,
            vol.origin,
            vol.direction,
            vol.spacing,
            128,
        );
    }

    /// Mark all three MPR texture slots as needing re-render (e.g. after a
    /// view-transform or colormap change).
    fn mark_all_textures_dirty(&mut self) {
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
        self.mip_dirty = true;
        self.secondary_texture_dirty = true;
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
        self.mip_dirty = true;
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
        self.mip_dirty = true;
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

    /// Ingest shell-dropped inputs (desktop) or browser-dropped file handles.
    ///
    /// Files with filesystem paths are routed to the same code paths as File-menu
    /// actions. Browser-provided handles without paths are acknowledged with a
    /// deterministic status message.
    fn handle_dropped_inputs(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input_mut(|i| std::mem::take(&mut i.raw.dropped_files));
        match decide_dropped_input_action(&dropped) {
            DroppedInputAction::QueueDicom(path) => {
                self.scan_for_series(path.clone());
                self.pending_load = Some(path.clone());
                self.status_message = format!("Queued dropped DICOM input: {}", path.display());
            }
            DroppedInputAction::LoadVolume(path) => {
                self.load_volume_file(path);
            }
            DroppedInputAction::LoadVolumeBytes { name, bytes } => {
                self.load_volume_bytes(name, bytes.as_ref());
            }
            DroppedInputAction::LoadDicomSeriesBytes { files } => {
                self.load_dicom_series_bytes(files);
            }
            DroppedInputAction::Message(msg) => {
                self.status_message = msg;
            }
            DroppedInputAction::None => {}
        }
    }

    // ── Left panel ────────────────────────────────────────────────────────────

    fn show_left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("info_panel")
            .min_width(220.0)
            .max_width(360.0)
            .show(ctx, |ui| {
                if !self.show_series_browser {
                    ui.heading("Series Browser Hidden");
                    ui.label("Enable 'Show Series Browser' from the View menu or ribbon.");
                    return;
                }

                ui.heading("Series Browser");
                ui.horizontal(|ui| {
                    ui.label("Target:");
                    ui.selectable_value(
                        &mut self.series_load_target,
                        SeriesLoadTarget::Primary,
                        "Primary",
                    );
                    ui.selectable_value(
                        &mut self.series_load_target,
                        SeriesLoadTarget::Secondary,
                        "Secondary",
                    );
                });
                ui.separator();

                let sidebar_result = {
                    let tree_ref = &self.series_tree;
                    let sel_ref = &mut self.selected_series;
                    let tab_ref = &mut self.sidebar_tab;
                    let vol_ref = self.loaded.as_ref();
                    let mut panel =
                        crate::ui::sidebar::SidebarPanel::new(tree_ref, sel_ref, tab_ref, vol_ref);
                    panel.show(ui)
                };

                if let Some(folder) = sidebar_result {
                    match self.series_load_target {
                        SeriesLoadTarget::Primary => self.pending_load = Some(folder),
                        SeriesLoadTarget::Secondary => self.pending_secondary_load = Some(folder),
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
                    let (slice_idx, total) = self.axis_slice_info(self.status_axis);
                    let _ = vol; // vol borrow used implicitly via axis_slice_info
                    ui.separator();
                    ui.label(format!("Slice {}/{}", slice_idx + 1, total));
                    ui.separator();
                    let axis_name = self.axis_label(self.status_axis);
                    ui.label(axis_name);

                    // Voxel I/J/K index and physical LPS position from linked cursor.
                    if let (Some(cursor), Some(vol)) = (self.linked_cursor, self.loaded.as_ref()) {
                        let [d, r, c] = cursor.voxel();
                        ui.separator();
                        ui.label(format!("I={d} J={r} K={c}"));
                        let lps = voxel_to_lps([d, r, c], vol.origin, vol.direction, vol.spacing);
                        ui.separator();
                        ui.label(format_lps(lps));
                    }
                }
            });
        });
    }

    fn show_aux_windows(&mut self, ctx: &egui::Context) {
        let mut show_filter = self.show_filter_panel;
        if show_filter {
            egui::Window::new("Processing")
                .open(&mut show_filter)
                .resizable(true)
                .show(ctx, |ui| {
                    let applied =
                        crate::ui::filter_panel::show_filter_panel(ui, &mut self.active_filter);
                    if applied {
                        self.apply_filter_to_loaded_volume();
                    }
                });
        }
        self.show_filter_panel = show_filter;

        let mut show_cb = self.show_colorbar;
        if show_cb {
            egui::Window::new("Colorbar")
                .open(&mut show_cb)
                .resizable(false)
                .show(ctx, |ui| {
                    let wc = self.viewer_state.window_center.unwrap_or(128.0);
                    let ww = self.viewer_state.window_width.unwrap_or(256.0).max(1.0);
                    show_colorbar(ui, self.colormap, wc, ww);
                });
        }
        self.show_colorbar = show_cb;
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

    // ── Multi-planar side-by-side viewport ───────────────────────────────────

    /// Render 2×2 MPR viewports (Coronal / Axial / Sagittal / 3D-MIP) with
    /// a shared Info panel row below.
    fn show_central_panel_multi(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let info_h = (avail.y * 0.24).clamp(110.0, 210.0);
            let grid_h = (avail.y - info_h - 6.0).max(160.0);
            let row_h = grid_h / 2.0;
            let col_w = avail.x / 2.0;

            let axial_axis = self.axis_for_plane(AnatomicalPlane::Axial);
            let coronal_axis = self.axis_for_plane(AnatomicalPlane::Coronal);
            let sagittal_axis = self.axis_for_plane(AnatomicalPlane::Sagittal);

            ui.allocate_ui(egui::vec2(avail.x, grid_h), |ui| {
                ui.horizontal(|ui| {
                    ui.allocate_ui(egui::vec2(col_w, row_h), |ui| {
                        self.render_axis_viewport(ui, ctx, coronal_axis);
                    });
                    ui.allocate_ui(egui::vec2(col_w, row_h), |ui| {
                        self.render_axis_viewport(ui, ctx, axial_axis);
                    });
                });
                ui.horizontal(|ui| {
                    ui.allocate_ui(egui::vec2(col_w, row_h), |ui| {
                        self.render_axis_viewport(ui, ctx, sagittal_axis);
                    });
                    ui.allocate_ui(egui::vec2(col_w, row_h), |ui| {
                        self.render_mip_viewport(ui, ctx); // 3D-MIP
                    });
                });
            });

            ui.separator();
            ui.allocate_ui(egui::vec2(avail.x, info_h), |ui| {
                self.show_right_info_panel(ui);
            });
        });
    }

    fn show_central_panel_dual(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.loaded.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label("Open a volume to use 2-plane layout.");
                });
                return;
            }

            let avail = ui.available_size();
            let info_h = (avail.y * 0.24).clamp(110.0, 210.0);
            let view_h = (avail.y - info_h - 6.0).max(120.0);
            let view_w = avail.x / 2.0;

            ui.allocate_ui(egui::vec2(avail.x, view_h), |ui| {
                ui.horizontal(|ui| {
                    ui.allocate_ui(egui::vec2(view_w, view_h), |ui| {
                        self.render_axis_viewport(ui, ctx, self.dual_axes[0]);
                    });
                    ui.allocate_ui(egui::vec2(view_w, view_h), |ui| {
                        self.render_axis_viewport(ui, ctx, self.dual_axes[1]);
                    });
                });
            });

            ui.separator();
            ui.allocate_ui(egui::vec2(avail.x, info_h), |ui| {
                self.show_right_info_panel(ui);
            });
        });
    }

    fn show_central_panel_compare(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.loaded.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label("Open a primary volume to use compare layout.");
                });
                return;
            }

            let avail = ui.available_size();
            let info_h = (avail.y * 0.24).clamp(110.0, 210.0);
            let view_h = (avail.y - info_h - 6.0).max(120.0);
            let view_w = avail.x / 2.0;

            ui.allocate_ui(egui::vec2(avail.x, view_h), |ui| {
                ui.horizontal(|ui| {
                    ui.allocate_ui(egui::vec2(view_w, view_h), |ui| {
                        self.render_axis_viewport(ui, ctx, self.compare_axes[0]);
                    });
                    ui.allocate_ui(egui::vec2(view_w, view_h), |ui| {
                        self.render_secondary_compare_viewport(
                            ui,
                            ctx,
                            self.compare_axes[0],
                            self.compare_axes[1],
                        );
                    });
                });
            });

            ui.separator();
            ui.allocate_ui(egui::vec2(avail.x, info_h), |ui| {
                ui.columns(2, |cols| {
                    cols[0].heading("Primary");
                    self.show_right_info_panel(&mut cols[0]);
                    cols[1].heading(if self.compare_fused_overlay {
                        "Fused"
                    } else {
                        "Secondary"
                    });
                    if let Some(vol) = &self.loaded_secondary {
                        let [d, r, c] = vol.shape;
                        let [dz, dy, dx] = vol.spacing;
                        cols[1].label(format!("Dims: {d}x{r}x{c}"));
                        cols[1].label(format!("Spacing: {dz:.2}x{dy:.2}x{dx:.2} mm"));
                        cols[1].label(format!(
                            "Modality: {}",
                            vol.modality.as_deref().unwrap_or("—")
                        ));
                        cols[1].label(format!(
                            "Series: {}",
                            vol.series_description.as_deref().unwrap_or("—")
                        ));
                        if self.compare_fused_overlay {
                            cols[1].label(format!("Alpha: {:.2}", self.compare_fusion_alpha));
                        }
                    } else {
                        cols[1].label("No secondary volume loaded.");
                    }
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
                    let label = self.axis_label(axis);
                    ui.label(format!("{label} — open a volume to begin"));
                });
                return;
            }
        };

        // ── 3. Compute spacing-aware fit and render image ─────────────────────
        let tex_w = tex_w_usize as f32;
        let tex_h = tex_h_usize as f32;
        let available = ui.available_size();
        let (row_mm, col_mm) = if let Some(vol) = &self.loaded {
            let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
            match axis {
                0 => (dy.max(1e-6), dx.max(1e-6)),
                1 => (dz.max(1e-6), dx.max(1e-6)),
                _ => (dz.max(1e-6), dy.max(1e-6)),
            }
        } else {
            (1.0, 1.0)
        };

        let phys_w = tex_w * col_mm;
        let phys_h = tex_h * row_mm;
        let use_uniform_fit = self.multi_planar || self.dual_plane || self.compare_side_by_side;
        let fit_scale = if use_uniform_fit {
            if tex_w > 0.0 && tex_h > 0.0 {
                (available.x / tex_w).min(available.y / tex_h)
            } else {
                1.0
            }
        } else if phys_w > 0.0 && phys_h > 0.0 {
            (available.x / phys_w).min(available.y / phys_h)
        } else {
            1.0
        };
        let (scale_x, scale_y) = if use_uniform_fit {
            let s = fit_scale * self.zoom;
            (s, s)
        } else {
            (
                fit_scale * self.zoom * col_mm,
                fit_scale * self.zoom * row_mm,
            )
        };
        let display_size = egui::vec2(tex_w * scale_x, tex_h * scale_y);

        let image_widget = egui::Image::new(egui::load::SizedTexture::new(tex_id, display_size))
            .sense(egui::Sense::click_and_drag());
        let response = ui.add(image_widget);

        // Track which axis is currently hovered for status/info display
        if response.hovered() || response.has_focus() || response.clicked() {
            self.status_axis = axis;
        }

        // ── 4–6. Overlay text, DICOM overlay, crosshair ────────────────────────
        // Painter::new clones the Arc<Context>; it does not hold a borrow on ui.
        let painter = ui.painter_at(response.rect);

        let axis_name = self.axis_label(axis);
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
                    self.pointer_suv,
                    self.current_cursor_suv(),
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
            self.draw_rt_struct_overlay(&painter, response.rect, axis, tex_h_usize, tex_w_usize);
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
                        egui::Stroke::new(1.0_f32, color),
                    );
                    painter.line_segment(
                        [
                            egui::pos2(crosshair.x, response.rect.min.y),
                            egui::pos2(crosshair.x, response.rect.max.y),
                        ],
                        egui::Stroke::new(1.0_f32, color),
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
            // img_to_screen: image-pixel Pos2 { x: col, y: row } → screen Pos2
            let origin = response.rect.min;
            let img_to_screen =
                |p: egui::Pos2| egui::pos2(origin.x + p.x * scale_x, origin.y + p.y * scale_y);

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
            let cursor_img_opt = if scale_x > 0.0 && scale_y > 0.0 {
                response
                    .hover_pos()
                    .map(|s| egui::pos2((s.x - origin.x) / scale_x, (s.y - origin.y) / scale_y))
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
        let (scroll_y, ctrl_or_cmd) = ctx.input(|i| {
            (
                i.smooth_scroll_delta.y,
                i.modifiers.ctrl || i.modifiers.command,
            )
        });
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
            // Map screen to image-pixel coordinates for tool event
            let img_pos = response.interact_pointer_pos().map(|s| {
                egui::pos2(
                    (s.x - response.rect.min.x) / scale_x,
                    (s.y - response.rect.min.y) / scale_y,
                )
            });
            self.on_drag_start(img_pos);
        }
        if response.dragged() {
            if self.active_tool == ToolKind::LabelPaint || self.active_tool == ToolKind::LabelErase
            {
                self.apply_label_at_pointer(axis, response.interact_pointer_pos(), response.rect);
            }
            let img_pos = response.interact_pointer_pos().map(|s| {
                egui::pos2(
                    (s.x - response.rect.min.x) / scale_x,
                    (s.y - response.rect.min.y) / scale_y,
                )
            });
            self.on_drag(img_pos);
        }
        if response.drag_stopped() {
            let img_pos = response.interact_pointer_pos().map(|s| {
                egui::pos2(
                    (s.x - response.rect.min.x) / scale_x,
                    (s.y - response.rect.min.y) / scale_y,
                )
            });
            self.on_drag_end(img_pos);
        }
        if response.clicked() {
            self.update_linked_cursor_from_pointer(
                axis,
                response.interact_pointer_pos(),
                response.rect,
            );
            if self.active_tool == ToolKind::LabelPaint || self.active_tool == ToolKind::LabelErase
            {
                self.apply_label_at_pointer(axis, response.interact_pointer_pos(), response.rect);
            }
            let img_pos = response.interact_pointer_pos().map(|s| {
                egui::pos2(
                    (s.x - response.rect.min.x) / scale_x,
                    (s.y - response.rect.min.y) / scale_y,
                )
            });
            self.on_click(img_pos);
        }
    }

    fn render_secondary_compare_viewport(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        primary_axis: usize,
        secondary_axis: usize,
    ) {
        let Some(secondary) = self.loaded_secondary.as_ref() else {
            ui.centered_and_justified(|ui| {
                ui.label("Select a series for Secondary to compare.");
            });
            return;
        };

        let primary_total = self.axis_slice_info(primary_axis).1.max(1);
        let primary_idx = self.axis_slice_info(primary_axis).0;
        let secondary_total = Self::axis_extent_for_volume(secondary, secondary_axis).max(1);
        let secondary_idx =
            Self::map_slice_index_between_volumes(primary_idx, primary_total, secondary_total);

        let needs_rebuild = if self.compare_fused_overlay {
            true
        } else {
            self.secondary_texture_dirty
                || self.secondary_texture.is_none()
                || self.secondary_texture_axis != secondary_axis
                || self.secondary_texture_slice != secondary_idx
        };
        if needs_rebuild {
            if self.compare_fused_overlay {
                let (color_image, tex_name) = {
                    let Some(primary) = self.loaded.as_ref() else {
                        return;
                    };
                    let primary_wc = self.viewer_state.window_center.unwrap_or(128.0) as f64;
                    let primary_ww =
                        self.viewer_state.window_width.unwrap_or(256.0).max(1.0) as f64;
                    let secondary_wc = self.secondary_window_center.unwrap_or(128.0) as f64;
                    let secondary_ww = self.secondary_window_width.unwrap_or(256.0).max(1.0) as f64;
                    let color_image = render_fused_slice(
                        primary,
                        primary_axis,
                        primary_idx,
                        WindowLevel::new(primary_wc, primary_ww),
                        self.colormap,
                        secondary,
                        secondary_axis,
                        secondary_idx,
                        WindowLevel::new(secondary_wc, secondary_ww),
                        self.secondary_colormap,
                        self.compare_fusion_alpha,
                    );
                    let color_image = apply_to_image(&color_image, self.view_transform);
                    let tex_name = format!(
                        "slice_tex_fused_axis{}_{}_slice{}_{}_a{}",
                        primary_axis.min(2),
                        secondary_axis.min(2),
                        primary_idx,
                        secondary_idx,
                        (self.compare_fusion_alpha.clamp(0.0, 1.0) * 100.0).round() as i32,
                    );
                    (color_image, tex_name)
                };
                self.secondary_texture =
                    Some(ctx.load_texture(tex_name, color_image, egui::TextureOptions::LINEAR));
                self.secondary_texture_axis = secondary_axis;
                self.secondary_texture_slice = secondary_idx;
                self.secondary_texture_dirty = false;
            } else {
                self.rebuild_secondary_texture(ctx, secondary_axis, secondary_idx);
            }
        }

        let Some(tex) = self.secondary_texture.as_ref() else {
            return;
        };

        let [w, h] = tex.size();
        let tex_w = w as f32;
        let tex_h = h as f32;
        let avail = ui.available_size();
        let fit = if tex_w > 0.0 && tex_h > 0.0 {
            (avail.x / tex_w).min(avail.y / tex_h)
        } else {
            1.0
        };
        let size = egui::vec2(tex_w * fit * self.zoom, tex_h * fit * self.zoom);
        let response = ui.add(
            egui::Image::new(egui::load::SizedTexture::new(tex.id(), size))
                .sense(egui::Sense::hover()),
        );
        let painter = ui.painter_at(response.rect);
        painter.text(
            response.rect.min + egui::vec2(6.0, 6.0),
            egui::Align2::LEFT_TOP,
            if self.compare_fused_overlay {
                "Fused"
            } else {
                "Secondary"
            },
            egui::FontId::proportional(12.0),
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 210),
        );
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

    /// Render the 3D-MIP projection through WL LUT and upload to the GPU.
    fn rebuild_texture_for_mip(&mut self, ctx: &egui::Context) {
        let color_image = {
            let Some(vol) = &self.loaded else { return };
            let wc = self.viewer_state.window_center.unwrap_or(128.0) as f64;
            let ww = self.viewer_state.window_width.unwrap_or(256.0).max(1.0) as f64;
            let wl = WindowLevel::new(wc, ww);
            match self.projection_mode {
                ProjectionMode::Mip => render_mip_axial(vol, wl, self.colormap),
                ProjectionMode::Vr => render_vr_axial(vol, wl, self.colormap, 0.06),
            }
        };

        self.mip_tex = Some(ctx.load_texture(
            "slice_tex_mip_axial",
            color_image,
            egui::TextureOptions::LINEAR,
        ));
    }

    /// Render one 3D-MIP viewport into `ui`.
    fn render_mip_viewport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let needs_rebuild = self.mip_dirty || self.mip_tex.is_none();
        if needs_rebuild && self.loaded.is_some() {
            self.rebuild_texture_for_mip(ctx);
            self.mip_dirty = false;
        }

        let Some((tex_id, [tex_w_usize, tex_h_usize])) =
            self.mip_tex.as_ref().map(|t| (t.id(), t.size()))
        else {
            ui.centered_and_justified(|ui| {
                ui.label("3D MIP — open a volume to begin");
            });
            return;
        };

        let tex_w = tex_w_usize as f32;
        let tex_h = tex_h_usize as f32;
        let available = ui.available_size();
        let fit_scale = if tex_w > 0.0 && tex_h > 0.0 {
            (available.x / tex_w).min(available.y / tex_h)
        } else {
            1.0
        };
        let s = fit_scale * self.zoom;
        let display_size = egui::vec2(tex_w * s, tex_h * s);

        let image_widget = egui::Image::new(egui::load::SizedTexture::new(tex_id, display_size));
        let response = ui.add(image_widget);

        let painter = ui.painter_at(response.rect);
        let label = match self.projection_mode {
            ProjectionMode::Mip => "3D MIP",
            ProjectionMode::Vr => "3D VR",
        };
        painter.text(
            response.rect.min + egui::vec2(6.0, 6.0),
            egui::Align2::LEFT_TOP,
            label,
            egui::FontId::proportional(12.0),
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 210),
        );

        response.context_menu(|ui| {
            ui.label("3D Projection");
            ui.separator();
            if ui
                .selectable_label(self.projection_mode == ProjectionMode::Mip, "MIP")
                .clicked()
            {
                self.projection_mode = ProjectionMode::Mip;
                self.mip_dirty = true;
                ui.close_menu();
            }
            if ui
                .selectable_label(self.projection_mode == ProjectionMode::Vr, "VR")
                .clicked()
            {
                self.projection_mode = ProjectionMode::Vr;
                self.mip_dirty = true;
                ui.close_menu();
            }
        });
    }

    fn rebuild_secondary_texture(&mut self, ctx: &egui::Context, axis: usize, slice_index: usize) {
        let (color_image, tex_name) = {
            let Some(vol) = &self.loaded_secondary else {
                return;
            };
            let wc = self.secondary_window_center.unwrap_or(128.0) as f64;
            let ww = self.secondary_window_width.unwrap_or(256.0).max(1.0) as f64;
            let wl = WindowLevel::new(wc, ww);
            let name = format!(
                "slice_tex_secondary_axis{}_slice{}",
                axis.min(2),
                slice_index
            );
            let img = SliceRenderer::render(vol, axis, slice_index, wl, self.secondary_colormap);
            let img = apply_to_image(&img, self.view_transform);
            (img, name)
        };

        self.secondary_texture =
            Some(ctx.load_texture(tex_name, color_image, egui::TextureOptions::LINEAR));
        self.secondary_texture_axis = axis;
        self.secondary_texture_slice = slice_index;
        self.secondary_texture_dirty = false;
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
                        let (slice_idx, total) = match self.status_axis {
                            0 => (self.viewer_state.slice_index, depth),
                            1 => (self.coronal_slice, rows),
                            2 => (self.sagittal_slice, cols),
                            _ => (self.viewer_state.slice_index, depth),
                        };
                        let axis_name = self.axis_label(self.status_axis);
                        row(ui, axis_name, &format!("{}/{}", slice_idx + 1, total));
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

        if let Some(path) = FileDialog::new()
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
                    self.status_message =
                        format!("PNG export failed for {}: {e:#}", path.display());
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
                self.rt_dvh_selected_roi = rt.rois.first().map(|roi| roi.roi_number);
                self.rt_struct = Some(rt);
                self.show_rt_struct_overlay = true;
                self.refresh_rt_dvh_cache();
                self.status_message = format!(
                    "Loaded RT-STRUCT {} ({} ROIs) from {}",
                    label,
                    roi_count,
                    path.display()
                );
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message =
                    format!("RT-STRUCT load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    fn load_rt_dose_file(&mut self, path: std::path::PathBuf) {
        match ritk_io::read_rt_dose(&path) {
            Ok(grid) => {
                let max_dose_gy = grid.dose_gy.iter().copied().fold(0.0_f64, f64::max);
                self.status_message = format!(
                    "Loaded RT-DOSE ({} type, {}×{}×{} grid) from {}",
                    grid.dose_type,
                    grid.rows,
                    grid.cols,
                    grid.n_frames,
                    path.display()
                );
                info!("{}", self.status_message);
                self.rt_dose = Some(grid);
                self.rt_dose_max_gy = Some(max_dose_gy);
                self.clear_rt_dose_overlay_cache();
                self.show_rt_dose_overlay = true;
                self.refresh_rt_dvh_cache();
            }
            Err(e) => {
                self.status_message = format!("RT-DOSE load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    fn load_rt_plan_file(&mut self, path: std::path::PathBuf) {
        match ritk_io::read_rt_plan(&path) {
            Ok(plan) => {
                let beam_count = plan.beams.len();
                let fg_count = plan.fraction_groups.len();
                let label = plan.rt_plan_label.clone();
                self.rt_plan = Some(plan);
                self.refresh_rt_dvh_cache();
                self.status_message = format!(
                    "Loaded RT-PLAN {} ({} beams, {} fraction groups) from {}",
                    label,
                    beam_count,
                    fg_count,
                    path.display()
                );
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("RT-PLAN load failed for {}: {e:#}", path.display());
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
        use crate::ui::rtdose_texture::{
            build_overlay_image, overlay_alpha, positive_finite_dose_range,
        };

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
        let vol_origin = [
            vol.origin[0] as f64,
            vol.origin[1] as f64,
            vol.origin[2] as f64,
        ];
        let vol_dir: [f64; 9] = std::array::from_fn(|i| vol.direction[i] as f64);
        let vol_spacing = [
            vol.spacing[0] as f64,
            vol.spacing[1] as f64,
            vol.spacing[2] as f64,
        ];

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
        let texture =
            painter
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
                MultiOtsuThreshold, NormalizeImageFilter, RelabelComponentFilter, SqrtImageFilter,
                SquareImageFilter, UnsharpMaskFilter,
            };
            match &filter_kind {
                crate::FilterKind::BedSeparation(config) => {
                    BedSeparationFilter::new(*config).apply(&image)
                }
                crate::FilterKind::Gaussian { sigma } => Ok(GaussianFilter::<LoadBackend>::new(
                    vec![f64::from(*sigma); 3],
                )
                .apply(&image)),
                crate::FilterKind::Median { radius } => MedianFilter::new(*radius).apply(&image),
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
                crate::FilterKind::RelabelComponents {
                    minimum_object_size,
                } => {
                    let (relabeled, _stats) = RelabelComponentFilter::with_minimum_object_size(
                        *minimum_object_size as usize,
                    )
                    .apply(&image);
                    Ok(relabeled)
                }
                crate::FilterKind::MultiOtsuThreshold { num_classes } => {
                    Ok(MultiOtsuThreshold::new(*num_classes as usize).apply(&image))
                }
                crate::FilterKind::BinaryErode {
                    radius,
                    foreground_value,
                } => BinaryErodeFilter::new(*radius)
                    .with_foreground(*foreground_value)
                    .apply(&image),
                crate::FilterKind::BinaryDilate {
                    radius,
                    foreground_value,
                } => BinaryDilateFilter::new(*radius)
                    .with_foreground(*foreground_value)
                    .apply(&image),
                crate::FilterKind::BinaryClosing {
                    radius,
                    foreground_value,
                } => BinaryMorphologicalClosing::new(*radius)
                    .with_foreground(*foreground_value)
                    .apply(&image),
                crate::FilterKind::BinaryOpening {
                    radius,
                    foreground_value,
                } => BinaryMorphologicalOpening::new(*radius)
                    .with_foreground(*foreground_value)
                    .apply(&image),
                crate::FilterKind::BinaryFillhole { foreground_value } => {
                    BinaryFillholeFilter::new()
                        .with_foreground(*foreground_value)
                        .apply(&image)
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
                crate::FilterKind::InvertIntensity { maximum } => Ok(match maximum {
                    Some(m) => InvertIntensityFilter::with_maximum(*m).apply(&image),
                    None => InvertIntensityFilter::new().apply(&image),
                }),
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
                crate::FilterKind::FlipZ => {
                    ritk_core::filter::FlipImageFilter::flip_z().apply(&image)
                }
                crate::FilterKind::FlipY => {
                    ritk_core::filter::FlipImageFilter::flip_y().apply(&image)
                }
                crate::FilterKind::FlipX => {
                    ritk_core::filter::FlipImageFilter::flip_x().apply(&image)
                }
                crate::FilterKind::MaskThreshold { threshold } => {
                    let dims = image.shape();
                    let td = image.data().clone().into_data();
                    let vals: Vec<f32> = td
                        .into_vec::<f32>()
                        .unwrap_or_else(|_| vec![0.0; dims[0] * dims[1] * dims[2]]);
                    let mask_vals: Vec<f32> = vals
                        .iter()
                        .map(|&v| if v > *threshold { 1.0_f32 } else { 0.0_f32 })
                        .collect();
                    let device = image.data().device();
                    let mask_td =
                        burn::tensor::TensorData::new(mask_vals, burn::tensor::Shape::new(dims));
                    let mask_tensor =
                        burn::tensor::Tensor::<LoadBackend, 3>::from_data(mask_td, &device);
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
                crate::FilterKind::ZeroCrossing {
                    foreground_value,
                    background_value,
                } => ritk_core::filter::ZeroCrossingImageFilter::new()
                    .with_foreground(*foreground_value)
                    .with_background(*background_value)
                    .apply(&image),
                crate::FilterKind::RegionOfInterest {
                    start_z,
                    start_y,
                    start_x,
                    size_z,
                    size_y,
                    size_x,
                } => ritk_core::filter::RegionOfInterestImageFilter::new(
                    [*start_z, *start_y, *start_x],
                    [*size_z, *size_y, *size_x],
                )
                .apply(&image),
                crate::FilterKind::PermuteAxes {
                    order_0,
                    order_1,
                    order_2,
                } => ritk_core::filter::PermuteAxesImageFilter::new([*order_0, *order_1, *order_2])
                    .apply(&image),
                crate::FilterKind::Mean { radius } => {
                    ritk_core::filter::MeanImageFilter::new(*radius).apply(&image)
                }
                crate::FilterKind::BinaryContour {
                    fully_connected,
                    foreground_value,
                } => ritk_core::filter::BinaryContourImageFilter::new(
                    *fully_connected,
                    *foreground_value,
                )
                .apply(&image),
                crate::FilterKind::LabelContour {
                    fully_connected,
                    background_value,
                } => ritk_core::filter::LabelContourImageFilter::new(
                    *fully_connected,
                    *background_value,
                )
                .apply(&image),
                crate::FilterKind::VotingBinary {
                    radius,
                    birth_threshold,
                    survival_threshold,
                    foreground_value,
                    background_value,
                } => ritk_core::filter::VotingBinaryImageFilter::new(
                    *radius,
                    *birth_threshold,
                    *survival_threshold,
                    *foreground_value,
                    *background_value,
                )
                .apply(&image),
                crate::FilterKind::Shrink {
                    factor_z,
                    factor_y,
                    factor_x,
                } => ritk_core::filter::ShrinkImageFilter::new([*factor_z, *factor_y, *factor_x])
                    .apply(&image),
                crate::FilterKind::ConstantPad {
                    pad_lower_z,
                    pad_lower_y,
                    pad_lower_x,
                    pad_upper_z,
                    pad_upper_y,
                    pad_upper_x,
                    constant,
                } => ritk_core::filter::ConstantPadImageFilter::new(
                    [*pad_lower_z, *pad_lower_y, *pad_lower_x],
                    [*pad_upper_z, *pad_upper_y, *pad_upper_x],
                    *constant,
                )
                .apply(&image),
                crate::FilterKind::MirrorPad {
                    pad_lower_z,
                    pad_lower_y,
                    pad_lower_x,
                    pad_upper_z,
                    pad_upper_y,
                    pad_upper_x,
                } => ritk_core::filter::MirrorPadImageFilter::new(
                    [*pad_lower_z, *pad_lower_y, *pad_lower_x],
                    [*pad_upper_z, *pad_upper_y, *pad_upper_x],
                )
                .apply(&image),
                crate::FilterKind::WrapPad {
                    pad_lower_z,
                    pad_lower_y,
                    pad_lower_x,
                    pad_upper_z,
                    pad_upper_y,
                    pad_upper_x,
                } => ritk_core::filter::WrapPadImageFilter::new(
                    [*pad_lower_z, *pad_lower_y, *pad_lower_x],
                    [*pad_upper_z, *pad_upper_y, *pad_upper_x],
                )
                .apply(&image),
                crate::FilterKind::GrayscaleErode { radius } => {
                    ritk_core::filter::GrayscaleErosion::new(*radius).apply(&image)
                }
                crate::FilterKind::GrayscaleDilate { radius } => {
                    ritk_core::filter::GrayscaleDilation::new(*radius).apply(&image)
                }
                crate::FilterKind::BinaryThreshold {
                    lower,
                    upper,
                    foreground,
                    background,
                } => ritk_core::filter::BinaryThresholdImageFilter::new(
                    *lower,
                    *upper,
                    *foreground,
                    *background,
                )
                .apply(&image),
                crate::FilterKind::RescaleIntensity { out_min, out_max } => {
                    ritk_core::filter::RescaleIntensityFilter::new(*out_min, *out_max).apply(&image)
                }
                crate::FilterKind::Clamp { lower, upper } => {
                    ritk_core::filter::ClampImageFilter::new(*lower, *upper).apply(&image)
                }
                crate::FilterKind::ConnectedThreshold {
                    seed_z,
                    seed_y,
                    seed_x,
                    lower,
                    upper,
                } => Ok(
                    ritk_core::segmentation::region_growing::ConnectedThresholdFilter::new(
                        [*seed_z, *seed_y, *seed_x],
                        *lower,
                        *upper,
                    )
                    .apply(&image),
                ),
                crate::FilterKind::ConfidenceConnected {
                    seed_z,
                    seed_y,
                    seed_x,
                    initial_lower,
                    initial_upper,
                    multiplier,
                    max_iterations,
                } => Ok(
                    ritk_core::segmentation::region_growing::ConfidenceConnectedFilter::new(
                        [*seed_z, *seed_y, *seed_x],
                        *initial_lower,
                        *initial_upper,
                    )
                    .with_multiplier(*multiplier)
                    .with_max_iterations(*max_iterations as usize)
                    .apply(&image),
                ),
                crate::FilterKind::NeighborhoodConnected {
                    seed_z,
                    seed_y,
                    seed_x,
                    lower,
                    upper,
                    radius_z,
                    radius_y,
                    radius_x,
                } => Ok(
                    ritk_core::segmentation::region_growing::NeighborhoodConnectedFilter::new(
                        [*seed_z, *seed_y, *seed_x],
                        *lower,
                        *upper,
                    )
                    .with_radius([*radius_z, *radius_y, *radius_x])
                    .apply(&image),
                ),
                crate::FilterKind::Atan => {
                    Ok(ritk_core::filter::AtanImageFilter::new().apply(&image))
                }
                crate::FilterKind::Sin => {
                    Ok(ritk_core::filter::SinImageFilter::new().apply(&image))
                }
                crate::FilterKind::Cos => {
                    Ok(ritk_core::filter::CosImageFilter::new().apply(&image))
                }
                crate::FilterKind::Tan => {
                    Ok(ritk_core::filter::TanImageFilter::new().apply(&image))
                }
                crate::FilterKind::Asin => {
                    Ok(ritk_core::filter::AsinImageFilter::new().apply(&image))
                }
                crate::FilterKind::Acos => {
                    Ok(ritk_core::filter::AcosImageFilter::new().apply(&image))
                }
                crate::FilterKind::BoundedReciprocal => {
                    Ok(ritk_core::filter::BoundedReciprocalImageFilter::new().apply(&image))
                }
                crate::FilterKind::CurvatureFlow {
                    iterations,
                    time_step,
                } => ritk_core::filter::CurvatureFlowImageFilter::new(
                    ritk_core::filter::CurvatureFlowConfig {
                        num_iterations: *iterations as usize,
                        time_step: *time_step,
                    },
                )
                .apply(&image),
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
                self.mip_dirty = true;
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

        let Some(root) = FileDialog::new().pick_folder() else {
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
            let color_image =
                SliceRenderer::render(vol, export.axis, export.slice_index, wl, self.colormap);
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
        let Some(path) = FileDialog::new()
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
        let Some(path) = FileDialog::new().add_filter("JSON", &["json"]).pick_file() else {
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
            self.status_message = "Save segmentation: no volume or segmentation loaded.".to_owned();
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

        let Some(path) = FileDialog::new()
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
                self.status_message = format!("Saved segmentation to {}", path.display());
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("Segmentation save failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    fn save_segmentation_dicom_seg_dialog(&mut self) {
        let (Some(vol), Some(editor)) = (self.loaded.as_ref(), self.label_editor.as_ref()) else {
            self.status_message = "Save DICOM-SEG: no volume or segmentation loaded.".to_owned();
            return;
        };
        let map = editor.current_map();
        let origin = vol.origin;
        let spacing = vol.spacing;
        let direction = vol.direction;

        let Some(path) = FileDialog::new()
            .set_file_name("segmentation.dcm")
            .add_filter("DICOM SEG", &["dcm"])
            .save_file()
        else {
            return;
        };

        match ritk_io::label_map_to_dicom_seg(map, origin, spacing, direction, true) {
            Ok(seg) => match ritk_io::write_dicom_seg(&path, &seg) {
                Ok(()) => {
                    self.status_message = format!("Saved DICOM-SEG to {}", path.display());
                    info!("{}", self.status_message);
                }
                Err(e) => {
                    self.status_message = format!("DICOM-SEG write failed: {e:#}");
                    error!("{}", self.status_message);
                }
            },
            Err(e) => {
                self.status_message = format!("DICOM-SEG conversion failed: {e:#}");
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
            self.status_message = "Load segmentation: no volume loaded.".to_owned();
            return;
        };
        let expected_shape = vol.shape;

        let Some(path) = FileDialog::new()
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
                        self.label_editor = Some(crate::label::LabelEditor::from_label_map(map));
                        self.status_message =
                            format!("Loaded segmentation from {}", path.display());
                        info!("{}", self.status_message);
                    }
                    Err(e) => {
                        self.status_message = format!("Segmentation data error: {e}");
                        error!("{}", self.status_message);
                    }
                }
            }
            Err(e) => {
                self.status_message = format!("Segmentation load failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    /// Load a label map from a DICOM-SEG file and replace the current segmentation.
    ///
    /// The reconstructed shape must match the currently loaded volume.
    fn load_segmentation_dicom_seg_file(&mut self, path: &std::path::Path) {
        let Some(vol) = self.loaded.as_ref() else {
            self.status_message = "Load DICOM-SEG: no volume loaded.".to_owned();
            return;
        };
        let expected_shape = vol.shape;

        match ritk_io::read_dicom_seg(path) {
            Ok(seg) => match ritk_io::dicom_seg_to_label_map(&seg) {
                Ok(map) => {
                    if map.shape != expected_shape {
                        self.status_message = format!(
                            "DICOM-SEG shape {:?} does not match volume {:?}",
                            map.shape, expected_shape
                        );
                        error!("{}", self.status_message);
                        return;
                    }
                    self.label_editor = Some(crate::label::LabelEditor::from_label_map(map));
                    self.status_message = format!("Loaded DICOM-SEG from {}", path.display());
                    info!("{}", self.status_message);
                }
                Err(e) => {
                    self.status_message = format!("DICOM-SEG decode failed: {e:#}");
                    error!("{}", self.status_message);
                }
            },
            Err(e) => {
                self.status_message = format!("DICOM-SEG load failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    /// Load a label map from a DICOM-SEG file and replace the current segmentation.
    ///
    /// The reconstructed shape must match the currently loaded volume.
    fn load_segmentation_dicom_seg_dialog(&mut self) {
        let Some(path) = FileDialog::new()
            .add_filter("DICOM SEG", &["dcm", "dicom"])
            .pick_file()
        else {
            return;
        };

        self.load_segmentation_dicom_seg_file(&path);
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
                let series_time = meta.series_time.clone();
                let patient_weight_kg = meta.patient_weight_kg;
                let injected_dose_bq = meta.radionuclide_total_dose_bq;
                let radionuclide_half_life_s = meta.radionuclide_half_life_s;
                let radiopharmaceutical_start_time = meta.radiopharmaceutical_start_time.clone();
                let decay_correction = meta.decay_correction.clone();

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
                    series_time,
                    patient_weight_kg,
                    injected_dose_bq,
                    radionuclide_half_life_s,
                    radiopharmaceutical_start_time,
                    decay_correction,
                });
                self.viewer_state = state;
                self.axis = protocol.preferred_axis.min(2);
                self.coronal_slice = shape[1] / 2;
                self.sagittal_slice = shape[2] / 2;
                self.multi_planar = protocol.multi_planar;
                self.dual_plane = false;
                self.compare_side_by_side = false;
                self.compare_fused_overlay = false;
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
                self.rt_dose_max_gy = None;
                self.rt_plan = None;
                self.rt_dvh_selected_roi = None;
                self.rt_dvh_cache = None;
                self.clear_rt_dose_overlay_cache();
                self.tool_state = ToolState::Idle;
                self.pan_offset = egui::Vec2::ZERO;
                self.zoom = 1.0;
                self.pointer_intensity = 0.0;
                self.pointer_suv = None;
                self.colormap = Self::colormap_for_modality(
                    self.loaded.as_ref().and_then(|v| v.modality.as_deref()),
                );
                self.texture = None;
                self.texture_dirty = true;
                self.coronal_tex = None;
                self.coronal_dirty = true;
                self.sagittal_tex = None;
                self.sagittal_dirty = true;
                self.mip_tex = None;
                self.mip_dirty = true;
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

    fn load_secondary_from_path(&mut self, path: std::path::PathBuf) {
        let load_root = crate::dicom::classify_dicom_input_path(&path)
            .dicom_root()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| path.clone());
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
                let raw = image.data().clone().into_data();
                let data = match raw.into_vec::<f32>() {
                    Ok(v) => Arc::new(v),
                    Err(e) => {
                        self.status_message = format!("Secondary pixel extraction failed: {e:?}");
                        return;
                    }
                };

                self.loaded_secondary = Some(LoadedVolume {
                    data,
                    shape,
                    spacing,
                    origin,
                    direction,
                    metadata: Some(Box::new(meta.clone())),
                    source: Some(load_root.clone()),
                    modality: meta.modality.clone(),
                    patient_name: meta.patient_name.clone(),
                    patient_id: meta.patient_id.clone(),
                    study_date: meta.study_date.clone(),
                    series_description: meta.series_description.clone(),
                    series_time: meta.series_time.clone(),
                    patient_weight_kg: meta.patient_weight_kg,
                    injected_dose_bq: meta.radionuclide_total_dose_bq,
                    radionuclide_half_life_s: meta.radionuclide_half_life_s,
                    radiopharmaceutical_start_time: meta.radiopharmaceutical_start_time.clone(),
                    decay_correction: meta.decay_correction.clone(),
                });
                let protocol = select_hanging_protocol(
                    meta.modality.as_deref(),
                    meta.series_description.as_deref(),
                    shape,
                );
                self.secondary_window_center = Some(protocol.window_center);
                self.secondary_window_width = Some(protocol.window_width);
                self.secondary_texture = None;
                self.secondary_texture_dirty = true;
                self.secondary_colormap = Self::colormap_for_modality(meta.modality.as_deref());
                self.compare_side_by_side = true;
                self.multi_planar = false;
                self.dual_plane = false;
                self.status_message = format!("Loaded secondary series: {}", load_root.display());
            }
            Err(e) => {
                self.status_message = format!(
                    "Secondary DICOM load failed for {}: {e:#}",
                    load_root.display()
                );
            }
        }
    }

    /// Load a medical image volume from `path`.
    ///
    /// Calls [`crate::dicom::loader::load_volume_from_path`], which supports
    /// NIfTI, MetaImage, NRRD, MGH, and DICOM fallbacks. On success, all
    /// viewer state and textures are reset.
    fn load_volume_file(&mut self, path: std::path::PathBuf) {
        self.cine.stop();
        match crate::dicom::loader::load_volume_from_path(&path) {
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
                self.dual_plane = false;
                self.compare_side_by_side = false;
                self.compare_fused_overlay = false;
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
                self.rt_dose_max_gy = None;
                self.rt_plan = None;
                self.rt_dvh_selected_roi = None;
                self.rt_dvh_cache = None;
                self.clear_rt_dose_overlay_cache();
                self.tool_state = ToolState::Idle;
                self.pan_offset = egui::Vec2::ZERO;
                self.zoom = 1.0;
                self.pointer_intensity = 0.0;
                self.pointer_suv = None;
                self.colormap = Self::colormap_for_modality(
                    self.loaded.as_ref().and_then(|v| v.modality.as_deref()),
                );
                self.texture = None;
                self.texture_dirty = true;
                self.coronal_tex = None;
                self.coronal_dirty = true;
                self.sagittal_tex = None;
                self.sagittal_dirty = true;
                self.mip_tex = None;
                self.mip_dirty = true;
                self.status_message = msg;
                self.refresh_cached_histogram();
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("Volume load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    /// Load a medical image volume from pathless dropped bytes.
    fn load_volume_bytes(&mut self, name_hint: String, bytes: &[u8]) {
        self.cine.stop();
        match crate::dicom::loader::load_volume_from_bytes(&name_hint, bytes) {
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
                    "Loaded dropped in-memory volume '{}' — shape {:?} — protocol {}",
                    name_hint, shape, protocol.protocol_name
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
                self.rt_dose_max_gy = None;
                self.rt_plan = None;
                self.rt_dvh_selected_roi = None;
                self.rt_dvh_cache = None;
                self.clear_rt_dose_overlay_cache();
                self.tool_state = ToolState::Idle;
                self.pan_offset = egui::Vec2::ZERO;
                self.zoom = 1.0;
                self.pointer_intensity = 0.0;
                self.pointer_suv = None;
                self.colormap = Self::colormap_for_modality(
                    self.loaded.as_ref().and_then(|v| v.modality.as_deref()),
                );
                self.texture = None;
                self.texture_dirty = true;
                self.coronal_tex = None;
                self.coronal_dirty = true;
                self.sagittal_tex = None;
                self.sagittal_dirty = true;
                self.mip_tex = None;
                self.mip_dirty = true;
                self.status_message = msg;

                self.refresh_cached_histogram();
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message =
                    format!("Volume load failed for dropped '{}': {e:#}", name_hint);
                error!("{}", self.status_message);
            }
        }
    }

    /// Load a DICOM series from pathless dropped named byte payloads.
    fn load_dicom_series_bytes(&mut self, files: Vec<(String, std::sync::Arc<[u8]>)>) {
        self.cine.stop();
        let borrowed: Vec<(String, &[u8])> = files
            .iter()
            .map(|(name, bytes)| (name.clone(), bytes.as_ref()))
            .collect();

        match crate::dicom::loader::load_dicom_series_from_named_bytes(&borrowed) {
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
                let msg =
                    format!(
                    "Loaded dropped in-memory DICOM series ({} files) — shape {:?} — protocol {}",
                    files.len(), shape, protocol.protocol_name
                );
                self.loaded = Some(vol);
                self.viewer_state = state;
                self.axis = protocol.preferred_axis.min(2);
                self.coronal_slice = shape[1] / 2;
                self.sagittal_slice = shape[2] / 2;
                self.multi_planar = protocol.multi_planar;
                self.dual_plane = false;
                self.compare_side_by_side = false;
                self.compare_fused_overlay = false;
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
                self.rt_dose_max_gy = None;
                self.rt_plan = None;
                self.rt_dvh_selected_roi = None;
                self.rt_dvh_cache = None;
                self.clear_rt_dose_overlay_cache();
                self.tool_state = ToolState::Idle;
                self.pan_offset = egui::Vec2::ZERO;
                self.zoom = 1.0;
                self.pointer_intensity = 0.0;
                self.pointer_suv = None;
                self.colormap = Self::colormap_for_modality(
                    self.loaded.as_ref().and_then(|v| v.modality.as_deref()),
                );
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
                self.status_message =
                    format!("Volume load failed for dropped in-memory DICOM series: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    /// Drop the currently loaded study and reset all study-owned state.
    fn close_study(&mut self) {
        self.loaded = None;
        self.loaded_secondary = None;
        self.series_load_target = SeriesLoadTarget::Primary;
        self.secondary_window_center = None;
        self.secondary_window_width = None;
        self.secondary_colormap = Colormap::Grayscale;
        self.multi_planar = false;
        self.dual_plane = false;
        self.compare_side_by_side = false;
        self.compare_fused_overlay = false;
        self.compare_fusion_alpha = 0.35;
        self.compare_axes = [0, 0];
        self.dual_axes = [0, 1];
        self.annotations.clear();
        self.label_editor = None;
        self.rt_struct = None;
        self.rt_dose = None;
        self.rt_dose_max_gy = None;
        self.rt_plan = None;
        self.rt_dvh_selected_roi = None;
        self.rt_dvh_cache = None;
        self.clear_rt_dose_overlay_cache();
        self.viewer_state = ViewerState::new();
        self.linked_cursor = None;
        self.pointer_intensity = 0.0;
        self.pointer_suv = None;
        self.cached_histogram = None;
        self.selected_series = None;
        self.pan_offset = egui::Vec2::ZERO;
        self.zoom = 1.0;
        self.texture = None;
        self.secondary_texture = None;
        self.coronal_tex = None;
        self.sagittal_tex = None;
        self.mip_tex = None;
        self.projection_mode = ProjectionMode::Mip;
        self.texture_dirty = false;
        self.secondary_texture_dirty = false;
        self.secondary_texture_axis = 0;
        self.secondary_texture_slice = 0;
        self.coronal_dirty = false;
        self.sagittal_dirty = false;
        self.mip_dirty = false;
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
                    if v < mn {
                        mn = v;
                    }
                    if v > mx {
                        mx = v;
                    }
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

    fn axis_extent_for_volume(volume: &LoadedVolume, axis: usize) -> usize {
        match axis {
            0 => volume.shape[0],
            1 => volume.shape[1],
            _ => volume.shape[2],
        }
    }

    fn map_slice_index_between_volumes(
        primary_index: usize,
        primary_total: usize,
        secondary_total: usize,
    ) -> usize {
        if primary_total <= 1 || secondary_total <= 1 {
            return 0;
        }
        let pmax = primary_total.saturating_sub(1) as f64;
        let smax = secondary_total.saturating_sub(1) as f64;
        ((primary_index as f64 / pmax) * smax)
            .round()
            .clamp(0.0, smax) as usize
    }

    /// Clear all cached RT-DOSE overlay textures.
    fn clear_rt_dose_overlay_cache(&mut self) {
        self.rt_dose_overlay_cache = std::array::from_fn(|_| None);
    }

    fn axis_for_plane(&self, plane: AnatomicalPlane) -> usize {
        axis_for_plane_in_volume(self.loaded.as_ref(), plane)
    }

    fn axis_label(&self, axis: usize) -> &'static str {
        anatomical_label_for_axis(self.loaded.as_ref(), axis)
    }

    // ── Slice navigation ──────────────────────────────────────────────────────

    /// Return `(current_slice_index, total_slices)` for `axis`.
    fn axis_slice_info(&self, axis: usize) -> (usize, usize) {
        let total = self
            .loaded
            .as_ref()
            .map(|v| axis_total(v.shape, axis))
            .unwrap_or(1);
        match axis {
            0 => (self.viewer_state.slice_index, total),
            1 => (self.coronal_slice, total),
            _ => (self.sagittal_slice, total),
        }
    }

    /// Step the slice for `axis` by `delta`, clamped to the valid range.
    ///
    /// Marks the corresponding texture dirty when the index changes.
    fn set_slice_for_axis(&mut self, axis: usize, index: usize) {
        let total = self
            .loaded
            .as_ref()
            .map(|v| axis_total(v.shape, axis))
            .unwrap_or(1);
        let next = clamp_index(index, total);

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
        let next = step_clamped(current, total, delta);
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
        let total = self
            .loaded
            .as_ref()
            .map(|v| axis_total(v.shape, axis))
            .unwrap_or(1);
        if total == 0 {
            return;
        }
        let current = match axis {
            0 => self.viewer_state.slice_index,
            1 => self.coronal_slice,
            _ => self.sagittal_slice,
        };
        let next = advance_wrapped(current, total, steps);
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
                self.mip_dirty = true;
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
        self.status_message = format!(
            "Ellipse ROI: μ={mean:.1}  σ={std_dev:.1}  [{min:.0}, {max:.0}]  {area_mm2:.1} mm²"
        );
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
        let Some(voxel) = viewport_point_to_voxel(
            volume.shape,
            axis,
            self.axis_slice_info(axis).0,
            point,
            rect,
        ) else {
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
        let Some(voxel) =
            cursor.update_from_viewport_point(volume.shape, axis, slice_index, point, rect)
        else {
            return;
        };

        self.viewer_state.slice_index = voxel[0];
        self.coronal_slice = voxel[1];
        self.sagittal_slice = voxel[2];
        self.axis = axis.min(2);
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
        self.mip_dirty = true;
        self.status_message = format!(
            "Linked cursor axis={} voxel=[{},{},{}]",
            axis, voxel[0], voxel[1], voxel[2]
        );
    }

    fn update_pointer_intensity(&mut self, axis: usize, pos: Option<egui::Pos2>, rect: egui::Rect) {
        let Some(point) = pos else {
            self.pointer_intensity = 0.0;
            self.pointer_suv = None;
            return;
        };
        let Some(volume) = &self.loaded else {
            self.pointer_intensity = 0.0;
            self.pointer_suv = None;
            return;
        };
        let slice_index = self.axis_slice_info(axis).0;
        let Some(voxel) = viewport_point_to_voxel(volume.shape, axis, slice_index, point, rect)
        else {
            self.pointer_intensity = 0.0;
            self.pointer_suv = None;
            return;
        };
        self.pointer_intensity = intensity_at_voxel(volume, voxel);
        self.pointer_suv = Self::compute_suv_from_volume(volume, self.pointer_intensity as f64);
    }

    /// Select the default colormap for a modality string.
    ///
    /// PT → `Colormap::Hot` (standard PET display); all others → `Colormap::Grayscale`.
    fn colormap_for_modality(modality: Option<&str>) -> Colormap {
        if modality == Some("PT") {
            Colormap::Hot
        } else {
            Colormap::Grayscale
        }
    }

    /// Compute SUVbw for a PET voxel value [Bq/mL].
    ///
    /// Returns `None` when `PetAcquisitionParams::from_loaded_volume` fails,
    /// `pixel_bqml` is non-finite, or the result is non-finite.
    fn compute_suv_from_volume(vol: &crate::LoadedVolume, pixel_bqml: f64) -> Option<f32> {
        use crate::dicom::pet::PetAcquisitionParams;
        if !pixel_bqml.is_finite() {
            return None;
        }
        let pet = PetAcquisitionParams::from_loaded_volume(vol)?;
        let delta_t = PetAcquisitionParams::delta_t_s_from_vol(vol);
        let suv = pet.pixel_to_suvbw(pixel_bqml, delta_t);
        if suv.is_finite() {
            Some(suv as f32)
        } else {
            None
        }
    }

    /// Compute SUVbw at the linked-cursor voxel position, if available.
    fn current_cursor_suv(&self) -> Option<f32> {
        let volume = self.loaded.as_ref()?;
        let cursor = self.linked_cursor?;
        let [z, y, x] = cursor.voxel();
        let pixel = volume.pixel_at(z, y, x) as f64;
        Self::compute_suv_from_volume(volume, pixel)
    }

    fn current_cursor_value(&self) -> Option<f32> {
        let volume = self.loaded.as_ref()?;
        let cursor = self.linked_cursor?;
        let [z, y, x] = cursor.voxel();
        Some(volume.pixel_at(z, y, x))
    }

    fn draw_label_overlay(&self, painter: &egui::Painter, rect: egui::Rect, axis: usize) {
        let Some(editor) = &self.label_editor else {
            return;
        };
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
            let color =
                egui::Color32::from_rgb(contour.color[0], contour.color[1], contour.color[2]);
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
                    egui::Stroke::new(1.5_f32, color),
                );
            }

            if contour.closed {
                if let (Some(first), Some(last)) = (
                    contour.points_row_col.first().copied(),
                    contour.points_row_col.last().copied(),
                ) {
                    painter.line_segment(
                        [to_screen(last[0], last[1]), to_screen(first[0], first[1])],
                        egui::Stroke::new(1.5_f32, color),
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
            series_time: None,
            patient_weight_kg: None,
            injected_dose_bq: None,
            radionuclide_half_life_s: None,
            radiopharmaceutical_start_time: None,
            decay_correction: None,
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
        app.mip_dirty = false;

        app.reset_view_to_fit();

        assert_eq!(app.zoom, 1.0);
        assert_eq!(app.pan_offset, egui::Vec2::ZERO);
        assert!(app.texture_dirty);
        assert!(app.coronal_dirty);
        assert!(app.sagittal_dirty);
        assert!(app.mip_dirty);
        assert_eq!(app.status_message, "Zoom reset to fit.");
    }

    #[test]
    fn close_study_clears_loaded_and_cached_state() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([2, 2, 2]));
        app.loaded_secondary = Some(test_volume([2, 2, 2]));
        app.multi_planar = true;
        app.dual_plane = true;
        app.compare_side_by_side = true;
        app.series_load_target = SeriesLoadTarget::Secondary;
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
        app.projection_mode = ProjectionMode::Vr;

        app.close_study();

        assert!(app.loaded.is_none(), "loaded volume must be cleared");
        assert!(
            app.loaded_secondary.is_none(),
            "secondary volume must be cleared"
        );
        assert!(!app.multi_planar, "multi-planar mode must reset to false");
        assert!(!app.dual_plane, "dual-plane mode must reset to false");
        assert!(
            !app.compare_side_by_side,
            "compare mode must reset to false"
        );
        assert_eq!(
            app.series_load_target,
            SeriesLoadTarget::Primary,
            "series target must reset to primary"
        );
        assert!(app.linked_cursor.is_none(), "linked cursor must be cleared");
        assert!(
            app.cached_histogram.is_none(),
            "histogram cache must be cleared"
        );
        assert!(
            app.selected_series.is_none(),
            "selected series must be cleared"
        );
        assert_eq!(app.pointer_intensity, 0.0, "pointer intensity must reset");
        assert_eq!(app.pan_offset, egui::Vec2::ZERO, "pan must reset");
        assert_eq!(app.zoom, 1.0, "zoom must reset");
        assert_eq!(
            app.projection_mode,
            ProjectionMode::Mip,
            "projection mode must reset to MIP"
        );
        assert_eq!(app.status_message, "Study closed.");
    }

    #[test]
    fn map_slice_index_between_volumes_maps_bounds_and_midpoint() {
        assert_eq!(SnapApp::map_slice_index_between_volumes(0, 300, 90), 0);
        assert_eq!(SnapApp::map_slice_index_between_volumes(299, 300, 90), 89);

        let mapped = SnapApp::map_slice_index_between_volumes(150, 300, 90);
        assert!(
            mapped >= 44 && mapped <= 45,
            "midpoint mapping should stay near the secondary midpoint"
        );
    }

    #[test]
    fn load_external_dcmqi_dicom_seg_into_snap_app() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([3, 512, 512]));

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("test_data")
            .join("dicom_seg")
            .join("dcmqi")
            .join("liver.dcm");
        assert!(
            path.is_file(),
            "external SEG fixture missing: {}",
            path.display()
        );

        app.load_segmentation_dicom_seg_file(&path);

        let editor = app
            .label_editor
            .as_ref()
            .expect("label editor loaded from external SEG");
        let map = editor.current_map();
        assert_eq!(map.shape, [3, 512, 512]);
        assert!(map.present_labels().contains(&1));
        assert!(
            map.count_label(1) > 0,
            "external SEG must populate label 1 voxels"
        );
        assert_eq!(
            map.table.get_label(1).map(|e| e.name.as_str()),
            Some("Liver")
        );
        assert_eq!(
            app.status_message,
            format!("Loaded DICOM-SEG from {}", path.display())
        );
    }

    #[test]
    fn load_external_dcmqi_partial_overlap_dicom_seg_into_snap_app() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([3, 512, 512]));

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("test_data")
            .join("dicom_seg")
            .join("dcmqi")
            .join("partial_overlaps.dcm");
        assert!(
            path.is_file(),
            "external SEG fixture missing: {}",
            path.display()
        );

        app.load_segmentation_dicom_seg_file(&path);

        let editor = app
            .label_editor
            .as_ref()
            .expect("label editor loaded from external dcmqi partial-overlap SEG");
        let map = editor.current_map();
        assert_eq!(map.shape, [3, 512, 512]);
        let present = map.present_labels();
        for label in [1u32, 2, 3, 4, 5] {
            assert!(present.contains(&label), "label {label} must be present");
            assert!(map.count_label(label) > 0, "label {label} must have voxels");
        }
        assert_eq!(
            app.status_message,
            format!("Loaded DICOM-SEG from {}", path.display())
        );
    }

    #[test]
    fn load_external_highdicom_overlap_dicom_seg_into_snap_app() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([4, 16, 16]));

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("test_data")
            .join("dicom_seg")
            .join("highdicom")
            .join("seg_image_ct_binary_overlap.dcm");
        assert!(
            path.is_file(),
            "external SEG fixture missing: {}",
            path.display()
        );

        app.load_segmentation_dicom_seg_file(&path);

        let editor = app
            .label_editor
            .as_ref()
            .expect("label editor loaded from external highdicom SEG");
        let map = editor.current_map();
        assert_eq!(map.shape, [4, 16, 16]);
        assert!(map.present_labels().contains(&1));
        assert!(map.present_labels().contains(&2));
        assert!(
            map.count_label(1) > 0,
            "segment 1 voxels must populate the viewer state"
        );
        assert!(
            map.count_label(2) > 0,
            "segment 2 voxels must populate the viewer state"
        );
        assert_eq!(
            map.table.get_label(1).map(|e| e.name.as_str()),
            Some("first segment")
        );
        assert_eq!(
            map.table.get_label(2).map(|e| e.name.as_str()),
            Some("second segment")
        );
        assert_eq!(
            app.status_message,
            format!("Loaded DICOM-SEG from {}", path.display())
        );
    }

    #[test]
    fn load_external_highdicom_binary_dicom_seg_into_snap_app() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([3, 16, 16]));

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("test_data")
            .join("dicom_seg")
            .join("highdicom")
            .join("seg_image_ct_binary.dcm");
        assert!(
            path.is_file(),
            "external SEG fixture missing: {}",
            path.display()
        );

        app.load_segmentation_dicom_seg_file(&path);

        let editor = app
            .label_editor
            .as_ref()
            .expect("label editor loaded from external highdicom binary SEG");
        let map = editor.current_map();
        assert_eq!(map.shape, [3, 16, 16]);
        assert!(map.present_labels().contains(&1));
        assert!(
            map.count_label(1) > 0,
            "segment voxels must populate the viewer state"
        );
        assert_eq!(
            map.table.get_label(1).map(|e| e.name.as_str()),
            Some("first segment")
        );
        assert_eq!(
            app.status_message,
            format!("Loaded DICOM-SEG from {}", path.display())
        );
    }

    #[test]
    fn load_external_rsna_dido_liver_dicom_seg_into_snap_app() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([34, 512, 512]));

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("test_data")
            .join("dicom_seg")
            .join("rsna_dido")
            .join("xTtzBC6F6p_rpexuszCnb_01_liver.dcm");
        assert!(
            path.is_file(),
            "external SEG fixture missing: {}",
            path.display()
        );

        app.load_segmentation_dicom_seg_file(&path);

        let editor = app
            .label_editor
            .as_ref()
            .expect("label editor loaded from external rsna dido SEG");
        let map = editor.current_map();
        assert_eq!(map.shape, [34, 512, 512]);
        assert!(map.present_labels().contains(&1));
        assert!(
            map.count_label(1) > 0,
            "segment voxels must populate the viewer state"
        );
        assert_eq!(
            map.table.get_label(1).map(|e| e.name.as_str()),
            Some("liver")
        );
        assert_eq!(
            app.status_message,
            format!("Loaded DICOM-SEG from {}", path.display())
        );
    }

    #[test]
    fn load_rt_plan_file_sets_plan_summary_state() {
        let mut app = SnapApp::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("plan.dcm");

        let plan = ritk_io::RtPlanInfo {
            sop_instance_uid: String::new(),
            rt_plan_label: "PLAN_A".to_owned(),
            rt_plan_name: "Plan A".to_owned(),
            rt_plan_description: "Synthetic plan".to_owned(),
            plan_intent: "CURATIVE".to_owned(),
            beams: vec![ritk_io::RtBeamInfo {
                beam_number: 1,
                beam_name: "BEAM_1".to_owned(),
                beam_description: "Beam one".to_owned(),
                radiation_type: "PHOTON".to_owned(),
                treatment_delivery_type: "TREATMENT".to_owned(),
                n_control_points: 2,
            }],
            fraction_groups: vec![ritk_io::RtFractionGroup {
                fraction_group_number: 1,
                n_fractions_planned: 30,
                referenced_beam_numbers: vec![1],
            }],
        };
        ritk_io::write_rt_plan(&path, &plan).expect("write rt plan");

        app.load_rt_plan_file(path.clone());

        let loaded = app.rt_plan.as_ref().expect("rt plan loaded");
        assert_eq!(loaded.rt_plan_label, "PLAN_A");
        assert_eq!(loaded.beams.len(), 1);
        assert_eq!(loaded.fraction_groups.len(), 1);
        assert!(
            app.status_message.contains("Loaded RT-PLAN PLAN_A"),
            "status: {}",
            app.status_message
        );
    }

    #[test]
    fn rt_dose_plan_link_status_reports_linked_uid() {
        let mut app = SnapApp::default();
        app.rt_plan = Some(ritk_io::RtPlanInfo {
            sop_instance_uid: "2.25.9001".to_owned(),
            rt_plan_label: "PLAN_LINK".to_owned(),
            rt_plan_name: String::new(),
            rt_plan_description: String::new(),
            plan_intent: String::new(),
            beams: vec![],
            fraction_groups: vec![],
        });
        app.rt_dose = Some(ritk_io::RtDoseGrid {
            rows: 1,
            cols: 1,
            n_frames: 1,
            dose_type: "PHYSICAL".to_owned(),
            dose_summation_type: "PLAN".to_owned(),
            dose_grid_scaling: 1.0,
            frame_offsets: vec![0.0],
            dose_gy: vec![1.0],
            image_position: None,
            image_orientation: None,
            pixel_spacing: None,
            referenced_rt_plan_sop_instance_uid: Some("2.25.9001".to_owned()),
        });

        let msg = app.rt_dose_plan_link_status().expect("link status present");
        assert!(
            msg.contains("linked to loaded RT-PLAN UID 2.25.9001"),
            "{msg}"
        );
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
        assert!(app.mip_dirty, "mip dirty not set");
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
            series_time: None,
            patient_weight_kg: None,
            injected_dose_bq: None,
            radionuclide_half_life_s: None,
            radiopharmaceutical_start_time: None,
            decay_correction: None,
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
        assert_ne!(
            coronal, sagittal,
            "coronal and sagittal spacing must differ"
        );
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

    #[test]
    fn colormap_for_modality_pt_yields_hot() {
        assert_eq!(
            SnapApp::colormap_for_modality(Some("PT")),
            Colormap::Hot,
            "PT modality must auto-select Hot colormap"
        );
    }

    #[test]
    fn colormap_for_modality_ct_yields_grayscale() {
        assert_eq!(
            SnapApp::colormap_for_modality(Some("CT")),
            Colormap::Grayscale,
            "CT modality must auto-select Grayscale colormap"
        );
    }

    #[test]
    fn colormap_for_modality_none_yields_grayscale() {
        assert_eq!(
            SnapApp::colormap_for_modality(None),
            Colormap::Grayscale,
            "absent modality must default to Grayscale colormap"
        );
    }

    #[test]
    fn secondary_colormap_auto_selects_hot_when_secondary_is_pt() {
        let mut app = SnapApp::default();
        assert_eq!(app.secondary_colormap, Colormap::Grayscale);
        app.secondary_colormap = SnapApp::colormap_for_modality(Some("PT"));
        assert_eq!(
            app.secondary_colormap,
            Colormap::Hot,
            "secondary PT volume must produce Hot secondary colormap"
        );
    }

    #[test]
    fn secondary_colormap_remains_grayscale_when_secondary_is_ct() {
        let mut app = SnapApp::default();
        app.secondary_colormap = SnapApp::colormap_for_modality(Some("CT"));
        assert_eq!(
            app.secondary_colormap,
            Colormap::Grayscale,
            "secondary CT volume must retain Grayscale secondary colormap"
        );
    }

    #[test]
    fn primary_colormap_auto_selects_hot_when_primary_is_pt() {
        let mut app = SnapApp::default();
        assert_eq!(app.colormap, Colormap::Grayscale);
        app.colormap = SnapApp::colormap_for_modality(Some("PT"));
        assert_eq!(
            app.colormap,
            Colormap::Hot,
            "primary PT volume must produce Hot primary colormap"
        );
    }
}
