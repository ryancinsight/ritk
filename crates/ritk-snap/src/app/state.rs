use crate::label::LabelEditor;
use crate::render::colormap::Colormap;
use crate::render::RenderBufferPool;
use crate::tools::interaction::{Annotation, ToolState};
use crate::tools::kind::ToolKind;
use crate::ui::RoiDoseAnalytics;
use crate::ui::{CinePlayback, LinkedCursor, ViewTransform};
use crate::{LoadedVolume, ViewerState};

/// Default opacity for the fused-overlay compare mode.
pub(crate) const DEFAULT_FUSION_ALPHA: f32 = 0.35;

// â”€â”€ Helper types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Cached RT-DOSE overlay texture for one axis.
pub(crate) struct RtDoseOverlayCacheEntry {
    pub(crate) slice_idx: usize,
    pub(crate) vol_shape: [usize; 3],
    pub(crate) dose_dims: [usize; 3],
    pub(crate) opacity_alpha: u8,
    pub(crate) texture: egui::TextureHandle }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SeriesLoadTarget {
    Primary,
    Secondary }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProjectionMode {
    Mip,
    Vr }

// â”€â”€ SnapApp â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

pub(crate) struct SnapApp {
    // â”€â”€ Volume â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    /// Currently loaded volume, if any.
    pub(crate) loaded: Option<LoadedVolume>,
    /// Secondary loaded volume for cross-study compare.
    pub(crate) loaded_secondary: Option<LoadedVolume>,
    /// Viewer navigation state (slice index, W/L).
    pub(crate) viewer_state: ViewerState,
    /// Secondary compare viewport W/L center.
    pub(crate) secondary_window_center: Option<f32>,
    /// Secondary compare viewport W/L width.
    pub(crate) secondary_window_width: Option<f32>,
    /// Active colormap for intensity mapping.
    pub(crate) colormap: Colormap,
    /// Secondary colormap for compare panel.
    pub(crate) secondary_colormap: Colormap,
    /// Primary MPR axis for single-viewport and tool operations:
    /// 0 = axial, 1 = coronal, 2 = sagittal.
    pub(crate) axis: usize,

    // â”€â”€ Tools â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    /// Active interaction tool.
    pub(crate) active_tool: ToolKind,
    /// In-progress gesture state for the active tool.
    pub(crate) tool_state: ToolState,
    /// Completed measurement annotations.
    pub(crate) annotations: Vec<Annotation>,
    /// Last hovered or interacted axis for status/info display.
    pub(crate) status_axis: usize,
    /// Segmentation label editor for the currently loaded volume.
    pub(crate) label_editor: Option<LabelEditor>,
    /// Brush radius in voxels for paint/erase tools.
    pub(crate) label_brush_radius: usize,
    /// Whether label overlays are rendered on viewports.
    pub(crate) show_label_overlay: bool,
    /// RT-STRUCT contour overlay visibility.
    pub(crate) show_rt_struct_overlay: bool,
    /// Currently loaded RT Structure Set.
    pub(crate) rt_struct: Option<ritk_io::RtStructureSet>,
    /// Currently loaded RT Dose grid.
    pub(crate) rt_dose: Option<ritk_io::RtDoseGrid>,
    /// Cached RT-DOSE maximum Gy value (computed once at load time).
    pub(crate) rt_dose_max_gy: Option<f64>,
    /// Currently loaded RT Plan metadata.
    pub(crate) rt_plan: Option<ritk_io::RtPlanInfo>,
    /// Selected ROI number for RT dose analytics.
    pub(crate) rt_dvh_selected_roi: Option<u32>,
    /// Cached ROI dose analytics for selected ROI.
    pub(crate) rt_dvh_cache: Option<RoiDoseAnalytics>,
    /// Whether to render the RT-DOSE heat-map overlay on viewports.
    pub(crate) show_rt_dose_overlay: bool,
    /// Opacity of the RT-DOSE overlay (0.0 transparent â€¦ 1.0 opaque).
    pub(crate) rt_dose_opacity: f32,
    /// Per-axis RT-DOSE overlay texture cache (bounded to three entries).
    pub(crate) rt_dose_overlay_cache: [Option<RtDoseOverlayCacheEntry>; 3],
    /// Active filter configuration shown in the processing panel.
    pub(crate) active_filter: crate::FilterKind,
    /// Whether the filter processing panel is visible.
    pub(crate) show_filter_panel: bool,

    // â”€â”€ Texture cache â€” axial â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    /// Cached egui texture for the axial slice.
    pub(crate) texture: Option<egui::TextureHandle>,
    /// Cached egui texture for secondary compare panel.
    pub(crate) secondary_texture: Option<egui::TextureHandle>,
    /// `true` when the axial texture must be rebuilt before the next frame.
    pub(crate) texture_dirty: bool,
    /// `true` when secondary texture must be rebuilt.
    pub(crate) secondary_texture_dirty: bool,
    /// Axis used by current secondary texture.
    pub(crate) secondary_texture_axis: usize,
    /// Slice index used by current secondary texture.
    pub(crate) secondary_texture_slice: usize,

    // â”€â”€ Texture cache â€” coronal / sagittal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    /// Cached egui texture for the coronal slice (MPR mode).
    pub(crate) coronal_tex: Option<egui::TextureHandle>,
    /// `true` when the coronal texture must be rebuilt.
    pub(crate) coronal_dirty: bool,
    /// Current coronal slice index (fixed row `r`).
    pub(crate) coronal_slice: usize,
    /// Cached egui texture for the sagittal slice (MPR mode).
    pub(crate) sagittal_tex: Option<egui::TextureHandle>,
    /// `true` when the sagittal texture must be rebuilt.
    pub(crate) sagittal_dirty: bool,
    /// Current sagittal slice index (fixed column `c`).
    pub(crate) sagittal_slice: usize,
    /// Cached egui texture for the 3D-MIP viewport (axial projection).
    pub(crate) mip_tex: Option<egui::TextureHandle>,
    /// `true` when the MIP projection texture must be rebuilt.
    pub(crate) mip_dirty: bool,
    /// Active projection mode for the bottom-right 3D viewport.
    pub(crate) projection_mode: ProjectionMode,

    // â”€â”€ Surface mesh overlay â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    /// Currently loaded surface mesh for overlay rendering on the MIP viewport.
    pub(crate) loaded_mesh: Option<ritk_io::VtkPolyData>,
    /// Cached egui texture for the Phong-shaded mesh overlay.
    pub(crate) mesh_tex: Option<egui::TextureHandle>,
    /// `true` when the mesh overlay texture must be rebuilt before next frame.
    pub(crate) mesh_dirty: bool,
    /// Whether the mesh overlay is composited on the 3D-MIP viewport.
    pub(crate) show_mesh_overlay: bool,

    // â”€â”€ Viewport â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    /// Viewport pan offset in screen pixels.
    pub(crate) pan_offset: egui::Vec2,
    /// Viewport zoom multiplier (1.0 = fit-to-panel).
    pub(crate) zoom: f32,
    /// Viewport image orientation transform (flip/rotate).
    pub(crate) view_transform: ViewTransform,
    /// Whether to show the colorbar overlay in each viewport.
    pub(crate) show_colorbar: bool,

    // â”€â”€ UI state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    /// `true` when the 2Ã—2 multi-planar layout is active.
    pub(crate) multi_planar: bool,
    /// `true` when 2-panel same-volume layout is active.
    pub(crate) dual_plane: bool,
    /// `true` when primary/secondary compare layout is active.
    pub(crate) compare_side_by_side: bool,
    /// `true` when compare panel renders fused primary/secondary overlay.
    pub(crate) compare_fused_overlay: bool,
    /// Secondary contribution weight in fused compare mode.
    pub(crate) compare_fusion_alpha: f32,
    /// Axis assignment for dual-plane same-volume layout.
    pub(crate) dual_axes: [usize; 2],
    /// Axis assignment for compare layout: [primary_axis, secondary_axis].
    pub(crate) compare_axes: [usize; 2],
    /// `true` when the DICOM 4-corner overlay is drawn on viewports.
    pub(crate) show_overlay: bool,
    /// `true` when crosshair lines are drawn on viewports.
    pub(crate) show_crosshair: bool,
    /// Shared voxel cursor used to synchronize all MPR viewports.
    pub(crate) linked_cursor: Option<LinkedCursor>,
    /// Cine playback controller for automatic slice stepping.
    pub(crate) cine: CinePlayback,
    /// `true` when the series browser left panel is visible.
    pub(crate) show_series_browser: bool,
    /// Current voxel intensity value under the pointer (HU or relative).
    pub(crate) pointer_intensity: f32,
    /// SUVbw value under the pointer for PET volumes; `None` for non-PET or unavailable params.
    pub(crate) pointer_suv: Option<f32>,
    /// Cached voxel intensity histogram for the loaded volume.
    ///
    /// Computed once when a volume is loaded; `None` when no volume is loaded.
    /// Used to render the W/L histogram panel in the sidebar.
    pub(crate) cached_histogram: Option<crate::render::histogram::Histogram>,

    /// Pre-allocated scratch buffers for per-frame texture rebuild.
    ///
    /// Eliminates per-call heap allocations on the slice-render and MIP-render
    /// hot paths. Capacity grows monotonically to the maximum observed dimension.
    pub(crate) render_buffer_pool: RenderBufferPool,

    // â”€â”€ Series browser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    /// Hierarchical DICOM series tree.
    pub(crate) series_tree: crate::dicom::series_tree::SeriesTree<'static>,
    /// The folder path currently highlighted in the series browser.
    pub(crate) selected_series: Option<std::path::PathBuf>,
    /// Which tab is active in the series browser sidebar.
    pub(crate) sidebar_tab: crate::ui::sidebar::SidebarTab,
    /// Active load target for series selection.
    pub(crate) series_load_target: SeriesLoadTarget,

    // â”€â”€ Status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    /// Message shown in the bottom status bar.
    pub(crate) status_message: String,
    /// Path queued for loading on the next [`eframe::App::update`] cycle.
    pub(crate) pending_load: Option<std::path::PathBuf>,
    /// Secondary path queued for load on next update cycle.
    pub(crate) pending_secondary_load: Option<std::path::PathBuf>,

    // â”€â”€ PACS panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    /// PACS server connection configuration.
    pub(crate) pacs_config: crate::pacs::PacsConfig,
    /// Current PACS query state machine (Idle / Pending / Results / Error).
    pub(crate) pacs_query_state: crate::pacs::QueryState,
    /// Whether the PACS panel window is visible.
    pub(crate) show_pacs_panel: bool,
    /// Patient name filter string for C-FIND queries (DICOM wildcard format).
    pub(crate) pacs_patient_filter: String,
    /// Modality filter for C-FIND queries; empty = all modalities.
    pub(crate) pacs_modality_filter: String,
    /// Study date range filter for C-FIND queries.
    /// DICOM date range format: `YYYYMMDD-YYYYMMDD`, `YYYYMMDD-`, `-YYYYMMDD`, or `""` (all).
    pub(crate) pacs_study_date_filter: String,
    /// Accession number filter for C-FIND queries; empty string = all.
    pub(crate) pacs_accession_filter: String,
    /// Human-readable result of the last C-ECHO test.
    pub(crate) pacs_echo_display: String,
    /// Index of the currently selected C-FIND study-level result row.
    pub(crate) pacs_selected_row: Option<usize>,
    /// Index of the currently selected series-level result row.
    pub(crate) pacs_selected_series_row: Option<usize>,
    /// StudyInstanceUID of the study currently being explored in series drill-down.
    pub(crate) pacs_study_context_uid: String,
    /// Handle to an in-flight background PACS operation, if any.
    pub(crate) pacs_worker: Option<crate::pacs::PacsWorkerHandle>,
    /// Embedded C-STORE SCP handle; `Some` when the SCP is running.
    ///
    /// Receives instances forwarded by the PACS during C-MOVE sub-operations.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) pacs_scp_handle: Option<ritk_io::StoreScpHandle>,
    /// Count of DICOM instances received by the embedded SCP since last start.
    pub(crate) pacs_received_count: u32,
    /// Buffered SCP-received instances awaiting load into the viewer.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) pacs_pending_instances: Vec<ritk_io::StoredInstance>,

    /// Number of instances auto-loaded this frame (set by `poll_pacs_scp`, consumed by UI).
    ///
    /// Set to `Some(N)` when auto-load fires, `None` otherwise. Cleared at the
    /// start of each frame so the notification is shown for exactly one frame.
    pub(crate) pacs_auto_loaded_this_frame: Option<usize>,

    // â”€â”€ GPU renderer (native only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    /// GPU-accelerated volume renderer.  `None` when no suitable GPU is
    /// available or when running on wasm32.  CPU path is the fallback.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) gpu_renderer: Option<crate::render::gpu_volume::GpuVolumeRenderer> }

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
            loaded_mesh: None,
            mesh_tex: None,
            mesh_dirty: false,
            show_mesh_overlay: false,
            pan_offset: egui::Vec2::ZERO,
            zoom: 1.0,
            view_transform: ViewTransform::default(),
            show_colorbar: false,
            multi_planar: false,
            dual_plane: false,
            compare_side_by_side: false,
            compare_fused_overlay: false,
            compare_fusion_alpha: DEFAULT_FUSION_ALPHA,
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
            render_buffer_pool: RenderBufferPool::default(),
            series_tree: crate::dicom::series_tree::SeriesTree::new(),
            selected_series: None,
            sidebar_tab: crate::ui::sidebar::SidebarTab::Series,
            series_load_target: SeriesLoadTarget::Primary,
            status_message: "No study loaded â€” use File > Open to load a DICOM folder.".to_owned(),
            pending_load: None,
            pending_secondary_load: None,
            pacs_config: crate::pacs::PacsConfig::default(),
            pacs_query_state: crate::pacs::QueryState::Idle,
            show_pacs_panel: false,
            pacs_patient_filter: String::new(),
            pacs_modality_filter: String::new(),
            pacs_study_date_filter: String::new(),
            pacs_accession_filter: String::new(),
            pacs_echo_display: String::new(),
            pacs_selected_row: None,
            pacs_selected_series_row: None,
            pacs_study_context_uid: String::new(),
            pacs_worker: None,
            #[cfg(not(target_arch = "wasm32"))]
            pacs_scp_handle: None,
            pacs_received_count: 0,
            #[cfg(not(target_arch = "wasm32"))]
            pacs_pending_instances: Vec::new(),
            pacs_auto_loaded_this_frame: None,
            status_axis: 0,
            #[cfg(not(target_arch = "wasm32"))]
            gpu_renderer: crate::render::gpu_volume::GpuVolumeRenderer::try_create() }
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

// â”€â”€ eframe::App â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

        // Poll background PACS worker on every frame (must run even when the
        // PACS panel is closed so responses are applied promptly).
        self.poll_pacs_worker();

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
