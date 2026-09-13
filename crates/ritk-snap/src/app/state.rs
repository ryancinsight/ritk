#[cfg(not(target_arch = "wasm32"))]
use super::volume_input::VolumeInput;
use crate::label::LabelEditor;
use crate::presentation::PresentationDispatcher;
use crate::render::NamedColorMap;
#[cfg(not(target_arch = "wasm32"))]
use crate::render::RenderBufferPool;
use crate::tools::interaction::{Annotation, ToolState, ViewportOffset};
use crate::tools::kind::ToolKind;
use crate::ui::LinkedCursor;
#[cfg(not(target_arch = "wasm32"))]
use crate::ui::RoiDoseAnalytics;
#[cfg(not(target_arch = "wasm32"))]
use crate::ui::{CinePlayback, ViewTransform};
use crate::{LoadedVolume, ViewerState};

/// Default opacity for the fused-overlay compare mode.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const DEFAULT_FUSION_ALPHA: f32 = 0.35;

// ── Helper types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(not(target_arch = "wasm32"))]
pub(crate) enum SeriesLoadTarget {
    Primary,
    Secondary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(not(target_arch = "wasm32"))]
pub(crate) enum ProjectionMode {
    Mip,
    Vr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(not(target_arch = "wasm32"))]
pub(crate) enum ProjectionBackend {
    Cpu,
    Gpu,
    Pending,
}

// ── SnapApp ───────────────────────────────────────────────────────────────────

pub(crate) struct SnapApp {
    // ── Volume ────────────────────────────────────────────────────────────────
    /// Currently loaded volume, if any.
    pub(crate) loaded: Option<LoadedVolume>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Secondary loaded volume for cross-study compare.
    pub(crate) loaded_secondary: Option<LoadedVolume>,
    /// Viewer navigation state (slice index, W/L).
    pub(crate) viewer_state: ViewerState,
    /// Monotonic revision for transitions that invalidate retained render resources.
    ///
    /// Host adapters use this value to invalidate their own retained render
    /// resources. It is deliberately independent of any GUI texture type.
    pub(crate) visual_revision: u64,
    #[cfg(not(target_arch = "wasm32"))]
    /// Secondary compare viewport W/L center.
    pub(crate) secondary_window_center: Option<f32>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Secondary compare viewport W/L width.
    pub(crate) secondary_window_width: Option<f32>,
    /// Active colormap for intensity mapping.
    pub(crate) colormap: NamedColorMap,
    #[cfg(not(target_arch = "wasm32"))]
    /// Secondary colormap for compare panel.
    pub(crate) secondary_colormap: NamedColorMap,
    /// Primary MPR axis for single-viewport and tool operations:
    /// 0 = axial, 1 = coronal, 2 = sagittal.
    pub(crate) axis: usize,

    // ── Tools ─────────────────────────────────────────────────────────────────
    /// Active interaction tool.
    pub(crate) active_tool: ToolKind,
    /// In-progress gesture state for the active tool.
    pub(crate) tool_state: ToolState,
    /// Stateful reducer for host pointer gestures at the presentation seam.
    pub(crate) presentation_dispatcher: PresentationDispatcher,
    /// Completed measurement annotations.
    pub(crate) annotations: Vec<Annotation>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Last hovered or interacted axis for status/info display.
    pub(crate) status_axis: usize,
    /// Segmentation label editor for the currently loaded volume.
    pub(crate) label_editor: Option<LabelEditor>,
    /// Brush radius in voxels for paint/erase tools.
    pub(crate) label_brush_radius: usize,
    #[cfg(not(target_arch = "wasm32"))]
    /// Whether label overlays are rendered on viewports.
    pub(crate) show_label_overlay: bool,
    #[cfg(not(target_arch = "wasm32"))]
    /// RT-STRUCT contour overlay visibility.
    pub(crate) show_rt_struct_overlay: bool,
    #[cfg(not(target_arch = "wasm32"))]
    /// Currently loaded RT Structure Set.
    pub(crate) rt_struct: Option<ritk_io::RtStructureSet>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Currently loaded RT Dose grid.
    pub(crate) rt_dose: Option<ritk_io::RtDoseGrid>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Cached RT-DOSE maximum Gy value (computed once at load time).
    pub(crate) rt_dose_max_gy: Option<f64>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Currently loaded RT Plan metadata.
    pub(crate) rt_plan: Option<ritk_io::RtPlanInfo>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Selected ROI number for RT dose analytics.
    pub(crate) rt_dvh_selected_roi: Option<u32>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Cached ROI dose analytics for selected ROI.
    pub(crate) rt_dvh_cache: Option<RoiDoseAnalytics>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Whether to render the RT-DOSE heat-map overlay on viewports.
    pub(crate) show_rt_dose_overlay: bool,
    #[cfg(not(target_arch = "wasm32"))]
    /// Opacity of the RT-DOSE overlay (0.0 transparent … 1.0 opaque).
    pub(crate) rt_dose_opacity: f32,
    #[cfg(not(target_arch = "wasm32"))]
    /// Active filter configuration shown in the processing panel.
    pub(crate) active_filter: crate::FilterKind,
    #[cfg(not(target_arch = "wasm32"))]
    /// Whether the filter processing panel is visible.
    pub(crate) show_filter_panel: bool,

    // ── Texture cache — coronal / sagittal ────────────────────────────────────
    /// Current coronal slice index (fixed row `r`).
    pub(crate) coronal_slice: usize,
    /// Current sagittal slice index (fixed column `c`).
    pub(crate) sagittal_slice: usize,
    #[cfg(not(target_arch = "wasm32"))]
    /// Active projection mode for the bottom-right 3D viewport.
    pub(crate) projection_mode: ProjectionMode,
    #[cfg(not(target_arch = "wasm32"))]
    /// Renderer that produced the current 3D projection texture.
    pub(crate) projection_backend: ProjectionBackend,

    // ── Surface mesh overlay ──────────────────────────────────────────────────
    /// Currently loaded surface mesh for overlay rendering on the MIP viewport.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) loaded_mesh: Option<ritk_io::VtkPolyData>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Whether the mesh overlay is composited on the 3D-MIP viewport.
    pub(crate) show_mesh_overlay: bool,

    // ── Viewport ──────────────────────────────────────────────────────────────
    /// Viewport pan offset in screen pixels.
    pub(crate) pan_offset: ViewportOffset,
    /// Viewport zoom multiplier (1.0 = fit-to-panel).
    pub(crate) zoom: f32,
    #[cfg(not(target_arch = "wasm32"))]
    /// Viewport image orientation transform (flip/rotate).
    pub(crate) view_transform: ViewTransform,
    #[cfg(not(target_arch = "wasm32"))]
    /// Whether to show the colorbar overlay in each viewport.
    pub(crate) show_colorbar: bool,

    // ── UI state ──────────────────────────────────────────────────────────────
    /// `true` when the 2×2 multi-planar layout is active.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) multi_planar: bool,
    #[cfg(not(target_arch = "wasm32"))]
    /// `true` when 2-panel same-volume layout is active.
    pub(crate) dual_plane: bool,
    #[cfg(not(target_arch = "wasm32"))]
    /// `true` when primary/secondary compare layout is active.
    pub(crate) compare_side_by_side: bool,
    #[cfg(not(target_arch = "wasm32"))]
    /// `true` when compare panel renders fused primary/secondary overlay.
    pub(crate) compare_fused_overlay: bool,
    #[cfg(not(target_arch = "wasm32"))]
    /// Secondary contribution weight in fused compare mode.
    pub(crate) compare_fusion_alpha: f32,
    #[cfg(not(target_arch = "wasm32"))]
    /// Axis assignment for dual-plane same-volume layout.
    pub(crate) dual_axes: [usize; 2],
    #[cfg(not(target_arch = "wasm32"))]
    /// Axis assignment for compare layout: [primary_axis, secondary_axis].
    pub(crate) compare_axes: [usize; 2],
    #[cfg(not(target_arch = "wasm32"))]
    /// `true` when the DICOM 4-corner overlay is drawn on viewports.
    pub(crate) show_overlay: bool,
    #[cfg(not(target_arch = "wasm32"))]
    /// `true` when crosshair lines are drawn on viewports.
    pub(crate) show_crosshair: bool,
    /// Shared voxel cursor used to synchronize all MPR viewports.
    pub(crate) linked_cursor: Option<LinkedCursor>,
    /// Cine playback controller for automatic slice stepping.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) cine: CinePlayback,
    #[cfg(not(target_arch = "wasm32"))]
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

    #[cfg(not(target_arch = "wasm32"))]
    /// Pre-allocated scratch buffers for per-frame texture rebuild.
    ///
    /// Eliminates per-call heap allocations on the slice-render and MIP-render
    /// hot paths. Capacity grows monotonically to the maximum observed dimension.
    pub(crate) render_buffer_pool: RenderBufferPool,

    // ── Series browser ────────────────────────────────────────────────────────
    /// Hierarchical DICOM series tree.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) series_tree: crate::dicom::series_tree::SeriesTree<'static>,
    #[cfg(not(target_arch = "wasm32"))]
    /// The folder path currently highlighted in the series browser.
    pub(crate) selected_series: Option<std::sync::Arc<ritk_io::DicomSeriesInfo>>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Which tab is active in the series browser sidebar.
    pub(crate) sidebar_tab: crate::ui::sidebar::SidebarTab,
    #[cfg(not(target_arch = "wasm32"))]
    /// Active load target for series selection.
    pub(crate) series_load_target: SeriesLoadTarget,

    // ── Status ────────────────────────────────────────────────────────────────
    /// Message shown in the bottom status bar.
    pub(crate) status_message: String,
    #[cfg(not(target_arch = "wasm32"))]
    /// Path queued for loading on the next [`eframe::App::update`] cycle.
    pub(crate) pending_load: Option<VolumeInput>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Secondary path queued for load on next update cycle.
    pub(crate) pending_secondary_load: Option<VolumeInput>,

    /// Monotonic publication generations for primary and secondary loads.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) load_generations: [u64; 2],
    /// At most one bounded decode task per viewer target.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) load_tasks: [Option<super::load_tasks::LoadTask>; 2],

    // ── PACS panel ────────────────────────────────────────────────────────────
    #[cfg(not(target_arch = "wasm32"))]
    /// PACS server connection configuration.
    pub(crate) pacs_config: crate::pacs::PacsConfig,
    #[cfg(not(target_arch = "wasm32"))]
    /// Current PACS query state machine (Idle / Pending / Results / Error).
    pub(crate) pacs_query_state: crate::pacs::QueryState,
    #[cfg(not(target_arch = "wasm32"))]
    /// Whether the PACS panel window is visible.
    pub(crate) show_pacs_panel: bool,
    #[cfg(not(target_arch = "wasm32"))]
    /// Patient name filter string for C-FIND queries (DICOM wildcard format).
    pub(crate) pacs_patient_filter: String,
    #[cfg(not(target_arch = "wasm32"))]
    /// Modality filter for C-FIND queries; empty = all modalities.
    pub(crate) pacs_modality_filter: String,
    #[cfg(not(target_arch = "wasm32"))]
    /// Study date range filter for C-FIND queries.
    /// DICOM date range format: `YYYYMMDD-YYYYMMDD`, `YYYYMMDD-`, `-YYYYMMDD`, or `""` (all).
    pub(crate) pacs_study_date_filter: String,
    #[cfg(not(target_arch = "wasm32"))]
    /// Accession number filter for C-FIND queries; empty string = all.
    pub(crate) pacs_accession_filter: String,
    #[cfg(not(target_arch = "wasm32"))]
    /// Human-readable result of the last C-ECHO test.
    pub(crate) pacs_echo_display: String,
    #[cfg(not(target_arch = "wasm32"))]
    /// Index of the currently selected C-FIND study-level result row.
    pub(crate) pacs_selected_row: Option<usize>,
    #[cfg(not(target_arch = "wasm32"))]
    /// Index of the currently selected series-level result row.
    pub(crate) pacs_selected_series_row: Option<usize>,
    #[cfg(not(target_arch = "wasm32"))]
    /// StudyInstanceUID of the study currently being explored in series drill-down.
    pub(crate) pacs_study_context_uid: String,
    #[cfg(not(target_arch = "wasm32"))]
    /// Handle to an in-flight background PACS operation, if any.
    pub(crate) pacs_worker: Option<crate::pacs::PacsWorkerHandle>,
    /// Embedded C-STORE SCP handle; `Some` when the SCP is running.
    ///
    /// Receives instances forwarded by the PACS during C-MOVE sub-operations.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) pacs_scp_handle: Option<ritk_io::StoreScpHandle>,
    /// Count of DICOM instances received by the embedded SCP since last start.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) pacs_received_count: u32,
    /// Buffered SCP-received instances awaiting load into the viewer.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) pacs_pending_instances: Vec<ritk_io::StoredInstance>,

    /// Number of instances auto-loaded this frame (set by `poll_pacs_scp`, consumed by UI).
    ///
    /// Set to `Some(N)` when auto-load fires, `None` otherwise. Cleared at the
    /// start of each frame so the notification is shown for exactly one frame.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) pacs_auto_loaded_this_frame: Option<usize>,

    // ── GPU renderer (native only) ────────────────────────────────────────────
    /// GPU-accelerated volume renderer.  `None` when no suitable GPU is
    /// available or when running on wasm32.  CPU path is the fallback.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) gpu_renderer: Option<crate::render::gpu_volume::GpuVolumeRenderer>,
}

impl Default for SnapApp {
    fn default() -> Self {
        Self {
            loaded: None,
            #[cfg(not(target_arch = "wasm32"))]
            loaded_secondary: None,
            viewer_state: ViewerState::new(),
            visual_revision: 0,
            #[cfg(not(target_arch = "wasm32"))]
            secondary_window_center: None,
            #[cfg(not(target_arch = "wasm32"))]
            secondary_window_width: None,
            colormap: NamedColorMap::Grayscale,
            #[cfg(not(target_arch = "wasm32"))]
            secondary_colormap: NamedColorMap::Grayscale,
            axis: 0,
            active_tool: ToolKind::WindowLevel,
            tool_state: ToolState::Idle,
            presentation_dispatcher: PresentationDispatcher::new(),
            annotations: Vec::new(),
            label_editor: None,
            label_brush_radius: 1,
            #[cfg(not(target_arch = "wasm32"))]
            show_label_overlay: true,
            #[cfg(not(target_arch = "wasm32"))]
            show_rt_struct_overlay: true,
            #[cfg(not(target_arch = "wasm32"))]
            rt_struct: None,
            #[cfg(not(target_arch = "wasm32"))]
            rt_dose: None,
            #[cfg(not(target_arch = "wasm32"))]
            rt_dose_max_gy: None,
            #[cfg(not(target_arch = "wasm32"))]
            rt_plan: None,
            #[cfg(not(target_arch = "wasm32"))]
            rt_dvh_selected_roi: None,
            #[cfg(not(target_arch = "wasm32"))]
            rt_dvh_cache: None,
            #[cfg(not(target_arch = "wasm32"))]
            show_rt_dose_overlay: false,
            #[cfg(not(target_arch = "wasm32"))]
            rt_dose_opacity: 0.5,
            #[cfg(not(target_arch = "wasm32"))]
            active_filter: crate::FilterKind::Gaussian { sigma: 1.0 },
            #[cfg(not(target_arch = "wasm32"))]
            show_filter_panel: false,
            coronal_slice: 0,
            sagittal_slice: 0,
            #[cfg(not(target_arch = "wasm32"))]
            projection_mode: ProjectionMode::Mip,
            #[cfg(not(target_arch = "wasm32"))]
            projection_backend: ProjectionBackend::Cpu,
            #[cfg(not(target_arch = "wasm32"))]
            loaded_mesh: None,
            #[cfg(not(target_arch = "wasm32"))]
            show_mesh_overlay: false,
            pan_offset: ViewportOffset::new(0.0, 0.0),
            zoom: 1.0,
            #[cfg(not(target_arch = "wasm32"))]
            view_transform: ViewTransform::default(),
            #[cfg(not(target_arch = "wasm32"))]
            show_colorbar: false,
            #[cfg(not(target_arch = "wasm32"))]
            multi_planar: false,
            #[cfg(not(target_arch = "wasm32"))]
            dual_plane: false,
            #[cfg(not(target_arch = "wasm32"))]
            compare_side_by_side: false,
            #[cfg(not(target_arch = "wasm32"))]
            compare_fused_overlay: false,
            #[cfg(not(target_arch = "wasm32"))]
            compare_fusion_alpha: DEFAULT_FUSION_ALPHA,
            #[cfg(not(target_arch = "wasm32"))]
            dual_axes: [0, 1],
            #[cfg(not(target_arch = "wasm32"))]
            compare_axes: [0, 0],
            #[cfg(not(target_arch = "wasm32"))]
            show_overlay: true,
            #[cfg(not(target_arch = "wasm32"))]
            show_crosshair: false,
            linked_cursor: None,
            #[cfg(not(target_arch = "wasm32"))]
            cine: CinePlayback::default(),
            #[cfg(not(target_arch = "wasm32"))]
            show_series_browser: true,
            pointer_intensity: 0.0,
            pointer_suv: None,
            cached_histogram: None,
            #[cfg(not(target_arch = "wasm32"))]
            render_buffer_pool: RenderBufferPool::default(),
            #[cfg(not(target_arch = "wasm32"))]
            series_tree: crate::dicom::series_tree::SeriesTree::new(),
            #[cfg(not(target_arch = "wasm32"))]
            selected_series: None,
            #[cfg(not(target_arch = "wasm32"))]
            sidebar_tab: crate::ui::sidebar::SidebarTab::Series,
            #[cfg(not(target_arch = "wasm32"))]
            series_load_target: SeriesLoadTarget::Primary,
            status_message: "No study loaded — use File > Open to load a DICOM folder.".to_owned(),
            #[cfg(not(target_arch = "wasm32"))]
            pending_load: None,
            #[cfg(not(target_arch = "wasm32"))]
            pending_secondary_load: None,
            #[cfg(not(target_arch = "wasm32"))]
            load_generations: [0; 2],
            #[cfg(not(target_arch = "wasm32"))]
            load_tasks: std::array::from_fn(|_| None),
            #[cfg(not(target_arch = "wasm32"))]
            pacs_config: crate::pacs::PacsConfig::default(),
            #[cfg(not(target_arch = "wasm32"))]
            pacs_query_state: crate::pacs::QueryState::Idle,
            #[cfg(not(target_arch = "wasm32"))]
            show_pacs_panel: false,
            #[cfg(not(target_arch = "wasm32"))]
            pacs_patient_filter: String::new(),
            #[cfg(not(target_arch = "wasm32"))]
            pacs_modality_filter: String::new(),
            #[cfg(not(target_arch = "wasm32"))]
            pacs_study_date_filter: String::new(),
            #[cfg(not(target_arch = "wasm32"))]
            pacs_accession_filter: String::new(),
            #[cfg(not(target_arch = "wasm32"))]
            pacs_echo_display: String::new(),
            #[cfg(not(target_arch = "wasm32"))]
            pacs_selected_row: None,
            #[cfg(not(target_arch = "wasm32"))]
            pacs_selected_series_row: None,
            #[cfg(not(target_arch = "wasm32"))]
            pacs_study_context_uid: String::new(),
            #[cfg(not(target_arch = "wasm32"))]
            pacs_worker: None,
            #[cfg(not(target_arch = "wasm32"))]
            pacs_scp_handle: None,
            #[cfg(not(target_arch = "wasm32"))]
            pacs_received_count: 0,
            #[cfg(not(target_arch = "wasm32"))]
            pacs_pending_instances: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            pacs_auto_loaded_this_frame: None,
            #[cfg(not(target_arch = "wasm32"))]
            status_axis: 0,
            #[cfg(not(target_arch = "wasm32"))]
            gpu_renderer: crate::render::gpu_volume::GpuVolumeRenderer::try_create(),
        }
    }
}

impl SnapApp {
    /// Advance the render revision after a state transition changes pixels.
    pub(crate) fn bump_visual_revision(&mut self) {
        self.visual_revision = self.visual_revision.saturating_add(1);
    }

    /// Construct an app that loads `path` on the first update cycle.
    ///
    /// Directory paths are scanned immediately so the series browser is
    /// populated before the deferred volume load runs. When a series UID is
    /// supplied, its discovered acquisition is queued directly; otherwise the
    /// path is queued for the ordinary format loader.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn with_initial_path(
        path: std::path::PathBuf,
        initial_series_uid: Option<String>,
    ) -> Self {
        let mut app = Self::default();
        let is_dicom_input = crate::dicom::classify_dicom_input_path(&path)
            .dicom_root()
            .is_some();
        if is_dicom_input {
            app.scan_for_series(path.clone());
        }
        match initial_series_uid {
            Some(series_uid) if is_dicom_input => {
                if let Some(entry) = app.series_tree.find_by_uid(&series_uid) {
                    app.status_message = format!(
                        "Queued initial series {} from {}",
                        series_uid,
                        path.display()
                    );
                    app.pending_load = Some(VolumeInput::Series(std::sync::Arc::clone(
                        &entry.acquisition,
                    )));
                } else {
                    app.status_message = format!(
                        "Initial SeriesInstanceUID not found in {}: {}",
                        path.display(),
                        series_uid
                    );
                }
            }
            Some(_) => {
                app.status_message = format!(
                    "Initial SeriesInstanceUID requires a DICOM input: {}",
                    path.display()
                );
            }
            None => {
                app.status_message = format!("Queued initial load: {}", path.display());
                app.pending_load = Some(VolumeInput::Path(path));
            }
        }
        app
    }
}
