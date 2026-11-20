//! Default RITK viewer state construction.
//!
//! The [Default] implementation lives beside the state definition without
//! growing it past the file budget; [SnapApp] and its helper types stay in
//! [super::state].

use super::state::SnapApp;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
use super::state::{ProjectionBackend, ProjectionMode, SeriesLoadTarget, DEFAULT_FUSION_ALPHA};
use crate::presentation::PresentationDispatcher;
use crate::render::NamedColorMap;
use crate::tools::interaction::{ToolState, ViewportOffset};
use crate::tools::kind::ToolKind;
use crate::ui::{CinePlayback, ViewTransform};
use crate::ViewerState;

impl Default for SnapApp {
    fn default() -> Self {
        Self {
            loaded: None,
            measurement_pixel_spacing: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            loaded_secondary: None,
            viewer_state: ViewerState::new(),
            visual_revision: 0,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            secondary_window_center: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            secondary_window_width: None,
            colormap: NamedColorMap::Grayscale,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            secondary_colormap: NamedColorMap::Grayscale,
            axis: 0,
            active_tool: ToolKind::WindowLevel,
            tool_state: ToolState::Idle,
            pointer: Default::default(),
            presentation_dispatcher: PresentationDispatcher::new(),
            annotations: Vec::new(),
            #[cfg(windows)]
            length_calibrations: Vec::new(),
            label_editor: None,
            label_brush_radius: 1,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            show_label_overlay: true,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            show_rt_struct_overlay: true,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            rt_struct: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            rt_dose: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            rt_dose_max_gy: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            rt_plan: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            rt_dvh_selected_roi: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            rt_dvh_cache: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            show_rt_dose_overlay: false,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            rt_dose_opacity: 0.5,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            active_filter: crate::FilterKind::Gaussian { sigma: 1.0 },
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            show_filter_panel: false,
            coronal_slice: 0,
            sagittal_slice: 0,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            projection_mode: ProjectionMode::Mip,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            projection_backend: ProjectionBackend::Cpu,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            loaded_mesh: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            show_mesh_overlay: false,
            pan_offset: ViewportOffset::new(0.0, 0.0),
            zoom: 1.0,
            view_transform: ViewTransform::default(),
            #[cfg(windows)]
            invert_display: false,
            #[cfg(windows)]
            custom_presets: [None; super::state::MAX_CUSTOM_PRESETS],
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            show_colorbar: false,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            multi_planar: false,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            dual_plane: false,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            compare_side_by_side: false,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            compare_fused_overlay: false,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            compare_fusion_alpha: DEFAULT_FUSION_ALPHA,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            dual_axes: [0, 1],
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            compare_axes: [0, 0],
            #[cfg(any(windows, all(not(target_arch = "wasm32"), feature = "eframe-shell")))]
            show_overlay: true,
            show_crosshair: false,
            linked_cursor: None,
            cine: CinePlayback::default(),
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            show_series_browser: true,
            pointer_intensity: 0.0,
            pointer_suv: None,
            cached_histogram: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            series_tree: crate::dicom::series_tree::SeriesTree::new(),
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            selected_series: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            sidebar_tab: crate::ui::sidebar::SidebarTab::Series,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            series_load_target: SeriesLoadTarget::Primary,
            status_message: "No study loaded — use File > Open to load a DICOM folder.".to_owned(),
            #[cfg(not(target_arch = "wasm32"))]
            pending_load: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pending_secondary_load: None,
            #[cfg(not(target_arch = "wasm32"))]
            load_generations: [0; 2],
            #[cfg(not(target_arch = "wasm32"))]
            load_tasks: std::array::from_fn(|_| None),
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_config: crate::pacs::PacsConfig::default(),
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_query_state: crate::pacs::QueryState::Idle,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            show_pacs_panel: false,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_patient_filter: String::new(),
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_modality_filter: String::new(),
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_study_date_filter: String::new(),
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_accession_filter: String::new(),
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_echo_display: String::new(),
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_selected_row: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_selected_series_row: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_study_context_uid: String::new(),
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_worker: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_scp_handle: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_received_count: 0,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_pending_instances: Vec::new(),
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            pacs_auto_loaded_this_frame: None,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            status_axis: 0,
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            gpu_renderer: crate::render::gpu_volume::GpuVolumeRenderer::try_create(),
        }
    }
}
