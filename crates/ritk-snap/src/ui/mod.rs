//! Viewer UI components.
//!
//! # Sub-modules
//!
//! | Module           | Contents                                                |
//! |------------------|---------------------------------------------------------|
//! | [`layout`]       | [`LayoutMode`] and [`ViewportId`] enumerations.        |
//! | [`viewport`]     | [`ViewportState`] and [`ViewportPanel`] — the core MPR  |
//! |                  | slice display widget.                                   |
//! | [`toolbar`]      | [`ToolbarState`] and [`ToolbarPanel`] — top toolbar.    |
//! | [`sidebar`]      | [`SidebarPanel`] — series browser + metadata tab.       |
//! | [`overlay`]      | [`OverlayRenderer`] — DICOM 4-corner text overlays.     |
//! | [`measurements`] | [`MeasurementLayer`] — annotation drawing helpers.     |
//! | [`tool_shortcuts`] | Tool keyboard shortcuts SSOT (L=length, A=angle, etc.). |
//! | [`mpr_cursor`]   | [`LinkedCursor`] and viewport/voxel transforms.         |
//! | [`cine`]         | [`CinePlayback`] — per-frame playback timing state.     |
//! | [`cursor_info`]  | [`voxel_to_lps`] — ITK affine voxel → LPS mm transform. |
//! | [`pointer_intensity`] | [`intensity_at_voxel`] — voxel intensity lookup SSOT. |//! | [`live_preview`]     | [`live_length_mm`] and [`live_angle_deg`] — live measurement labels. |
//! | [`histogram`]    | [`draw_histogram`] — voxel intensity histogram + W/L range overlay.  |
//! | [`histogram_interact`] | [`x_to_intensity`], [`wl_from_histogram_drag`], [`wl_center_from_click`] — histogram canvas interaction SSOT. |
//! | [`preset_panel`] | [`draw_preset_buttons`] — W/L preset quick-select button strip SSOT.  |
//! | [`export_plan`]  | Deterministic all-axis MPR PNG export planning.          |
//! | [`rtstruct_overlay`] | RT-STRUCT patient-space contour projection.          |
//! | [`pan`]          | Pan drag mapping SSOT for viewport offset updates.      |
//! | [`zoom`]         | Scroll-wheel zoom policy and clamped zoom mapping.       |
//! | [`window_level`] | W/L drag mapping SSOT and sensitivity constant.          |
//! | [`window_presets`] | [`WindowPreset`] with standard CT/MR presets.         |

pub mod histogram;
pub mod histogram_interact;
pub mod preset_panel;
pub mod layout;
pub mod measurements;
pub mod cine;
pub mod cursor_info;
pub mod export_plan;
pub mod live_preview;
pub mod mpr_cursor;
pub mod overlay;
pub mod pan;
pub mod pointer_intensity;
pub mod rtstruct_overlay;
pub mod sidebar;
pub mod toolbar;
pub mod tool_shortcuts;
pub mod viewport;
pub mod window_level;
pub mod window_presets;
pub mod zoom;

pub use cine::CinePlayback;
pub use cursor_info::{format_lps, voxel_to_lps};
pub use export_plan::{axis_folder_name, axis_slice_total, plan_all_mpr_exports, PlannedSliceExport};
pub use layout::{LayoutMode, ViewportId};
pub use measurements::MeasurementLayer;
pub use mpr_cursor::{axis_slice_dimensions, map_view_row_col_to_voxel, viewport_point_to_voxel, LinkedCursor};
pub use overlay::OverlayRenderer;
pub use pan::pan_from_drag_delta;
pub use pointer_intensity::intensity_at_voxel;
pub use preset_panel::draw_preset_buttons;
pub use rtstruct_overlay::{project_rt_struct_contours_for_slice, ProjectedRtContour};
pub use sidebar::SidebarPanel;
pub use toolbar::{ToolbarPanel, ToolbarState};
pub use tool_shortcuts::tool_kind_for_key;
pub use viewport::{ViewportPanel, ViewportState};
pub use window_level::{clamp_window_width, window_level_from_drag_delta, WINDOW_LEVEL_SENSITIVITY};
pub use window_presets::WindowPreset;
pub use zoom::{
	fit_view_transform, should_zoom_with_scroll, zoom_from_drag_delta, zoom_from_scroll, FIT_ZOOM,
	MAX_ZOOM, MIN_ZOOM,
};
pub use window_level::MIN_WINDOW_WIDTH;
