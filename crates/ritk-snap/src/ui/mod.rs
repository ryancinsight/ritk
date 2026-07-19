//! Viewer UI components.
//!
//! # Sub-modules
//!
//! | Module           | Contents                                                |
//! |------------------|---------------------------------------------------------|
//! | [`layout`]       | [`LayoutMode`] and [`ViewportId`] enumerations.        |
//! | [`viewport`]     | [`ViewportState`] and [`ViewportPanel`] â€” the core MPR  |
//! |                  | slice display widget.                                   |
//! | [`toolbar`]      | [`ToolbarState`] and [`ToolbarPanel`] â€” top toolbar.    |
//! | [`sidebar`]      | [`SidebarPanel`] â€” series browser + metadata tab.       |
//! | [`overlay`]      | [`OverlayRenderer`] â€” DICOM 4-corner text overlays.     |
//! | [`measurements`] | [`MeasurementLayer`] â€” annotation drawing helpers.     |
//! | [`tool_shortcuts`] | Tool keyboard shortcuts SSOT (L=length, A=angle, etc.). |
//! | [`mpr_cursor`]   | [`LinkedCursor`] and viewport/voxel transforms.         |
//! | [`cine`]         | [`CinePlayback`] â€” per-frame playback timing state.     |
//! | [`cursor_info`]  | [`voxel_to_lps`] â€” ITK affine voxel â†’ LPS mm transform. |
//! | [`pointer_intensity`] | [`intensity_at_voxel`] â€” voxel intensity lookup SSOT. |
//! | [`live_preview`] | `live_length_mm` and `live_angle_deg` â€” live measurement labels. |
//! | [`histogram`] | `draw_histogram` â€” voxel intensity histogram + W/L range overlay. |
//! | [`histogram_interact`] | `x_to_intensity`, `wl_from_histogram_drag`, `wl_center_from_click` â€” histogram canvas interaction SSOT. |
//! | [`preset_panel`] | [`draw_preset_buttons`] â€” W/L preset quick-select button strip SSOT.  |
//! | [`annotation_panel`] | [`draw_annotation_panel`] â€” per-entry delete and CSV export SSOT.    |
//! | [`export_plan`]  | Deterministic all-axis MPR PNG export planning.          |
//! | [`rtstruct_overlay`] | RT-STRUCT patient-space contour projection.          |
//! | [`rtdose_overlay`]   | RT-DOSE grid slice projection and dose-colormap mapping.|
//! | [`rtdose_texture`]   | RT-DOSE scalar-to-texture colorization helpers.      |
//! | [`filter_panel`]     | Image processing filter selection panel for viewer.    |
//! | [`pan`]          | Pan drag mapping SSOT for viewport offset updates.      |
//! | [`zoom`]         | Scroll-wheel zoom policy and clamped zoom mapping.       |
//! | [`window_level`] | W/L drag mapping SSOT and sensitivity constant.          |
//! | [`window_presets`] | [`WindowPreset`] with standard CT/MR presets.         |
//! | [`view_transform`] | [`ViewTransform`] viewport flip/rotate state + pixel transforms. |
//! | [`colorbar`]     | [`draw_colorbar`] / [`show_colorbar`] â€” W/L colorbar widget.     |
//! | [`dropped_input`] | Dropped-file routing SSOT for app-shell ingestion decisions. |

pub mod anatomical_plane;
pub mod annotation_panel;
pub mod cine;
pub mod colorbar;
pub mod cursor_info;
pub mod dropped_input;
pub mod export_plan;
pub mod filter_panel;
pub mod histogram;
pub mod histogram_interact;
pub mod layout;
pub mod live_preview;
pub mod measurements;
pub mod mpr_cursor;
pub mod overlay;
pub mod pacs_panel;
pub mod pan;
pub mod pet_suv_panel;
pub mod pointer_intensity;
pub mod preset_panel;
pub mod rt_dose_analytics;
pub mod rtdose_overlay;
pub mod rtdose_texture;
pub mod rtstruct_overlay;
pub mod sidebar;
pub mod slice_navigation;
pub mod tool_shortcuts;
pub mod toolbar;
pub mod view_transform;
pub mod viewport;
pub mod window_level;
pub mod window_presets;
pub mod zoom;

pub use anatomical_plane::{anatomical_label_for_axis, axis_for_plane_in_volume, AnatomicalPlane};
pub use annotation_panel::{draw_annotation_panel, AnnotationPanelAction};
pub use cine::CinePlayback;
pub use colorbar::{draw_colorbar, show_colorbar, COLORBAR_PANEL_WIDTH, COLORBAR_WIDTH};
pub use cursor_info::{format_lps, voxel_to_lps};
pub use dropped_input::{decide_dropped_input_action, DroppedInputAction};
pub use export_plan::{
    axis_folder_name, axis_slice_total, plan_all_mpr_exports, PlannedSliceExport,
};
pub use layout::{LayoutMode, ViewportId};
pub use measurements::MeasurementLayer;
pub use mpr_cursor::{
    axis_slice_dimensions, map_view_row_col_to_voxel, map_voxel_to_view_row_col,
    viewport_point_to_voxel, LinkedCursor,
};
pub use overlay::OverlayRenderer;
pub use pan::pan_from_drag_delta;
pub use pet_suv_panel::{draw_pet_suv_panel, PetSuvPanelAction};
pub use pointer_intensity::intensity_at_voxel;
pub use preset_panel::draw_preset_buttons;
pub use rt_dose_analytics::{
    compute_roi_dose_analytics, draw_dvh_curve, RoiDoseAnalytics, VolumeGeometry,
};
pub use rtstruct_overlay::{project_rt_struct_contours_for_slice, ProjectedRtContour};
pub use sidebar::SidebarPanel;
pub use slice_navigation::{advance_wrapped, axis_total, clamp_index, step_clamped};
pub use tool_shortcuts::tool_kind_for_key;
pub use toolbar::{ToolbarPanel, ToolbarState};
pub(crate) use view_transform::apply_to_image_into;
pub use view_transform::{
    apply_to_image, flip_h_image, flip_v_image, rotate_90_cw_image, RotationSteps, ViewTransform,
};
pub use viewport::{ViewportPanel, ViewportState};
pub use window_level::MIN_WINDOW_WIDTH;
pub use window_level::{
    clamp_window_width, window_level_from_drag_delta, WINDOW_LEVEL_SENSITIVITY,
};
pub use window_presets::WindowPreset;
pub use zoom::{
    fit_view_transform, should_zoom_with_scroll, zoom_from_drag_delta, zoom_from_scroll, FIT_ZOOM,
    MAX_ZOOM, MIN_ZOOM,
};
