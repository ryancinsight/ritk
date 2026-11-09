//! Viewer UI components.
//!
//! # Sub-modules
//!
//! | Module           | Contents                                                |
//! |------------------|---------------------------------------------------------|
//! | `layout`       | `LayoutMode` and `ViewportId` enumerations.        |
//! | `sidebar`      | `SidebarTab` — active sidebar tab state.              |
//! | `tool_shortcuts` | Tool keyboard shortcuts SSOT (L=length, A=angle, etc.). |
//! | `annotation_csv` | Annotation CSV export and row labels. |
//! | `histogram` | Log-scaled bar height and W/L x-position geometry. |
//! | `measurements` | Measurement annotation geometry. |
//! | `overlay` | Anatomical edge labels and intensity readout strings. |
//! | `pet_suv_readout` | PET acquisition and SUV readout text. |
//! | `rtdose_texture` | RT-DOSE scalar-to-RGBA colorization helpers. |
//! | `mpr_cursor`   | `LinkedCursor` and viewport/voxel transforms.         |
//! | `cine`         | `CinePlayback` — per-frame playback timing state.     |
//! | `coordinate_system` | `AnatomicalFrame`, `lps_to_ras`, `ras_to_lps` — LPS/RAS frame conversion. |
//! | `cursor_info`  | `voxel_to_lps` — ITK affine voxel → LPS mm transform. |
//! | `pointer_intensity` | `intensity_at_voxel` — voxel intensity lookup SSOT. |
//! | `live_preview` | `live_length_mm` and `live_angle_deg` — live measurement labels. |
//! | `histogram_interact` | `x_to_intensity`, `wl_from_histogram_drag`, `wl_center_from_click` — histogram canvas interaction SSOT. |
//! | `export_plan`  | Deterministic all-axis MPR PNG export planning.          |
//! | `rtstruct_overlay` | RT-STRUCT patient-space contour projection.          |
//! | `rtdose_overlay`   | RT-DOSE grid slice projection and dose-colormap mapping.|
//! | `pan`          | Pan drag mapping SSOT for viewport offset updates.      |
//! | `zoom`         | Scroll-wheel zoom policy and clamped zoom mapping.       |
//! | `window_level` | W/L drag mapping SSOT and sensitivity constant.          |
//! | `window_presets` | `WindowPreset` with standard CT/MR presets.         |
//! | `view_transform` | `ViewTransform` viewport flip/rotate state + pixel transforms. |
//! | `dropped_input` | Dropped-file routing SSOT for app-shell ingestion decisions. |

pub mod anatomical_plane;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-EXPORT-001")
)]
pub(crate) mod annotation_csv;
pub mod cine;
pub mod coordinate_system;
pub mod cursor_info;
pub mod dropped_input;
pub mod export_plan;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-DISPLAY-001")
)]
pub(crate) mod histogram;
pub mod histogram_interact;
pub mod layout;
pub mod live_preview;
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "ratchet RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001"
    )
)]
pub(crate) mod measurements;
pub mod mpr_cursor;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-OVERLAY-TEXT-001")
)]
pub(crate) mod overlay;
pub mod pan;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-FUSION-001")
)]
pub(crate) mod pet_suv_readout;
pub mod pointer_intensity;
pub mod rt_dose_analytics;
pub mod rtdose_overlay;
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-RT-001")
)]
pub(crate) mod rtdose_texture;
pub mod rtstruct_overlay;
pub mod sidebar;
pub mod slice_navigation;
pub mod tool_shortcuts;
pub mod view_transform;
pub mod window_level;
pub mod window_presets;
pub mod zoom;

pub use anatomical_plane::{anatomical_label_for_axis, axis_for_plane_in_volume, AnatomicalPlane};
pub use cine::CinePlayback;
pub use coordinate_system::{
    format_point_mm, lps_to_ras, ras_to_lps, AnatomicalFrame, PatientPosition,
};
pub use cursor_info::{format_lps, voxel_to_lps};
pub use dropped_input::{decide_dropped_input_action, DroppedInput, DroppedInputAction};
pub use export_plan::{
    axis_folder_name, axis_slice_total, plan_all_mpr_exports, PlannedSliceExport,
};
pub use layout::{LayoutMode, ViewportId};
pub use mpr_cursor::{
    axis_slice_dimensions, image_point_to_voxel, map_view_row_col_to_voxel,
    map_voxel_to_view_row_col, LinkedCursor,
};
pub use pan::pan_from_drag_delta;
pub use pointer_intensity::intensity_at_voxel;
pub use rt_dose_analytics::{
    compute_roi_dose_analytics, DvhPoint, RoiDoseAnalytics, VolumeGeometry,
};
pub use rtstruct_overlay::{project_rt_struct_contours_for_slice, ProjectedRtContour};
pub use sidebar::SidebarTab;
pub use slice_navigation::{advance_wrapped, axis_total, clamp_index, step_clamped};
pub use tool_shortcuts::tool_kind_for_virtual_key;
pub use view_transform::apply_to_rgba_into;
pub use view_transform::{RotationSteps, ViewTransform};
pub use window_level::MIN_WINDOW_WIDTH;
pub use window_level::{
    clamp_window_width, window_level_from_drag_delta, WINDOW_LEVEL_SENSITIVITY,
};
pub use window_presets::WindowPreset;
pub use zoom::{
    fit_view_transform, should_zoom_with_scroll, zoom_from_drag_delta, zoom_from_scroll, FIT_ZOOM,
    MAX_ZOOM, MIN_ZOOM,
};
