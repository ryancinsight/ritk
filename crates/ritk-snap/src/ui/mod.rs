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
//! | [`mpr_cursor`]   | [`LinkedCursor`] and viewport/voxel transforms.         |
//! | [`cine`]         | [`CinePlayback`] — per-frame playback timing state.     |
//! | [`cursor_info`]  | [`voxel_to_lps`] — ITK affine voxel → LPS mm transform. |
//! | [`zoom`]         | Scroll-wheel zoom policy and clamped zoom mapping.       |
//! | [`window_presets`] | [`WindowPreset`] with standard CT/MR presets.         |

pub mod layout;
pub mod measurements;
pub mod cine;
pub mod cursor_info;
pub mod mpr_cursor;
pub mod overlay;
pub mod sidebar;
pub mod toolbar;
pub mod viewport;
pub mod window_presets;
pub mod zoom;

pub use cine::CinePlayback;
pub use cursor_info::{format_lps, voxel_to_lps};
pub use layout::{LayoutMode, ViewportId};
pub use measurements::MeasurementLayer;
pub use mpr_cursor::{axis_slice_dimensions, map_view_row_col_to_voxel, viewport_point_to_voxel, LinkedCursor};
pub use overlay::OverlayRenderer;
pub use sidebar::SidebarPanel;
pub use toolbar::{ToolbarPanel, ToolbarState};
pub use viewport::{ViewportPanel, ViewportState};
pub use window_presets::WindowPreset;
pub use zoom::{should_zoom_with_scroll, zoom_from_scroll, MAX_ZOOM, MIN_ZOOM};
