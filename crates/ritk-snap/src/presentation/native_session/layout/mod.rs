//! Native framebuffer layout and pixel composition for the viewer session.

mod compare;
mod composition;
mod crosshair;
mod geometry;
mod overlay;
mod overlay_clinical;
pub(crate) mod overlay_display;
mod overlay_measurements;
mod reference;
mod responsive;
pub(super) use reference::reference_segment;
pub(super) mod text;

pub(super) use compare::{
    panel_header_action_at, surface_frames_grid, GridPanel, PanelGrid, PanelHeaderAction,
    WorkspaceLayout, MAX_COMPARISON_PANELS, MAX_GRID_COLUMNS, MAX_GRID_PANELS, MAX_GRID_ROWS,
};
pub(super) use composition::{surface_frames, surface_frames_with_projection};
#[cfg(test)]
pub(super) use crosshair::CROSSHAIR_COLOR;
pub(super) use crosshair::{crosshair_overlay, screen_coordinate};
#[cfg(test)]
pub(super) use geometry::VIEW_GAP_PIXELS;
pub(super) use geometry::{NativeViewport, ViewportArea};
pub(super) use overlay::{application_overlay, ClinicalOverlay, OverlayCommands};
#[cfg(test)]
pub(super) use overlay::{
    clinical_labels_for_view, projection_overlay, OVERLAY_BAR_HEIGHT, OVERLAY_TEXT,
};
#[cfg(test)]
pub(super) use overlay_clinical::scale_bar_for_scale;
pub(super) use overlay_measurements::append_measurement_overlay;
#[cfg(test)]
pub(super) use overlay_measurements::append_measurement_preview;
pub(super) use responsive::surface_frames_responsive;
pub(super) use text::{draw_text, text_style, TextStyle};
