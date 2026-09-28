//! Native framebuffer layout and pixel composition for the viewer session.

mod compare;
mod composition;
mod crosshair;
mod geometry;
mod overlay;
mod responsive;
pub(super) mod text;

pub(super) use compare::{
    surface_frames_grid, GridPanel, PanelGrid, WorkspaceLayout, MAX_COMPARISON_PANELS,
    MAX_GRID_COLUMNS, MAX_GRID_PANELS, MAX_GRID_ROWS,
};
pub(super) use composition::{surface_frames, surface_frames_with_projection};
pub(super) use crosshair::crosshair_overlay;
#[cfg(test)]
pub(super) use crosshair::CROSSHAIR_COLOR;
#[cfg(test)]
pub(super) use geometry::VIEW_GAP_PIXELS;
pub(super) use geometry::{NativeViewport, ViewportArea};
#[cfg(test)]
pub(super) use overlay::{
    application_overlay, projection_overlay, OVERLAY_BAR_HEIGHT, OVERLAY_TEXT,
};
pub(super) use responsive::surface_frames_responsive;
pub(super) use text::text_style;
