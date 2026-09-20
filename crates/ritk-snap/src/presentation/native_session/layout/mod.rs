//! Native framebuffer layout and pixel composition for the viewer session.

mod composition;
mod crosshair;
mod geometry;

#[cfg(test)]
pub(super) use composition::{
    application_overlay, projection_overlay, OVERLAY_BAR_HEIGHT, OVERLAY_TEXT,
};
pub(super) use composition::{surface_frames, surface_frames_with_projection};
pub(super) use crosshair::crosshair_overlay;
#[cfg(test)]
pub(super) use crosshair::CROSSHAIR_COLOR;
pub(super) use geometry::NativeViewport;
