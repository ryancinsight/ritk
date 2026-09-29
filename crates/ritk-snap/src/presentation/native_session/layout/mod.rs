//! Native framebuffer layout and pixel composition for the viewer session.

mod composition;
mod crosshair;
mod geometry;
mod measurement;
mod overlay;
mod responsive;
pub(super) mod text;

pub(super) use composition::{
    surface_frames, surface_frames_with_oblique, surface_frames_with_projection,
};
#[cfg(test)]
pub(super) use crosshair::CROSSHAIR_COLOR;
pub(super) use crosshair::{crosshair_overlay, oblique_crosshair_overlay};
pub(super) use geometry::NativeViewport;
#[cfg(test)]
pub(crate) use geometry::screen_coordinate;
pub(crate) use measurement::patient_measurement_overlay;
#[cfg(test)]
pub(super) use overlay::{
    OVERLAY_BAR_HEIGHT, OVERLAY_TEXT, application_overlay, projection_overlay,
};
pub(super) use responsive::surface_frames_responsive;
