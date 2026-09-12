//! Native framebuffer layout and pixel composition for the viewer session.

mod composition;
mod geometry;

pub(super) use composition::{surface_frames, surface_frames_with_mip};
#[cfg(test)]
pub(super) use composition::{OVERLAY_BAR_HEIGHT, OVERLAY_TEXT};
pub(super) use geometry::NativeViewport;
