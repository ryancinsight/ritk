//! ritk-snap native application shell (eframe/egui backend).
//!
//! Owns the top-level [`SnapApp`] struct and its [`eframe::App`]
//! implementation. All domain logic (intensity mapping, slice extraction,
//! annotation computation) lives in the `render` and `tools` sub-modules;
//! this module wires events, drives state transitions, and builds the egui
//! widget tree.
//!
//! # Layout modes
//!
//! | `multi_planar` | Layout |
//! |----------------|---------------------------------------------|
//! | `false`        | Single viewport — current axis fills panel. |
//! | `true`         | 2×2 grid: Axial / Coronal / Sagittal / 3D-MIP, with Info below.|

mod filter;
mod io_ops;
mod menu;
mod panels;
mod pointer_ops;
mod render_cache;
mod rt_overlay;
mod shortcuts;
mod slice_ops;
pub(crate) mod state;
mod surface_export;
mod toolbar;
mod viewport;
mod viewport_render;
mod volume_ops;
mod volume_state;

#[cfg(test)]
mod tests;

pub(crate) use state::SnapApp;
