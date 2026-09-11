//! ritk-snap viewer application state and desktop-shell adapters.
//!
//! Owns the top-level `SnapApp` struct and its eframe implementation. All
//! domain logic (intensity mapping, slice extraction, annotation computation)
//! lives in the `render` and `tools` sub-modules; this module wires events and
//! drives state transitions. The format-neutral presentation module supplies
//! the Métis native host without moving DICOM or viewer state into that host.
//!
//! # Layout modes
//!
//! | `multi_planar` | Layout |
//! |----------------|---------------------------------------------|
//! | `false`        | Single viewport — current axis fills panel. |
//! | `true`         | 2×2 grid: Axial / Coronal / Sagittal / 3D-MIP, with Info below.|

pub(crate) mod action_adapter;
pub(crate) mod clinical_distribution;
mod filter;
mod image_placement;
mod io_ops;
#[cfg(not(target_arch = "wasm32"))]
mod load_tasks;
mod menu;
mod mesh_ops;
mod pacs_ops;
mod panels;
mod pointer_ops;
mod render_cache;
mod rt_overlay;
mod rt_struct_export;
mod shortcuts;
mod slice_ops;
pub(crate) mod state;
mod surface_export;
mod toolbar;
mod viewport;
mod viewport_compare;
mod viewport_render;
mod volume_input;
mod volume_ops;
mod volume_state;

#[cfg(target_arch = "wasm32")]
mod browser_input;
#[cfg(target_arch = "wasm32")]
mod web_viewer;

#[cfg(test)]
mod tests;

pub(crate) use state::SnapApp;

#[cfg(target_arch = "wasm32")]
pub(crate) use web_viewer::{start_web_canvas, start_web_orthogonal_canvases, stop_web_canvas};
