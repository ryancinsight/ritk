//! ritk-snap viewer application state and desktop-shell adapters.
//!
//! Owns the top-level `SnapApp` struct and optional desktop-shell adapters. All
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
#[cfg(any(target_arch = "wasm32", test))]
mod browser_geometry;
#[cfg(any(target_arch = "wasm32", test))]
mod browser_semantics;
#[cfg(any(target_arch = "wasm32", test))]
mod browser_slice_selection;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
pub(crate) mod clinical_distribution;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod eframe;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod filter;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod image_placement;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod io_ops;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod load_tasks;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod menu;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod mesh_ops;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod pacs_ops;
#[cfg(feature = "eframe-shell")]
mod panels;
mod pointer_ops;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod render_cache;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod rt_overlay;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod rt_struct_export;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod shortcuts;
mod slice_ops;
pub(crate) mod state;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod surface_export;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod toolbar;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod viewport;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod viewport_compare;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod viewport_render;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod volume_input;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
mod volume_ops;
mod volume_state;

#[cfg(target_arch = "wasm32")]
mod browser_canvas;
#[cfg(target_arch = "wasm32")]
mod browser_input;
#[cfg(target_arch = "wasm32")]
mod web_viewer;

#[cfg(test)]
mod tests;

#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
pub(crate) use eframe::EguiApp;
#[cfg(windows)]
pub(crate) use slice_ops::CineTick;
pub(crate) use state::SnapApp;

#[cfg(target_arch = "wasm32")]
pub(crate) use web_viewer::{
    select_web_slice, start_web_canvas, start_web_canvas_gpu, start_web_orthogonal_canvases,
    start_web_orthogonal_canvases_gpu, stop_web_canvas, web_canvas_listener_count,
};
