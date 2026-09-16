//! ritk-snap viewer application state and desktop-shell adapters.
//!
//! Owns the top-level `SnapApp` struct and its eframe shell wrapper. All
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
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod clinical_distribution;
#[cfg(not(target_arch = "wasm32"))]
mod eframe;
#[cfg(not(target_arch = "wasm32"))]
mod filter;
#[cfg(not(target_arch = "wasm32"))]
mod image_placement;
#[cfg(not(target_arch = "wasm32"))]
mod io_ops;
#[cfg(not(target_arch = "wasm32"))]
mod load_tasks;
#[cfg(not(target_arch = "wasm32"))]
mod menu;
#[cfg(not(target_arch = "wasm32"))]
mod mesh_ops;
#[cfg(not(target_arch = "wasm32"))]
mod pacs_ops;
mod panels;
mod pointer_ops;
#[cfg(not(target_arch = "wasm32"))]
mod render_cache;
#[cfg(not(target_arch = "wasm32"))]
mod rt_overlay;
#[cfg(not(target_arch = "wasm32"))]
mod rt_struct_export;
#[cfg(not(target_arch = "wasm32"))]
mod shortcuts;
mod slice_ops;
pub(crate) mod state;
#[cfg(not(target_arch = "wasm32"))]
mod surface_export;
#[cfg(not(target_arch = "wasm32"))]
mod toolbar;
#[cfg(not(target_arch = "wasm32"))]
mod viewport;
#[cfg(not(target_arch = "wasm32"))]
mod viewport_compare;
#[cfg(not(target_arch = "wasm32"))]
mod viewport_render;
#[cfg(any(test, not(target_arch = "wasm32")))]
mod volume_input;
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

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use eframe::EguiApp;
#[cfg(windows)]
pub(crate) use slice_ops::CineTick;
pub(crate) use state::SnapApp;

#[cfg(target_arch = "wasm32")]
pub(crate) use web_viewer::{
    select_web_slice, start_web_canvas, start_web_canvas_gpu, start_web_orthogonal_canvases,
    start_web_orthogonal_canvases_gpu, stop_web_canvas, web_canvas_listener_count,
};
