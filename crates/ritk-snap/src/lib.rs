//! `ritk-snap` viewer core.
//!
//! This crate defines the viewer domain model and backend abstraction for
//! DICOM and other medical image studies. It does not perform I/O itself;
//! loading is delegated to `ritk-io` or another data source.
//!
//! The design goal is to keep the viewer frontend/backend split explicit:
//! - core state and navigation live here,
//! - rendering and presentation live behind a backend trait,
//! - DICOM/volume loading remains in `ritk-io`.
//!
//! This crate is intended to support multiple presentation targets, including
//! native desktop and web-backed shells, without duplicating viewer logic.
//!
//! Geometry handling is modality-aware at the summary layer:
//! - CT summaries may be derived from DICOM spatial metadata or loaded image geometry.
//! - MRI summaries preserve the same affine contract but do not assume CT-specific table/bed semantics.
//! - Ultrasound summaries must respect acquisition-specific orientation metadata and may not use CT-only display heuristics.

pub mod app;
pub mod dicom;
pub mod filter;
pub mod geometry;
pub mod label;
pub mod launch;
pub mod loaded_volume;
pub mod pacs;
pub mod presentation;
pub mod render;
pub mod session;
pub mod tools;
pub mod ui;
pub mod viewer;

// Re-export flat API surface so downstream crates don't need path changes.
pub use filter::{BedSeparationConfigSerde, FilterKind};
pub use geometry::{GeometrySummary, ModalityDisplay, ViewerResult, ViewerStatus};
#[cfg(not(target_arch = "wasm32"))]
pub use launch::{
    run_app, run_app_with_options, AppLaunchOptions, CompatibilityPresentation,
    NativePresentationMode,
};
#[cfg(target_arch = "wasm32")]
pub use launch::{run_app_with_options, AppLaunchOptions, NativePresentationMode};
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
pub use launch::{
    run_eframe_app, run_eframe_app_with_options, run_eframe_app_with_presentation,
    run_eframe_app_with_viewport, EframeViewport, EframeViewportError,
};
#[cfg(target_arch = "wasm32")]
pub use launch::{
    select_web_slice, select_web_tool, set_web_cine_rate, set_web_window_preset, start_web,
    start_web_canvas, start_web_canvas_gpu, start_web_orthogonal_canvases,
    start_web_orthogonal_canvases_gpu, stop_web_canvas, toggle_web_cine, web_canvas_listener_count,
    web_tool_count, web_tool_name, web_window_preset_count, web_window_preset_name,
};
pub use loaded_volume::LoadedVolume;
pub use viewer::ViewerState;

#[cfg(test)]
#[path = "tests_lib.rs"]
mod tests;
