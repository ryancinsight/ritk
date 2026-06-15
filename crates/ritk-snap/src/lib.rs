#![allow(
    clippy::too_many_arguments,
    clippy::field_reassign_with_default, // stylistic; test code patterns
)]

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
pub mod render;
pub mod session;
pub mod tools;
pub mod ui;
pub mod viewer;

// Re-export flat API surface so downstream crates don't need path changes.
pub use filter::{BedSeparationConfigSerde, FilterKind};
pub use geometry::{GeometrySummary, ModalityDisplay, ViewerResult, ViewerStatus};
#[cfg(target_arch = "wasm32")]
pub use launch::start_web;
pub use launch::{run_app, run_app_with_options, AppLaunchOptions};
pub use loaded_volume::LoadedVolume;
pub use viewer::{DefaultBackend, Study, ViewerBackend, ViewerCore, ViewerEvent, ViewerState};

#[cfg(test)]
#[path = "tests_lib.rs"]
mod tests;
