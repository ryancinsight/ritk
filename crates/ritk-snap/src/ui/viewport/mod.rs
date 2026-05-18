//! Per-viewport MPR slice display widget.
//!
//! See sub-modules for details:
//! - [`state`] — [`ViewportRenderMode`], [`ViewportState`], and [`ViewportPanel`] struct.
//! - [`panel`] — [`ViewportPanel`] methods and private helpers.
//! - [`tests`] — unit tests.

mod panel;
mod state;
#[cfg(test)]
mod tests;

pub use state::{ViewportPanel, ViewportRenderMode, ViewportState};
