//! Viewer interaction tools.
//!
//! Two sub-modules:
//! - [`kind`]        — [`ToolKind`] discriminant enum (toolbar items).
//! - [`interaction`] — per-viewport tool state machine and completed
//!                     [`Annotation`] types with their computation functions.

pub mod interaction;
pub mod kind;

pub use interaction::{Annotation, RoiKind, ToolState};
pub use kind::ToolKind;
