//! Tool interaction state and measurement annotation types.
//!
//! Two orthogonal concerns:
//!
//! 1. **In-progress state** — [`ToolState`] tracks the partial interaction for
//!    the currently active tool. Held in memory only; never persisted.
//! 2. **Completed annotations** — [`Annotation`] stores the result of a
//!    finished measurement with computed values. Serialisable; may be saved
//!    to a session file.
//!
//! See sub-module documentation for mathematical specifications.

mod annotation;
mod tool_state;

#[cfg(test)]
mod tests;

pub use annotation::Annotation;
pub use tool_state::{RoiKind, ToolState};
