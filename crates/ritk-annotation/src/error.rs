//! Typed errors for the annotation subsystem.

use thiserror::Error;

use crate::types::LabelId;

/// Errors produced by annotation operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AnnotationError {
    /// A path (contour or polyline) requires at least 2 points.
    #[error("{kind} requires >= 2 points, got {count}")]
    TooFewPoints {
        /// `"contour"` or `"polyline"`
        kind: &'static str,
        /// Actual number of points supplied.
        count: usize,
    },
    /// A label table already holds an entry with this ID.
    #[error("label id {id} already exists in the table")]
    DuplicateLabel {
        /// The repeated ID.
        id: LabelId,
    },
}
