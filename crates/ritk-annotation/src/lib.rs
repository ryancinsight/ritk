//! ITK-SNAP workflow annotation primitives.
//!
//! Provides the data structures for interactive segmentation workflows:
//! label tables, dense label maps, seed/contour annotations, undo/redo history,
//! and overlay composition state.
//!
//! # Module Structure
//! - [`color`]: `RgbaBytes` and `RgbaLinear` â€” RGBA color newtypes.
//! - [`label_table`]: `LabelTable` and `LabelEntry` â€” display properties per label.
//! - [`label_map`]: `LabelMap` â€” dense 3-D volume of integer label IDs.
//! - [`annotation_state`]: `AnnotationState` â€” seed points, contours, polylines.
//! - [`undo_redo`]: `UndoRedoStack<S>` â€” generic command-pattern undo/redo history.
//! - [`overlay`]: `OverlayState` â€” composite image/contour/mask overlay layers.
//! - [`error`]: `AnnotationError` â€” typed errors for annotation operations.

pub mod annotation_state;
pub mod color;
pub mod error;
pub mod label_map;
pub mod label_table;
pub mod overlay;
pub mod types;
pub mod undo_redo;

pub use annotation_state::{AnnotationState, PointAnnotation};
pub use color::{RgbaBytes, RgbaLinear};
pub use error::AnnotationError;
pub use label_map::LabelMap;
pub use label_table::{LabelEntry, LabelTable};
pub use overlay::{
    Colormap, ContourOverlay, ImageOverlay, MaskOverlay, Opacity, OverlayState, Visibility,
};
pub use types::LabelId;
pub use undo_redo::UndoRedoStack;
