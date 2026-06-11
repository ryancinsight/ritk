//! ITK-SNAP workflow annotation primitives.
//!
//! Provides the data structures for interactive segmentation workflows:
//! label tables, dense label maps, seed/contour annotations, undo/redo history,
//! and overlay composition state.
//!
//! # Module Structure
//! - [`color`]: `RgbaU8` and `RgbaF32` — RGBA color newtypes.
//! - [`label_table`]: `LabelTable` and `LabelEntry` — display properties per label.
//! - [`label_map`]: `LabelMap` — dense 3-D volume of integer label IDs.
//! - [`annotation_state`]: `AnnotationState` — seed points, contours, polylines.
//! - [`undo_redo`]: `UndoRedoStack<S>` — generic command-pattern undo/redo history.
//! - [`overlay`]: `OverlayState` — composite image/contour/mask overlay layers.
//! - [`error`]: `AnnotationError` — typed errors for annotation operations.

pub mod annotation_state;
pub mod color;
pub mod error;
pub mod label_map;
pub mod label_table;
pub mod overlay;
pub mod types;
pub mod undo_redo;

pub use annotation_state::{AnnotationState, PointAnnotation};
pub use color::{RgbaF32, RgbaU8};
pub use error::AnnotationError;
pub use label_map::LabelMap;
pub use label_table::{LabelEntry, LabelTable};
pub use overlay::{
    Colormap, ContourOverlay, ImageOverlay, MaskOverlay, Opacity, OverlayState, Visibility,
};
pub use types::LabelId;
pub use undo_redo::UndoRedoStack;
