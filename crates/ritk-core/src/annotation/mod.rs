//! ITK-SNAP workflow annotation primitives.
//!
//! Provides the data structures for interactive segmentation workflows:
//! label tables, dense label maps, seed/contour annotations, undo/redo history,
//! and overlay composition state.
//!
//! # Module Structure
//! - [`label_table`]: `LabelTable` and `LabelEntry` — display properties per label.
//! - [`label_map`]: `LabelMap` — dense 3-D volume of integer label IDs.
//! - [`annotation_state`]: `AnnotationState` — seed points, contours, polylines.
//! - [`undo_redo`]: `UndoRedoStack<S>` — generic command-pattern undo/redo history.
//! - [`overlay`]: `OverlayState` — composite image/contour/mask overlay layers.

pub mod annotation_state;
pub mod label_map;
pub mod label_table;
pub mod overlay;
pub mod undo_redo;

pub use annotation_state::{AnnotationState, PointAnnotation};
pub use label_map::LabelMap;
pub use label_table::{LabelEntry, LabelTable};
pub use overlay::{Colormap, ContourOverlay, ImageOverlay, MaskOverlay, OverlayState};
pub use undo_redo::UndoRedoStack;
