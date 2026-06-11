//! Viewer-side segmentation label editing.
//!
//! This module is the `ritk-snap` application boundary for interactive label
//! editing. It composes the canonical annotation primitives from `ritk-core`
//! instead of duplicating label-map storage, label-table metadata, or undo/redo
//! history inside the viewer crate.
//!
//! # Specification
//!
//! A paint operation applies one label value to the discrete ball
//! `B(c, r) = { p in Z3 | ||p - c||_2^2 <= r^2 }` intersected with the label
//! map domain. Erase is the same operation with label `0`.
//!
//! # Invariants
//! - `LabelEditor::current_map()` is always defined because `UndoRedoStack`
//!   starts with one initial `LabelMap`.
//! - `active_label_id != 0` and exists in the current map's `LabelTable`.
//! - A no-op paint/erase does not create a new undo history entry.
//! - Out-of-bounds brush centers are rejected before touching the label map.

use ritk_core::annotation::{LabelId, LabelMap, LabelTable, UndoRedoStack};
use ritk_core::annotation::{RgbaU8, Visibility};

const DEFAULT_LABEL_ID: LabelId = LabelId(1);
const DEFAULT_LABEL_NAME: &str = "Label 1";
const DEFAULT_LABEL_COLOR: RgbaU8 = RgbaU8([255, 0, 0, 180]);

/// Application-level segmentation editor for a single 3-D label map.
#[derive(Debug, Clone)]
pub struct LabelEditor {
    history: UndoRedoStack<LabelMap>,
    active_label_id: LabelId,
}

impl LabelEditor {
    /// Construct an editor with a background-filled label map and default label.
    pub fn new(shape: [usize; 3]) -> Self {
        let table = default_label_table();
        Self {
            history: UndoRedoStack::new(LabelMap::new(shape, table)),
            active_label_id: DEFAULT_LABEL_ID,
        }
    }

    /// Construct an editor with a caller-provided table and active label.
    ///
    /// Returns `Err` when `active_label_id` is background or is absent from the
    /// provided label table. The label map is initialized as background.
    pub fn with_table(
        shape: [usize; 3],
        table: LabelTable,
        active_label_id: impl Into<LabelId>,
    ) -> Result<Self, String> {
        let active_label_id = active_label_id.into();
        validate_label_exists(&table, active_label_id)?;
        Ok(Self {
            history: UndoRedoStack::new(LabelMap::new(shape, table)),
            active_label_id,
        })
    }

    /// Construct an editor from an existing label map, setting the active label
    /// to the smallest non-background ID present in the map's table (or the
    /// default label ID `1` if none exist).
    pub fn from_label_map(map: LabelMap) -> Self {
        let active = map
            .table
            .entries()
            .iter()
            .map(|e| e.id)
            .next()
            .unwrap_or(DEFAULT_LABEL_ID);
        Self {
            history: UndoRedoStack::new(map),
            active_label_id: active,
        }
    }

    /// Current immutable label map snapshot.
    pub fn current_map(&self) -> &LabelMap {
        self.history.current()
    }

    /// Current active foreground label ID.
    pub fn active_label_id(&self) -> LabelId {
        self.active_label_id
    }

    /// Number of undoable snapshots after the initial state.
    pub fn history_depth(&self) -> usize {
        self.history.history_depth()
    }

    /// Select an existing foreground label as the active paint target.
    pub fn set_active_label(&mut self, label_id: impl Into<LabelId>) -> Result<(), String> {
        let label_id = label_id.into();
        validate_label_exists(&self.current_map().table, label_id)?;
        self.active_label_id = label_id;
        Ok(())
    }

    /// Add a label to the current table and make it active.
    ///
    /// The ID is the smallest positive integer absent from the current table.
    pub fn add_label(&mut self, name: impl Into<String>, color: RgbaU8) -> Result<LabelId, String> {
        let mut next = self.current_map().clone();
        let label_id = next.table.next_free_id();
        next.table.add_label(label_id, name, color)?;
        self.history.push(next);
        self.active_label_id = label_id;
        Ok(label_id)
    }

    /// Set label visibility in the current table.
    pub fn set_label_visibility(
        &mut self,
        label_id: impl Into<LabelId>,
        visible: Visibility,
    ) -> Result<(), String> {
        let label_id = label_id.into();
        let mut next = self.current_map().clone();
        if !next.table.set_visibility(label_id, visible) {
            return Err(format!("label id {label_id} is not present"));
        }
        if self
            .current_map()
            .table
            .get_label(label_id)
            .map(|e| e.visible)
            != Some(visible)
        {
            self.history.push(next);
        }
        Ok(())
    }

    /// Paint one voxel with the active label.
    pub fn paint_voxel(&mut self, idx: [usize; 3]) -> Result<usize, String> {
        self.paint_sphere(idx, 0)
    }

    /// Erase one voxel to background.
    pub fn erase_voxel(&mut self, idx: [usize; 3]) -> Result<usize, String> {
        self.erase_sphere(idx, 0)
    }

    /// Paint a closed integer ball with the active label.
    pub fn paint_sphere(&mut self, center: [usize; 3], radius: usize) -> Result<usize, String> {
        validate_label_exists(&self.current_map().table, self.active_label_id)?;
        self.apply_sphere(center, radius, self.active_label_id)
    }

    /// Erase a closed integer ball to background label `0`.
    pub fn erase_sphere(&mut self, center: [usize; 3], radius: usize) -> Result<usize, String> {
        self.apply_sphere(center, radius, LabelId::BACKGROUND)
    }

    /// Undo one committed label-map/table snapshot.
    pub fn undo(&mut self) -> bool {
        self.history.undo()
    }

    /// Redo one undone label-map/table snapshot.
    pub fn redo(&mut self) -> bool {
        self.history.redo()
    }

    /// Whether undo is currently available.
    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    /// Whether redo is currently available.
    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Sorted `(label_id, voxel_count)` pairs for labels present in the map.
    pub fn label_counts(&self) -> Vec<(LabelId, usize)> {
        self.current_map()
            .present_labels()
            .into_iter()
            .map(|id| (id, self.current_map().count_label(id)))
            .collect()
    }

    fn apply_sphere(
        &mut self,
        center: [usize; 3],
        radius: usize,
        label_id: LabelId,
    ) -> Result<usize, String> {
        validate_index(self.current_map().shape.0, center)?;

        let mut next = self.current_map().clone();
        let shape = next.shape.0;
        let [z_min, y_min, x_min] = [
            center[0].saturating_sub(radius),
            center[1].saturating_sub(radius),
            center[2].saturating_sub(radius),
        ];
        let [z_max, y_max, x_max] = [
            center[0].saturating_add(radius).min(shape[0] - 1),
            center[1].saturating_add(radius).min(shape[1] - 1),
            center[2].saturating_add(radius).min(shape[2] - 1),
        ];
        let radius_squared = radius.saturating_mul(radius);
        let mut changed = 0usize;

        for z in z_min..=z_max {
            let dz = z.abs_diff(center[0]);
            for y in y_min..=y_max {
                let dy = y.abs_diff(center[1]);
                for x in x_min..=x_max {
                    let dx = x.abs_diff(center[2]);
                    let distance_squared = dz
                        .saturating_mul(dz)
                        .saturating_add(dy.saturating_mul(dy))
                        .saturating_add(dx.saturating_mul(dx));
                    if distance_squared <= radius_squared {
                        let idx = [z, y, x];
                        if next.label_at(idx) != label_id {
                            next.set_label_at(idx, label_id);
                            changed += 1;
                        }
                    }
                }
            }
        }

        if changed > 0 {
            self.history.push(next);
        }
        Ok(changed)
    }
}

/// Construct a default single-label table suitable for a freshly loaded
/// segmentation with no pre-existing labels.
pub fn default_label_table() -> LabelTable {
    let mut table = LabelTable::new();
    table
        .add_label(DEFAULT_LABEL_ID, DEFAULT_LABEL_NAME, DEFAULT_LABEL_COLOR)
        .expect("default label id is unique in a new label table");
    table
}

fn validate_label_exists(table: &LabelTable, label_id: impl Into<LabelId>) -> Result<(), String> {
    let label_id = label_id.into();
    if label_id == LabelId::BACKGROUND {
        return Err("background label 0 cannot be active".to_string());
    }
    if table.get_label(label_id).is_none() {
        return Err(format!("label id {label_id} is not present"));
    }
    Ok(())
}

fn validate_index(shape: [usize; 3], [z, y, x]: [usize; 3]) -> Result<(), String> {
    if shape.contains(&0) {
        return Err(format!("label map shape {:?} has an empty axis", shape));
    }
    if z >= shape[0] || y >= shape[1] || x >= shape[2] {
        return Err(format!(
            "label index [{z},{y},{x}] out of bounds for shape {:?}",
            shape
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
