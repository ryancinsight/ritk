//! Label table for segmentation overlays.
//!
//! # Mathematical Specification
//!
//! A label table T is a partial function T: N -> LabelEntry, where each entry
//! carries a display name, RGBA color, and visibility flag. The domain of T
//! is the set of active label IDs. The ID 0 is conventionally reserved for
//! the background label; the table may but need not include it.
//!
//! # Invariants
//! - Label IDs within a table are unique.
//! -  returns min { k in N | k not in dom(T) }, guaranteed >= 1
//!    (so auto-assigned IDs never collide with existing entries).

use super::color::RgbaBytes;
use super::overlay::Visibility;
use super::types::LabelId;
use serde::{Deserialize, Serialize};

/// A single entry in a label table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelEntry {
    /// Unique label identifier. 0 is conventionally the background.
    pub id: LabelId,
    /// Human-readable name (e.g., Left Ventricle).
    pub name: String,
    /// RGBA color \[R, G, B, A\] each in \[0, 255\].
    pub color: RgbaBytes,
    /// Whether this label is currently visible in overlays.
    pub visible: Visibility,
}

impl LabelEntry {
    /// Construct a new fully-visible label entry.
    pub fn new(id: impl Into<LabelId>, name: impl Into<String>, color: RgbaBytes) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            color,
            visible: Visibility::Visible,
        }
    }
}

/// Ordered collection of segmentation labels.
///
/// Invariant: all entries have distinct IDs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LabelTable {
    entries: Vec<LabelEntry>,
}

impl LabelTable {
    /// Construct an empty label table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a label entry. Returns Err if the ID already exists in the table.
    pub fn add_label(
        &mut self,
        id: impl Into<LabelId>,
        name: impl Into<String>,
        color: RgbaBytes,
    ) -> Result<(), String> {
        let id = id.into();
        if self.entries.iter().any(|e| e.id == id) {
            return Err(format!("label id {} already exists", u32::from(id)));
        }
        self.entries.push(LabelEntry::new(id, name, color));
        Ok(())
    }

    /// Remove the label with the given ID. Returns true if the label was present.
    pub fn remove_label(&mut self, id: impl Into<LabelId>) -> bool {
        let id = id.into();
        let before = self.entries.len();
        self.entries.retain(|e| e.id != id);
        self.entries.len() < before
    }

    /// Return a reference to the entry with the given ID, or None if absent.
    pub fn get_label(&self, id: impl Into<LabelId>) -> Option<&LabelEntry> {
        let id = id.into();
        self.entries.iter().find(|e| e.id == id)
    }

    /// Return a mutable reference to the entry with the given ID, or None if absent.
    pub fn get_label_mut(&mut self, id: impl Into<LabelId>) -> Option<&mut LabelEntry> {
        let id = id.into();
        self.entries.iter_mut().find(|e| e.id == id)
    }

    /// View all entries in insertion order.
    pub fn entries(&self) -> &[LabelEntry] {
        &self.entries
    }

    /// Set the visibility of the label with id. Returns false if the ID is not present.
    pub fn set_visibility(&mut self, id: impl Into<LabelId>, visible: Visibility) -> bool {
        let id = id.into();
        match self.get_label_mut(id) {
            Some(e) => {
                e.visible = visible;
                true
            }
            None => false,
        }
    }

    /// Number of labels in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the table contains no labels.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the smallest positive integer not already used as a label ID.
    ///
    /// Guarantees: result >= 1, result not in { e.id | e in entries }.
    ///
    /// Algorithm: linear scan from 1, O(n) per call.
    pub fn next_free_id(&self) -> LabelId {
        let mut candidate = 1u32;
        loop {
            if !self.entries.iter().any(|e| e.id == candidate) {
                return LabelId(candidate);
            }
            candidate += 1;
        }
    }
}

#[cfg(test)]
#[path = "tests_label_table.rs"]
mod tests;
