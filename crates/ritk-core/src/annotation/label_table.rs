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
//!   (so auto-assigned IDs never collide with existing entries).

use serde::{Deserialize, Serialize};

/// A single entry in a label table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelEntry {
    /// Unique label identifier. 0 is conventionally the background.
    pub id: u32,
    /// Human-readable name (e.g., Left Ventricle).
    pub name: String,
    /// RGBA color \[R, G, B, A\] each in \[0, 255\].
    pub color: [u8; 4],
    /// Whether this label is currently visible in overlays.
    pub visible: bool,
}

impl LabelEntry {
    /// Construct a new fully-visible label entry.
    pub fn new(id: u32, name: impl Into<String>, color: [u8; 4]) -> Self {
        Self { id, name: name.into(), color, visible: true }
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
    pub fn new() -> Self { Self::default() }

    /// Add a label entry. Returns Err if the ID already exists in the table.
    pub fn add_label(&mut self, id: u32, name: impl Into<String>, color: [u8; 4]) -> Result<(), String> {
        if self.entries.iter().any(|e| e.id == id) {
            return Err(format!("label id {} already exists", id));
        }
        self.entries.push(LabelEntry::new(id, name, color));
        Ok(())
    }

    /// Remove the label with the given ID. Returns true if the label was present.
    pub fn remove_label(&mut self, id: u32) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.id != id);
        self.entries.len() < before
    }

    /// Return a reference to the entry with the given ID, or None if absent.
    pub fn get_label(&self, id: u32) -> Option<&LabelEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Return a mutable reference to the entry with the given ID, or None if absent.
    pub fn get_label_mut(&mut self, id: u32) -> Option<&mut LabelEntry> {
        self.entries.iter_mut().find(|e| e.id == id)
    }

    /// View all entries in insertion order.
    pub fn entries(&self) -> &[LabelEntry] { &self.entries }

    /// Set the visibility of the label with id. Returns false if the ID is not present.
    pub fn set_visibility(&mut self, id: u32, visible: bool) -> bool {
        match self.get_label_mut(id) {
            Some(e) => { e.visible = visible; true }
            None => false,
        }
    }

    /// Number of labels in the table.
    pub fn len(&self) -> usize { self.entries.len() }

    /// True if the table contains no labels.
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Return the smallest positive integer not already used as a label ID.
    ///
    /// Guarantees: result >= 1, result not in { e.id | e in entries }.
    ///
    /// Algorithm: linear scan from 1, O(n) per call.
    pub fn next_free_id(&self) -> u32 {
        let mut candidate = 1u32;
        loop {
            if !self.entries.iter().any(|e| e.id == candidate) {
                return candidate;
            }
            candidate += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_table_add_and_get() {
        let mut table = LabelTable::new();
        table.add_label(1, "Brain", [255, 0, 0, 255]).unwrap();
        let entry = table.get_label(1).expect("entry must be present");
        assert_eq!(entry.id, 1);
        assert_eq!(entry.name, "Brain");
        assert_eq!(entry.color, [255, 0, 0, 255]);
        assert!(entry.visible);
    }

    #[test]
    fn test_label_table_duplicate_id_error() {
        let mut table = LabelTable::new();
        table.add_label(1, "Brain", [255, 0, 0, 255]).unwrap();
        let result = table.add_label(1, "Duplicate", [0, 0, 0, 255]);
        assert!(result.is_err(), "duplicate id must return Err");
        let msg = result.unwrap_err();
        assert!(msg.contains("1"), "error message must mention the id");
    }

    #[test]
    fn test_label_table_remove_present() {
        let mut table = LabelTable::new();
        table.add_label(3, "Liver", [0, 255, 0, 255]).unwrap();
        let removed = table.remove_label(3);
        assert!(removed, "remove must return true for present label");
        assert!(table.get_label(3).is_none(), "label must be absent after removal");
    }

    #[test]
    fn test_label_table_remove_absent() {
        let mut table = LabelTable::new();
        let removed = table.remove_label(99);
        assert!(!removed, "remove must return false for absent label");
    }

    #[test]
    fn test_label_table_set_visibility() {
        let mut table = LabelTable::new();
        table.add_label(2, "Kidney", [0, 0, 255, 255]).unwrap();
        let ok = table.set_visibility(2, false);
        assert!(ok, "set_visibility must return true for present label");
        assert!(
            !table.get_label(2).unwrap().visible,
            "label must be invisible after set_visibility(false)"
        );
    }

    #[test]
    fn test_label_table_next_free_id_empty() {
        let table = LabelTable::new();
        assert_eq!(table.next_free_id(), 1, "next_free_id on empty table must be 1");
    }

    #[test]
    fn test_label_table_next_free_id_with_gaps() {
        let mut table = LabelTable::new();
        table.add_label(1, "A", [0, 0, 0, 255]).unwrap();
        table.add_label(3, "B", [0, 0, 0, 255]).unwrap();
        table.add_label(4, "C", [0, 0, 0, 255]).unwrap();
        // IDs present: {1, 3, 4}; smallest positive absent: 2
        assert_eq!(table.next_free_id(), 2, "next_free_id must return 2 when 1,3,4 are occupied");
    }

    #[test]
    fn test_label_table_len_and_is_empty() {
        let mut table = LabelTable::new();
        assert!(table.is_empty(), "new table must be empty");
        assert_eq!(table.len(), 0);
        table.add_label(5, "Spleen", [128, 0, 128, 255]).unwrap();
        assert!(!table.is_empty(), "table must be non-empty after add");
        assert_eq!(table.len(), 1);
    }
}
