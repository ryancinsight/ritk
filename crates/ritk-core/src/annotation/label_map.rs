//! Dense label map for volumetric segmentation.
//!
//! # Mathematical Specification
//!
//! A label map L: Z_nz x Z_ny x Z_nx -> N is a discrete function mapping each
//! voxel position to a non-negative integer label ID. Label 0 conventionally
//! denotes background (unlabeled). The spatial extent is shape = [nz, ny, nx].
//!
//! Index mapping (ZYX order): flat(z, y, x) = z * ny * nx + y * nx + x.
//!
//! # Invariants
//! - All label IDs are valid u32 values (backing storage is Vec<u32>).
//! - shape[0] * shape[1] * shape[2] == data.len() exactly.

use std::collections::HashSet;

use super::label_table::LabelTable;

/// Dense 3-D label map with associated label table.
///
/// Layout: ZYX (z varies slowest, x varies fastest).
/// Invariant: `shape[0] * shape[1] * shape[2] == data.len()`.
#[derive(Debug, Clone)]
pub struct LabelMap {
    /// Volume dimensions [nz, ny, nx].
    pub shape: [usize; 3],
    /// Flat label buffer in ZYX layout. Label 0 denotes background.
    data: Vec<u32>,
    /// Label-to-display-properties table.
    pub table: LabelTable,
}

impl LabelMap {
    /// Construct a LabelMap filled with background (0) for the given shape and table.
    pub fn new(shape: [usize; 3], table: LabelTable) -> Self {
        let n = shape[0] * shape[1] * shape[2];
        Self { shape, data: vec![0u32; n], table }
    }

    /// Construct a LabelMap from an existing flat buffer.
    ///
    /// Returns `Err` if `data.len() != shape[0] * shape[1] * shape[2]`.
    pub fn from_data(
        shape: [usize; 3],
        data: Vec<u32>,
        table: LabelTable,
    ) -> Result<Self, String> {
        let expected = shape[0] * shape[1] * shape[2];
        if data.len() != expected {
            return Err(format!(
                "data length {} != shape product {}",
                data.len(),
                expected
            ));
        }
        Ok(Self { shape, data, table })
    }

    /// Total number of voxels.
    pub fn num_voxels(&self) -> usize { self.data.len() }

    /// Get the label at voxel [z, y, x]. Panics if the index is out of bounds.
    pub fn label_at(&self, idx: [usize; 3]) -> u32 {
        self.data[self.flat_index(idx)]
    }

    /// Set the label at voxel [z, y, x]. Panics if the index is out of bounds.
    pub fn set_label_at(&mut self, idx: [usize; 3], label_id: u32) {
        let flat = self.flat_index(idx);
        self.data[flat] = label_id;
    }

    /// Return the flat buffer (read-only).
    pub fn as_slice(&self) -> &[u32] { &self.data }

    /// Compute a binary mask: `mask[i] = true` iff `data[i] == label_id`.
    ///
    /// Result is a flat `Vec<bool>` of length `num_voxels()` in ZYX order.
    pub fn mask_for_label(&self, label_id: u32) -> Vec<bool> {
        self.data.iter().map(|&v| v == label_id).collect()
    }

    /// Count voxels assigned the given label.
    pub fn count_label(&self, label_id: u32) -> usize {
        self.data.iter().filter(|&&v| v == label_id).count()
    }

    /// Returns all unique label IDs present in the map, sorted ascending.
    pub fn present_labels(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .data
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Compute ZYX flat index with bounds assertion.
    fn flat_index(&self, [z, y, x]: [usize; 3]) -> usize {
        assert!(
            z < self.shape[0] && y < self.shape[1] && x < self.shape[2],
            "LabelMap index [{},{},{}] out of bounds for shape {:?}",
            z, y, x, self.shape
        );
        z * self.shape[1] * self.shape[2] + y * self.shape[2] + x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_table() -> LabelTable { LabelTable::new() }

    #[test]
    fn test_label_map_new_all_background() {
        let lm = LabelMap::new([2, 3, 4], empty_table());
        assert!(
            lm.as_slice().iter().all(|&v| v == 0),
            "all voxels must be 0 (background) on construction"
        );
        assert_eq!(lm.num_voxels(), 24);
    }

    #[test]
    fn test_label_map_set_and_get() {
        let mut lm = LabelMap::new([4, 5, 6], empty_table());
        lm.set_label_at([1, 2, 3], 5);
        assert_eq!(lm.label_at([1, 2, 3]), 5);
        assert_eq!(lm.label_at([0, 0, 0]), 0);
        assert_eq!(lm.label_at([3, 4, 5]), 0);
    }

    #[test]
    fn test_label_map_from_data_valid() {
        let data: Vec<u32> = (0u32..60).collect();
        let lm = LabelMap::from_data([3, 4, 5], data.clone(), empty_table()).unwrap();
        assert_eq!(lm.as_slice(), data.as_slice());
    }

    #[test]
    fn test_label_map_from_data_wrong_len() {
        let data = vec![0u32; 10];
        let result = LabelMap::from_data([3, 4, 5], data, empty_table());
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("10") && msg.contains("60"), "{}", msg);
    }

    #[test]
    fn test_label_map_mask_for_label() {
        let mut lm = LabelMap::new([2, 2, 2], empty_table());
        lm.set_label_at([0, 0, 1], 2);
        lm.set_label_at([1, 1, 0], 2);
        let mask = lm.mask_for_label(2);
        assert_eq!(mask.len(), 8);
        // flat([0,0,1]) = 0*4 + 0*2 + 1 = 1
        // flat([1,1,0]) = 1*4 + 1*2 + 0 = 6
        assert!(mask[1], "voxel [0,0,1] must be true");
        assert!(mask[6], "voxel [1,1,0] must be true");
        assert_eq!(mask.iter().filter(|&&b| b).count(), 2);
    }

    #[test]
    fn test_label_map_count_label() {
        let mut lm = LabelMap::new([3, 3, 3], empty_table());
        for pos in [[0,0,0],[0,0,1],[1,1,1],[2,2,2]] {
            lm.set_label_at(pos, 7);
        }
        assert_eq!(lm.count_label(7), 4);
        assert_eq!(lm.count_label(0), 27 - 4);
    }

    #[test]
    fn test_label_map_present_labels() {
        let mut lm = LabelMap::new([2, 2, 2], empty_table());
        lm.set_label_at([0, 0, 0], 1);
        lm.set_label_at([0, 0, 1], 3);
        lm.set_label_at([1, 0, 0], 7);
        let present = lm.present_labels();
        assert_eq!(present, vec![0u32, 1, 3, 7]);
    }

    #[test]
    fn test_label_map_num_voxels() {
        let lm = LabelMap::new([4, 5, 6], empty_table());
        assert_eq!(lm.num_voxels(), 120);
    }
}
