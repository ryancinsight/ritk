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
//! - All label IDs are valid `u32` values (backing storage is `Arc<[u32]>`).
//! - shape\[0\] * shape\[1\] * shape\[2\] == data.len() exactly.

use std::collections::HashSet;
use std::sync::Arc;

use super::label_table::LabelTable;
use super::types::LabelId;
use ritk_spatial::VolumeDims;

/// Dense 3-D label map with associated label table.
///
/// Layout: ZYX (z varies slowest, x varies fastest).
/// Invariant: `shape.total_voxels() == data.len()`.
///
/// # Memory model (Copy-on-Write)
///
/// The flat label buffer is wrapped in `Arc<[u32]>` so that `clone()` (used
/// by the label editor before each edit) increments a reference count instead of
/// deep-copying every voxel. The deep copy is deferred to the first `set_label_at`
/// call via `Arc::make_mut`, which materializes a new buffer only when the `Arc`
/// has multiple references. Read-only operations (`label_at`, `as_slice`,
/// `present_labels`, etc.) incur zero copy overhead regardless of the reference
/// count.
#[derive(Debug, Clone)]
pub struct LabelMap {
    /// Volume dimensions [nz, ny, nx].
    pub shape: VolumeDims,
    /// Flat label buffer in ZYX layout. Label 0 denotes background.
    data: Arc<[u32]>,
    /// Label-to-display-properties table.
    pub table: LabelTable,
}

impl LabelMap {
    /// Construct a LabelMap filled with background (0) for the given shape and table.
    pub fn new(shape: impl Into<VolumeDims>, table: LabelTable) -> Self {
        let shape = shape.into();
        let n = shape.total_voxels();
        Self {
            shape,
            data: Arc::from(vec![0u32; n]),
            table,
        }
    }

    /// Construct a LabelMap from an existing flat buffer.
    ///
    /// Returns `Err` if `data.len() != shape.total_voxels()`.
    pub fn from_data(
        shape: impl Into<VolumeDims>,
        data: Vec<u32>,
        table: LabelTable,
    ) -> Result<Self, String> {
        let shape = shape.into();
        let expected = shape.total_voxels();
        if data.len() != expected {
            return Err(format!(
                "data length {} != shape product {}",
                data.len(),
                expected
            ));
        }
        Ok(Self {
            shape,
            data: Arc::from(data),
            table,
        })
    }

    /// Total number of voxels.
    pub fn num_voxels(&self) -> usize {
        self.data.len()
    }

    /// Get the label at voxel [z, y, x]. Panics if the index is out of bounds.
    pub fn label_at(&self, idx: [usize; 3]) -> LabelId {
        LabelId(self.data[self.flat_index(idx)])
    }

    /// Set the label at voxel [z, y, x]. Panics if the index is out of bounds.
    ///
    /// Uses `Arc::make_mut` for copy-on-write: if the `Arc` is shared (multiple
    /// references from undo history), the underlying buffer is deep-copied on the
    /// *first* mutation and subsequently mutated in place.  Subsequent calls with
    /// exclusive ownership run without any copy.
    pub fn set_label_at(&mut self, idx: [usize; 3], label_id: impl Into<LabelId>) {
        let flat = self.flat_index(idx);
        Arc::make_mut(&mut self.data)[flat] = u32::from(label_id.into());
    }

    /// Return the flat buffer (read-only).
    pub fn as_slice(&self) -> &[u32] {
        &self.data[..]
    }

    /// Compute a binary mask: `mask[i] = true` iff `data[i] == label_id`.
    ///
    /// Result is a flat `Vec<bool>` of length `num_voxels()` in ZYX order.
    pub fn mask_for_label(&self, label_id: impl Into<LabelId>) -> Vec<bool> {
        let label_id = u32::from(label_id.into());
        self.data.iter().map(|&v| v == label_id).collect()
    }

    /// Count voxels assigned the given label.
    pub fn count_label(&self, label_id: impl Into<LabelId>) -> usize {
        let label_id = u32::from(label_id.into());
        self.data.iter().filter(|&&v| v == label_id).count()
    }

    /// Returns all unique label IDs present in the map, sorted ascending.
    pub fn present_labels(&self) -> Vec<LabelId> {
        let mut ids: Vec<u32> = self
            .data
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        ids.sort_unstable();
        ids.into_iter().map(LabelId).collect()
    }

    /// Compute ZYX flat index with bounds assertion.
    fn flat_index(&self, [z, y, x]: [usize; 3]) -> usize {
        let [nz, ny, nx] = self.shape.0;
        assert!(
            z < nz && y < ny && x < nx,
            "LabelMap index [{},{},{}] out of bounds for shape {:?}",
            z,
            y,
            x,
            self.shape
        );
        z * ny * nx + y * nx + x
    }
}

#[cfg(test)]
#[path = "tests_label_map.rs"]
mod tests;
