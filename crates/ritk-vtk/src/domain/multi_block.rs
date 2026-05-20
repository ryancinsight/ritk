//! VTK composite multi-block dataset.
//!
//! # Mathematical Specification
//!
//! A multi-block dataset is a rooted tree MB = (N, E) where:
//! - Leaf nodes N_leaf ⊆ N each hold one `VtkDataObject`.
//! - Inner nodes N_inner ⊆ N each hold an ordered sequence of `Block` children.
//! - `Block` = `Leaf(VtkDataObject)` | `Composite(VtkMultiBlockDataSet)`.
//!
//! Invariant: the tree is acyclic (ownership is direct, no shared references).
//!
//! `leaf_count()` = |N_leaf| (recursive sum over the full subtree).
//! `iter_leaves()` performs a depth-first traversal of all leaves using an
//! explicit stack (`LeafIter`) — no heap allocation per iteration step.

use crate::domain::vtk_data_object::VtkDataObject;

/// A block entry inside a `VtkMultiBlockDataSet`.
///
/// Either a terminal leaf holding a single dataset, or a nested composite
/// that recursively contains more blocks.
#[derive(Debug, Clone)]
pub enum Block {
    /// Terminal block containing one VTK dataset.
    Leaf(VtkDataObject),
    /// Nested multi-block subtree.
    Composite(VtkMultiBlockDataSet),
}

/// VTK composite multi-block dataset — an ordered, optionally named collection
/// of `Block` entries that can be nested to arbitrary depth.
#[derive(Debug, Clone, Default)]
pub struct VtkMultiBlockDataSet {
    blocks: Vec<(Option<String>, Block)>,
}

impl VtkMultiBlockDataSet {
    /// Construct an empty multi-block dataset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a block with an optional name.
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn add_block(&mut self, name: Option<&str>, block: Block) -> &mut Self {
        self.blocks.push((name.map(|s| s.to_owned()), block));
        self
    }

    /// Total number of top-level blocks (direct children only).
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Access the top-level block at `idx`, or `None` if out of range.
    pub fn get_block(&self, idx: usize) -> Option<&Block> {
        self.blocks.get(idx).map(|(_, b)| b)
    }

    /// Access the optional name of the top-level block at `idx`.
    pub fn get_block_name(&self, idx: usize) -> Option<&str> {
        self.blocks.get(idx)?.0.as_deref()
    }

    /// Total number of leaf datasets in the full subtree (recursive).
    pub fn leaf_count(&self) -> usize {
        self.blocks
            .iter()
            .map(|(_, b)| match b {
                Block::Leaf(_) => 1,
                Block::Composite(sub) => sub.leaf_count(),
            })
            .sum()
    }

    /// Return a depth-first iterator over all leaf `VtkDataObject`s in the tree.
    ///
    /// Uses an explicit stack; no allocation per `next()` call (amortised O(1)).
    pub fn iter_leaves(&self) -> LeafIter<'_> {
        LeafIter {
            stack: vec![self.blocks.iter()],
        }
    }
}

/// Depth-first leaf iterator for `VtkMultiBlockDataSet`.
///
/// Maintains an explicit stack of iterators (one per nesting level) to avoid
/// recursive calls and `Box<dyn Iterator>` overhead.
pub struct LeafIter<'a> {
    stack: Vec<std::slice::Iter<'a, (Option<String>, Block)>>,
}

impl<'a> Iterator for LeafIter<'a> {
    type Item = &'a VtkDataObject;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let top = self.stack.last_mut()?;
            match top.next() {
                None => {
                    self.stack.pop();
                }
                Some((_, Block::Leaf(obj))) => return Some(obj),
                Some((_, Block::Composite(sub))) => {
                    self.stack.push(sub.blocks.iter());
                }
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::vtk_data_object::{VtkDataObject, VtkPolyData};

    fn leaf_poly() -> Block {
        Block::Leaf(VtkDataObject::PolyData(VtkPolyData::default()))
    }

    fn poly_points(n: usize) -> VtkPolyData {
        VtkPolyData {
            points: vec![[0.0f32; 3]; n],
            ..Default::default()
        }
    }

    #[test]
    fn empty_dataset_has_zero_counts() {
        let mb = VtkMultiBlockDataSet::new();
        assert_eq!(mb.block_count(), 0);
        assert_eq!(mb.leaf_count(), 0);
        assert_eq!(mb.iter_leaves().count(), 0);
    }

    #[test]
    fn add_single_leaf_increments_block_and_leaf_counts() {
        let mut mb = VtkMultiBlockDataSet::new();
        mb.add_block(None, leaf_poly());
        assert_eq!(mb.block_count(), 1);
        assert_eq!(mb.leaf_count(), 1);
    }

    #[test]
    fn get_block_returns_correct_variant() {
        let mut mb = VtkMultiBlockDataSet::new();
        mb.add_block(Some("mesh"), leaf_poly());
        let block = mb.get_block(0).expect("index 0 must exist");
        assert!(matches!(block, Block::Leaf(_)));
    }

    #[test]
    fn get_block_out_of_range_returns_none() {
        let mb = VtkMultiBlockDataSet::new();
        assert!(mb.get_block(0).is_none());
    }

    #[test]
    fn get_block_name_round_trips() {
        let mut mb = VtkMultiBlockDataSet::new();
        mb.add_block(Some("surfaces"), leaf_poly());
        mb.add_block(None, leaf_poly());
        assert_eq!(mb.get_block_name(0), Some("surfaces"));
        assert_eq!(mb.get_block_name(1), None);
    }

    #[test]
    fn nested_composite_leaf_count_is_recursive() {
        let mut inner = VtkMultiBlockDataSet::new();
        inner.add_block(None, leaf_poly());
        inner.add_block(None, leaf_poly());

        let mut outer = VtkMultiBlockDataSet::new();
        outer.add_block(None, leaf_poly());
        outer.add_block(None, Block::Composite(inner));

        // outer has 1 direct leaf + 2 inner leaves = 3 total
        assert_eq!(outer.leaf_count(), 3);
    }

    #[test]
    fn iter_leaves_flat_dataset_visits_all_in_order() {
        let mut mb = VtkMultiBlockDataSet::new();
        for n in [1usize, 2, 3] {
            mb.add_block(
                None,
                Block::Leaf(VtkDataObject::PolyData(poly_points(n))),
            );
        }
        let counts: Vec<usize> = mb
            .iter_leaves()
            .map(|obj| match obj {
                VtkDataObject::PolyData(p) => p.points.len(),
                _ => panic!("expected PolyData"),
            })
            .collect();
        assert_eq!(counts, vec![1, 2, 3]);
    }

    #[test]
    fn iter_leaves_nested_composite_depth_first_order() {
        let mut inner = VtkMultiBlockDataSet::new();
        inner.add_block(None, Block::Leaf(VtkDataObject::PolyData(poly_points(10))));
        inner.add_block(None, Block::Leaf(VtkDataObject::PolyData(poly_points(20))));

        let mut outer = VtkMultiBlockDataSet::new();
        outer.add_block(None, Block::Leaf(VtkDataObject::PolyData(poly_points(1))));
        outer.add_block(None, Block::Composite(inner));
        outer.add_block(None, Block::Leaf(VtkDataObject::PolyData(poly_points(99))));

        let counts: Vec<usize> = outer
            .iter_leaves()
            .map(|obj| match obj {
                VtkDataObject::PolyData(p) => p.points.len(),
                _ => panic!("expected PolyData"),
            })
            .collect();
        // DFS: outer[0]=1, then composite's leaves 10,20, then outer[2]=99
        assert_eq!(counts, vec![1, 10, 20, 99]);
    }
}
