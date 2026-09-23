//! The layer lists: one index-based doubly linked list per layer.
//!
//! Relinking a voxel — [`move_to`] — is the inner loop's most frequent mutation,
//! and it is `O(1)`: unlink through the voxel's handle, push at the front. The
//! order of a layer is load-bearing (the active layer's iteration order feeds the
//! update sweep, which writes into neighbouring layers' `phi`, and the crate's
//! oracles pin the float-exact ITK result), so the store is a list rather than an
//! array: an array frees in `O(1)` only by `swap_remove`, which reorders.
//!
//! Membership is the handle, not `status`. The filters deliberately leave a voxel
//! linked for the rest of the sweep after marking it `ST_CUP`/`ST_CDN`, dropping
//! it in the layer rebuild, so deriving membership from `status` would unlink a
//! voxel the algorithm keeps linked.
//!
//! Memory is band-proportional: one `u32` handle per voxel plus a 16-byte node per
//! band member, recycled through the free list.

/// Sentinel for "no node" / "no link" in every `u32` slot.
pub(crate) const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Default)]
struct Node {
    prev: u32,
    next: u32,
    voxel: u32,
    layer: u32,
}

/// Layer lists for one sparse-field run: `layers` doubly linked lists over
/// `voxels` voxels, allocated once and reused for the whole filter run.
pub(crate) struct SparseFieldLayers {
    /// Per-voxel node handle, or [`NONE`] when the voxel is not in a layer.
    handle: Vec<u32>,
    /// Node slab; `free` names the recycled entries.
    nodes: Vec<Node>,
    free: Vec<u32>,
    head: Vec<u32>,
    tail: Vec<u32>,
}

impl SparseFieldLayers {
    /// Empty lists over `voxels` voxels and `layer_count` layers.
    pub(crate) fn new(voxels: usize, layer_count: usize) -> Self {
        assert!(
            voxels < NONE as usize,
            "sparse-field layers address voxels by u32"
        );
        Self {
            handle: vec![NONE; voxels],
            nodes: Vec::new(),
            free: Vec::new(),
            head: vec![NONE; layer_count],
            tail: vec![NONE; layer_count],
        }
    }

    /// Push `f` at the front of `layer`'s list.
    ///
    /// The voxel must not already be linked; the callers maintain that
    /// through [`move_to`] and by rebuilding a layer with [`Self::replace`].
    #[inline]
    pub(crate) fn push_front(&mut self, layer: usize, f: usize) {
        debug_assert_eq!(self.handle[f], NONE, "voxel already linked");
        let h = self.alloc(f, layer);
        let old_head = self.head[layer];
        self.nodes[h as usize].next = old_head;
        if old_head == NONE {
            self.tail[layer] = h;
        } else {
            self.nodes[old_head as usize].prev = h;
        }
        self.head[layer] = h;
    }

    /// Push `f` at the back of `layer`'s list.
    #[inline]
    pub(crate) fn push_back(&mut self, layer: usize, f: usize) {
        debug_assert_eq!(self.handle[f], NONE, "voxel already linked");
        let h = self.alloc(f, layer);
        let old_tail = self.tail[layer];
        self.nodes[h as usize].prev = old_tail;
        if old_tail == NONE {
            self.head[layer] = h;
        } else {
            self.nodes[old_tail as usize].next = h;
        }
        self.tail[layer] = h;
    }

    /// Unlink `f` from whichever layer holds it. `O(1)`.
    ///
    /// Returns `false` (and does nothing) when the voxel is not linked,
    /// which mirrors the layer-scan removal this replaces: that scan was a
    /// no-op when the voxel was absent from its status layer.
    #[inline]
    pub(crate) fn unlink(&mut self, f: usize) -> bool {
        let h = self.handle[f];
        if h == NONE {
            return false;
        }
        let node = self.nodes[h as usize];
        let layer = node.layer as usize;
        if node.prev == NONE {
            self.head[layer] = node.next;
        } else {
            self.nodes[node.prev as usize].next = node.next;
        }
        if node.next == NONE {
            self.tail[layer] = node.prev;
        } else {
            self.nodes[node.next as usize].prev = node.prev;
        }
        self.handle[f] = NONE;
        self.free.push(h);
        true
    }

    /// Whether `f` is currently linked in some layer.
    #[cfg(test)]
    #[inline]
    pub(crate) fn is_linked(&self, f: usize) -> bool {
        self.handle[f] != NONE
    }

    /// Walk `layer` from front to back.
    pub(crate) fn iter(&self, layer: usize) -> LayerIter<'_> {
        LayerIter {
            lists: self,
            cursor: self.head[layer],
        }
    }

    /// Replace `layer`'s membership with `items`, in order (front to back).
    ///
    /// Frees every node the layer held — including voxels the caller has
    /// already marked out of the band — then links `items` at the back so the
    /// order matches a rebuild through `push_back`.
    pub(crate) fn replace(&mut self, layer: usize, items: &[usize]) {
        self.clear(layer);
        for &f in items {
            self.push_back(layer, f);
        }
    }

    /// Empty `layer`, releasing its nodes.
    pub(crate) fn clear(&mut self, layer: usize) {
        let mut cursor = self.head[layer];
        while cursor != NONE {
            let node = self.nodes[cursor as usize];
            self.handle[node.voxel as usize] = NONE;
            self.free.push(cursor);
            cursor = node.next;
        }
        self.head[layer] = NONE;
        self.tail[layer] = NONE;
    }

    #[inline]
    fn alloc(&mut self, f: usize, layer: usize) -> u32 {
        let node = Node {
            prev: NONE,
            next: NONE,
            voxel: f as u32,
            layer: layer as u32,
        };
        let h = if let Some(recycled) = self.free.pop() {
            self.nodes[recycled as usize] = node;
            recycled
        } else {
            self.nodes.push(node);
            (self.nodes.len() - 1) as u32
        };
        self.handle[f] = h;
        h
    }
}

/// Iterator over one layer's voxels, front to back.
pub(crate) struct LayerIter<'a> {
    lists: &'a SparseFieldLayers,
    cursor: u32,
}

impl Iterator for LayerIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor == NONE {
            return None;
        }
        let node = self.lists.nodes[self.cursor as usize];
        self.cursor = node.next;
        Some(node.voxel as usize)
    }
}

/// Remove `f` from its layer (if linked), set its status, and — when `s` is a
/// real layer — push it to the front of that layer.
///
/// This is ITK's `ChangeLayer`/relink step. The membership test is the list's
/// own handle, so the "voxel was never in the layer its status names" case is a
/// no-op exactly as the layer scan was.
#[inline]
pub(crate) fn move_to(lists: &mut SparseFieldLayers, status: &mut [i32], f: usize, s: i32) {
    lists.unlink(f);
    status[f] = s;
    if s >= 0 {
        lists.push_front(s as usize, f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlink_reports_false_for_an_unlinked_voxel() {
        let mut lists = SparseFieldLayers::new(4, 3);
        assert!(!lists.is_linked(2));
        assert!(!lists.unlink(2));
        lists.push_front(1, 2);
        assert!(lists.is_linked(2));
        assert!(lists.unlink(2));
        assert!(!lists.is_linked(2));
    }

    #[test]
    fn push_front_iterates_the_most_recent_first() {
        let mut lists = SparseFieldLayers::new(4, 2);
        lists.push_front(0, 1);
        lists.push_front(0, 2);
        lists.push_back(0, 3);
        assert_eq!(lists.iter(0).collect::<Vec<_>>(), vec![2, 1, 3]);
    }

    #[test]
    fn unlink_repairs_both_neighbours_and_the_ends() {
        let mut lists = SparseFieldLayers::new(5, 1);
        for f in 0..5 {
            lists.push_back(0, f);
        }
        // Middle, head, and tail.
        assert!(lists.unlink(2));
        assert!(lists.unlink(0));
        assert!(lists.unlink(4));
        assert_eq!(lists.iter(0).collect::<Vec<_>>(), vec![1, 3]);
        // Re-push reuses the freed nodes.
        lists.push_front(0, 4);
        assert_eq!(lists.iter(0).collect::<Vec<_>>(), vec![4, 1, 3]);
    }

    #[test]
    fn replace_frees_the_old_membership_even_when_status_disagrees() {
        let mut lists = SparseFieldLayers::new(4, 2);
        let mut status = vec![0i32; 4];
        for f in 0..3 {
            lists.push_front(0, f);
        }
        // The filters mark voxels out of the band without unlinking them.
        status[1] = -2;
        lists.replace(0, &[2, 0]);
        assert!(!lists.is_linked(1));
        assert_eq!(lists.iter(0).collect::<Vec<_>>(), vec![2, 0]);
        // move_to on the stale voxel is a no-op removal, then re-links.
        move_to(&mut lists, &mut status, 1, 1);
        assert_eq!(status[1], 1);
        assert_eq!(lists.iter(1).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn move_to_moves_between_layers_in_one_step() {
        let mut lists = SparseFieldLayers::new(3, 3);
        let mut status = vec![0i32; 3];
        lists.push_front(0, 0);
        lists.push_front(0, 1);
        move_to(&mut lists, &mut status, 1, 2);
        assert_eq!(lists.iter(0).collect::<Vec<_>>(), vec![0]);
        assert_eq!(lists.iter(2).collect::<Vec<_>>(), vec![1]);
        move_to(&mut lists, &mut status, 0, -1);
        assert!(lists.iter(0).next().is_none());
        assert!(!lists.is_linked(0));
    }
}
