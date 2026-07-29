//! Tag trees (ISO 15444-1 §B.10.2).
//!
//! A tag tree codes a 2-D grid of non-negative integers (one leaf per
//! code-block) through a quad-tree whose internal nodes hold the minimum of
//! their children. Coding is incremental against a growing threshold: a `0`
//! bit means "the node's value is larger than the current lower bound"
//! (increment), a `1` bit means "the value equals the lower bound" (known) —
//! the standard polarity used by conformant codecs.
//!
//! One tree instance carries decoder/encoder state (`low`, `known`) across
//! packets, so inclusion and missing-MSB information accumulate correctly
//! over quality layers.

use anyhow::{Context, Result};

use super::packet::{BitReader, BitWriter};

/// A tag tree halves each dimension at every level. Starting from the largest
/// representable dimension therefore needs at most one leaf level plus
/// `usize::BITS` parent levels.
const MAX_TAG_TREE_LEVELS: usize = usize::BITS as usize + 1;

#[derive(Clone, Copy, Debug)]
struct Node {
    /// Leaf/internal value (min of children for internal nodes).
    value: u32,
    /// Lower bound communicated so far.
    low: u32,
    /// Whether `value` has been fully communicated.
    known: bool,
    /// Parent index (`usize::MAX` for the root).
    parent: usize,
}

/// Quad-tree over a `w × h` leaf grid.
pub(crate) struct TagTree {
    nodes: Vec<Node>,
    /// Per-level grid widths/heights, finest (leaves) first.
    level_dims: Vec<(usize, usize)>,
    /// Node-index offset of each level, finest first.
    level_offsets: Vec<usize>,
}

impl TagTree {
    fn required_levels(mut w: usize, mut h: usize) -> usize {
        let mut levels = 1;
        while w > 1 || h > 1 {
            w = w.div_ceil(2);
            h = h.div_ceil(2);
            levels += 1;
        }
        levels
    }

    /// Build a tree for a `w × h` leaf grid (`w, h ≥ 1`); all values start 0.
    pub(crate) fn new(w: usize, h: usize) -> Self {
        assert!(w >= 1 && h >= 1, "tag tree requires a non-empty leaf grid");
        let mut level_dims = Vec::with_capacity(Self::required_levels(w, h));
        level_dims.push((w, h));
        let (mut lw, mut lh) = (w, h);
        while lw > 1 || lh > 1 {
            lw = lw.div_ceil(2);
            lh = lh.div_ceil(2);
            level_dims.push((lw, lh));
        }
        let mut level_offsets = Vec::with_capacity(level_dims.len());
        let mut total = 0usize;
        for &(dw, dh) in &level_dims {
            level_offsets.push(total);
            total += dw * dh;
        }
        let mut nodes = vec![
            Node {
                value: 0,
                low: 0,
                known: false,
                parent: usize::MAX
            };
            total
        ];
        // Wire parents: node (x, y) at level l → (x/2, y/2) at level l+1.
        for l in 0..level_dims.len() - 1 {
            let (dw, dh) = level_dims[l];
            let (pw, _) = level_dims[l + 1];
            for y in 0..dh {
                for x in 0..dw {
                    nodes[level_offsets[l] + y * dw + x].parent =
                        level_offsets[l + 1] + (y / 2) * pw + (x / 2);
                }
            }
        }
        Self {
            nodes,
            level_dims,
            level_offsets,
        }
    }

    #[inline]
    fn leaf_index(&self, x: usize, y: usize) -> usize {
        let (w, _) = self.level_dims[0];
        self.level_offsets[0] + y * w + x
    }

    /// Fill `path` from leaf to root and return its initialized length.
    fn fill_path(&self, x: usize, y: usize, path: &mut [usize; MAX_TAG_TREE_LEVELS]) -> usize {
        let mut idx = self.leaf_index(x, y);
        let mut len = 0;
        loop {
            let Some(slot) = path.get_mut(len) else {
                unreachable!("invariant: halving usize dimensions fits MAX_TAG_TREE_LEVELS");
            };
            *slot = idx;
            len += 1;
            let p = self.nodes[idx].parent;
            if p == usize::MAX {
                break;
            }
            idx = p;
        }
        len
    }

    /// Set the value of leaf `(x, y)` (encoder side). Call [`Self::finalize`]
    /// after all leaves are set.
    pub(crate) fn set_value(&mut self, x: usize, y: usize, value: u32) {
        let idx = self.leaf_index(x, y);
        self.nodes[idx].value = value;
    }

    /// Propagate minima from leaves to the root (encoder side).
    pub(crate) fn finalize(&mut self) {
        // Internal nodes = min of children: initialise to MAX, then sweep.
        for l in 1..self.level_dims.len() {
            let (dw, dh) = self.level_dims[l];
            for i in 0..dw * dh {
                self.nodes[self.level_offsets[l] + i].value = u32::MAX;
            }
        }
        for l in 0..self.level_dims.len() - 1 {
            let (dw, dh) = self.level_dims[l];
            for i in 0..dw * dh {
                let idx = self.level_offsets[l] + i;
                let p = self.nodes[idx].parent;
                let v = self.nodes[idx].value;
                if self.nodes[p].value > v {
                    self.nodes[p].value = v;
                }
            }
        }
    }

    /// Encode information about leaf `(x, y)` up to `threshold`
    /// (§B.10.2 / jasper `jpc_tagtree_encode`).
    pub(crate) fn encode(&mut self, bw: &mut BitWriter, x: usize, y: usize, threshold: u32) {
        let mut path = [0; MAX_TAG_TREE_LEVELS];
        let path_len = self.fill_path(x, y, &mut path);
        let mut low = 0u32;
        for &idx in path[..path_len].iter().rev() {
            let node = &mut self.nodes[idx];
            if node.low < low {
                node.low = low;
            }
            while node.low < threshold {
                if node.low >= node.value {
                    if !node.known {
                        bw.write_bit(1);
                        node.known = true;
                    }
                    break;
                }
                bw.write_bit(0);
                node.low += 1;
            }
            low = node.low;
        }
    }

    /// Decode information about leaf `(x, y)` up to `threshold`. Returns
    /// `true` when the leaf value is known to be `< threshold`.
    pub(crate) fn decode(
        &mut self,
        br: &mut BitReader<'_>,
        x: usize,
        y: usize,
        threshold: u32,
    ) -> Result<bool> {
        let mut path = [0; MAX_TAG_TREE_LEVELS];
        let path_len = self.fill_path(x, y, &mut path);
        self.decode_path(br, &path[..path_len], threshold)
    }

    /// Decode one threshold using a leaf-to-root path prepared by the caller.
    fn decode_path(
        &mut self,
        br: &mut BitReader<'_>,
        path: &[usize],
        threshold: u32,
    ) -> Result<bool> {
        let mut low = 0u32;
        for &idx in path.iter().rev() {
            let node = &mut self.nodes[idx];
            if node.low < low {
                node.low = low;
            }
            while node.low < threshold && !node.known {
                if br.read_bit()? == 1 {
                    node.value = node.low;
                    node.known = true;
                } else {
                    node.low += 1;
                }
            }
            low = node.low;
        }
        let Some(&leaf_index) = path.first() else {
            unreachable!("invariant: every tag-tree path contains its leaf");
        };
        let leaf = &self.nodes[leaf_index];
        Ok(leaf.known && leaf.value < threshold)
    }

    /// Decode the exact value of leaf `(x, y)` by growing the threshold until
    /// it is known (used for missing-MSB coding, §B.10.5).
    pub(crate) fn decode_value(
        &mut self,
        br: &mut BitReader<'_>,
        x: usize,
        y: usize,
    ) -> Result<u32> {
        let mut path = [0; MAX_TAG_TREE_LEVELS];
        let path_len = self.fill_path(x, y, &mut path);
        let path = &path[..path_len];
        let Some(&leaf_index) = path.first() else {
            unreachable!("invariant: every tag-tree path contains its leaf");
        };
        let mut t = self.nodes[leaf_index]
            .low
            .checked_add(1)
            .context("J2K tag-tree value exceeds u32 range")?;
        while !self.nodes[leaf_index].known {
            self.decode_path(br, path, t)?;
            t = t
                .checked_add(1)
                .context("J2K tag-tree value exceeds u32 range")?;
        }
        Ok(self.nodes[leaf_index].value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(w: usize, h: usize, values: &[u32]) {
        assert_eq!(values.len(), w * h);
        let mut enc = TagTree::new(w, h);
        for y in 0..h {
            for x in 0..w {
                enc.set_value(x, y, values[y * w + x]);
            }
        }
        enc.finalize();
        let mut bw = BitWriter::new();
        for y in 0..h {
            for x in 0..w {
                enc.encode(&mut bw, x, y, values[y * w + x] + 1);
            }
        }
        let bytes = bw.flush();
        let mut dec = TagTree::new(w, h);
        let mut br = BitReader::new(&bytes);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    dec.decode_value(&mut br, x, y)
                        .expect("encoded tag-tree value must decode"),
                    values[y * w + x],
                    "leaf ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn single_leaf_round_trip() {
        for v in [0u32, 1, 5, 17] {
            round_trip(1, 1, &[v]);
        }
    }

    #[test]
    fn grid_2x2_round_trip() {
        round_trip(2, 2, &[3, 0, 7, 2]);
    }

    #[test]
    fn grid_3x2_round_trip_iso_example_shape() {
        round_trip(3, 2, &[1, 3, 2, 3, 2, 0]);
    }

    #[test]
    fn grid_2x2_bitstream_is_stable() {
        let mut tree = TagTree::new(2, 2);
        let values = [3, 0, 7, 2];
        for (index, &value) in values.iter().enumerate() {
            tree.set_value(index % 2, index / 2, value);
        }
        tree.finalize();

        let mut writer = BitWriter::new();
        for (index, &value) in values.iter().enumerate() {
            tree.encode(&mut writer, index % 2, index / 2, value + 1);
        }

        assert_eq!(writer.flush(), vec![140, 4, 128]);
    }

    #[test]
    fn maximum_dimension_level_count_fits_fixed_path() {
        assert_eq!(
            TagTree::required_levels(usize::MAX, usize::MAX),
            MAX_TAG_TREE_LEVELS
        );
        assert_eq!(TagTree::required_levels(1, usize::MAX), MAX_TAG_TREE_LEVELS);
        assert_eq!(TagTree::required_levels(1, 1), 1);
    }

    #[test]
    fn deep_rectangular_tree_round_trip() {
        let values = (0..257)
            .map(|index| {
                u32::try_from(index * 17 % 31)
                    .expect("invariant: generated tag-tree value is at most 30")
            })
            .collect::<Vec<_>>();
        round_trip(1, values.len(), &values);
    }

    #[test]
    fn threshold_decode_partial_knowledge() {
        // Leaf value 2: decode at threshold 1 and 2 must answer "not < t"
        // without claiming knowledge of the exact value prematurely.
        let mut enc = TagTree::new(1, 1);
        enc.set_value(0, 0, 2);
        enc.finalize();
        let mut bw = BitWriter::new();
        enc.encode(&mut bw, 0, 0, 1);
        enc.encode(&mut bw, 0, 0, 2);
        enc.encode(&mut bw, 0, 0, 3);
        let bytes = bw.flush();
        let mut dec = TagTree::new(1, 1);
        let mut br = BitReader::new(&bytes);
        assert!(!dec
            .decode(&mut br, 0, 0, 1)
            .expect("threshold 1 must decode"));
        assert!(!dec
            .decode(&mut br, 0, 0, 2)
            .expect("threshold 2 must decode"));
        assert!(dec
            .decode(&mut br, 0, 0, 3)
            .expect("threshold 3 must decode"));
    }
}
