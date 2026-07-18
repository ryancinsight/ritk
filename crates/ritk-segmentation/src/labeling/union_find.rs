//! Path-compressed union-find (disjoint-set) data structure.
//!
//! Used by the Hoshen-Kopelman connected-component labeling algorithm to
//! track and merge provisional component labels in O(n Â· Î±(n)) â‰ˆ O(n) time.

/// Union-find with path-halving and union-by-rank.
pub(super) struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    pub(super) fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Find with path-halving (iterative, safe variant of path compression).
    pub(super) fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            // Path halving: point x to its grandparent.
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Union by rank.
    pub(super) fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}
