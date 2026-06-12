//! Connectivity enforcement for SLIC superpixels.
//!
//! Relabels connected components smaller than `min_component_size` into
//! the nearest large neighbor by intensity distance, using union-find
//! for connected-component detection and iterative merging.
//!
//! All neighbor lookups use precomputed C-contiguous strides for pure
//! arithmetic O(1) neighbor-index computation — no per-voxel Vec allocation.

/// Compute C-contiguous (row-major) strides for the given shape.
///
/// `strides[d]` is the number of elements stepped when coordinate `d`
/// increments by 1. For flat index `i`, coordinate in dimension `d` is
/// `(i / strides[d]) % shape[d]`.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    // Iterate over `shape` in reverse with `enumerate()`, using the
    // index to write into `strides` and accumulating the stride
    // product. This avoids `clippy::needless_range_loop` (the loop
    // variable would otherwise only index into `strides`/`shape`).
    let mut stride = 1usize;
    for (i, &s) in shape.iter().enumerate().rev() {
        strides[i] = stride;
        stride *= s;
    }
    strides
}

/// Return the flat neighbor index at offset `delta` along dimension `d`.
///
/// Returns `None` when the neighbor is out of the array boundary.
/// No heap allocation; uses the precomputed stride to compute the offset
/// in O(1) with a single bounds check.
#[inline]
fn neighbor_index(
    i: usize,
    d: usize,
    delta: isize,
    shape: &[usize],
    stride: usize,
) -> Option<usize> {
    let coord = (i / stride) % shape[d];
    let nbr_coord = coord as isize + delta;
    if nbr_coord < 0 || nbr_coord >= shape[d] as isize {
        None
    } else {
        Some((i as isize + delta * stride as isize) as usize)
    }
}

/// Relabel connected components smaller than `min_size` into the nearest
/// large neighbor by intensity distance. Iterates until no more small
/// components remain or the pass limit is reached.
pub fn enforce_connectivity(
    labels: &mut [u32],
    shape: &[usize],
    _ndim: usize,
    intensities: &[f64],
    min_size: usize,
) {
    let n: usize = shape.iter().product();
    if n == 0 || min_size == 0 {
        return;
    }

    // If all voxels share the same label, skip connectivity enforcement.
    let first = labels[0];
    if labels.iter().all(|&l| l == first) {
        return;
    }

    // Precompute C-contiguous strides once — reused across all passes and
    // both inner loops, eliminating per-voxel decode/encode allocations.
    let strides = compute_strides(shape);

    let mut new_label = labels.to_vec();
    let mut changed = true;
    let mut pass = 0;
    let max_passes = 10;

    while changed && pass < max_passes {
        changed = false;
        pass += 1;

        // Re-compute connected components with current labels.
        let mut uf = UnionFind::new(n);
        for i in 0..n {
            // Forward adjacency: +1 along each axis (face-connected).
            for (d, &stride) in strides.iter().enumerate() {
                if let Some(nbr) = neighbor_index(i, d, 1, shape, stride) {
                    if new_label[i] == new_label[nbr] {
                        uf.union(i, nbr);
                    }
                }
            }
        }

        // Compute component sizes and mean intensities.
        let mut comp_sizes: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::with_capacity(n);
        let mut comp_int_sum: std::collections::HashMap<usize, f64> =
            std::collections::HashMap::with_capacity(n);

        for (i, intensity) in intensities.iter().enumerate().take(n) {
            let root = uf.find(i);
            *comp_sizes.entry(root).or_insert(0) += 1;
            *comp_int_sum.entry(root).or_insert(0.0) += intensity;
        }

        let mut comp_mean: std::collections::HashMap<usize, f64> =
            std::collections::HashMap::with_capacity(n);
        for (&root, &size) in &comp_sizes {
            let sum = comp_int_sum.get(&root).copied().unwrap_or(0.0);
            comp_mean.insert(root, sum / size as f64);
        }

        // Identify small components.
        let small_roots: Vec<usize> = comp_sizes
            .iter()
            .filter(|(_, &size)| size < min_size)
            .map(|(&root, _)| root)
            .collect();

        if small_roots.is_empty() {
            break;
        }

        for &small_root in &small_roots {
            let small_label = new_label[small_root];
            let small_mean = comp_mean.get(&small_root).copied().unwrap_or(0.0);

            let mut best_label = small_label;
            let mut best_dist = f64::MAX;

            // Find the neighboring label with minimum intensity distance.
            for i in 0..n {
                if uf.find(i) != small_root {
                    continue;
                }

                // Check all face-adjacent neighbors (±1 per axis).
                for (d, &stride) in strides.iter().enumerate() {
                    for &delta in &[-1isize, 1] {
                        if let Some(nbr) = neighbor_index(i, d, delta, shape, stride) {
                            let nbr_root = uf.find(nbr);
                            if nbr_root != small_root {
                                let nbr_label = new_label[nbr];
                                if nbr_label != small_label {
                                    let nbr_mean = comp_mean.get(&nbr_root).copied().unwrap_or(0.0);
                                    let dist = (small_mean - nbr_mean).abs();
                                    if dist < best_dist {
                                        best_dist = dist;
                                        best_label = nbr_label;
                                    }
                                }
                            }
                        }
                    }
                }

                // Early exit once any neighbor is found.
                if best_dist < f64::MAX {
                    break;
                }
            }

            if best_label != small_label {
                for (i, label) in new_label.iter_mut().enumerate().take(n) {
                    if uf.find(i) == small_root {
                        *label = best_label;
                    }
                }
                changed = true;
            }
        }
    }

    // Relabel to consecutive integers 0..actual_k-1.
    let mut label_map = std::collections::HashMap::with_capacity(n);
    let mut next = 0u32;
    for (i, &l) in new_label.iter().enumerate().take(n) {
        let entry = label_map.entry(l).or_insert_with(|| {
            let v = next;
            next += 1;
            v
        });
        labels[i] = *entry;
    }
}

// ── Union-Find ────────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
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
