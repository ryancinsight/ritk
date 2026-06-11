//! Connected-component labeling for 3-D binary images.
//!
//! # Mathematical Specification
//!
//! Given a binary mask M where M\[z,y,x\] ∈ {0, 1}, a connected component C is
//! the maximal set of voxels such that every pair (p, q) ∈ C × C is connected
//! by a path through M where every step moves to an adjacent foreground voxel.
//!
//! Two adjacency models are supported:
//! - **6-connectivity**:  faces only — each voxel has at most 6 neighbours.
//! - **26-connectivity**: faces + edges + corners — up to 26 neighbours.
//!
//! # Algorithm — Hoshen-Kopelman (two-pass)
//!
//! **Pass 1** (Z→Y→X scan order):
//!   For each foreground voxel v, examine the 3 (6-conn) or 13 (26-conn)
//!   already-visited backward neighbours. If none are labelled, assign a new
//!   provisional label. Otherwise assign the minimum neighbour label and union
//!   all neighbour labels using path-compressed union-find.
//!
//! **Pass 2**:
//!   Resolve all provisional labels to their canonical roots; renumber
//!   components with consecutive integers [1, K].
//!
//! # Complexity
//! - Time:  O(n · α(n)) ≈ O(n) with union-find path compression.
//! - Space: O(n) for provisional labels + O(K) for statistics.

use crate::filter::ops::extract_vec_infallible;
use crate::image::Image;
use crate::spatial::Point;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

pub mod relabel;
pub use relabel::{RelabelComponentFilter, RelabelStatistics};

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

    /// Find with path-halving (a safe, iterative variant of path compression).
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            // Path halving: point x to its grandparent.
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Union by rank.
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

// ── Public types ──────────────────────────────────────────────────────────────

/// Per-component statistics produced by `ConnectedComponentsFilter`.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelStatistics {
    /// Component label (1-based integer, cast to u32).
    pub label: u32,
    /// Number of foreground voxels in this component.
    pub voxel_count: usize,
    /// Physical centroid in image-index coordinates.
    pub centroid: Point<3>,
    /// Axis-aligned bounding box: (min_corner, max_corner) in [z, y, x] index space.
    pub bounding_box: ([usize; 3], [usize; 3]),
}

/// Connected-component labeling filter.
///
/// Applies Hoshen-Kopelman labeling to a binary mask image, returning:
/// - A label image whose voxels carry integer class indices (1…K) cast to f32
///   (background voxels remain 0.0).
/// - Per-component `LabelStatistics`.
///
/// # ITK parity
/// Matches `itk::ConnectedComponentImageFilter` semantics: any pixel whose
/// value differs from `background_value` is treated as foreground. Default
/// `background_value = 0.0`.
pub struct ConnectedComponentsFilter {
    /// Adjacency model: 6 (faces only) or 26 (faces + edges + corners).
    pub connectivity: u32,
    /// Pixel value that designates background (ITK `SetBackgroundValue`).
    /// Any voxel with this value is excluded from labeling. Default: 0.0.
    pub background_value: f32,
}

impl ConnectedComponentsFilter {
    /// Create a filter with 6-connectivity and background_value = 0.0 (ITK defaults).
    pub fn new() -> Self {
        Self {
            connectivity: 6,
            background_value: 0.0,
        }
    }

    /// Create a filter with explicit connectivity (6 or 26) and background_value = 0.0.
    ///
    /// # Panics
    /// Panics if `connectivity` is neither 6 nor 26.
    pub fn with_connectivity(connectivity: u32) -> Self {
        assert!(
            connectivity == 6 || connectivity == 26,
            "connectivity must be 6 or 26, got {connectivity}"
        );
        Self {
            connectivity,
            background_value: 0.0,
        }
    }

    /// Set the background pixel value (builder pattern).
    ///
    /// Voxels with this exact value are excluded from labeling. Matches
    /// `itk::ConnectedComponentImageFilter::SetBackgroundValue`.
    pub fn with_background(mut self, background_value: f32) -> Self {
        self.background_value = background_value;
        self
    }

    /// Apply labeling to a binary mask.
    ///
    /// Returns `(label_image, statistics)` where:
    /// - `label_image` has the same shape and spatial metadata as `mask`.
    /// - `statistics` has one entry per component, ordered by label index.
    pub fn apply<B: Backend>(&self, mask: &Image<B, 3>) -> (Image<B, 3>, Vec<LabelStatistics>) {
        let (mask_vals, shape) = extract_vec_infallible(mask);
        let device = mask.data().device();
        let mask_slice: &[f32] = &mask_vals;

        let (label_vec, stats) =
            hoshen_kopelman(mask_slice, shape, self.connectivity, self.background_value);

        let td = TensorData::new(label_vec, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td, &device);

        let label_image = Image::new(tensor, *mask.origin(), *mask.spacing(), *mask.direction());

        (label_image, stats)
    }
}

impl Default for ConnectedComponentsFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: label connected components with `connectivity` ∈ {6, 26}.
///
/// Returns `(label_image, num_components)`.
/// Voxel values in `label_image` are component indices in [1, K] as f32;
/// background voxels are 0.0.
///
/// Uses `background_value = 0.0`. For a custom background value, construct
/// `ConnectedComponentsFilter` directly with `with_background(v)`.
pub fn connected_components<B: Backend>(
    mask: &Image<B, 3>,
    connectivity: u32,
) -> (Image<B, 3>, usize) {
    let filter = ConnectedComponentsFilter::with_connectivity(connectivity);
    let (label_image, stats) = filter.apply(mask);
    (label_image, stats.len())
}

// ── Core algorithm ────────────────────────────────────────────────────────────

/// Hoshen-Kopelman two-pass connected-component labeling.
///
/// Returns `(label_vec, statistics)` where `label_vec` has the same length as
/// `mask` (flat Z×Y×X order).
///
/// `background_value`: pixel value that designates background (ITK parity).
/// Any voxel whose value equals `background_value` is excluded from labeling.
fn hoshen_kopelman(
    mask: &[f32],
    dims: [usize; 3],
    connectivity: u32,
    background_value: f32,
) -> (Vec<f32>, Vec<LabelStatistics>) {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let n = nz * ny * nx;

    // Maximum possible number of components = n (every voxel is isolated).
    // Labels are 1-based; index 0 is reserved for background in the UnionFind.
    let mut uf = UnionFind::new(n + 1);
    let mut provisional = vec![0usize; n];
    let mut next_label = 1usize;

    // Flat index helper.
    let idx = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;

    // Backward-neighbour offsets in scan order.
    // For 6-conn: 3 offsets. For 26-conn: 13 offsets.
    let backward_6: &[(isize, isize, isize)] = &[(-1, 0, 0), (0, -1, 0), (0, 0, -1)];

    let backward_26: &[(isize, isize, isize)] = &[
        (-1, -1, -1),
        (-1, -1, 0),
        (-1, -1, 1),
        (-1, 0, -1),
        (-1, 0, 0),
        (-1, 0, 1),
        (-1, 1, -1),
        (-1, 1, 0),
        (-1, 1, 1),
        (0, -1, -1),
        (0, -1, 0),
        (0, -1, 1),
        (0, 0, -1),
    ];

    let offsets: &[(isize, isize, isize)] = if connectivity == 6 {
        backward_6
    } else {
        backward_26
    };

    // ── Pass 1: assign provisional labels ────────────────────────────────────
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = idx(iz, iy, ix);
                if mask[flat] == background_value {
                    continue; // background
                }

                // Collect labels of already-visited foreground neighbours.
                let mut nbr_labels: Vec<usize> = Vec::with_capacity(offsets.len());
                for &(dz, dy, dx) in offsets {
                    let nz_i = iz as isize + dz;
                    let ny_i = iy as isize + dy;
                    let nx_i = ix as isize + dx;
                    if nz_i < 0
                        || nz_i >= nz as isize
                        || ny_i < 0
                        || ny_i >= ny as isize
                        || nx_i < 0
                        || nx_i >= nx as isize
                    {
                        continue;
                    }
                    let n_flat = idx(nz_i as usize, ny_i as usize, nx_i as usize);
                    let lbl = provisional[n_flat];
                    if lbl > 0 {
                        nbr_labels.push(lbl);
                    }
                }

                if nbr_labels.is_empty() {
                    // No labelled foreground neighbour → new component.
                    provisional[flat] = next_label;
                    next_label += 1;
                } else {
                    // Find canonical root among all neighbours; union them all.
                    let mut root = uf.find(nbr_labels[0]);
                    for &lbl in &nbr_labels[1..] {
                        let r = uf.find(lbl);
                        if r != root {
                            uf.union(root, r);
                            root = uf.find(root);
                        }
                    }
                    provisional[flat] = root;
                }
            }
        }
    }

    if next_label == 1 {
        // No foreground voxels at all.
        return (vec![0.0_f32; n], Vec::new());
    }

    // ── Resolve all provisional labels to canonical roots ─────────────────────
    // Map canonical root → consecutive final label [1, K].
    let mut root_to_final = vec![0usize; next_label];
    let mut num_components = 0usize;
    for prov in provisional.iter_mut().take(n) {
        if *prov > 0 {
            let root = uf.find(*prov);
            if root_to_final[root] == 0 {
                num_components += 1;
                root_to_final[root] = num_components;
            }
            *prov = root_to_final[root];
        }
    }

    // ── Compute LabelStatistics ───────────────────────────────────────────────
    // Accumulate per-component: count, centroid sums, bounding box.
    let mut counts = vec![0usize; num_components + 1];
    let mut sum_z = vec![0.0f64; num_components + 1];
    let mut sum_y = vec![0.0f64; num_components + 1];
    let mut sum_x = vec![0.0f64; num_components + 1];
    let mut bb_min = vec![[usize::MAX; 3]; num_components + 1];
    let mut bb_max = vec![[0usize; 3]; num_components + 1];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = idx(iz, iy, ix);
                let lbl = provisional[flat];
                if lbl == 0 {
                    continue;
                }
                counts[lbl] += 1;
                sum_z[lbl] += iz as f64;
                sum_y[lbl] += iy as f64;
                sum_x[lbl] += ix as f64;
                // Bounding box min
                if iz < bb_min[lbl][0] {
                    bb_min[lbl][0] = iz;
                }
                if iy < bb_min[lbl][1] {
                    bb_min[lbl][1] = iy;
                }
                if ix < bb_min[lbl][2] {
                    bb_min[lbl][2] = ix;
                }
                // Bounding box max
                if iz > bb_max[lbl][0] {
                    bb_max[lbl][0] = iz;
                }
                if iy > bb_max[lbl][1] {
                    bb_max[lbl][1] = iy;
                }
                if ix > bb_max[lbl][2] {
                    bb_max[lbl][2] = ix;
                }
            }
        }
    }

    let mut stats = Vec::with_capacity(num_components);
    for lbl in 1..=num_components {
        let c = counts[lbl] as f64;
        stats.push(LabelStatistics {
            label: lbl as u32,
            voxel_count: counts[lbl],
            centroid: Point::new([sum_z[lbl] / c, sum_y[lbl] / c, sum_x[lbl] / c]),
            bounding_box: (bb_min[lbl], bb_max[lbl]),
        });
    }

    // ── Build output label image ──────────────────────────────────────────────
    let label_vec: Vec<f32> = provisional.iter().map(|&l| l as f32).collect();

    (label_vec, stats)
}

#[cfg(test)]
#[path = "tests_labeling.rs"]
mod tests_labeling;
