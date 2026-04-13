//! Python-exposed segmentation functions: Otsu thresholding and connected-component labeling.
//!
//! # Implementation strategy
//! Both algorithms are implemented inline on CPU-side `Vec<f32>` data, independent
//! of the ritk-core segmentation module state.  This guarantees ritk-python compiles
//! and runs correctly regardless of whether the ritk-core segmentation sub-modules
//! are finalised.
//!
//! # Algorithms
//!
//! ## Otsu's method (Otsu, 1979)
//! Maximises between-class variance σ²_B(t) over a 256-bin histogram:
//!
//!   σ²_B(t) = w₀(t) · w₁(t) · (μ₀(t) − μ₁(t))²
//!
//! where w₀, w₁ are class probabilities and μ₀, μ₁ are class mean bin indices.
//! Solved in O(n + L) using prefix-sum scan over L = 256 bins.
//!
//! ## Hoshen-Kopelman connected-component labeling (two-pass)
//! - Pass 1: scan Z→Y→X; for each foreground voxel examine backward neighbours;
//!   assign provisional labels via union-find with path-halving.
//! - Pass 2: resolve all provisional labels to consecutive final labels [1, K].
//! Complexity: O(n · α(n)) ≈ O(n).

use crate::image::{image_to_vec, into_py_image, vec_to_image_like, PyImage};
use pyo3::prelude::*;

// ── otsu_threshold ────────────────────────────────────────────────────────────

/// Compute the Otsu threshold and produce a binary mask.
///
/// Builds a 256-bin histogram from the image intensities and finds the threshold
/// t* that maximises between-class variance.  Returns both the threshold value
/// and a binary mask where voxels ≥ t* are 1.0 (foreground) and < t* are 0.0.
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     (threshold, mask): threshold value as f32 and binary mask as PyImage.
///
/// Raises:
///     RuntimeError: on internal tensor operation failure.
#[pyfunction]
pub fn otsu_threshold(image: &PyImage) -> PyResult<(f32, PyImage)> {
    let (values, shape) = image_to_vec(image.inner.as_ref())?;

    let threshold = compute_otsu_threshold(&values);

    let mask_vec: Vec<f32> = values
        .iter()
        .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
        .collect();

    let mask_image = vec_to_image_like(mask_vec, shape, image.inner.as_ref());
    Ok((threshold, into_py_image(mask_image)))
}

// ── connected_components ──────────────────────────────────────────────────────

/// Label connected components in a binary mask.
///
/// Applies Hoshen-Kopelman two-pass labeling.  Foreground voxels (value > 0.5)
/// are labeled with consecutive integers [1, K] cast to f32; background voxels
/// remain 0.0.
///
/// Args:
///     mask:         Binary mask PyImage (values 0 or 1).
///     connectivity: 6 (face-adjacent, default) or 26 (face + edge + corner).
///
/// Returns:
///     (labeled_image, num_components): label image and component count K.
///
/// Raises:
///     RuntimeError: on internal tensor operation failure.
///     ValueError:   if connectivity is not 6 or 26.
#[pyfunction]
#[pyo3(signature = (mask, connectivity=6))]
pub fn connected_components(mask: &PyImage, connectivity: u32) -> PyResult<(PyImage, usize)> {
    if connectivity != 6 && connectivity != 26 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "connectivity must be 6 or 26, got {connectivity}"
        )));
    }

    let (values, shape) = image_to_vec(mask.inner.as_ref())?;
    let (label_vec, num_components) = hoshen_kopelman(&values, shape, connectivity);

    let label_image = vec_to_image_like(label_vec, shape, mask.inner.as_ref());
    Ok((into_py_image(label_image), num_components))
}

// ── Otsu core ─────────────────────────────────────────────────────────────────

const NUM_BINS: usize = 256;

/// Compute the Otsu threshold over a flat slice of f32 values.
///
/// # Algorithm
/// 1. Determine [x_min, x_max].  Return x_min for constant images.
/// 2. Build a normalised L=256 bin histogram h[i].
/// 3. Prefix-sum scan: at threshold index t, σ²_B(t) = w₀·w₁·(μ₀−μ₁)².
/// 4. t* = argmax σ²_B; convert to intensity: x_min + t*/(L−1)·range.
pub(crate) fn compute_otsu_threshold(values: &[f32]) -> f32 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }

    // ── Intensity range ───────────────────────────────────────────────────────
    let x_min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let x_max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if (x_max - x_min).abs() < f32::EPSILON {
        return x_min; // constant image — degenerate case
    }

    let range = x_max - x_min;
    let scale = (NUM_BINS - 1) as f32;

    // ── Normalised histogram ─────────────────────────────────────────────────
    let mut counts = [0u64; NUM_BINS];
    for &v in values {
        let bin = ((v - x_min) / range * scale).floor() as usize;
        counts[bin.min(NUM_BINS - 1)] += 1;
    }
    let inv_n = 1.0_f64 / n as f64;
    let h: [f64; NUM_BINS] = std::array::from_fn(|i| counts[i] as f64 * inv_n);

    // ── Total mean (bin index units) ─────────────────────────────────────────
    let total_mu: f64 = (0..NUM_BINS).map(|i| i as f64 * h[i]).sum();

    // ── Prefix-sum scan ──────────────────────────────────────────────────────
    // At threshold t: class 0 = bins [0, t−1], class 1 = bins [t, L−1].
    let mut best_sigma2 = 0.0_f64;
    let mut best_t = 0_usize;
    let mut w0 = 0.0_f64; // Σ h[0..t−1]
    let mut mu0_partial = 0.0_f64; // Σ i·h[i] for i ∈ [0, t−1]

    for t in 1..NUM_BINS {
        // Extend class 0 to include bin t−1.
        w0 += h[t - 1];
        mu0_partial += (t - 1) as f64 * h[t - 1];

        let w1 = 1.0 - w0;
        if w0 < 1e-12 || w1 < 1e-12 {
            continue;
        }

        let mu0 = mu0_partial / w0;
        let mu1 = (total_mu - mu0_partial) / w1;
        let sigma2 = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);

        if sigma2 > best_sigma2 {
            best_sigma2 = sigma2;
            best_t = t;
        }
    }

    // Convert best bin index to intensity units.
    x_min + best_t as f32 / scale * range
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

    /// Find with path-halving (iterative, safe).
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
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

// ── Hoshen-Kopelman core ──────────────────────────────────────────────────────

/// Two-pass Hoshen-Kopelman connected-component labeling on a flat Z×Y×X volume.
///
/// Returns `(label_vec, num_components)` where `label_vec` has the same length
/// as `mask`.
///
/// # Backward-neighbour sets
/// 6-connectivity:  3 offsets (−z, −y, −x face neighbors in scan order).
/// 26-connectivity: 13 offsets (all backward neighbors in a 3×3×3 cube).
fn hoshen_kopelman(mask: &[f32], dims: [usize; 3], connectivity: u32) -> (Vec<f32>, usize) {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let n = nz * ny * nx;

    // Maximum provisional labels = n (worst case: every voxel is isolated).
    // Labels are 1-based; index 0 is background in the union-find.
    let mut uf = UnionFind::new(n + 1);
    let mut provisional = vec![0usize; n];
    let mut next_label = 1usize;

    let flat = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;

    // Backward neighbours (already visited in Z→Y→X scan order).
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
                let center = flat(iz, iy, ix);
                if mask[center] <= 0.5 {
                    continue; // background
                }

                // Collect canonical labels of already-visited foreground neighbours.
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
                    let lbl = provisional[flat(nz_i as usize, ny_i as usize, nx_i as usize)];
                    if lbl > 0 {
                        nbr_labels.push(lbl);
                    }
                }

                if nbr_labels.is_empty() {
                    // No labelled neighbour — start a new component.
                    provisional[center] = next_label;
                    next_label += 1;
                } else {
                    // Union all neighbour labels; assign canonical root to center.
                    let mut root = uf.find(nbr_labels[0]);
                    for &lbl in &nbr_labels[1..] {
                        let r = uf.find(lbl);
                        if r != root {
                            uf.union(root, r);
                            root = uf.find(root);
                        }
                    }
                    provisional[center] = root;
                }
            }
        }
    }

    if next_label == 1 {
        // No foreground voxels.
        return (vec![0.0_f32; n], 0);
    }

    // ── Pass 2: resolve provisional labels to consecutive final labels ─────────
    // root_to_final[canonical_root] = final_label (1-based).
    let mut root_to_final = vec![0usize; next_label];
    let mut num_components = 0usize;

    for i in 0..n {
        if provisional[i] > 0 {
            let root = uf.find(provisional[i]);
            if root_to_final[root] == 0 {
                num_components += 1;
                root_to_final[root] = num_components;
            }
            provisional[i] = root_to_final[root];
        }
    }

    let label_vec: Vec<f32> = provisional.iter().map(|&l| l as f32).collect();
    (label_vec, num_components)
}

/// Register the `segmentation` submodule.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "segmentation")?;
    m.add_function(wrap_pyfunction!(otsu_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(connected_components, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
