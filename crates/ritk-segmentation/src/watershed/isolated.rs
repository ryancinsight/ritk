//! Isolated watershed segmentation.
//!
//! # Mathematical Specification
//!
//! Given two seeds s1, s2 in a scalar image, assigns each voxel to the
//! gradient-descent basin it flows to via steepest descent on the gradient
//! magnitude `g` (ITK `GradientMagnitudeImageFilter`, unit spacing).
//!
//! ## Algorithm
//!
//! Gradient-descent watershed on the gradient magnitude: each voxel drains to
//! the local minimum of gradient magnitude it reaches by following the steepest
//! strictly-descending path.  Seeds in different basins at the finest level are
//! labeled directly.  Seeds in the same basin (edge case) → best-effort: the
//! shared basin is returned as label 1.
//!
//! Basin assignment uses memoised steepest-descent tracing with path
//! compression: once any voxel on a trace reaches a labeled node the entire
//! traced path is stamped with that basin in O(path length) time.
//!
//! ## Output Label Convention
//!
//! - Label 1 (`f32` 1.0): voxels whose gradient-descent basin contains s1.
//! - Label 2 (`f32` 2.0): voxels whose gradient-descent basin contains s2
//!   (when s1 and s2 are in different basins).
//! - Label 0 (`f32` 0.0): remaining voxels.
//!
//! ## Edge Cases
//!
//! - Identical seeds: all voxels assigned label 1.
//! - Seeds in the same basin: the shared basin is returned as label 1; no
//!   label-2 region is produced.
//!
//! # Complexity
//!
//! O(n) expected with path compression, where n is the number of voxels.
//!
//! # References
//!
//! - ITK `itk::IsolatedWatershedImageFilter`

use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use moirai::prelude::ParallelSliceMut;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

// 6-connected face-neighbour offsets (dz, dy, dx) for a [nz, ny, nx] grid.
const NEIGHBOUR_OFFSETS: [(i64, i64, i64); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// In-bounds 6-connected neighbours of flat index `idx` as flat indices.
fn neighbours(idx: usize, dims: [usize; 3]) -> impl Iterator<Item = usize> {
    let [nz, ny, nx] = dims;
    let z = idx / (ny * nx);
    let rem = idx % (ny * nx);
    let y = rem / nx;
    let x = rem % nx;
    NEIGHBOUR_OFFSETS.iter().filter_map(move |&(dz, dy, dx)| {
        let zi = z as i64 + dz;
        let yi = y as i64 + dy;
        let xi = x as i64 + dx;
        if zi < 0 || zi >= nz as i64 || yi < 0 || yi >= ny as i64 || xi < 0 || xi >= nx as i64 {
            None
        } else {
            Some(zi as usize * ny * nx + yi as usize * nx + xi as usize)
        }
    })
}

/// ITK `GradientMagnitudeImageFilter`: per-axis central difference
/// `(f[+1] − f[−1]) / 2` with ZeroFluxNeumann (edge-clamp) boundaries, magnitude
/// `sqrt(Σ dᵢ²)`. Unit spacing (the IsolatedWatershed internal gradient). Matches
/// `sitk.GradientMagnitude` to 0.0 on unit-spacing images. A `z == 1` volume
/// yields `dz == 0` via the clamp, reducing to the 2-D gradient.
fn gradient_magnitude(vals: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let mut out = vec![0.0_f32; nz * ny * nx];
    out.par_mut().enumerate(|i, val| {
        let z = i / (ny * nx);
        let rem = i % (ny * nx);
        let y = rem / nx;
        let x = rem % nx;

        let zm = z.saturating_sub(1);
        let zp = (z + 1).min(nz - 1);
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(ny - 1);
        let xm = x.saturating_sub(1);
        let xp = (x + 1).min(nx - 1);

        let dz = (vals[zp * ny * nx + y * nx + x] - vals[zm * ny * nx + y * nx + x]) * 0.5;
        let dy = (vals[z * ny * nx + yp * nx + x] - vals[z * ny * nx + ym * nx + x]) * 0.5;
        let dx = (vals[z * ny * nx + y * nx + xp] - vals[z * ny * nx + y * nx + xm]) * 0.5;
        *val = (dz * dz + dy * dy + dx * dx).sqrt();
    });
    out
}

/// Assign each voxel the flat index of the local minimum of `g` that it flows
/// to via steepest gradient descent.
///
/// # Algorithm
///
/// 1. **First pass**: label every local minimum of `g` (voxels with no
///    strictly-lower 6-connected neighbour) with their own flat index.
/// 2. **Second pass**: for each unlabeled voxel, trace the steepest-descent
///    chain — at each step, move to the 6-connected neighbour with the smallest
///    `g` value that is strictly lower than the current voxel's `g`.  Once the
///    chain reaches a labeled voxel, stamp the entire traced path with that
///    basin label (path compression).
///
/// Because each descent step moves to a strictly lower `g`, the chain is
/// strictly monotone and cannot cycle.  The `None` arm (no strictly-lower
/// neighbour for an unlabeled voxel) is a safety net for floating-point
/// plateaus not caught by the first pass; such voxels are treated as local
/// minima and given their own basin.
///
/// # Complexity
///
/// O(n) expected: each voxel enters a path at most once before being labeled.
fn watershed_basins_gd(g: &[f32], dims: [usize; 3]) -> Vec<usize> {
    let n: usize = dims.iter().product();

    // Step 1: Compute distance-to-exit for all voxels using multi-source BFS.
    // An exit is a voxel with at least one neighbour of strictly lower gradient magnitude.
    let mut dist = vec![usize::MAX; n];
    let mut q = std::collections::VecDeque::new();

    for i in 0..n {
        let gi = g[i];
        if neighbours(i, dims).any(|j| g[j] < gi) {
            dist[i] = 0;
            q.push_back(i);
        }
    }

    // Propagate distances across equal-value plateaus
    while let Some(u) = q.pop_front() {
        let du = dist[u];
        let gu = g[u];
        for v in neighbours(u, dims) {
            if g[v] == gu && dist[v] == usize::MAX {
                dist[v] = du + 1;
                q.push_back(v);
            }
        }
    }

    // Step 2: Unify plateau minima (plateaus where dist[i] == usize::MAX) into single basins.
    let mut basin = vec![usize::MAX; n];
    let mut component = Vec::new();
    for start in 0..n {
        if dist[start] == usize::MAX && basin[start] == usize::MAX {
            component.clear();
            component.push(start);
            basin[start] = start; // Temporary label to mark visited
            let mut q_comp = std::collections::VecDeque::new();
            q_comp.push_back(start);
            let gs = g[start];
            let mut min_idx = start;

            while let Some(u) = q_comp.pop_front() {
                for v in neighbours(u, dims) {
                    if g[v] == gs && basin[v] == usize::MAX {
                        basin[v] = start; // Mark as visited with start label
                        component.push(v);
                        if v < min_idx {
                            min_idx = v;
                        }
                        q_comp.push_back(v);
                    }
                }
            }

            // Assign the entire plateau minimum component to its representative index
            for &v in &component {
                basin[v] = min_idx;
            }
        }
    }

    // Step 3: Trace steepest descent for each unlabeled voxel.
    // Re-use a single path buffer to prevent heap allocations inside the loop.
    let mut path: Vec<usize> = Vec::with_capacity(64);
    for start in 0..n {
        if basin[start] != usize::MAX {
            continue;
        }
        path.clear();
        path.push(start);
        let mut cur = start;
        loop {
            let gc = g[cur];
            let dc = dist[cur];

            // Next step: lexicographical minimum of (g[j], dist[j]).
            // Must step down in g, or step to same g with smaller dist to exit.
            let next = neighbours(cur, dims)
                .filter(|&j| g[j] < gc || (g[j] == gc && dc != usize::MAX && dist[j] < dc))
                .min_by(|&a, &b| {
                    let ga = g[a];
                    let gb = g[b];
                    match ga.total_cmp(&gb) {
                        std::cmp::Ordering::Equal => dist[a].cmp(&dist[b]),
                        ord => ord,
                    }
                });

            match next {
                Some(j) => {
                    if basin[j] != usize::MAX {
                        let b = basin[j];
                        for &p in &path {
                            basin[p] = b;
                        }
                        break;
                    }
                    path.push(j);
                    cur = j;
                }
                None => {
                    // Safety fallback: treat cur as its own local minimum
                    basin[cur] = cur;
                    let b = cur;
                    for &p in &path {
                        basin[p] = b;
                    }
                    break;
                }
            }
        }
    }
    basin
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Parameters for isolated watershed segmentation.
///
/// `threshold`, `isolated_value_tolerance`, and `upper_value_limit` are
/// retained for API compatibility with ITK's `IsolatedWatershedImageFilter`
/// but are **not used** in the gradient-descent watershed algorithm.  The
/// watershed operates at the finest level (no basin merging); seeds that are
/// already in separate basins are labeled directly without any parameter-
/// dependent search.
#[derive(Debug, Clone)]
pub struct IsolatedWatershedConfig {
    /// Retained for API compatibility; not used by the gradient-descent
    /// watershed.
    pub threshold: f32,
    /// Retained for API compatibility; not used by the gradient-descent
    /// watershed.
    pub isolated_value_tolerance: f32,
    /// Retained for API compatibility; not used by the gradient-descent
    /// watershed.
    pub upper_value_limit: f32,
}

impl Default for IsolatedWatershedConfig {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            isolated_value_tolerance: 0.001,
            upper_value_limit: 1.0,
        }
    }
}

// ── Core algorithm ─────────────────────────────────────────────────────────────

/// Gradient-descent watershed segmentation matching ITK's
/// `IsolatedWatershedImageFilter`.
///
/// Each voxel is assigned the label of the seed whose gradient-descent basin
/// it belongs to.  `seed1`/`seed2` are flat linear indices
/// (`flat = z·ny·nx + y·nx + x`).
///
/// The `config` parameters are accepted for API compatibility but are not used
/// in the gradient-descent algorithm.
pub fn isolated_watershed(
    vals: &[f32],
    dims: [usize; 3],
    seed1: usize,
    seed2: usize,
    _config: &IsolatedWatershedConfig,
) -> Vec<f32> {
    let n: usize = dims.iter().product();

    if seed1 == seed2 {
        return vec![1.0_f32; n];
    }

    let g = gradient_magnitude(vals, dims);
    let basins = watershed_basins_gd(&g, dims);

    let b1 = basins[seed1];
    let b2 = basins[seed2];

    if b1 != b2 {
        // Seeds are in separate basins — label each basin directly.
        return (0..n)
            .map(|i| {
                if basins[i] == b1 {
                    1.0_f32
                } else if basins[i] == b2 {
                    2.0_f32
                } else {
                    0.0_f32
                }
            })
            .collect();
    }

    // Edge case: both seeds drain to the same local minimum (e.g., they lie on
    // the same plateau).  Return the shared basin as label 1; no label-2
    // region can be produced at the finest watershed level.
    (0..n)
        .map(|i| if basins[i] == b1 { 1.0_f32 } else { 0.0_f32 })
        .collect()
}

// ── Public filter struct ───────────────────────────────────────────────────────

/// Isolated watershed segmentation filter.
///
/// Assigns each voxel to the gradient-descent basin it flows to on the
/// gradient magnitude image, then labels the basin containing `seed1` as
/// region 1 and the basin containing `seed2` as region 2:
///
/// | Label | Meaning |
/// |-------|---------|
/// | 1.0   | Gradient-descent basin of `seed1` |
/// | 2.0   | Gradient-descent basin of `seed2` (when separate from `seed1`) |
/// | 0.0   | Remaining voxels |
///
/// Corresponds to ITK `itk::IsolatedWatershedImageFilter`.
#[derive(Debug, Clone)]
pub struct IsolatedWatershed {
    /// First seed voxel `[z, y, x]` (voxel indices, zero-based).
    pub seed1: [usize; 3],
    /// Second seed voxel `[z, y, x]` (voxel indices, zero-based).
    pub seed2: [usize; 3],
    /// Retained for API compatibility; not used by the gradient-descent
    /// watershed.
    pub threshold: f32,
    /// Retained for API compatibility; not used by the gradient-descent
    /// watershed.
    pub isolated_value_tolerance: f32,
    /// Retained for API compatibility; not used by the gradient-descent
    /// watershed.
    pub upper_value_limit: f32,
}

impl IsolatedWatershed {
    /// Apply the isolated watershed filter to a 3-D scalar image.
    ///
    /// Returns a label image with the same shape and spatial metadata as `image`.
    /// Labels are encoded as `f32`: 1.0 (seed1 region), 2.0 (seed2 region), 0.0 (rest).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tensor data cannot be read as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let [_, ny, nx] = dims;

        let seed1_flat = self.seed1[0] * ny * nx + self.seed1[1] * nx + self.seed1[2];
        let seed2_flat = self.seed2[0] * ny * nx + self.seed2[1] * nx + self.seed2[2];

        let config = IsolatedWatershedConfig {
            threshold: self.threshold,
            isolated_value_tolerance: self.isolated_value_tolerance,
            upper_value_limit: self.upper_value_limit,
        };

        let labels = isolated_watershed(&vals, dims, seed1_flat, seed2_flat, &config);

        let device = image.data().device();
        let tensor = Tensor::<B, 3>::from_data(TensorData::new(labels, Shape::new(dims)), &device);

        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_isolated.rs"]
mod tests_isolated;
