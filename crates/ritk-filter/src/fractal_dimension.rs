//! Stochastic fractal dimension filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Ports `itk::StochasticFractalDimensionImageFilter`. For each voxel the filter
//! estimates the local fractal dimension from the scaling of intensity
//! differences with physical distance inside a `(2R+1)^3` neighborhood (default
//! radius `R = 2` per axis).
//!
//! For every ordered pair of in-bounds neighborhood members `(p, q)`, `p ≠ q`,
//! the squared physical distance
//!
//! ```text
//! d²(p, q) = Σ_a ( spacing_a · (offset_p,a − offset_q,a) )²
//! ```
//!
//! is greedily binned: the pair joins the first existing bin whose stored
//! squared distance differs by less than `0.5·min_a spacing_a`, otherwise it
//! opens a new bin. Each bin accumulates the running mean absolute intensity
//! difference `mean_k = ⟨|I_p − I_q|⟩`. A least-squares line is fitted to
//! `(ln√d², ln mean_k)` over the `N` bins, and the output is
//!
//! ```text
//! D = 3 − slope,    slope = (N·Σxy − Σx·Σy) / (N·Σxx − Σx²)
//! ```
//!
//! with `x = ln√d²` and `y = ln mean_k`. The squared distance reduces to the
//! offset/spacing form above because `TransformIndexToPhysicalPoint` differs
//! between two members of one neighborhood only by the (orthonormal) direction
//! matrix applied to the index offset, which preserves Euclidean distance.
//!
//! # Boundary
//!
//! Neighborhood members whose index leaves the image are skipped (ITK
//! `IsInBounds` on the `ConstNeighborhoodIterator`), so edge voxels fit the
//! regression over fewer pairs — no padding or reflection is introduced.
//!
//! # ITK parity
//!
//! Corresponds to `itk::StochasticFractalDimensionImageFilter` (default
//! `NeighborhoodRadius = 2` on every axis). No mask is applied (the unmasked
//! ITK path). A constant neighborhood yields `mean_k = 0`, whose logarithm is
//! `−∞`; ITK produces the same non-finite output there, so constant inputs are
//! degenerate by construction and excluded from value tests.
//!
//! # Complexity
//!
//! O(M² · B) per voxel, with `M = (2R+1)^3` neighborhood members and `B` the
//! number of distinct distance bins.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Stochastic fractal dimension filter (`itk::StochasticFractalDimensionImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct StochasticFractalDimensionFilter {
    /// Neighborhood radius per axis in `[z, y, x]` order. ITK default `[2, 2, 2]`.
    pub radius: [usize; 3],
}

impl Default for StochasticFractalDimensionFilter {
    fn default() -> Self {
        Self { radius: [2, 2, 2] }
    }
}

impl StochasticFractalDimensionFilter {
    /// Construct with a per-axis neighborhood radius (`[z, y, x]`).
    pub fn new(radius: [usize; 3]) -> Self {
        Self { radius }
    }

    /// Estimate the per-voxel stochastic fractal dimension of a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec_infallible(image);
        let [nz, ny, nx] = dims;
        let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];
        let origin = [image.origin()[0], image.origin()[1], image.origin()[2]];
        let mut dir = [[0.0f64; 3]; 3];
        for (c, row) in dir.iter_mut().enumerate() {
            for (axis, d) in row.iter_mut().enumerate() {
                *d = image.direction()[(c, axis)];
            }
        }
        let min_spacing = spacing.iter().cloned().fold(f64::INFINITY, f64::min);
        let tol = 0.5 * min_spacing;
        let [rz, ry, rx] = self.radius;

        // f64 physical point of an absolute `[z, y, x]` index, matching ITK's
        // `TransformIndexToPhysicalPoint` (`origin + Direction·(index⊙spacing)`).
        // Distances are formed by subtracting two such absolute points so the
        // greedy distance binning reproduces ITK's exact floating-point ties,
        // including the catastrophic cancellation of the absolute-coordinate
        // form — a direct `spacing·Δ` shortcut is more accurate but flips
        // bin-boundary ties against the reference on anisotropic spacing.
        let physical = |iz: f64, iy: f64, ix: f64| -> [f64; 3] {
            let scaled = [iz * spacing[0], iy * spacing[1], ix * spacing[2]];
            let mut pt = origin;
            for (c, p) in pt.iter_mut().enumerate() {
                *p += dir[c][0] * scaled[0] + dir[c][1] * scaled[1] + dir[c][2] * scaled[2];
            }
            pt
        };

        // Neighborhood offsets in ITK raster order (x fastest), shared by every
        // voxel so the greedy binning order matches the reference exactly.
        let mut offsets: Vec<[isize; 3]> =
            Vec::with_capacity((2 * rz + 1) * (2 * ry + 1) * (2 * rx + 1));
        for dz in -(rz as isize)..=(rz as isize) {
            for dy in -(ry as isize)..=(ry as isize) {
                for dx in -(rx as isize)..=(rx as isize) {
                    offsets.push([dz, dy, dx]);
                }
            }
        }

        // Each output voxel is independent (read-only neighborhood), so the grid
        // fans out across threads; per-voxel scratch is thread-local and the
        // result is deterministic (bitwise identical to a serial run) because no
        // voxel depends on another's output.
        let out: Vec<f32> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(vals.len(), |flat| {
                let cz = flat / (ny * nx);
                let rem = flat % (ny * nx);
                let cy = rem / nx;
                let cx = rem % nx;

                let mut members: Vec<([f64; 3], f32)> = Vec::with_capacity(offsets.len());
                for off in &offsets {
                    let iz = cz as isize + off[0];
                    let iy = cy as isize + off[1];
                    let ix = cx as isize + off[2];
                    if iz < 0
                        || iy < 0
                        || ix < 0
                        || iz >= nz as isize
                        || iy >= ny as isize
                        || ix >= nx as isize
                    {
                        continue;
                    }
                    let v = vals[(iz as usize) * ny * nx + (iy as usize) * nx + ix as usize];
                    members.push((physical(iz as f64, iy as f64, ix as f64), v));
                }

                let mut bin_dist: Vec<f64> = Vec::new();
                let mut bin_freq: Vec<f64> = Vec::new();
                let mut bin_accum: Vec<f64> = Vec::new();
                for (i, &(pi, vi)) in members.iter().enumerate() {
                    for (j, &(pj, vj)) in members.iter().enumerate() {
                        if i == j {
                            continue;
                        }
                        // Squared physical distance in ITK's (x, y, z) axis order:
                        // internal point columns are [z, y, x], so the x term is
                        // column 2, y is 1, z is 0. Absolute points are subtracted
                        // (not `spacing·Δ`) to match ITK's exact rounding at
                        // bin-boundary ties.
                        let ex = pi[2] - pj[2];
                        let ey = pi[1] - pj[1];
                        let ez = pi[0] - pj[0];
                        let d2 = ex * ex + ey * ey + ez * ez;
                        let diff = (vi - vj).abs() as f64;
                        let mut found = false;
                        for k in 0..bin_dist.len() {
                            if (bin_dist[k] - d2).abs() < tol {
                                bin_freq[k] += 1.0;
                                bin_accum[k] += diff;
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            bin_dist.push(d2);
                            bin_freq.push(1.0);
                            bin_accum.push(diff);
                        }
                    }
                }

                let (mut sum_x, mut sum_y, mut sum_xx, mut sum_xy) = (0.0, 0.0, 0.0, 0.0);
                for k in 0..bin_dist.len() {
                    if bin_freq[k] == 0.0 {
                        continue;
                    }
                    let mean = bin_accum[k] / bin_freq[k];
                    let y = mean.ln();
                    let x = bin_dist[k].sqrt().ln();
                    sum_y += y;
                    sum_x += x;
                    sum_xx += x * x;
                    sum_xy += y * x;
                }
                let n = bin_dist.len() as f64;
                let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
                (3.0 - slope) as f32
            });

        rebuild(out, dims, image)
    }
}

#[cfg(test)]
#[path = "tests_fractal_dimension.rs"]
mod tests_fractal_dimension;
