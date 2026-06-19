//! Isolated watershed segmentation.
//!
//! # Mathematical Specification
//!
//! Given two seeds s1, s2 in a scalar image I, finds the threshold T* such
//! that at T* the seeds are in separate connected components of {x : I(x) ≤ T*}.
//!
//! ## Algorithm
//!
//! Binary search on T in [threshold, upper_value_limit] with precision
//! `isolated_value_tolerance`:
//! - At each T: BFS from s1 through {x : I(x) ≤ T}; if s2 reachable → T is too
//!   high (seeds still merge); lower the ceiling.
//! - T* = supremum of T such that s1 and s2 are separated.
//!
//! The binary search maintains:
//! - `lo`: highest T seen where seeds are separated (starts at `threshold`)
//! - `hi`: lowest T seen where seeds are connected (starts at `upper_value_limit`)
//!
//! ## Output Label Convention
//!
//! - Label 1 (`f32` 1.0): voxels reachable from s1 through {I ≤ T*}
//! - Label 2 (`f32` 2.0): voxels reachable from s2 through {I ≤ T*}, not already in label 1
//! - Label 3 (`f32` 3.0): remaining voxels (above T* or unreachable from either seed)
//!
//! ## Edge Cases
//!
//! - Identical seeds: all voxels assigned label 1.
//! - Seeds inseparable in `[threshold, upper_value_limit]` (connected even at `lo`):
//!   seed1's reachable region receives label 1, the rest label 3.
//!
//! # Complexity
//!
//! O(log((upper − lower) / tol) · n) where n is the number of voxels.
//!
//! # References
//!
//! - ITK `itk::IsolatedWatershedImageFilter`

use std::collections::VecDeque;

use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
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

// ── Configuration ─────────────────────────────────────────────────────────────

/// Parameters for isolated watershed segmentation.
#[derive(Debug, Clone)]
pub struct IsolatedWatershedConfig {
    /// Lower bound of the binary search range.
    pub threshold: f32,
    /// Convergence tolerance for the binary search.
    pub isolated_value_tolerance: f32,
    /// Upper bound of the binary search range.
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

// ── Core primitives ────────────────────────────────────────────────────────────

/// Returns `true` if `seed2` is reachable from `seed1` through voxels with value ≤ `t`.
///
/// Uses 6-connected BFS. If `seed1` itself has value > `t` it is inactive and
/// the function immediately returns `false`.
fn seeds_connected(vals: &[f32], dims: [usize; 3], seed1: usize, seed2: usize, t: f32) -> bool {
    if vals[seed1] > t {
        return false;
    }
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    visited[seed1] = true;
    queue.push_back(seed1);
    while let Some(idx) = queue.pop_front() {
        if idx == seed2 {
            return true;
        }
        let z = idx / (ny * nx);
        let rem = idx % (ny * nx);
        let y = rem / nx;
        let x = rem % nx;
        for &(dz, dy, dx) in &NEIGHBOUR_OFFSETS {
            let zi = z as i64 + dz;
            let yi = y as i64 + dy;
            let xi = x as i64 + dx;
            if zi < 0 || zi >= nz as i64 || yi < 0 || yi >= ny as i64 || xi < 0 || xi >= nx as i64 {
                continue;
            }
            let ni = zi as usize * ny * nx + yi as usize * nx + xi as usize;
            if !visited[ni] && vals[ni] <= t {
                visited[ni] = true;
                queue.push_back(ni);
            }
        }
    }
    false
}

/// Flood-fill BFS from `start` through voxels with value ≤ `t`.
///
/// Returns a boolean mask (`true` = reachable). If `start` has value > `t`
/// the mask is entirely `false`.
fn bfs_flood(vals: &[f32], dims: [usize; 3], start: usize, t: f32) -> Vec<bool> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut reached = vec![false; n];
    if vals[start] > t {
        return reached;
    }
    let mut queue = VecDeque::new();
    reached[start] = true;
    queue.push_back(start);
    while let Some(idx) = queue.pop_front() {
        let z = idx / (ny * nx);
        let rem = idx % (ny * nx);
        let y = rem / nx;
        let x = rem % nx;
        for &(dz, dy, dx) in &NEIGHBOUR_OFFSETS {
            let zi = z as i64 + dz;
            let yi = y as i64 + dy;
            let xi = x as i64 + dx;
            if zi < 0 || zi >= nz as i64 || yi < 0 || yi >= ny as i64 || xi < 0 || xi >= nx as i64 {
                continue;
            }
            let ni = zi as usize * ny * nx + yi as usize * nx + xi as usize;
            if !reached[ni] && vals[ni] <= t {
                reached[ni] = true;
                queue.push_back(ni);
            }
        }
    }
    reached
}

// ── Core algorithm ─────────────────────────────────────────────────────────────

/// Isolated watershed on a flat voxel array with shape `[nz, ny, nx]`.
///
/// Returns a label vector (`Vec<f32>`) of the same length as `vals`:
/// - `1.0`: region reachable from `seed1` at T*
/// - `2.0`: region reachable from `seed2` at T* (disjoint from label-1 set)
/// - `3.0`: all remaining voxels
///
/// `seed1` and `seed2` are flat linear indices into `vals` using row-major
/// layout: `flat = z * ny * nx + y * nx + x`.
pub fn isolated_watershed(
    vals: &[f32],
    dims: [usize; 3],
    seed1: usize,
    seed2: usize,
    config: &IsolatedWatershedConfig,
) -> Vec<f32> {
    let n: usize = dims.iter().product();

    // Edge case: identical seeds → single region.
    if seed1 == seed2 {
        return vec![1.0_f32; n];
    }

    let lower = config.threshold;
    let upper = config.upper_value_limit;
    let tol = config.isolated_value_tolerance.max(f32::EPSILON);

    // Iteration budget: ceil(log2((upper − lower) / tol)), capped at 50.
    let range = (upper - lower).max(0.0_f32);
    let max_iter: usize = if range <= 0.0 {
        1
    } else {
        let ratio = range / tol;
        if ratio <= 1.0 {
            1
        } else {
            // SAFETY: ratio > 1.0 ⟹ log2(ratio) > 0 ⟹ ceil gives a positive integer.
            (ratio.log2().ceil() as usize).min(50)
        }
    };

    // Binary search for T* = sup{T ∈ [lower, upper] : seeds separated at T}.
    //
    // Invariant:
    //   lo — highest T confirmed separated (or lower as the initial lower bound)
    //   hi — lowest T confirmed connected (or upper as the initial upper bound)
    //
    // At each step:
    //   seeds_connected(mid) → hi = mid   (T too high, bring ceiling down)
    //   seeds_separated(mid) → lo = mid   (T still valid, try to raise the floor)
    // Converges when hi − lo < tol. T* = lo.
    let mut lo = lower;
    let mut hi = upper;

    for _ in 0..max_iter {
        if hi - lo < tol {
            break;
        }
        let mid = (lo + hi) * 0.5;
        if seeds_connected(vals, dims, seed1, seed2, mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let t_star = lo;

    // Flood-fill from seed1 at T*.
    let reached1 = bfs_flood(vals, dims, seed1, t_star);

    // Can't-isolate case: seeds are connected even at T* (binary search was unable
    // to push lo above the always-connected floor).  Label seed1's entire region
    // as 1 and the rest as 3.
    if reached1[seed2] {
        return reached1
            .iter()
            .map(|&r| if r { 1.0_f32 } else { 3.0_f32 })
            .collect();
    }

    // Flood-fill from seed2, excluding voxels already in seed1's region.
    let reached2_raw = bfs_flood(vals, dims, seed2, t_star);
    // Mask out any overlap with seed1's region (shouldn't occur post-separation
    // check, but guards against floating-point edge cases at the boundary).
    let reached2: Vec<bool> = reached2_raw
        .iter()
        .zip(reached1.iter())
        .map(|(&r2, &r1)| r2 && !r1)
        .collect();

    // Assign final labels.
    reached1
        .iter()
        .zip(reached2.iter())
        .map(|(&r1, &r2)| {
            if r1 {
                1.0_f32
            } else if r2 {
                2.0_f32
            } else {
                3.0_f32
            }
        })
        .collect()
}

// ── Public filter struct ───────────────────────────────────────────────────────

/// Isolated watershed segmentation filter.
///
/// Finds T* — the highest threshold at which `seed1` and `seed2` remain in
/// separate connected components of {I ≤ T*} — then labels voxels by region:
///
/// | Label | Meaning |
/// |-------|---------|
/// | 1.0   | Reachable from `seed1` at T* |
/// | 2.0   | Reachable from `seed2` at T* (disjoint from label 1) |
/// | 3.0   | Remaining voxels |
///
/// Corresponds to ITK `itk::IsolatedWatershedImageFilter`.
#[derive(Debug, Clone)]
pub struct IsolatedWatershed {
    /// First seed voxel `[z, y, x]` (voxel indices, zero-based).
    pub seed1: [usize; 3],
    /// Second seed voxel `[z, y, x]` (voxel indices, zero-based).
    pub seed2: [usize; 3],
    /// Lower bound of the binary search range.
    pub threshold: f32,
    /// Convergence tolerance for the binary search (stops when `hi − lo < tol`).
    pub isolated_value_tolerance: f32,
    /// Upper bound of the binary search range.
    pub upper_value_limit: f32,
}

impl IsolatedWatershed {
    /// Apply the isolated watershed filter to a 3-D scalar image.
    ///
    /// Returns a label image with the same shape and spatial metadata as `image`.
    /// Labels are encoded as `f32`: 1.0 (seed1 region), 2.0 (seed2 region), 3.0 (rest).
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
