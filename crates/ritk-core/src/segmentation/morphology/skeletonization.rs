//! Topology-preserving skeletonization (thinning) for binary images.
//!
//! # Mathematical Specification
//!
//! Given a binary image M: ℤᴰ → {0, 1}, skeletonization produces a thin
//! subset S ⊆ M that preserves the topology of M (same number of connected
//! components, tunnels, and cavities) while removing as many foreground
//! voxels as possible.
//!
//! ## Definition (Simple Point)
//!
//! A foreground point p is **simple** if its removal does not change the
//! topology of the image. For the (FG=26, BG=6) adjacency pair in 3-D:
//!
//!   p is simple ⟺ T₂₆(p) = 1
//!
//! where T₂₆(p) is the number of 26-connected foreground components in
//! N₂₆(p) \ {p} (the 26-neighborhood excluding the center).
//!
//! **Proof sketch (Bertrand & Malandain, 1994):** For (26, 6) adjacency,
//! T₂₆(p) = 1 implies T̄₆(p) = 1 (exactly one 6-connected background
//! component 6-adjacent to p in N₁₈(p) \ {p}). Both conditions together
//! are necessary and sufficient for simplicity. Since the first implies the
//! second under (26, 6) adjacency, T₂₆(p) = 1 alone suffices.
//!
//! ## Definition (Endpoint)
//!
//! A foreground point p is an **endpoint** if it has at most one foreground
//! neighbor in its full connectivity neighborhood:
//! - D = 2: |{q ∈ N₈(p) \ {p} : M(q) = 1}| ≤ 1
//! - D = 3: |{q ∈ N₂₆(p) \ {p} : M(q) = 1}| ≤ 1
//!
//! Endpoints are never removed, ensuring the skeleton retains branch tips
//! and produces a **curve skeleton** (medial axis).
//!
//! ## Algorithm — D = 2: Zhang–Suen Thinning (1984)
//!
//! Two sub-iterations per pass on 8-connected foreground / 4-connected background.
//! Neighbors P₂..P₉ are labeled clockwise from north. Let:
//! - B(p) = Σ Pᵢ (count of foreground 8-neighbors)
//! - A(p) = number of 0→1 transitions in the cyclic sequence P₂..P₉,P₂
//!
//! **Sub-iteration 1:** Delete p if 2 ≤ B(p) ≤ 6, A(p) = 1,
//! P₂·P₄·P₆ = 0, and P₄·P₆·P₈ = 0.
//!
//! **Sub-iteration 2:** Delete p if 2 ≤ B(p) ≤ 6, A(p) = 1,
//! P₂·P₄·P₈ = 0, and P₂·P₆·P₈ = 0.
//!
//! Repeat until neither sub-iteration removes any pixel.
//!
//! ## Algorithm — D = 3: 6-Directional Sequential Thinning
//!
//! Each iteration comprises 6 sub-iterations (one per face direction:
//! ±z, ±y, ±x). In each sub-iteration:
//!
//! 1. Identify **border voxels**: foreground voxels whose face-neighbor in
//!    the current direction is background (or out of bounds).
//! 2. For each border voxel p (raster order), re-check:
//!    a. p is still foreground (may have been removed earlier in this pass).
//!    b. p is not an endpoint (|N₂₆(p) ∩ FG| > 1).
//!    c. p is a simple point (T₂₆(p) = 1).
//!    If all hold, delete p immediately (sequential deletion).
//! 3. Repeat until a full pass (all 6 directions) removes zero voxels.
//!
//! **Correctness:** Each individual deletion is of a verified simple point
//! in the current configuration, which preserves topology by definition.
//! Sequential re-checking ensures no two conflicting deletions occur.
//!
//! ## Algorithm — D = 1: Medial Point Extraction
//!
//! For each maximal connected foreground run [a, b], retain only the
//! midpoint ⌊(a + b) / 2⌋. This preserves one point per connected
//! component (topology) and selects the medial position.
//!
//! ## Complexity
//!
//! - D = 1: O(n) single pass.
//! - D = 2: O(n · k) where k is the number of passes (bounded by
//!   max(ny, nx) / 2).
//! - D = 3: O(n · k · 6) where k is the iteration count, plus O(26²)
//!   per simple-point test (constant).
//!
//! # References
//!
//! - Zhang, T.Y. & Suen, C.Y. (1984). "A Fast Parallel Algorithm for
//!   Thinning Digital Patterns." *Communications of the ACM*, 27(3), 236-239.
//! - Bertrand, G. & Malandain, G. (1994). "A New Characterization of
//!   Three-Dimensional Simple Points." *Pattern Recognition Letters*,
//!   15(2), 169-175.
//! - Lee, T.C., Kashyap, R.L. & Chu, C.N. (1994). "Building Skeleton
//!   Models via 3-D Medial Surface/Axis Thinning Algorithms." *CVGIP:
//!   Graphical Models and Image Processing*, 56(6), 462-478.
//! - Palágyi, K. & Kuba, A. (1999). "A Parallel 3D 12-Subiteration
//!   Thinning Algorithm." *Graphical Models and Image Processing*, 61(4),
//!   199-221.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

// ── Public types ─────────────────────────────────────────────────────────────

/// Topology-preserving skeletonization (thinning) filter.
///
/// Reduces a binary mask to its medial axis / curve skeleton while
/// preserving topology (connected components, tunnels, cavities).
/// Endpoints are preserved to retain branch structure.
///
/// Supports D = 1, 2, 3.
pub struct Skeletonization;

impl Skeletonization {
    /// Create a `Skeletonization` filter.
    pub fn new() -> Self {
        Self
    }

    /// Apply skeletonization to a binary mask image.
    ///
    /// Returns a binary mask containing the skeleton (values in {0.0, 1.0})
    /// with the same shape and spatial metadata as `mask`.
    ///
    /// # Panics
    /// Panics if D is not 1, 2, or 3.
    pub fn apply<B: Backend, const D: usize>(&self, mask: &Image<B, D>) -> Image<B, D> {
        let shape: [usize; D] = mask.shape();
        let device = mask.data().device();

        let mask_data = mask.data().clone().into_data();
        let flat = mask_data.as_slice::<f32>().expect("f32 mask tensor data");

        let output = skeleton_nd(flat, &shape);

        let tensor = Tensor::<B, D>::from_data(TensorData::new(output, Shape::new(shape)), &device);

        Image::new(
            tensor,
            mask.origin().clone(),
            mask.spacing().clone(),
            mask.direction().clone(),
        )
    }
}

impl Default for Skeletonization {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend, const D: usize> super::MorphologicalOperation<B, D> for Skeletonization {
    fn apply(&self, mask: &Image<B, D>) -> Image<B, D> {
        self.apply(mask)
    }
}

// ── Dispatcher ───────────────────────────────────────────────────────────────

fn skeleton_nd(flat: &[f32], shape: &[usize]) -> Vec<f32> {
    match shape.len() {
        1 => skeleton_1d(flat, shape[0]),
        2 => skeleton_2d(flat, shape[0], shape[1]),
        3 => skeleton_3d(flat, shape[0], shape[1], shape[2]),
        d => {
            panic!("Skeletonization: unsupported dimensionality D={d}; only D=1,2,3 are supported")
        }
    }
}

// ── D = 1 : Medial point extraction ──────────────────────────────────────────

/// For each maximal connected foreground run, retain only the midpoint.
fn skeleton_1d(flat: &[f32], nx: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; nx];
    let mut i = 0;
    while i < nx {
        if flat[i] > 0.5 {
            let start = i;
            while i < nx && flat[i] > 0.5 {
                i += 1;
            }
            // Run is [start, i-1] inclusive.
            let mid = (start + (i - 1)) / 2;
            output[mid] = 1.0;
        } else {
            i += 1;
        }
    }
    output
}

// ── D = 2 : Zhang–Suen thinning ─────────────────────────────────────────────

/// Read a pixel from the mask, treating out-of-bounds as background.
#[inline]
fn pixel(mask: &[bool], ny: usize, nx: usize, y: isize, x: isize) -> u8 {
    if y < 0 || y >= ny as isize || x < 0 || x >= nx as isize {
        0
    } else {
        mask[y as usize * nx + x as usize] as u8
    }
}

/// Count 0→1 transitions in the cyclic neighbor sequence P₂..P₉,P₂.
#[inline]
fn transitions(nb: &[u8; 8]) -> u8 {
    let mut count = 0u8;
    for i in 0..8 {
        if nb[i] == 0 && nb[(i + 1) % 8] == 1 {
            count += 1;
        }
    }
    count
}

/// One sub-iteration of Zhang–Suen. Returns the number of pixels removed.
fn zhang_suen_step(mask: &mut [bool], ny: usize, nx: usize, step1: bool) -> usize {
    let mut to_remove: Vec<usize> = Vec::new();

    for iy in 0..ny {
        for ix in 0..nx {
            if !mask[iy * nx + ix] {
                continue;
            }
            let y = iy as isize;
            let x = ix as isize;

            // Clockwise from north: P2, P3, P4, P5, P6, P7, P8, P9.
            let nb: [u8; 8] = [
                pixel(mask, ny, nx, y - 1, x),     // P2 north
                pixel(mask, ny, nx, y - 1, x + 1), // P3 northeast
                pixel(mask, ny, nx, y, x + 1),     // P4 east
                pixel(mask, ny, nx, y + 1, x + 1), // P5 southeast
                pixel(mask, ny, nx, y + 1, x),     // P6 south
                pixel(mask, ny, nx, y + 1, x - 1), // P7 southwest
                pixel(mask, ny, nx, y, x - 1),     // P8 west
                pixel(mask, ny, nx, y - 1, x - 1), // P9 northwest
            ];

            let b: u8 = nb.iter().sum();
            if b < 2 || b > 6 {
                continue;
            }

            if transitions(&nb) != 1 {
                continue;
            }

            let (p2, p4, p6, p8) = (nb[0], nb[2], nb[4], nb[6]);

            if step1 {
                // Sub-iteration 1: P2·P4·P6 = 0 AND P4·P6·P8 = 0
                if p2 * p4 * p6 != 0 {
                    continue;
                }
                if p4 * p6 * p8 != 0 {
                    continue;
                }
            } else {
                // Sub-iteration 2: P2·P4·P8 = 0 AND P2·P6·P8 = 0
                if p2 * p4 * p8 != 0 {
                    continue;
                }
                if p2 * p6 * p8 != 0 {
                    continue;
                }
            }

            to_remove.push(iy * nx + ix);
        }
    }

    let count = to_remove.len();
    for idx in to_remove {
        mask[idx] = false;
    }
    count
}

fn skeleton_2d(flat: &[f32], ny: usize, nx: usize) -> Vec<f32> {
    let mut mask: Vec<bool> = flat.iter().map(|&v| v > 0.5).collect();

    loop {
        let removed1 = zhang_suen_step(&mut mask, ny, nx, true);
        let removed2 = zhang_suen_step(&mut mask, ny, nx, false);
        if removed1 == 0 && removed2 == 0 {
            break;
        }
    }

    mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
}

// ── D = 3 : Directional sequential thinning ──────────────────────────────────

/// Count the number of 26-connected foreground neighbors of voxel (z, y, x).
#[inline]
fn count_26_neighbors(
    mask: &[bool],
    nz: usize,
    ny: usize,
    nx: usize,
    z: usize,
    y: usize,
    x: usize,
) -> usize {
    let mut count = 0usize;
    for dz in -1isize..=1 {
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                if dz == 0 && dy == 0 && dx == 0 {
                    continue;
                }
                let gz = z as isize + dz;
                let gy = y as isize + dy;
                let gx = x as isize + dx;
                if gz < 0
                    || gz >= nz as isize
                    || gy < 0
                    || gy >= ny as isize
                    || gx < 0
                    || gx >= nx as isize
                {
                    continue;
                }
                if mask[gz as usize * ny * nx + gy as usize * nx + gx as usize] {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Count the number of 26-connected foreground components in a local
/// 3×3×3 neighborhood array (center at index 13 excluded).
///
/// Uses depth-first search on a 27-element array. Each position
/// (lz, ly, lx) ∈ {0,1,2}³ maps to flat index lz·9 + ly·3 + lx.
/// Two positions are 26-adjacent iff they differ by at most 1 in each axis.
///
/// # Returns
/// The number of 26-connected foreground components (excluding center).
fn fg_components_26(local: &[bool; 27]) -> usize {
    let mut visited = [false; 27];
    visited[13] = true; // center is excluded

    let mut count = 0usize;

    for i in 0..27 {
        if i == 13 || !local[i] || visited[i] {
            continue;
        }

        count += 1;

        // DFS using a fixed-size stack (max 26 elements).
        let mut stack = [0usize; 26];
        let mut top: usize = 0;
        stack[top] = i;
        top += 1;
        visited[i] = true;

        while top > 0 {
            top -= 1;
            let curr = stack[top];
            let cz = curr / 9;
            let cy = (curr % 9) / 3;
            let cx = curr % 3;

            for dz in -1isize..=1 {
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        if dz == 0 && dy == 0 && dx == 0 {
                            continue;
                        }
                        let nz_l = cz as isize + dz;
                        let ny_l = cy as isize + dy;
                        let nx_l = cx as isize + dx;
                        if nz_l < 0 || nz_l >= 3 || ny_l < 0 || ny_l >= 3 || nx_l < 0 || nx_l >= 3 {
                            continue;
                        }
                        let ni = nz_l as usize * 9 + ny_l as usize * 3 + nx_l as usize;
                        if ni == 13 || !local[ni] || visited[ni] {
                            continue;
                        }
                        visited[ni] = true;
                        stack[top] = ni;
                        top += 1;
                    }
                }
            }
        }
    }

    count
}

/// Test whether the foreground voxel at (z, y, x) is a simple point
/// under (26, 6) adjacency.
///
/// Returns `true` iff T₂₆(p) = 1: exactly one 26-connected foreground
/// component exists in N₂₆(p) \ {p}.
#[inline]
fn is_simple_3d(
    mask: &[bool],
    nz: usize,
    ny: usize,
    nx: usize,
    z: usize,
    y: usize,
    x: usize,
) -> bool {
    // Extract 3×3×3 neighborhood into a local array.
    let mut local = [false; 27];
    for lz in 0..3usize {
        for ly in 0..3usize {
            for lx in 0..3usize {
                let gz = z as isize + lz as isize - 1;
                let gy = y as isize + ly as isize - 1;
                let gx = x as isize + lx as isize - 1;
                if gz >= 0
                    && gz < nz as isize
                    && gy >= 0
                    && gy < ny as isize
                    && gx >= 0
                    && gx < nx as isize
                {
                    local[lz * 9 + ly * 3 + lx] =
                        mask[gz as usize * ny * nx + gy as usize * nx + gx as usize];
                }
            }
        }
    }
    // Exclude center.
    local[13] = false;

    fg_components_26(&local) == 1
}

/// 6-directional sequential thinning for 3-D binary images.
fn skeleton_3d(flat: &[f32], nz: usize, ny: usize, nx: usize) -> Vec<f32> {
    let n = nz * ny * nx;
    let mut mask: Vec<bool> = flat.iter().map(|&v| v > 0.5).collect();

    let flat_idx = |z: usize, y: usize, x: usize| -> usize { z * ny * nx + y * nx + x };

    // 6 face-direction offsets for sub-iterations.
    let directions: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    loop {
        let mut any_removed = false;

        for &(dz, dy, dx) in &directions {
            // Collect candidate border voxels for this direction.
            let mut candidates: Vec<usize> = Vec::new();

            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let idx = flat_idx(iz, iy, ix);
                        if !mask[idx] {
                            continue;
                        }

                        // Border condition: face-neighbor in this direction is background.
                        let gz = iz as isize + dz;
                        let gy = iy as isize + dy;
                        let gx = ix as isize + dx;

                        let is_border = if gz < 0
                            || gz >= nz as isize
                            || gy < 0
                            || gy >= ny as isize
                            || gx < 0
                            || gx >= nx as isize
                        {
                            true
                        } else {
                            !mask[flat_idx(gz as usize, gy as usize, gx as usize)]
                        };

                        if is_border {
                            candidates.push(idx);
                        }
                    }
                }
            }

            // Sequential processing with re-check.
            for &idx in &candidates {
                if !mask[idx] {
                    continue; // already removed earlier in this sub-iteration
                }

                let iz = idx / (ny * nx);
                let rem = idx % (ny * nx);
                let iy = rem / nx;
                let ix = rem % nx;

                // Endpoint preservation: skip if ≤ 1 foreground 26-neighbor.
                if count_26_neighbors(&mask, nz, ny, nx, iz, iy, ix) <= 1 {
                    continue;
                }

                // Simple point test: T₂₆ = 1.
                if is_simple_3d(&mask, nz, ny, nx, iz, iy, ix) {
                    mask[idx] = false;
                    any_removed = true;
                }
            }
        }

        if !any_removed {
            break;
        }
    }

    let mut output = vec![0.0_f32; n];
    for (i, &b) in mask.iter().enumerate() {
        if b {
            output[i] = 1.0;
        }
    }
    output
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    // ── 1-D helpers ───────────────────────────────────────────────────────

    fn make_mask_1d(values: Vec<f32>, nx: usize) -> Image<TestBackend, 1> {
        let device: <TestBackend as Backend>::Device = Default::default();
        let td = TensorData::new(values, Shape::new([nx]));
        let tensor = Tensor::<TestBackend, 1>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0]),
            Spacing::new([1.0]),
            Direction::identity(),
        )
    }

    fn values_1d(image: &Image<TestBackend, 1>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    fn count_fg_1d(image: &Image<TestBackend, 1>) -> usize {
        values_1d(image).iter().filter(|&&v| v > 0.5).count()
    }

    // ── 2-D helpers ───────────────────────────────────────────────────────

    fn make_mask_2d(values: Vec<f32>, shape: [usize; 2]) -> Image<TestBackend, 2> {
        let device: <TestBackend as Backend>::Device = Default::default();
        let td = TensorData::new(values, Shape::new(shape));
        let tensor = Tensor::<TestBackend, 2>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 2]),
            Spacing::new([1.0; 2]),
            Direction::identity(),
        )
    }

    fn values_2d(image: &Image<TestBackend, 2>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    fn count_fg_2d(image: &Image<TestBackend, 2>) -> usize {
        values_2d(image).iter().filter(|&&v| v > 0.5).count()
    }

    // ── 3-D helpers ───────────────────────────────────────────────────────

    fn make_mask_3d(values: Vec<f32>, shape: [usize; 3]) -> Image<TestBackend, 3> {
        let device: <TestBackend as Backend>::Device = Default::default();
        let td = TensorData::new(values, Shape::new(shape));
        let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    fn values_3d(image: &Image<TestBackend, 3>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    fn count_fg_3d(image: &Image<TestBackend, 3>) -> usize {
        values_3d(image).iter().filter(|&&v| v > 0.5).count()
    }

    // ── Connected component counting (for topology checks) ───────────────

    /// Count 8-connected foreground components in a 2-D flat mask.
    fn count_components_2d(flat: &[f32], ny: usize, nx: usize) -> usize {
        let n = ny * nx;
        let mask: Vec<bool> = flat.iter().map(|&v| v > 0.5).collect();
        let mut visited = vec![false; n];
        let mut components = 0usize;

        for start in 0..n {
            if !mask[start] || visited[start] {
                continue;
            }
            components += 1;
            let mut stack = vec![start];
            visited[start] = true;
            while let Some(idx) = stack.pop() {
                let iy = idx / nx;
                let ix = idx % nx;
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        if dy == 0 && dx == 0 {
                            continue;
                        }
                        let ny_i = iy as isize + dy;
                        let nx_i = ix as isize + dx;
                        if ny_i < 0 || ny_i >= ny as isize || nx_i < 0 || nx_i >= nx as isize {
                            continue;
                        }
                        let ni = ny_i as usize * nx + nx_i as usize;
                        if mask[ni] && !visited[ni] {
                            visited[ni] = true;
                            stack.push(ni);
                        }
                    }
                }
            }
        }
        components
    }

    /// Count 26-connected foreground components in a 3-D flat mask.
    fn count_components_3d(flat: &[f32], nz: usize, ny: usize, nx: usize) -> usize {
        let n = nz * ny * nx;
        let mask: Vec<bool> = flat.iter().map(|&v| v > 0.5).collect();
        let mut visited = vec![false; n];
        let mut components = 0usize;

        for start in 0..n {
            if !mask[start] || visited[start] {
                continue;
            }
            components += 1;
            let mut stack = vec![start];
            visited[start] = true;
            while let Some(idx) = stack.pop() {
                let iz = idx / (ny * nx);
                let rem = idx % (ny * nx);
                let iy = rem / nx;
                let ix = rem % nx;
                for dz in -1isize..=1 {
                    for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            if dz == 0 && dy == 0 && dx == 0 {
                                continue;
                            }
                            let gz = iz as isize + dz;
                            let gy = iy as isize + dy;
                            let gx = ix as isize + dx;
                            if gz < 0
                                || gz >= nz as isize
                                || gy < 0
                                || gy >= ny as isize
                                || gx < 0
                                || gx >= nx as isize
                            {
                                continue;
                            }
                            let ni = gz as usize * ny * nx + gy as usize * nx + gx as usize;
                            if mask[ni] && !visited[ni] {
                                visited[ni] = true;
                                stack.push(ni);
                            }
                        }
                    }
                }
            }
        }
        components
    }

    /// Check that no 2×2 foreground block exists (thinness in 2-D).
    fn has_2x2_block(flat: &[f32], ny: usize, nx: usize) -> bool {
        for iy in 0..ny.saturating_sub(1) {
            for ix in 0..nx.saturating_sub(1) {
                if flat[iy * nx + ix] > 0.5
                    && flat[iy * nx + ix + 1] > 0.5
                    && flat[(iy + 1) * nx + ix] > 0.5
                    && flat[(iy + 1) * nx + ix + 1] > 0.5
                {
                    return true;
                }
            }
        }
        false
    }

    // ── D = 1 tests ──────────────────────────────────────────────────────

    #[test]
    fn test_1d_empty_stays_empty() {
        let image = make_mask_1d(vec![0.0; 5], 5);
        let result = Skeletonization::new().apply(&image);
        assert_eq!(count_fg_1d(&result), 0);
    }

    #[test]
    fn test_1d_single_voxel_preserved() {
        let mut vals = vec![0.0_f32; 7];
        vals[3] = 1.0;
        let image = make_mask_1d(vals, 7);
        let result = Skeletonization::new().apply(&image);
        let v = values_1d(&result);
        assert_eq!(v[3], 1.0);
        assert_eq!(count_fg_1d(&result), 1);
    }

    #[test]
    fn test_1d_run_produces_midpoint() {
        // Run of 5: indices 1..=5 → midpoint = (1+5)/2 = 3.
        let mut vals = vec![0.0_f32; 8];
        for i in 1..=5 {
            vals[i] = 1.0;
        }
        let image = make_mask_1d(vals, 8);
        let result = Skeletonization::new().apply(&image);
        let v = values_1d(&result);
        assert_eq!(count_fg_1d(&result), 1, "single run → single midpoint");
        assert_eq!(v[3], 1.0, "midpoint at index 3");
    }

    #[test]
    fn test_1d_two_runs_two_midpoints() {
        // Two runs: [0,1,1,0,0,1,1,1,0]
        let vals = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let image = make_mask_1d(vals, 9);
        let result = Skeletonization::new().apply(&image);
        let v = values_1d(&result);
        assert_eq!(count_fg_1d(&result), 2, "two runs → two midpoints");
        // Run 1: [1,2] → midpoint 1. Run 2: [5,6,7] → midpoint 6.
        assert_eq!(v[1], 1.0);
        assert_eq!(v[6], 1.0);
    }

    #[test]
    fn test_1d_all_foreground() {
        // Entire image foreground: run [0, nx-1] → midpoint at nx/2.
        let nx = 9;
        let image = make_mask_1d(vec![1.0; nx], nx);
        let result = Skeletonization::new().apply(&image);
        assert_eq!(count_fg_1d(&result), 1);
        let v = values_1d(&result);
        assert_eq!(v[4], 1.0, "midpoint of [0,8] is 4");
    }

    // ── D = 2 tests ──────────────────────────────────────────────────────

    #[test]
    fn test_2d_empty_stays_empty() {
        let image = make_mask_2d(vec![0.0; 9], [3, 3]);
        let result = Skeletonization::new().apply(&image);
        assert_eq!(count_fg_2d(&result), 0);
    }

    #[test]
    fn test_2d_single_pixel_preserved() {
        let mut vals = vec![0.0_f32; 9];
        vals[4] = 1.0; // center of 3×3
        let image = make_mask_2d(vals, [3, 3]);
        let result = Skeletonization::new().apply(&image);
        let v = values_2d(&result);
        assert_eq!(v[4], 1.0);
        assert_eq!(count_fg_2d(&result), 1);
    }

    #[test]
    fn test_2d_3x3_square_thins_to_center() {
        // Filled 3×3 square → Zhang-Suen thins to center pixel (1,1).
        let image = make_mask_2d(vec![1.0_f32; 9], [3, 3]);
        let result = Skeletonization::new().apply(&image);
        let v = values_2d(&result);
        assert_eq!(count_fg_2d(&result), 1, "3×3 square thins to 1 pixel");
        assert_eq!(v[4], 1.0, "remaining pixel is at center (1,1)");
    }

    #[test]
    fn test_2d_horizontal_line_preserved() {
        // A 1-pixel-wide horizontal line is already a skeleton.
        // 3 rows × 7 cols, middle row is foreground.
        let (ny, nx) = (3, 7);
        let mut vals = vec![0.0_f32; ny * nx];
        for ix in 0..nx {
            vals[1 * nx + ix] = 1.0;
        }
        let image = make_mask_2d(vals.clone(), [ny, nx]);
        let result = Skeletonization::new().apply(&image);
        let v = values_2d(&result);
        // Endpoints (row 1, col 0) and (row 1, col 6) should remain.
        // Interior pixels: each has exactly 2 neighbors (left and right).
        // Zhang-Suen A = 1 check: for a horizontal line interior pixel:
        // P2=0, P3=0, P4=1, P5=0, P6=0, P7=0, P8=1, P9=0
        // B=2, A=2 (0→1 at P3→P4 and P7→P8). A≠1 → not deleted. Line preserved.
        for ix in 0..nx {
            assert_eq!(
                v[1 * nx + ix],
                1.0,
                "horizontal line pixel at x={ix} must be preserved"
            );
        }
    }

    #[test]
    fn test_2d_skeleton_is_subset() {
        // 5×5 filled square.
        let image = make_mask_2d(vec![1.0_f32; 25], [5, 5]);
        let result = Skeletonization::new().apply(&image);
        let orig = vec![1.0_f32; 25];
        let skel = values_2d(&result);
        for (i, (&s, &o)) in skel.iter().zip(orig.iter()).enumerate() {
            if s > 0.5 {
                assert!(o > 0.5, "skeleton voxel {i} must be within original mask");
            }
        }
        assert!(
            count_fg_2d(&result) < 25,
            "skeleton must be strictly smaller than filled square"
        );
    }

    #[test]
    fn test_2d_binary_output() {
        let image = make_mask_2d(vec![1.0_f32; 25], [5, 5]);
        let result = Skeletonization::new().apply(&image);
        for &v in values_2d(&result).iter() {
            assert!(v == 0.0 || v == 1.0, "output must be binary, got {v}");
        }
    }

    #[test]
    fn test_2d_topology_preserved() {
        // Two separate 3×3 squares in a 3×9 image (with gap between).
        // Each should thin independently; component count preserved.
        let (ny, nx) = (3, 9);
        let mut vals = vec![0.0_f32; ny * nx];
        // Square 1: rows 0-2, cols 0-2
        for iy in 0..3 {
            for ix in 0..3 {
                vals[iy * nx + ix] = 1.0;
            }
        }
        // Square 2: rows 0-2, cols 6-8
        for iy in 0..3 {
            for ix in 6..9 {
                vals[iy * nx + ix] = 1.0;
            }
        }
        let cc_before = count_components_2d(&vals, ny, nx);
        assert_eq!(cc_before, 2, "two components before thinning");

        let image = make_mask_2d(vals, [ny, nx]);
        let result = Skeletonization::new().apply(&image);
        let skel = values_2d(&result);
        let cc_after = count_components_2d(&skel, ny, nx);
        assert_eq!(
            cc_after, cc_before,
            "connected component count must be preserved"
        );
    }

    #[test]
    fn test_2d_no_2x2_block() {
        // The skeleton of a filled rectangle must be thin (no 2×2 blocks).
        let (ny, nx) = (7, 11);
        let image = make_mask_2d(vec![1.0_f32; ny * nx], [ny, nx]);
        let result = Skeletonization::new().apply(&image);
        let skel = values_2d(&result);
        assert!(
            !has_2x2_block(&skel, ny, nx),
            "skeleton must contain no 2×2 foreground block (thinness)"
        );
    }

    #[test]
    fn test_2d_spatial_metadata_preserved() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let td = TensorData::new(vec![1.0_f32; 9], Shape::new([3, 3]));
        let tensor = Tensor::<TestBackend, 2>::from_data(td, &device);
        let origin = Point::new([1.0, 2.0]);
        let spacing = Spacing::new([0.5, 1.5]);
        let direction = Direction::identity();
        let image = Image::new(tensor, origin, spacing, direction);

        let result = Skeletonization::new().apply(&image);
        assert_eq!(result.origin(), &origin);
        assert_eq!(result.spacing(), &spacing);
        assert_eq!(result.direction(), &direction);
    }

    // ── D = 3 tests ──────────────────────────────────────────────────────

    #[test]
    fn test_3d_empty_stays_empty() {
        let image = make_mask_3d(vec![0.0; 27], [3, 3, 3]);
        let result = Skeletonization::new().apply(&image);
        assert_eq!(count_fg_3d(&result), 0);
    }

    #[test]
    fn test_3d_single_voxel_preserved() {
        let mut vals = vec![0.0_f32; 27];
        vals[13] = 1.0; // center of 3×3×3
        let image = make_mask_3d(vals, [3, 3, 3]);
        let result = Skeletonization::new().apply(&image);
        let v = values_3d(&result);
        assert_eq!(v[13], 1.0);
        assert_eq!(count_fg_3d(&result), 1);
    }

    #[test]
    fn test_3d_straight_line_preserved() {
        // 1×1×7 line (all foreground): already 1-voxel wide.
        // Endpoints have 1 neighbor each; interior voxels have 2 neighbors
        // and are NOT simple (T₂₆ = 2 when neighbors are disconnected).
        // The line is preserved.
        let image = make_mask_3d(vec![1.0_f32; 7], [1, 1, 7]);
        let result = Skeletonization::new().apply(&image);
        assert_eq!(
            count_fg_3d(&result),
            7,
            "1-voxel-wide line must be fully preserved"
        );
    }

    #[test]
    fn test_3d_cube_thins_to_smaller() {
        // 5×5×5 filled cube → skeleton is strictly smaller.
        let n = 5 * 5 * 5;
        let image = make_mask_3d(vec![1.0_f32; n], [5, 5, 5]);
        let result = Skeletonization::new().apply(&image);
        let skel_count = count_fg_3d(&result);
        assert!(skel_count > 0, "skeleton must be non-empty");
        assert!(
            skel_count < n,
            "skeleton ({skel_count}) must be strictly smaller than cube ({n})"
        );
    }

    #[test]
    fn test_3d_skeleton_is_subset() {
        let orig = vec![1.0_f32; 125]; // 5×5×5
        let image = make_mask_3d(orig.clone(), [5, 5, 5]);
        let result = Skeletonization::new().apply(&image);
        let skel = values_3d(&result);
        for (i, (&s, &o)) in skel.iter().zip(orig.iter()).enumerate() {
            if s > 0.5 {
                assert!(o > 0.5, "skeleton voxel {i} must be within original mask");
            }
        }
    }

    #[test]
    fn test_3d_binary_output() {
        let image = make_mask_3d(vec![1.0_f32; 27], [3, 3, 3]);
        let result = Skeletonization::new().apply(&image);
        for &v in values_3d(&result).iter() {
            assert!(v == 0.0 || v == 1.0, "output must be binary, got {v}");
        }
    }

    #[test]
    fn test_3d_topology_preserved() {
        // Two separate 3×3×3 cubes in a 3×3×9 image.
        let (nz, ny, nx) = (3, 3, 9);
        let mut vals = vec![0.0_f32; nz * ny * nx];
        // Cube 1: z=0..3, y=0..3, x=0..3
        for iz in 0..3 {
            for iy in 0..3 {
                for ix in 0..3 {
                    vals[iz * ny * nx + iy * nx + ix] = 1.0;
                }
            }
        }
        // Cube 2: z=0..3, y=0..3, x=6..9
        for iz in 0..3 {
            for iy in 0..3 {
                for ix in 6..9 {
                    vals[iz * ny * nx + iy * nx + ix] = 1.0;
                }
            }
        }
        let cc_before = count_components_3d(&vals, nz, ny, nx);
        assert_eq!(cc_before, 2);

        let image = make_mask_3d(vals, [nz, ny, nx]);
        let result = Skeletonization::new().apply(&image);
        let skel = values_3d(&result);
        let cc_after = count_components_3d(&skel, nz, ny, nx);
        assert_eq!(
            cc_after, cc_before,
            "3-D connected component count must be preserved"
        );
    }

    #[test]
    fn test_3d_spatial_metadata_preserved() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let td = TensorData::new(vec![1.0_f32; 27], Shape::new([3, 3, 3]));
        let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 1.0, 2.0]);
        let direction = Direction::identity();
        let image = Image::new(tensor, origin, spacing, direction);

        let result = Skeletonization::new().apply(&image);
        assert_eq!(result.origin(), &origin);
        assert_eq!(result.spacing(), &spacing);
        assert_eq!(result.direction(), &direction);
    }

    // ── Internal predicate tests ─────────────────────────────────────────

    #[test]
    fn test_fg_components_26_single_component() {
        // All 26 neighbors are foreground → 1 component.
        let mut local = [true; 27];
        local[13] = false; // center excluded by convention in the function
        assert_eq!(fg_components_26(&local), 1);
    }

    #[test]
    fn test_fg_components_26_two_components() {
        // Only two opposite corners: (0,0,0)=idx 0 and (2,2,2)=idx 26.
        // Chebyshev distance = 2 → not 26-adjacent → 2 components.
        let mut local = [false; 27];
        local[0] = true; // (0,0,0)
        local[26] = true; // (2,2,2)
        assert_eq!(fg_components_26(&local), 2);
    }

    #[test]
    fn test_fg_components_26_empty() {
        let local = [false; 27];
        assert_eq!(fg_components_26(&local), 0);
    }

    #[test]
    fn test_fg_components_26_adjacent_corners() {
        // (0,0,0) and (1,1,1)=center is excluded, (0,0,1) and (0,1,0).
        // (0,0,0)=0, (0,0,1)=1, (0,1,0)=3.
        // (0,0,0) is 26-adjacent to (0,0,1) (differ by 1 in x) → connected.
        // (0,0,0) is 26-adjacent to (0,1,0) (differ by 1 in y) → connected.
        // All form 1 component.
        let mut local = [false; 27];
        local[0] = true; // (0,0,0)
        local[1] = true; // (0,0,1)
        local[3] = true; // (0,1,0)
        assert_eq!(fg_components_26(&local), 1);
    }

    // ── Trait / API parity test ──────────────────────────────────────────

    #[test]
    fn test_morphological_operation_trait() {
        use super::super::MorphologicalOperation;
        let image = make_mask_3d(vec![1.0_f32; 27], [3, 3, 3]);
        let via_method = Skeletonization::new().apply(&image);
        let via_trait: Image<TestBackend, 3> = <Skeletonization as MorphologicalOperation<
            TestBackend,
            3,
        >>::apply(&Skeletonization::new(), &image);
        assert_eq!(values_3d(&via_method), values_3d(&via_trait));
    }

    #[test]
    fn test_default_impl() {
        let s = Skeletonization::default();
        let image = make_mask_2d(vec![1.0_f32; 9], [3, 3]);
        let result = s.apply(&image);
        assert_eq!(count_fg_2d(&result), 1);
    }
}
