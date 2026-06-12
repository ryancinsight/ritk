//! D = 3: 6-directional sequential thinning.
//!
//! Each iteration comprises 6 sub-iterations (one per face direction:
//! ±z, ±y, ±x). Sequential deletion of simple points preserves topology.

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
pub fn fg_components_26(local: &[bool; 27]) -> usize {
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
                        if !(0..3).contains(&nz_l)
                            || !(0..3).contains(&ny_l)
                            || !(0..3).contains(&nx_l)
                        {
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
pub(super) fn sequential_thin(flat: &[f32], nz: usize, ny: usize, nx: usize) -> Vec<f32> {
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
            // Heuristic: each direction targets a fraction of the volume surface.
            let mut candidates: Vec<usize> = Vec::with_capacity(n / 16);
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
