//! Grid-to-center search index and voxel assignment for SLIC.
//!
//! The search-window optimization maps each voxel to its grid cell and
//! checks only the cluster centers whose ±2S search region overlaps
//! that cell. This yields O(N · 2^D) amortized cost per iteration
//! instead of O(N · K).

use super::coords::{decode_coords, encode_coords};
use super::grid::Center;

/// Build a mapping from grid cell index to the list of center indices whose
/// search region (±2S per axis) overlaps that cell.
pub fn build_grid_map(
    centers: &[Center],
    grid_sizes: &[usize],
    shape: &[usize],
    ndim: usize,
) -> Vec<Vec<usize>> {
    let n_cells_per_axis: Vec<usize> = shape
        .iter()
        .zip(grid_sizes.iter())
        .map(|(&s, &gs)| if s == 0 { 1 } else { (s - 1) / gs.max(1) + 1 })
        .collect();

    let total_cells: usize = n_cells_per_axis.iter().copied().product::<usize>().max(1);
    // Each cell overlaps ~5^D / total_cells centers on average; add 1 to avoid zero.
    let avg_per_cell = centers.len() / total_cells + 1;
    let mut grid_map: Vec<Vec<usize>> = vec![Vec::with_capacity(avg_per_cell); total_cells];

    for (ci, center) in centers.iter().enumerate() {
        let mut lo_cell = vec![0usize; ndim];
        let mut hi_cell = vec![0usize; ndim];

        for d in 0..ndim {
            let step = grid_sizes[d] as f64;
            let gs = grid_sizes[d].max(1);
            let nc = n_cells_per_axis[d];

            let search_lo = (center.pos[d] - 2.0 * step).max(0.0) as usize;
            let search_hi = ((center.pos[d] + 2.0 * step) as usize).min(shape[d].saturating_sub(1));

            lo_cell[d] = (search_lo / gs).min(nc - 1);
            hi_cell[d] = (search_hi / gs).min(nc - 1);
        }

        // Enumerate all cells in [lo_cell, hi_cell]^D and register this center.
        let mut cell_coords = vec![0usize; ndim];
        enumerate_cells_range(
            &mut cell_coords,
            0,
            &lo_cell,
            &hi_cell,
            &n_cells_per_axis,
            &mut grid_map,
            ci,
            ndim,
        );
    }

    grid_map
}

/// Recursively enumerate grid cells in a hyper-rectangular range.
#[allow(clippy::too_many_arguments)]
fn enumerate_cells_range(
    cell_coords: &mut [usize],
    depth: usize,
    lo: &[usize],
    hi: &[usize],
    n_cells: &[usize],
    grid_map: &mut Vec<Vec<usize>>,
    center_idx: usize,
    ndim: usize,
) {
    if depth == ndim {
        let cell_flat = encode_coords(cell_coords, n_cells);
        if cell_flat < grid_map.len() {
            grid_map[cell_flat].push(center_idx);
        }
        return;
    }

    for c in lo[depth]..=hi[depth] {
        cell_coords[depth] = c;
        enumerate_cells_range(
            cell_coords,
            depth + 1,
            lo,
            hi,
            n_cells,
            grid_map,
            center_idx,
            ndim,
        );
    }
}

/// Assign each voxel to the nearest cluster center within a 2S search window.
///
/// Uses the grid-based index for O(2^D) amortized cost per voxel.
/// Parallelized via rayon with safe per-voxel result collection.
#[allow(clippy::too_many_arguments)]
pub fn assign_voxels(
    intensities: &[f64],
    shape: &[usize],
    ndim: usize,
    centers: &[Center],
    grid_map: &[Vec<usize>],
    grid_sizes: &[usize],
    m_c: f64,
    m_s: f64,
    compactness: f64,
    distances: &mut [f64],
    labels: &mut [u32],
) {
    let n: usize = shape.iter().product();

    let n_cells_per_axis: Vec<usize> = shape
        .iter()
        .zip(grid_sizes.iter())
        .map(|(&s, &gs)| if s == 0 { 1 } else { (s - 1) / gs.max(1) + 1 })
        .collect();

    let inv_m_c_sq = 1.0 / (m_c * m_c);
    let compactness_sq = compactness * compactness;
    let inv_m_s_sq = 1.0 / (m_s * m_s);
    let k = centers.len();

    // Parallel assignment over flat voxel indices; each index is independent.
    let results: Vec<(f64, u32)> =
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |i| {
            let coords = decode_coords(i, shape);
            let intensity = intensities[i];

            // Determine grid cell for this voxel.
            let mut cell = vec![0usize; ndim];
            for d in 0..ndim {
                cell[d] = (coords[d] / grid_sizes[d].max(1)).min(n_cells_per_axis[d] - 1);
            }

            let cell_flat = encode_coords(&cell, &n_cells_per_axis);

            let mut best_dist = distances[i];
            let mut best_label = labels[i];

            if cell_flat < grid_map.len() {
                for &ci in &grid_map[cell_flat] {
                    if ci >= k {
                        continue;
                    }
                    let center = &centers[ci];

                    // Check spatial proximity: each coordinate must be within 2*step.
                    let mut in_range = true;
                    for d in 0..ndim {
                        let diff = coords[d] as f64 - center.pos[d];
                        let step = grid_sizes[d] as f64;
                        if diff.abs() > 2.0 * step + 0.5 {
                            in_range = false;
                            break;
                        }
                    }
                    if !in_range {
                        continue;
                    }

                    // Compute SLIC distance squared.
                    let di = intensity - center.intensity;
                    let mut d_sq = di * di * inv_m_c_sq;
                    for (coord, pos) in coords.iter().zip(center.pos.iter()).take(ndim) {
                        let dp = *coord as f64 - pos;
                        d_sq += compactness_sq * dp * dp * inv_m_s_sq;
                    }

                    if d_sq < best_dist {
                        best_dist = d_sq;
                        best_label = ci as u32;
                    }
                }
            }

            (best_dist, best_label)
        });

    // Merge results back.
    for (i, (dist, label)) in results.into_iter().enumerate() {
        distances[i] = dist;
        labels[i] = label;
    }
}

/// Recompute each cluster center as the mean of all assigned voxels.
///
/// Returns the maximum center shift (combined intensity + spatial L2 distance).
pub fn update_centers(
    centers: &mut [Center],
    intensities: &[f64],
    labels: &[u32],
    shape: &[usize],
    ndim: usize,
    k: usize,
) -> f64 {
    let actual_k = centers.len().min(k);
    let n: usize = shape.iter().product();

    // Accumulate sums and counts for each center.
    let mut sum_intensity = vec![0.0_f64; actual_k];
    let mut sum_pos = vec![vec![0.0_f64; ndim]; actual_k];
    let mut counts = vec![0u64; actual_k];

    for i in 0..n {
        let ci = labels[i] as usize;
        if ci < actual_k {
            sum_intensity[ci] += intensities[i];
            counts[ci] += 1;

            let coords = decode_coords(i, shape);
            for d in 0..ndim {
                sum_pos[ci][d] += coords[d] as f64;
            }
        }
    }

    let mut max_shift = 0.0_f64;

    for ci in 0..actual_k {
        if counts[ci] > 0 {
            let c = counts[ci] as f64;
            let new_intensity = sum_intensity[ci] / c;

            let mut shift_sq = 0.0_f64;
            for (sum_p, pos) in sum_pos[ci]
                .iter()
                .zip(centers[ci].pos.iter_mut())
                .take(ndim)
            {
                let new_pos_d = *sum_p / c;
                let dp = new_pos_d - *pos;
                shift_sq += dp * dp;
                *pos = new_pos_d;
            }

            let di = new_intensity - centers[ci].intensity;
            shift_sq += di * di;

            centers[ci].intensity = new_intensity;

            let shift = shift_sq.sqrt();
            if shift > max_shift {
                max_shift = shift;
            }
        }
        // Empty clusters retain their previous position.
    }

    max_shift
}
