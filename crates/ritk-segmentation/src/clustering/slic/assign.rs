//! Grid-to-center search index and voxel assignment for SLIC.
//!
//! The search-window optimization maps each voxel to its grid cell and
//! checks only the cluster centers whose ±2S search region overlaps
//! that cell. This yields O(N · 2^D) amortized cost per iteration
//! instead of O(N · K).

use super::coords::{decode_coords, encode_coords, encode_coords_dyn};
use super::grid::Center;

const ASSIGN_CHUNK_LEN: usize = 1024;

/// Grid-cell to cluster-center search index in compressed sparse row form.
///
/// The centers whose search region (±2S per axis) overlaps cell `c` are
/// `members[offsets[c]..offsets[c + 1]]`, in ascending center index. That
/// order is the tie-break of the assignment scan, which keeps the first
/// center reaching the minimum distance. One contiguous member buffer replaces
/// a heap list per cell, so the per-voxel scan reads one slice of a single
/// allocation and a rebuild touches no per-cell headers.
#[derive(Debug, Clone, Default)]
pub struct CenterGrid {
    offsets: Vec<usize>,
    members: Vec<usize>,
}

impl CenterGrid {
    /// Rebuild the index for `centers`, reusing the buffers' capacity.
    ///
    /// Two passes over the centers: the first counts each cell's members, a
    /// prefix sum turns the counts into row offsets, and the second writes the
    /// members through a per-cell cursor held in `offsets` itself, which is
    /// then shifted back by one row to restore the row starts.
    pub fn rebuild(
        &mut self,
        centers: &[Center],
        grid_sizes: &[usize],
        shape: &[usize],
        ndim: usize,
    ) {
        let n_cells_per_axis: Vec<usize> = shape
            .iter()
            .zip(grid_sizes.iter())
            .map(|(&s, &gs)| if s == 0 { 1 } else { (s - 1) / gs.max(1) + 1 })
            .collect();
        let total_cells: usize = n_cells_per_axis.iter().copied().product::<usize>().max(1);

        let offsets = &mut self.offsets;
        offsets.clear();
        offsets.resize(total_cells + 1, 0);
        visit_center_cells(
            centers,
            grid_sizes,
            shape,
            &n_cells_per_axis,
            ndim,
            |_, cell| {
                offsets[cell + 1] += 1;
            },
        );
        for row in 1..=total_cells {
            offsets[row] += offsets[row - 1];
        }

        let members = &mut self.members;
        members.clear();
        members.resize(offsets[total_cells], 0);
        visit_center_cells(
            centers,
            grid_sizes,
            shape,
            &n_cells_per_axis,
            ndim,
            |center, cell| {
                members[offsets[cell]] = center;
                offsets[cell] += 1;
            },
        );
        offsets.copy_within(0..total_cells, 1);
        offsets[0] = 0;
    }

    /// Candidate centers for `cell`, in ascending center index; empty for a
    /// cell outside the grid.
    #[must_use]
    pub fn cell(&self, cell: usize) -> &[usize] {
        match self.offsets.get(cell..cell + 2) {
            Some(&[start, end]) => &self.members[start..end],
            _ => &[],
        }
    }
}

/// Call `visit(center, cell)` for every grid cell each center's search region
/// overlaps, centers in ascending index.
fn visit_center_cells(
    centers: &[Center],
    grid_sizes: &[usize],
    shape: &[usize],
    n_cells_per_axis: &[usize],
    ndim: usize,
    mut visit: impl FnMut(usize, usize),
) {
    // Pre-allocate coordinate buffers once; all positions are fully overwritten
    // per center, so no reset is needed.
    let mut lo_cell = vec![0usize; ndim];
    let mut hi_cell = vec![0usize; ndim];
    let mut cell_coords = vec![0usize; ndim];
    let total_cells: usize = n_cells_per_axis.iter().product();

    for (ci, center) in centers.iter().enumerate() {
        for d in 0..ndim {
            let step = grid_sizes[d] as f32;
            let gs = grid_sizes[d].max(1);
            let nc = n_cells_per_axis[d];

            let search_lo = (center.pos[d] - 2.0 * step).max(0.0) as usize;
            let search_hi = ((center.pos[d] + 2.0 * step) as usize).min(shape[d].saturating_sub(1));

            lo_cell[d] = (search_lo / gs).min(nc - 1);
            hi_cell[d] = (search_hi / gs).min(nc - 1);
        }

        // Enumerate all cells in [lo_cell, hi_cell]^D and register this center.
        enumerate_cells_range(
            &mut cell_coords,
            0,
            &lo_cell,
            &hi_cell,
            n_cells_per_axis,
            &mut |cell_flat| {
                if cell_flat < total_cells {
                    visit(ci, cell_flat);
                }
            },
        );
    }
}

/// Recursively enumerate grid cells in a hyper-rectangular range.
fn enumerate_cells_range(
    cell_coords: &mut [usize],
    depth: usize,
    lo: &[usize],
    hi: &[usize],
    n_cells: &[usize],
    visit: &mut impl FnMut(usize),
) {
    if depth == cell_coords.len() {
        visit(encode_coords_dyn(cell_coords, n_cells));
        return;
    }

    for c in lo[depth]..=hi[depth] {
        cell_coords[depth] = c;
        enumerate_cells_range(cell_coords, depth + 1, lo, hi, n_cells, visit);
    }
}

/// Assign each voxel to the nearest cluster center within a 2S search window.
///
/// Uses the grid-based index for O(2^D) amortized cost per voxel.
/// Parallelized via Moirai over disjoint `distances`/`labels` chunks.
///
/// Dispatches to a const-generic implementation for D ∈ {2, 3} to
/// eliminate per-voxel heap allocations (SLIC is only meaningful in 2-D/3-D).
///
/// # Panics
/// Panics if `ndim` is not in {2, 3}.
#[expect(clippy::too_many_arguments, reason = "ratchet RITK-LINT-1")]
pub fn assign_voxels(
    intensities: &[f32],
    shape: &[usize],
    ndim: usize,
    centers: &[Center],
    center_grid: &CenterGrid,
    grid_sizes: &[usize],
    m_c: f32,
    m_s: f32,
    compactness: f32,
    distances: &mut [f32],
    labels: &mut [u32],
) {
    match ndim {
        2 => assign_voxels_impl::<2>(
            intensities,
            shape,
            centers,
            center_grid,
            grid_sizes,
            m_c,
            m_s,
            compactness,
            distances,
            labels,
        ),
        3 => assign_voxels_impl::<3>(
            intensities,
            shape,
            centers,
            center_grid,
            grid_sizes,
            m_c,
            m_s,
            compactness,
            distances,
            labels,
        ),
        _ => panic!("assign_voxels: unsupported dimensionality {}", ndim),
    }
}

/// Const-generic implementation of [`assign_voxels`].
///
/// Uses stack-allocated `[usize; D]` arrays for coordinates and cell indices,
/// eliminating per-voxel `Vec` allocations in the hot parallel loop.
#[expect(clippy::too_many_arguments, reason = "ratchet RITK-LINT-1")]
fn assign_voxels_impl<const D: usize>(
    intensities: &[f32],
    shape: &[usize],
    centers: &[Center],
    center_grid: &CenterGrid,
    grid_sizes: &[usize],
    m_c: f32,
    m_s: f32,
    compactness: f32,
    distances: &mut [f32],
    labels: &mut [u32],
) {
    let shape_arr: [usize; D] = {
        let mut arr = [0usize; D];
        arr.copy_from_slice(shape);
        arr
    };

    let grid_sizes_arr: [usize; D] = {
        let mut arr = [0usize; D];
        arr.copy_from_slice(grid_sizes);
        arr
    };

    let n_cells_per_axis: [usize; D] = {
        let mut arr = [0usize; D];
        for d in 0..D {
            arr[d] = if shape_arr[d] == 0 {
                1
            } else {
                (shape_arr[d] - 1) / grid_sizes_arr[d].max(1) + 1
            };
        }
        arr
    };

    let k = centers.len();

    moirai::for_each_chunk_pair_mut_enumerated_with::<moirai::Adaptive, _, _, _>(
        distances,
        labels,
        ASSIGN_CHUNK_LEN,
        |chunk_idx, dist_chunk, label_chunk| {
            let base = chunk_idx * ASSIGN_CHUNK_LEN;
            for (local, (dist_mut, label_mut)) in dist_chunk
                .iter_mut()
                .zip(label_chunk.iter_mut())
                .enumerate()
            {
                let i = base + local;
                let coords = decode_coords(i, shape_arr);
                let intensity = intensities[i];

                let mut cell = [0usize; D];
                for d in 0..D {
                    cell[d] = (coords[d] / grid_sizes_arr[d].max(1)).min(n_cells_per_axis[d] - 1);
                }

                let cell_flat = encode_coords(&cell, n_cells_per_axis);

                let mut best_dist = *dist_mut;
                let mut best_label = *label_mut;

                for &ci in center_grid.cell(cell_flat) {
                    if ci >= k {
                        continue;
                    }
                    let center = &centers[ci];

                    let mut in_range = true;
                    for d in 0..D {
                        let diff = coords[d] as f32 - center.pos[d];
                        let step = grid_sizes_arr[d] as f32;
                        if diff.abs() > 2.0 * step + 0.5 {
                            in_range = false;
                            break;
                        }
                    }
                    if !in_range {
                        continue;
                    }

                    let normalized_intensity = (intensity - center.intensity) / m_c;
                    let mut distance = normalized_intensity.abs();
                    for (&coord_d, &pos_d) in coords.iter().zip(center.pos.iter()) {
                        let normalized_position = (((coord_d as f32 - pos_d) / m_s) * compactness)
                            .abs()
                            .min(f32::MAX);
                        distance = distance.hypot(normalized_position);
                    }

                    if distance < best_dist {
                        best_dist = distance;
                        best_label = ci as u32;
                    }
                }

                *dist_mut = best_dist;
                *label_mut = best_label;
            }
        },
    );
}

/// Recompute each cluster center as the mean of all assigned voxels.
///
/// Returns the maximum center shift (combined intensity + spatial L2 distance).
///
/// # Panics
/// Panics if `ndim` is not in {2, 3}.
pub fn update_centers(
    centers: &mut [Center],
    intensities: &[f32],
    labels: &[u32],
    shape: &[usize],
    ndim: usize,
    k: usize,
) -> f32 {
    match ndim {
        2 => update_centers_impl::<2>(centers, intensities, labels, shape, k),
        3 => update_centers_impl::<3>(centers, intensities, labels, shape, k),
        _ => panic!("update_centers: unsupported dimensionality {}", ndim),
    }
}

/// Const-generic implementation of [`update_centers`].
///
/// Uses stack-allocated `[usize; D]` for coordinate decoding, eliminating
/// per-voxel `Vec` allocations.
fn update_centers_impl<const D: usize>(
    centers: &mut [Center],
    intensities: &[f32],
    labels: &[u32],
    shape: &[usize],
    k: usize,
) -> f32 {
    let shape_arr: [usize; D] = {
        let mut arr = [0usize; D];
        arr.copy_from_slice(shape);
        arr
    };
    let actual_k = centers.len().min(k);
    let n: usize = shape_arr.iter().product();

    // Accumulate sums and counts for each center.
    let mut mean_intensity_offset = vec![0.0_f32; actual_k];
    let mut mean_pos_offset = vec![[0.0_f32; D]; actual_k];
    let mut counts = vec![0usize; actual_k];
    let mut intensity_anchors = vec![None; actual_k];
    let mut position_anchors = vec![None; actual_k];

    for i in 0..n {
        let ci = labels[i] as usize;
        if ci < actual_k {
            counts[ci] += 1;
            let coords = decode_coords(i, shape_arr);
            intensity_anchors[ci].get_or_insert(intensities[i]);
            position_anchors[ci].get_or_insert(coords);
        }
    }
    for i in 0..n {
        let ci = labels[i] as usize;
        if ci < actual_k {
            let intensity_anchor = intensity_anchors[ci].expect("nonempty cluster has an anchor");
            // The count remains exact; conversion supplies the nearest
            // divisor representable by this concrete-f32 operation.
            let count = counts[ci] as f32;
            mean_intensity_offset[ci] += (intensities[i] - intensity_anchor) / count;
            let coords = decode_coords(i, shape_arr);
            let position_anchor = position_anchors[ci].expect("nonempty cluster has an anchor");
            for ((offset, &coord), &anchor) in mean_pos_offset[ci]
                .iter_mut()
                .zip(coords.iter())
                .zip(position_anchor.iter())
            {
                *offset += (coord as f32 - anchor as f32) / count;
            }
        }
    }

    let mut max_shift = 0.0_f32;

    for ci in 0..actual_k {
        if counts[ci] > 0 {
            let new_intensity = intensity_anchors[ci].expect("nonempty cluster has an anchor")
                + mean_intensity_offset[ci];

            let mut shift = 0.0_f32;
            let position_anchor = position_anchors[ci].expect("nonempty cluster has an anchor");
            for ((&offset, &anchor), pos_d) in mean_pos_offset[ci]
                .iter()
                .zip(position_anchor.iter())
                .zip(centers[ci].pos.iter_mut())
            {
                let new_pos_d = anchor as f32 + offset;
                let dp = new_pos_d - *pos_d;
                shift = shift.hypot(dp);
                *pos_d = new_pos_d;
            }

            let di = new_intensity - centers[ci].intensity;
            shift = shift.hypot(di);

            centers[ci].intensity = new_intensity;

            if shift > max_shift {
                max_shift = shift;
            }
        }
        // Empty clusters retain their previous position.
    }

    max_shift
}
