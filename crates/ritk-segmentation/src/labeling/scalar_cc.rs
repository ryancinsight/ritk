//! Scalar connected-component labeling (`itk::ScalarConnectedComponentImageFilter`).
//!
//! Unlike binary connected components (which split a foreground mask), this
//! labels **every** voxel: two adjacent voxels belong to the same component when
//! their intensities differ by at most `distance_threshold`. Labels are assigned
//! consecutively (`1, 2, …`) in raster scan order of first encounter, matching
//! `itk::ConnectedComponentImageFilter`'s relabelling (the same Hoshen–Kopelman
//! two-pass union-find used by [`super::connected_components`]).

use super::union_find::UnionFind;

/// Backward (already-visited) neighbour offsets in raster scan order.
const BACKWARD_6: [(isize, isize, isize); 3] = [(-1, 0, 0), (0, -1, 0), (0, 0, -1)];
const BACKWARD_26: [(isize, isize, isize); 13] = [
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

/// Label voxels into scalar connected components.
///
/// Two raster-adjacent voxels (6- or 26-connectivity) join the same component
/// when `|vals[a] − vals[b]| ≤ distance_threshold`. Returns a flat `Z×Y×X`
/// label buffer with consecutive labels `1..=K`.
pub fn scalar_connected_components(
    vals: &[f32],
    dims: [usize; 3],
    distance_threshold: f32,
    connectivity: u32,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    if n == 0 {
        return Vec::new();
    }
    let offsets: &[(isize, isize, isize)] = if connectivity == 6 {
        &BACKWARD_6
    } else {
        &BACKWARD_26
    };

    let mut uf = UnionFind::new(n + 1);
    let mut provisional = vec![0usize; n];
    let mut next_label = 1usize;
    let idx = |z: usize, y: usize, x: usize| (z * ny + y) * nx + x;

    // ── Pass 1: provisional labels via backward neighbours within threshold ──
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = idx(iz, iy, ix);
                let v = vals[flat];
                let mut nbr_labels: Vec<usize> = Vec::with_capacity(offsets.len());
                for &(dz, dy, dx) in offsets {
                    let (nz_i, ny_i, nx_i) =
                        (iz as isize + dz, iy as isize + dy, ix as isize + dx);
                    if nz_i < 0
                        || nz_i >= nz as isize
                        || ny_i < 0
                        || ny_i >= ny as isize
                        || nx_i < 0
                        || nx_i >= nx as isize
                    {
                        continue;
                    }
                    let nflat = idx(nz_i as usize, ny_i as usize, nx_i as usize);
                    if (v - vals[nflat]).abs() <= distance_threshold {
                        nbr_labels.push(provisional[nflat]);
                    }
                }
                if nbr_labels.is_empty() {
                    provisional[flat] = next_label;
                    next_label += 1;
                } else {
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

    // ── Pass 2: relabel canonical roots to consecutive labels in scan order ──
    let mut root_to_final = vec![0usize; next_label];
    let mut num = 0usize;
    let mut out = vec![0.0f32; n];
    for (flat, slot) in out.iter_mut().enumerate() {
        let root = uf.find(provisional[flat]);
        if root_to_final[root] == 0 {
            num += 1;
            root_to_final[root] = num;
        }
        *slot = root_to_final[root] as f32;
    }
    out
}

#[cfg(test)]
#[path = "tests_scalar_cc.rs"]
mod tests_scalar_cc;
