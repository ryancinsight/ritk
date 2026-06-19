//! Scalar connected-component labeling (`itk::ScalarConnectedComponentImageFilter`).
//!
//! Unlike binary connected components (which split a foreground mask), this
//! labels **every** voxel: two adjacent voxels belong to the same component when
//! their intensities differ by at most `distance_threshold`. Labels are assigned
//! consecutively (`1, 2, …`) in raster scan order of first encounter, matching
//! `itk::ConnectedComponentImageFilter`'s relabelling (the same Hoshen–Kopelman
//! two-pass union-find used by [`super::connected_components`]).

use super::union_find::UnionFind;
use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

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
    connected_components_by(dims, connectivity, |a, b| {
        (vals[a] - vals[b]).abs() <= distance_threshold
    })
}

/// Label a multi-channel (vector) image into connected components by direction
/// similarity, matching `itk::VectorConnectedComponentImageFilter`.
///
/// Two raster-adjacent voxels join when `1 − |a · b| ≤ distance_threshold`, where
/// `a · b` is the dot product of their channel vectors (ITK assumes the vectors
/// are normalized; 180°-opposite vectors are treated as similar).  The dot
/// product is accumulated in `f64` and `1 − |dot|` is narrowed to `f32` before
/// the comparison, reproducing ITK's `static_cast<ValueType>` functor.
///
/// `channels` holds one flat `Z×Y×X` buffer per vector component.  Returns
/// consecutive labels `1..=K`; the **partition** matches SimpleITK (whose
/// non-compacted label integers differ, as for every connected-component filter
/// — the established ritk CC-parity convention compares the partition).
pub fn vector_connected_components(
    channels: &[Vec<f32>],
    dims: [usize; 3],
    distance_threshold: f32,
    connectivity: u32,
) -> Vec<f32> {
    connected_components_by(dims, connectivity, |a, b| {
        let mut dot = 0.0_f64;
        for ch in channels {
            dot += ch[a] as f64 * ch[b] as f64;
        }
        (1.0 - dot.abs()) as f32 <= distance_threshold
    })
}

/// Image-level [`vector_connected_components`]: extract each `images` entry as a
/// channel buffer (all must share dimensions) and return the rebuilt label image
/// carrying `images[0]`'s spatial metadata.
///
/// # Panics
/// If `images` is empty or the channel images differ in dimensions.
pub fn vector_connected_components_image<B: Backend>(
    images: &[&Image<B, 3>],
    distance_threshold: f32,
    connectivity: u32,
) -> Image<B, 3> {
    let (first, dims) = extract_vec_infallible(images[0]);
    let mut bufs: Vec<Vec<f32>> = Vec::with_capacity(images.len());
    bufs.push(first);
    for img in &images[1..] {
        let (vals, d) = extract_vec_infallible(*img);
        assert_eq!(
            d, dims,
            "vector_connected_components: channels differ in dimensions"
        );
        bufs.push(vals);
    }
    let labels = vector_connected_components(&bufs, dims, distance_threshold, connectivity);
    rebuild(labels, dims, images[0])
}

/// Label voxels into connected components by an arbitrary symmetric predicate.
///
/// Two raster-adjacent voxels (6- or 26-connectivity) join the same component
/// when `same(flat_a, flat_b)` is `true`.  Every voxel is labelled; returns a
/// flat `Z×Y×X` buffer with consecutive labels `1..=K` in raster scan order of
/// first appearance — the shared Hoshen–Kopelman two-pass core behind both
/// [`scalar_connected_components`] and [`vector_connected_components`].
pub fn connected_components_by<F>(dims: [usize; 3], connectivity: u32, mut same: F) -> Vec<f32>
where
    F: FnMut(usize, usize) -> bool,
{
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

    // ── Pass 1: provisional labels via backward neighbours satisfying `same` ──
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = idx(iz, iy, ix);
                let mut nbr_labels: Vec<usize> = Vec::with_capacity(offsets.len());
                for &(dz, dy, dx) in offsets {
                    let (nz_i, ny_i, nx_i) = (iz as isize + dz, iy as isize + dy, ix as isize + dx);
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
                    if same(flat, nflat) {
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
