//! Vector confidence-connected region growing, matching
//! `itk::VectorConfidenceConnectedImageFilter`.
//!
//! # Mathematical Specification
//!
//! From the seed voxels, compute a mean vector `μ` and population covariance `Σ`
//! over each seed's `(2r+1)^D` neighbourhood (averaged across seeds).  A voxel
//! `x` joins the region when its Mahalanobis distance
//! `√((x−μ)ᵀ Σ⁻¹ (x−μ)) ≤ threshold`, where `threshold = max(multiplier, max seed
//! distance)` (ITK bumps the multiplier so every seed is included).  The region
//! is grown by a face-connected flood from the seeds, then the statistics are
//! recomputed over the grown region (population mean/covariance) and the flood
//! re-run, for `iterations` passes (the threshold stays fixed at the bumped
//! value, per ITK).
//!
//! `Σ⁻¹` follows ITK's `MahalanobisDistanceMembershipFunction`: when
//! `|det Σ| > 1e-6` the true inverse is used, otherwise the inverse is taken as
//! `I · (f64::MAX^{1/3} / D)` (a large diagonal that pushes off-mean voxels far
//! out of the region).  All linear algebra is `f64`.
//!
//! # Parity scope
//! Region-exact to SimpleITK for well-conditioned inputs (verified at the default
//! and larger multipliers).  Where the recomputed covariance is near-singular —
//! tiny regions at very tight multipliers — the region can diverge: this is an
//! inherent cross-implementation sensitivity of the covariance inverse near the
//! singular threshold (ITK's own vnl SVD emits non-convergence/NaN there), not a
//! defect in this port.

// Dense small-matrix algebra (covariance, Gauss–Jordan inverse, Mahalanobis):
// nested `m[i][j]` index loops are the clearest form and avoid iterator-of-row
// gymnastics that obscure the linear algebra.
#![allow(clippy::needless_range_loop)]

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Mean vector + population covariance of a set of channel rows.
///
/// Mean and biased covariance (`cov[i][j] = E[x_i x_j] − E[x_i] E[x_j]`, divided
/// by `N`) over `count` pixels.  `fill(k, row)` writes pixel `k`'s `c` channel
/// values into `row`; the single `row` scratch is reused across all pixels, so
/// no per-pixel allocation occurs regardless of source (neighbourhood list or
/// masked region).  Accumulation order matches the pixel index order produced by
/// the caller, preserving f64 rounding.
fn mean_covariance<F: FnMut(usize, &mut [f64])>(
    count: usize,
    c: usize,
    mut fill: F,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = count as f64;
    let mut mean = vec![0.0; c];
    let mut cov = vec![vec![0.0; c]; c];
    let mut row = vec![0.0; c];
    for k in 0..count {
        fill(k, &mut row);
        for i in 0..c {
            mean[i] += row[i];
            for j in 0..c {
                cov[i][j] += row[i] * row[j];
            }
        }
    }
    for i in 0..c {
        mean[i] /= n;
    }
    for i in 0..c {
        for j in 0..c {
            cov[i][j] = cov[i][j] / n - mean[i] * mean[j];
        }
    }
    (mean, cov)
}

/// Inverse of a small symmetric matrix via Gauss–Jordan elimination, plus its
/// determinant magnitude.  Returns `(inverse, |det|)`.
fn invert(mat: &[Vec<f64>], c: usize) -> (Vec<Vec<f64>>, f64) {
    // Augment [mat | I] and reduce.
    let mut a = vec![vec![0.0; 2 * c]; c];
    for i in 0..c {
        for j in 0..c {
            a[i][j] = mat[i][j];
        }
        a[i][c + i] = 1.0;
    }
    let mut det = 1.0;
    for col in 0..c {
        // Partial pivot.
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..c {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                piv = r;
            }
        }
        if piv != col {
            a.swap(piv, col);
            det = -det;
        }
        let p = a[col][col];
        det *= p;
        if p == 0.0 {
            return (vec![vec![0.0; c]; c], 0.0);
        }
        let inv_p = 1.0 / p;
        for j in 0..(2 * c) {
            a[col][j] *= inv_p;
        }
        for r in 0..c {
            if r != col {
                let f = a[r][col];
                if f != 0.0 {
                    for j in 0..(2 * c) {
                        a[r][j] -= f * a[col][j];
                    }
                }
            }
        }
    }
    let mut inv = vec![vec![0.0; c]; c];
    for i in 0..c {
        for j in 0..c {
            inv[i][j] = a[i][c + j];
        }
    }
    (inv, det.abs())
}

/// Build the inverse-covariance used for the Mahalanobis distance, following
/// ITK's singular-threshold rule (`|det| > 1e-6` ⇒ true inverse, else a large
/// diagonal `I · f64::MAX^{1/3} / D`).
fn inverse_covariance(cov: &[Vec<f64>], c: usize) -> Vec<Vec<f64>> {
    let (inv, det) = invert(cov, c);
    if det > 1.0e-6 {
        inv
    } else {
        let large = f64::MAX.powf(1.0 / 3.0) / c as f64;
        let mut d = vec![vec![0.0; c]; c];
        for i in 0..c {
            d[i][i] = large;
        }
        d
    }
}

/// Squared Mahalanobis distance `(x−μ)ᵀ Σ⁻¹ (x−μ)` for the voxel at flat index
/// `i`, reading channel values directly from `channels`.  `d` is reusable
/// scratch (length `c`) so the per-voxel flood membership test allocates
/// nothing.  Arithmetic matches the explicit `(x−μ)` form it replaces.
fn maha_sq_at(
    channels: &[Vec<f64>],
    i: usize,
    mean: &[f64],
    inv: &[Vec<f64>],
    c: usize,
    d: &mut [f64],
) -> f64 {
    for j in 0..c {
        d[j] = channels[j][i] - mean[j];
    }
    let mut s = 0.0;
    for a in 0..c {
        let mut acc = 0.0;
        for b in 0..c {
            acc += inv[a][b] * d[b];
        }
        s += d[a] * acc;
    }
    s
}

/// Apply vector confidence-connected region growing.
///
/// `channels` holds one flat `Z×Y×X` buffer per vector component; `seeds` are
/// `[z, y, x]` indices.  Returns a flat label buffer (`replace_value` inside the
/// region, `0` outside).
#[allow(clippy::too_many_arguments)]
pub fn vector_confidence_connected(
    channels: &[Vec<f64>],
    dims: [usize; 3],
    seeds: &[[usize; 3]],
    multiplier: f64,
    iterations: u32,
    initial_radius: usize,
    replace_value: f32,
) -> Vec<f32> {
    let [zn, yn, xn] = dims;
    let n = zn * yn * xn;
    let c = channels.len();
    let mut out = vec![0.0_f32; n];
    if c == 0 || n == 0 {
        return out;
    }
    let stride = [yn * xn, xn, 1usize];
    let flat = |z: usize, y: usize, x: usize| z * stride[0] + y * stride[1] + x * stride[2];

    let valid_seeds: Vec<[usize; 3]> = seeds
        .iter()
        .copied()
        .filter(|s| s[0] < zn && s[1] < yn && s[2] < xn)
        .collect();
    if valid_seeds.is_empty() {
        return out;
    }

    // ── Initial statistics over each seed's neighbourhood (averaged) ────────
    let mut mean = vec![0.0; c];
    let mut cov = vec![vec![0.0; c]; c];
    let r = initial_radius as isize;
    let mut idx_scratch: Vec<usize> = Vec::new();
    for s in &valid_seeds {
        idx_scratch.clear();
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let (nz, ny, nx) = (s[0] as isize + dz, s[1] as isize + dy, s[2] as isize + dx);
                    if nz < 0
                        || nz >= zn as isize
                        || ny < 0
                        || ny >= yn as isize
                        || nx < 0
                        || nx >= xn as isize
                    {
                        continue;
                    }
                    idx_scratch.push(flat(nz as usize, ny as usize, nx as usize));
                }
            }
        }
        let (m, cv) = mean_covariance(idx_scratch.len(), c, |k, row| {
            let vi = idx_scratch[k];
            for (j, r) in row.iter_mut().enumerate() {
                *r = channels[j][vi];
            }
        });
        for i in 0..c {
            mean[i] += m[i];
            for j in 0..c {
                cov[i][j] += cv[i][j];
            }
        }
    }
    let sc = valid_seeds.len() as f64;
    for i in 0..c {
        mean[i] /= sc;
        for j in 0..c {
            cov[i][j] /= sc;
        }
    }

    // ── Threshold: bump multiplier so every seed is included ────────────────
    let mut inv = inverse_covariance(&cov, c);
    let mut threshold = multiplier;
    let mut d_scratch = vec![0.0; c];
    for s in &valid_seeds {
        let d = maha_sq_at(
            channels,
            flat(s[0], s[1], s[2]),
            &mean,
            &inv,
            c,
            &mut d_scratch,
        )
        .max(0.0)
        .sqrt();
        if d > threshold {
            threshold = d;
        }
    }

    // ── Flood, then recompute statistics over the region and re-flood ───────
    let mut mask = flood(channels, dims, &valid_seeds, &mean, &inv, threshold, c);
    for _ in 0..iterations {
        idx_scratch.clear();
        idx_scratch.extend((0..n).filter(|&i| mask[i]));
        if idx_scratch.is_empty() {
            break;
        }
        let (m, cv) = mean_covariance(idx_scratch.len(), c, |k, row| {
            let vi = idx_scratch[k];
            for (j, r) in row.iter_mut().enumerate() {
                *r = channels[j][vi];
            }
        });
        mean = m;
        cov = cv;
        inv = inverse_covariance(&cov, c);
        mask = flood(channels, dims, &valid_seeds, &mean, &inv, threshold, c);
    }

    for i in 0..n {
        if mask[i] {
            out[i] = replace_value;
        }
    }
    out
}

/// Face-connected flood from `seeds`, including voxels whose Mahalanobis distance
/// is `≤ threshold`.
#[allow(clippy::too_many_arguments)]
fn flood(
    channels: &[Vec<f64>],
    dims: [usize; 3],
    seeds: &[[usize; 3]],
    mean: &[f64],
    inv: &[Vec<f64>],
    threshold: f64,
    c: usize,
) -> Vec<bool> {
    let [zn, yn, xn] = dims;
    let n = zn * yn * xn;
    let stride = [yn * xn, xn, 1usize];
    let flat = |z: usize, y: usize, x: usize| z * stride[0] + y * stride[1] + x * stride[2];
    let thr_sq = threshold * threshold;
    // One scratch buffer reused for every membership test (the flood visits each
    // voxel's neighbourhood, so this is the dominant allocation hot path).
    let mut d_scratch = vec![0.0; c];
    let mut inside = |i: usize| -> bool {
        // Compare squared distances to avoid a sqrt; sqrt is monotone and the
        // membership clamps negatives to 0, which are always ≤ thr_sq.
        maha_sq_at(channels, i, mean, inv, c, &mut d_scratch) <= thr_sq
    };
    let mut mask = vec![false; n];
    let mut stack: Vec<[usize; 3]> = Vec::new();
    for s in seeds {
        let i = flat(s[0], s[1], s[2]);
        if !mask[i] && inside(i) {
            mask[i] = true;
            stack.push(*s);
        }
    }
    while let Some([z, y, x]) = stack.pop() {
        let neighbors = [
            (z as isize - 1, y as isize, x as isize),
            (z as isize + 1, y as isize, x as isize),
            (z as isize, y as isize - 1, x as isize),
            (z as isize, y as isize + 1, x as isize),
            (z as isize, y as isize, x as isize - 1),
            (z as isize, y as isize, x as isize + 1),
        ];
        for (nz, ny, nx) in neighbors {
            if nz < 0
                || nz >= zn as isize
                || ny < 0
                || ny >= yn as isize
                || nx < 0
                || nx >= xn as isize
            {
                continue;
            }
            let (nz, ny, nx) = (nz as usize, ny as usize, nx as usize);
            let ni = flat(nz, ny, nx);
            if !mask[ni] && inside(ni) {
                mask[ni] = true;
                stack.push([nz, ny, nx]);
            }
        }
    }
    mask
}

/// Image-level wrapper: extract channels, run [`vector_confidence_connected`],
/// rebuild the label image carrying `channels[0]`'s spatial metadata.
///
/// # Panics
/// If `channels` is empty or the channel images differ in dimensions.
pub fn vector_confidence_connected_image<B: Backend>(
    channels: &[&Image<B, 3>],
    seeds: &[[usize; 3]],
    multiplier: f64,
    iterations: u32,
    initial_radius: usize,
    replace_value: f32,
) -> Image<B, 3> {
    let (first, dims) = extract_vec_infallible(channels[0]);
    let mut bufs: Vec<Vec<f64>> = Vec::with_capacity(channels.len());
    bufs.push(first.iter().map(|&v| v as f64).collect());
    for img in &channels[1..] {
        let (vals, d) = extract_vec_infallible(*img);
        assert_eq!(
            d, dims,
            "vector_confidence_connected: channels differ in dimensions"
        );
        bufs.push(vals.iter().map(|&v| v as f64).collect());
    }
    let labels = vector_confidence_connected(
        &bufs,
        dims,
        seeds,
        multiplier,
        iterations,
        initial_radius,
        replace_value,
    );
    rebuild(labels, dims, channels[0])
}

#[cfg(test)]
#[path = "tests_vector_confidence_connected.rs"]
mod tests;
