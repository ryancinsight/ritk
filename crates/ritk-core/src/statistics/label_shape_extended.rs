//! Extended label shape statistics for 3-D label maps.
//!
//! # Mathematical Specification
//!
//! Given label map L\[z,y,x\] of shape \[Z,Y,X\] with voxel spacing s = \[sz,sy,sx\]:
//!
//! For each non-zero label k with voxel set V_k = { (z,y,x) : L=k }, |V_k|=N:
//!
//! **Centroid** (voxel coordinates):
//!   cz = (1/N) Σ z_i,  cy = (1/N) Σ y_i,  cx = (1/N) Σ x_i
//!
//! **Inertia tensor** (second central moments, physical coordinates):
//!   p = [(z−cz)·sz, (y−cy)·sy, (x−cx)·sx]
//!   I\[r,c\] = (1/N) Σ_{V_k} p\[r\]·p\[c\]
//!
//! **Principal moments** λ_0 ≤ λ_1 ≤ λ_2: eigenvalues of I, computed via the
//! Cardano / characteristic-polynomial method (Kopp 2008, Eberly 2021 §2).
//!
//! **Elongation** = √(λ_1/λ_2) ∈ \[0,1\];  returns 1.0 when λ_2 ≤ 0.
//! **Flatness**   = √(λ_0/λ_2) ∈ \[0,1\];  returns 1.0 when λ_2 ≤ 0.
//!
//! **Feret diameter** (approximate): maximum Euclidean distance between the 8
//! axis-aligned bounding-box corners in physical units.
//!
//! **Roundness** = clamp( V_phys / (π/6 · d³), 0, 1 );  0.0 when d = 0.
//! where V_phys = N·sz·sy·sx  and  d = feret_diameter.
//!
//! **Perimeter** = |{ v ∈ V_k : ∃ 6-connected neighbour n with L\[n\] ≠ k }|.
//!
//! # Complexity
//! Two serial passes per label (centroid+bbox, then moments), one parallel fold
//! to group voxel indices, and one parallel map over labels.

use crate::filter::ops::extract_vec_infallible;
use crate::image::Image;
use crate::spatial::Point;
use burn::tensor::backend::Backend;
use std::collections::HashMap;
use std::f64::consts::PI;

// ── Public struct ─────────────────────────────────────────────────────────────

/// Extended shape statistics for a single non-background label region.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelShapeStatisticsExtended {
    /// Integer label (≥1; background=0 excluded).
    pub label: u32,
    /// Voxel count.
    pub count: usize,
    /// Number of surface voxels (6-connected boundary).
    pub perimeter: usize,
    /// V_phys / (π/6 · feret³): ∈ (0,1], 1.0 = sphere. 0.0 if feret = 0.
    pub roundness: f64,
    /// √(λ_0/λ_2): smallest/largest principal moment ratio. ∈ \[0,1\].
    pub flatness: f64,
    /// √(λ_1/λ_2): middle/largest principal moment ratio. ∈ \[0,1\].
    pub elongation: f64,
    /// Approximate Feret diameter (bounding-box diagonal) in physical units.
    pub feret_diameter: f64,
    /// Principal moments of inertia [λ_0, λ_1, λ_2] ascending, physical units².
    pub principal_moments: [f64; 3],
    /// Centroid in voxel coordinates.
    pub centroid: Point<3>,
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Eigenvalues of a 3×3 real symmetric matrix via Cardano's formula.
///
/// # Formula
/// With q̄ = tr(M)/3,  p1 = Σ off-diagonal²,  p2 = Σ(M_ii−q̄)² + 2p1:
///   p = √(p2/6),  B = (M − q̄I)/p  (entries in [−2,2] when p > 0),
///   r = clamp(det(B)/2, −1, 1),  φ = arccos(r)/3.
/// Eigenvalues: q̄ + 2p·cos(φ + 2kπ/3) for k = 0,1,2.
/// Degenerate (p ≈ 0): all eigenvalues = q̄.
/// Returns values sorted ascending.
#[inline]
fn sym3_eigenvalues(m00: f64, m11: f64, m22: f64, m01: f64, m02: f64, m12: f64) -> [f64; 3] {
    let q = (m00 + m11 + m22) / 3.0;
    let p1 = m01 * m01 + m02 * m02 + m12 * m12;
    let p2 = (m00 - q) * (m00 - q) + (m11 - q) * (m11 - q) + (m22 - q) * (m22 - q) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();

    // Degenerate / spherical case: matrix is a scalar multiple of identity.
    if p < 1e-12 {
        return [q, q, q];
    }

    // Scaled matrix B = (M − q·I) / p; guaranteed det(B) ∈ [−2, 2].
    let b00 = (m00 - q) / p;
    let b11 = (m11 - q) / p;
    let b22 = (m22 - q) / p;
    let b01 = m01 / p;
    let b02 = m02 / p;
    let b12 = m12 / p;

    let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
        + b02 * (b01 * b12 - b11 * b02);

    let r = (det_b / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;

    let eig0 = q + 2.0 * p * phi.cos();
    let eig1 = q + 2.0 * p * (phi + 2.0 * PI / 3.0).cos();
    // Use the third cosine term instead of the trace invariant (3q − eig0 − eig1)
    // to avoid catastrophic cancellation when two eigenvalues are near-equal or zero.
    let eig2 = q + 2.0 * p * (phi + 4.0 * PI / 3.0).cos();

    let mut eigs = [eig0, eig1, eig2];
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigs
}

/// Returns `true` when voxel (z, y, x) has at least one 6-connected neighbour
/// that is out-of-bounds or carries a label different from `k`.
#[inline]
fn is_boundary(
    z: usize,
    y: usize,
    x: usize,
    label_slice: &[f32],
    dims: [usize; 3],
    k: u32,
) -> bool {
    const DELTA: [(i64, i64, i64); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];
    let [dim_z, dim_y, dim_x] = [dims[0] as i64, dims[1] as i64, dims[2] as i64];
    let yx = dims[1] * dims[2];
    for (dz, dy, dx) in DELTA {
        let nz = z as i64 + dz;
        let ny = y as i64 + dy;
        let nx = x as i64 + dx;
        if nz < 0 || ny < 0 || nx < 0 || nz >= dim_z || ny >= dim_y || nx >= dim_x {
            return true;
        }
        let idx = nz as usize * yx + ny as usize * dims[2] + nx as usize;
        if label_slice[idx] as u32 != k {
            return true;
        }
    }
    false
}

/// Maximum Euclidean distance between the 8 axis-aligned bounding-box corners
/// in physical coordinates.  This is the standard voxel-data approximation of
/// the Feret (caliper) diameter.
///
/// Corners: {z_min,z_max}·sz × {y_min,y_max}·sy × {x_min,x_max}·sx.
fn feret_from_bbox(
    z_min: i64,
    z_max: i64,
    y_min: i64,
    y_max: i64,
    x_min: i64,
    x_max: i64,
    spacing: [f64; 3],
) -> f64 {
    let [sz, sy, sx] = spacing;
    let corners: [[f64; 3]; 8] = [
        [z_min as f64 * sz, y_min as f64 * sy, x_min as f64 * sx],
        [z_min as f64 * sz, y_min as f64 * sy, x_max as f64 * sx],
        [z_min as f64 * sz, y_max as f64 * sy, x_min as f64 * sx],
        [z_min as f64 * sz, y_max as f64 * sy, x_max as f64 * sx],
        [z_max as f64 * sz, y_min as f64 * sy, x_min as f64 * sx],
        [z_max as f64 * sz, y_min as f64 * sy, x_max as f64 * sx],
        [z_max as f64 * sz, y_max as f64 * sy, x_min as f64 * sx],
        [z_max as f64 * sz, y_max as f64 * sy, x_max as f64 * sx],
    ];
    let mut max_sq = 0.0_f64;
    for i in 0..8 {
        for j in (i + 1)..8 {
            let dz = corners[i][0] - corners[j][0];
            let dy = corners[i][1] - corners[j][1];
            let dx = corners[i][2] - corners[j][2];
            let d2 = dz * dz + dy * dy + dx * dx;
            if d2 > max_sq {
                max_sq = d2;
            }
        }
    }
    max_sq.sqrt()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute extended label shape statistics from a 3-D `Image` tensor.
///
/// Spacing is read from `label_image.spacing()`; shape from `label_image.shape()`.
/// Background (label 0) is excluded from results.
///
/// # Panics
/// Panics if the backend tensor cannot be converted to `f32`.
pub fn compute_label_shape_statistics_extended<B: Backend>(
    label_image: &Image<B, 3>,
) -> Vec<LabelShapeStatisticsExtended> {
    let (label_vals, dims) = extract_vec_infallible(label_image);
    let sp = label_image.spacing();
    let spacing = [sp[0], sp[1], sp[2]];
    compute_label_shape_statistics_extended_from_slices(&label_vals, dims, spacing)
}

/// Compute extended label shape statistics from pre-extracted flat slices.
///
/// `dims`:    `[Z, Y, X]` voxel dimensions (row-major: z outermost).
/// `spacing`: `[sz, sy, sx]` physical voxel size in any consistent unit.
///
/// Returns a `Vec<LabelShapeStatisticsExtended>` sorted ascending by label.
pub fn compute_label_shape_statistics_extended_from_slices(
    label_slice: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
) -> Vec<LabelShapeStatisticsExtended> {
    let yx = dims[1] * dims[2];
    let [spc_z, spc_y, spc_x] = spacing;

    // ── Step 1: parallel grouping of flat voxel indices per label ─────────────
    let voxels_per_label: HashMap<u32, Vec<usize>> =
        moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
            label_slice.len(),
            HashMap::<u32, Vec<usize>>::new,
            |mut map, idx| {
                let label = label_slice[idx] as u32;
                if label != 0 {
                    map.entry(label).or_default().push(idx);
                }
                map
            },
            |mut a, b| {
                for (k, mut v) in b {
                    a.entry(k).or_default().append(&mut v);
                }
                a
            },
        );

    // ── Step 2: independent per-label computation (parallelised over labels) ──
    // Sequential over labels (their count is far below any parallel threshold).
    let mut result: Vec<LabelShapeStatisticsExtended> = voxels_per_label
        .into_iter()
        .map(|(label, indices)| {
            let n = indices.len();
            let n_f = n as f64;

            // Decode flat index to (z, y, x) voxel coordinates.
            let decode = |idx: usize| -> (usize, usize, usize) {
                let z = idx / yx;
                let y = (idx / dims[2]) % dims[1];
                let x = idx % dims[2];
                (z, y, x)
            };

            // ── Pass 1: centroid and bounding box ─────────────────────────────
            let (mut sum_z, mut sum_y, mut sum_x) = (0_i64, 0_i64, 0_i64);
            let (mut z_min, mut z_max) = (i64::MAX, i64::MIN);
            let (mut y_min, mut y_max) = (i64::MAX, i64::MIN);
            let (mut x_min, mut x_max) = (i64::MAX, i64::MIN);
            for &idx in &indices {
                let (z, y, x) = decode(idx);
                let (zi, yi, xi) = (z as i64, y as i64, x as i64);
                sum_z += zi;
                sum_y += yi;
                sum_x += xi;
                if zi < z_min {
                    z_min = zi;
                }
                if zi > z_max {
                    z_max = zi;
                }
                if yi < y_min {
                    y_min = yi;
                }
                if yi > y_max {
                    y_max = yi;
                }
                if xi < x_min {
                    x_min = xi;
                }
                if xi > x_max {
                    x_max = xi;
                }
            }
            let cz = sum_z as f64 / n_f;
            let cy = sum_y as f64 / n_f;
            let cx = sum_x as f64 / n_f;

            // ── Pass 2: second central moments in physical coordinates ─────────
            let (mut acc_zz, mut acc_yy, mut acc_xx) = (0.0_f64, 0.0_f64, 0.0_f64);
            let (mut acc_zy, mut acc_zx, mut acc_yx) = (0.0_f64, 0.0_f64, 0.0_f64);
            for &idx in &indices {
                let (z, y, x) = decode(idx);
                let pz = (z as f64 - cz) * spc_z;
                let py = (y as f64 - cy) * spc_y;
                let px = (x as f64 - cx) * spc_x;
                acc_zz += pz * pz;
                acc_yy += py * py;
                acc_xx += px * px;
                acc_zy += pz * py;
                acc_zx += pz * px;
                acc_yx += py * px;
            }
            // Normalise to get the second-moment tensor (per-voxel mean).
            let (m00, m11, m22) = (acc_zz / n_f, acc_yy / n_f, acc_xx / n_f);
            let (m01, m02, m12) = (acc_zy / n_f, acc_zx / n_f, acc_yx / n_f);

            // ── Eigenvalues of the 3×3 symmetric inertia tensor ───────────────
            let eigs = sym3_eigenvalues(m00, m11, m22, m01, m02, m12);
            let [lambda0, lambda1, lambda2] = eigs;

            // ── Shape descriptors ─────────────────────────────────────────────
            let elongation = if lambda2 <= 0.0 {
                1.0
            } else {
                (lambda1.max(0.0) / lambda2).sqrt()
            };
            let flatness = if lambda2 <= 0.0 {
                1.0
            } else {
                (lambda0.max(0.0) / lambda2).sqrt()
            };

            let feret = feret_from_bbox(z_min, z_max, y_min, y_max, x_min, x_max, spacing);

            let volume_physical = n_f * spc_z * spc_y * spc_x;
            let roundness = if feret <= 0.0 {
                0.0
            } else {
                (volume_physical / (PI / 6.0 * feret.powi(3))).clamp(0.0, 1.0)
            };

            // ── Perimeter: count 6-connected surface voxels ───────────────────
            let perimeter = indices
                .iter()
                .filter(|&&idx| {
                    let (z, y, x) = decode(idx);
                    is_boundary(z, y, x, label_slice, dims, label)
                })
                .count();

            LabelShapeStatisticsExtended {
                label,
                count: n,
                perimeter,
                roundness,
                flatness,
                elongation,
                feret_diameter: feret,
                principal_moments: eigs,
                centroid: Point::new([cz, cy, cx]),
            }
        })
        .collect();

    result.sort_by_key(|s| s.label);
    result
}
