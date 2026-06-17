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
//! **Elongation** = √(λ_2/λ_1) ∈ \[1, ∞) (ITK convention); 1.0 when λ_1 ≤ 0.
//! **Flatness**   = √(λ_1/λ_0) ∈ \[1, ∞) (ITK convention); 1.0 when λ_0 ≤ 0.
//!
//! **Feret diameter** (ITK `GetFeretDiameter`): maximum Euclidean distance in
//! physical units between any two surface voxels of the region.
//!
//! **Perimeter** (ITK `GetPerimeter`): physical surface area via the Crofton
//! formula over the 13 unique directions of the 3-D 26-neighbourhood. For each
//! direction offset `o`, the intercept count is
//!   I_o = Σ_{v∈V_k} ( [v+o ∉ V_k] + [v−o ∉ V_k] )
//! (out-of-bounds counts as background), and
//!   perimeter = 4 · Σ_o ( vol / |o|_phys · I_o / 2 · w_o ),
//! with vol = sz·sy·sx, |o|_phys = ‖(o_x·sx, o_y·sy, o_z·sz)‖, and the Voronoi
//! direction weights w_o (Lindblad/ITK): 0.04577789120476·2 for the 3 axis
//! directions, 0.03698062787608·2 for the 6 face-diagonals, 0.03519563978232·2
//! for the 4 body-diagonals. Float-exact match to ITK/SimpleITK `GetPerimeter`.
//!
//! **Equivalent spherical radius** r_eq = (3·V_phys / 4π)^(1/3);
//! **equivalent spherical perimeter** (surface area) = 4π·r_eq².
//!
//! **Roundness** (ITK `GetRoundness`) = equivalent_spherical_perimeter / perimeter.
//!
//! # Complexity
//! Two serial passes per label (centroid+bbox, then moments), one parallel fold
//! to group voxel indices, and one parallel map over labels.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_spatial::Point;
use ritk_tensor_ops::extract_vec_infallible;
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
    /// Physical surface area via the 13-direction Crofton estimator (ITK
    /// `GetPerimeter`).
    pub perimeter: f64,
    /// equivalent_spherical_perimeter / perimeter (ITK `GetRoundness`); 1.0 for a
    /// perfect sphere, < 1 otherwise. 0.0 when perimeter = 0.
    pub roundness: f64,
    /// √(λ_1/λ_0): second-smallest/smallest principal moment ratio (ITK
    /// `ShapeLabelObject::GetFlatness`). ∈ \[1, ∞); 1.0 = isotropic.
    pub flatness: f64,
    /// √(λ_2/λ_1): largest/second-largest principal moment ratio (ITK
    /// `ShapeLabelObject::GetElongation`). ∈ \[1, ∞); 1.0 = isotropic.
    pub elongation: f64,
    /// Feret diameter: max physical distance between two surface voxels (ITK
    /// `GetFeretDiameter`).
    pub feret_diameter: f64,
    /// Radius of the sphere with the same physical volume: (3·V/4π)^(1/3).
    pub equivalent_spherical_radius: f64,
    /// Surface area of the equivalent-volume sphere: 4π·r_eq² (ITK
    /// `GetEquivalentSphericalPerimeter`).
    pub equivalent_spherical_perimeter: f64,
    /// Principal moments of inertia [λ_0, λ_1, λ_2] ascending, physical units².
    pub principal_moments: [f64; 3],
    /// Centroid in voxel coordinates.
    pub centroid: Point<3>,
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Singularity epsilon for the Cardano eigenvalue solver: when the off-diagonal
/// norm `p` falls below this threshold the inertia tensor is essentially a scalar
/// multiple of the identity and all three eigenvalues equal the mean diagonal `q̄`.
const EIGENVALUE_SINGULARITY_EPS: f64 = 1e-12;

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
    if p < EIGENVALUE_SINGULARITY_EPS {
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

/// The 13 unique directions of the 3-D 26-neighbourhood (one per antipodal
/// pair), each as a `(dz, dy, dx)` offset and its Crofton/Voronoi weight: the
/// 3 axis directions, 6 face-diagonals, then 4 body-diagonals. Weights are the
/// ITK/Lindblad constants (already including ITK's ×2 factor).
const CROFTON_DIRECTIONS: [(i64, i64, i64, f64); 13] = {
    const W_AXIS: f64 = 0.04577789120476 * 2.0;
    const W_FACE: f64 = 0.03698062787608 * 2.0;
    const W_BODY: f64 = 0.03519563978232 * 2.0;
    [
        (0, 0, 1, W_AXIS),
        (0, 1, 0, W_AXIS),
        (1, 0, 0, W_AXIS),
        (0, 1, 1, W_FACE),
        (0, 1, -1, W_FACE),
        (1, 0, 1, W_FACE),
        (1, 0, -1, W_FACE),
        (1, 1, 0, W_FACE),
        (1, -1, 0, W_FACE),
        (1, 1, 1, W_BODY),
        (1, 1, -1, W_BODY),
        (1, -1, 1, W_BODY),
        (1, -1, -1, W_BODY),
    ]
};

/// Physical surface area of label `k` via the 13-direction Crofton estimator
/// (ITK `GetPerimeter`), float-exact to SimpleITK.
///
/// For each direction offset `o` the intercept count is
/// `I_o = Σ_{v∈label} ([v+o ∉ label] + [v−o ∉ label])` (out-of-bounds counts as
/// background), and `perimeter = 2 · vol · Σ_o (I_o · w_o / |o|_phys)`
/// (the `4 · … · I_o/2` of the ITK formulation folded into `2·`).
fn crofton_perimeter(
    indices: &[usize],
    label_slice: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    k: u32,
) -> f64 {
    let [dim_z, dim_y, dim_x] = [dims[0] as i64, dims[1] as i64, dims[2] as i64];
    let yx = dims[1] * dims[2];
    let [sz, sy, sx] = spacing;
    let vol = sz * sy * sx;

    let in_label = |z: i64, y: i64, x: i64| -> bool {
        if z < 0 || y < 0 || x < 0 || z >= dim_z || y >= dim_y || x >= dim_x {
            return false;
        }
        label_slice[z as usize * yx + y as usize * dims[2] + x as usize] as u32 == k
    };

    let mut sum = 0.0_f64;
    for &(oz, oy, ox, w) in CROFTON_DIRECTIONS.iter() {
        let d_phys = ((ox as f64 * sx).powi(2) + (oy as f64 * sy).powi(2) + (oz as f64 * sz).powi(2))
            .sqrt();
        let mut intercepts = 0_u64;
        for &idx in indices {
            let z = (idx / yx) as i64;
            let y = ((idx / dims[2]) % dims[1]) as i64;
            let x = (idx % dims[2]) as i64;
            if !in_label(z + oz, y + oy, x + ox) {
                intercepts += 1;
            }
            if !in_label(z - oz, y - oy, x - ox) {
                intercepts += 1;
            }
        }
        sum += intercepts as f64 * w / d_phys;
    }
    2.0 * vol * sum
}

/// Feret diameter: maximum physical distance between any two surface voxels
/// (ITK `GetFeretDiameter`). Interior voxels can never be the farthest pair, so
/// the search is restricted to the 6-connected boundary set (size B ≪ N); the
/// pairwise scan is O(B²), matching ITK's own complexity.
fn feret_from_boundary(boundary_phys: &[[f64; 3]]) -> f64 {
    let mut max_sq = 0.0_f64;
    for (i, a) in boundary_phys.iter().enumerate() {
        for b in &boundary_phys[i + 1..] {
            let dz = a[0] - b[0];
            let dy = a[1] - b[1];
            let dx = a[2] - b[2];
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

            // ── Pass 1: centroid ──────────────────────────────────────────────
            let (mut sum_z, mut sum_y, mut sum_x) = (0_i64, 0_i64, 0_i64);
            for &idx in &indices {
                let (z, y, x) = decode(idx);
                sum_z += z as i64;
                sum_y += y as i64;
                sum_x += x as i64;
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

            // ── Shape descriptors (ITK ShapeLabelObject convention) ───────────
            // Principal moments are ascending [λ0 ≤ λ1 ≤ λ2]. ITK defines both
            // ratios over ADJACENT moments and ≥ 1:
            //   elongation = √(λ2 / λ1)  (largest / second-largest)
            //   flatness   = √(λ1 / λ0)  (second-smallest / smallest)
            // For a solid ellipsoid with semi-axes a ≥ b ≥ c these reduce to
            // a/b and b/c respectively.
            let elongation = if lambda1 <= 0.0 {
                1.0
            } else {
                (lambda2 / lambda1).sqrt()
            };
            let flatness = if lambda0 <= 0.0 {
                1.0
            } else {
                (lambda1 / lambda0).sqrt()
            };

            // ── Surface (6-connected boundary) voxels in physical coords ──────
            // (used by the Feret diameter; B = |boundary| ≪ N).
            let boundary_phys: Vec<[f64; 3]> = indices
                .iter()
                .filter_map(|&idx| {
                    let (z, y, x) = decode(idx);
                    if is_boundary(z, y, x, label_slice, dims, label) {
                        Some([z as f64 * spc_z, y as f64 * spc_y, x as f64 * spc_x])
                    } else {
                        None
                    }
                })
                .collect();
            let feret = feret_from_boundary(&boundary_phys);

            // ── Perimeter (13-direction Crofton surface area, ITK) ────────────
            let perimeter = crofton_perimeter(&indices, label_slice, dims, spacing, label);

            // ── Equivalent-sphere descriptors and roundness ───────────────────
            let volume_physical = n_f * spc_z * spc_y * spc_x;
            let equivalent_spherical_radius = (3.0 * volume_physical / (4.0 * PI)).cbrt();
            let equivalent_spherical_perimeter =
                4.0 * PI * equivalent_spherical_radius * equivalent_spherical_radius;
            let roundness = if perimeter > 0.0 {
                equivalent_spherical_perimeter / perimeter
            } else {
                0.0
            };

            LabelShapeStatisticsExtended {
                label,
                count: n,
                perimeter,
                roundness,
                flatness,
                elongation,
                feret_diameter: feret,
                equivalent_spherical_radius,
                equivalent_spherical_perimeter,
                principal_moments: eigs,
                centroid: Point::new([cz, cy, cx]),
            }
        })
        .collect();

    result.sort_by_key(|s| s.label);
    result
}
