//! Jacobian determinant of a 3-D displacement field.
//!
//! # Mathematical Specification
//!
//! Given a displacement field **u**(x) = (u₀, u₁, u₂) at each voxel x, the
//! deformation φ(x) = x + **u**(x).  The Jacobian matrix of φ at x is:
//!
//!   J(x) = I + ∇**u**(x)
//!
//! where ∇**u** is the 3×3 matrix of partial derivatives:
//!
//!   ∇**u**ᵢⱼ = ∂u_i/∂x_j  for i,j ∈ {0,1,2}
//!
//! The **Jacobian determinant** det(J) measures local volume change:
//! - det(J) > 1 : local expansion
//! - det(J) = 1 : volume-preserving (topology-preserving ideal)
//! - 0 < det(J) < 1 : local compression
//! - det(J) ≤ 0 : folding / singularity (anatomically invalid)
//!
//! ## Jacobian matrix layout
//!
//! ```text
//! J = [ 1+∂u_z/∂z   ∂u_z/∂y   ∂u_z/∂x ]
//!     [   ∂u_y/∂z 1+∂u_y/∂y   ∂u_y/∂x ]
//!     [   ∂u_x/∂z   ∂u_x/∂y 1+∂u_x/∂x ]
//! ```
//!
//! ## Gradient computation
//!
//! Finite differences: central scheme at interior voxels, first-order one-sided
//! at boundaries.  For a 1-D array f with spacing h:
//!
//! Interior (0 < i < n−1): ∂f/∂x ≈ (f\[i+1\] − f\[i−1\]) / (2h)
//! Left boundary (i = 0): ∂f/∂x ≈ (f\[1\] − f\[0\]) / h
//! Right boundary (i = n−1): ∂f/∂x ≈ (f\[n−1\] − f\[n−2\]) / h
//!
//! ## Parallelism
//!
//! The outer Z loop is parallelised with `moirai::Adaptive`; each Z-slice is independent.

use anyhow::{anyhow, Result};
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

// ── Public structs ────────────────────────────────────────────────────────────

/// Summary statistics for a Jacobian determinant field.
///
/// Voxels are partitioned by their determinant value into three disjoint sets:
///
/// | Category    | Condition          | Meaning                         |
/// |-------------|--------------------|---------------------------------|
/// | folded      | det(J) ≤ 0         | topological singularity         |
/// | compressed  | 0 < det(J) < 1     | local volume shrinkage          |
/// | expanded    | det(J) ≥ 1         | volume-preserving or expanding  |
///
/// `num_valid` = `num_compressed` + `num_expanded` (all voxels with det(J) > 0).
#[derive(Debug, Clone)]
pub struct JacobianStats {
    /// Minimum determinant value across all voxels.
    pub min: f32,
    /// Maximum determinant value across all voxels.
    pub max: f32,
    /// Mean determinant value; f64 accumulator to prevent f32 precision loss.
    pub mean: f64,
    /// Number of voxels with det(J) ≤ 0 (folding / topological singularity).
    pub num_folded: usize,
    /// Number of voxels with 0 < det(J) < 1 (local compression).
    pub num_compressed: usize,
    /// Number of voxels with det(J) ≥ 1 (volume-preserving or expanding).
    pub num_expanded: usize,
    /// Number of voxels with det(J) > 0 (= `num_compressed` + `num_expanded`).
    pub num_valid: usize,
    /// Total number of voxels in the image.
    pub total_voxels: usize,
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Half-coefficient in the central-difference formula: (f[i+1] − f[i−1]) / (2·h) = (f[i+1] − f[i−1]) * (0.5 / h).
const CENTRAL_DIFF_HALF: f32 = 0.5;

/// Grid coordinate and shape for a 3-D volume of shape `[nz, ny, nx]`.
///
/// Groups the six index/extent parameters shared by `diff_z`, `diff_y`, `diff_x`
/// into a single struct to satisfy the clippy `too_many_arguments` lint.
struct GridCoord {
    z: usize,
    y: usize,
    x: usize,
    nz: usize,
    ny: usize,
    nx: usize,
}

/// Row-major flat index for a 3-D array of shape [nz, ny, nx].
///
/// Invariant: returned index < nz * ny * nx when z < nz, y < ny, x < nx.
#[inline(always)]
fn flat(z: usize, y: usize, x: usize, ny: usize, nx: usize) -> usize {
    z * ny * nx + y * nx + x
}

/// Finite-difference derivative of `field` along the Z axis (axis 0).
///
/// `sz_inv` = 1.0 / spacing_z. Returns 0.0 when nz == 1 (degenerate axis).
#[inline(always)]
fn diff_z(field: &[f32], gc: &GridCoord, sz_inv: f32) -> f32 {
    let GridCoord {
        z,
        y,
        x,
        nz,
        ny,
        nx,
    } = *gc;
    if nz == 1 {
        return 0.0;
    }
    if z == 0 {
        (field[flat(1, y, x, ny, nx)] - field[flat(0, y, x, ny, nx)]) * sz_inv
    } else if z == nz - 1 {
        (field[flat(nz - 1, y, x, ny, nx)] - field[flat(nz - 2, y, x, ny, nx)]) * sz_inv
    } else {
        (field[flat(z + 1, y, x, ny, nx)] - field[flat(z - 1, y, x, ny, nx)])
            * (CENTRAL_DIFF_HALF * sz_inv)
    }
}

/// Finite-difference derivative of `field` along the Y axis (axis 1).
///
/// `sy_inv` = 1.0 / spacing_y. Returns 0.0 when ny == 1.
#[inline(always)]
fn diff_y(field: &[f32], gc: &GridCoord, sy_inv: f32) -> f32 {
    let GridCoord {
        z,
        y,
        x,
        nz: _,
        ny,
        nx,
    } = *gc;
    if ny == 1 {
        return 0.0;
    }
    if y == 0 {
        (field[flat(z, 1, x, ny, nx)] - field[flat(z, 0, x, ny, nx)]) * sy_inv
    } else if y == ny - 1 {
        (field[flat(z, ny - 1, x, ny, nx)] - field[flat(z, ny - 2, x, ny, nx)]) * sy_inv
    } else {
        (field[flat(z, y + 1, x, ny, nx)] - field[flat(z, y - 1, x, ny, nx)])
            * (CENTRAL_DIFF_HALF * sy_inv)
    }
}

/// Finite-difference derivative of `field` along the X axis (axis 2).
///
/// `sx_inv` = 1.0 / spacing_x. Returns 0.0 when nx == 1.
#[inline(always)]
fn diff_x(field: &[f32], gc: &GridCoord, sx_inv: f32) -> f32 {
    let GridCoord {
        z,
        y,
        x,
        nz: _,
        ny,
        nx,
    } = *gc;
    if nx == 1 {
        return 0.0;
    }
    if x == 0 {
        (field[flat(z, y, 1, ny, nx)] - field[flat(z, y, 0, ny, nx)]) * sx_inv
    } else if x == nx - 1 {
        (field[flat(z, y, nx - 1, ny, nx)] - field[flat(z, y, nx - 2, ny, nx)]) * sx_inv
    } else {
        (field[flat(z, y, x + 1, ny, nx)] - field[flat(z, y, x - 1, ny, nx)])
            * (CENTRAL_DIFF_HALF * sx_inv)
    }
}

/// Determinant of a 3×3 matrix stored as a row-major `[f32; 9]` array.
///
/// Layout: `[[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]]`.
///
/// Cofactor expansion along the first row:
///   det = m00(m11·m22 − m12·m21)
///       − m01(m10·m22 − m12·m20)
///       + m02(m10·m21 − m11·m20)
#[inline(always)]
fn det3(m: [f32; 9]) -> f32 {
    let [m00, m01, m02, m10, m11, m12, m20, m21, m22] = m;
    m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute the Jacobian determinant of a displacement field at every voxel.
///
/// # Arguments
/// - `disp_z`, `disp_y`, `disp_x`: three components of the displacement field,
///   each of shape [nz, ny, nx] with the same physical spacing.
///
/// # Returns
/// `Image<f32, B, 3>` of shape [nz, ny, nx] where each voxel holds det(J(φ)).
///
/// # Errors
/// Returns `Err` when the three displacement components differ in shape or when
/// the backend tensor cannot be converted to f32.
///
/// # Invariants
/// - det(J) > 0 everywhere → topology-preserving deformation.
/// - det(J) ≤ 0 at any voxel → folding (anatomically invalid).
pub fn jacobian_determinant<B: Backend>(
    disp_z: &Image<f32, B, 3>,
    disp_y: &Image<f32, B, 3>,
    disp_x: &Image<f32, B, 3>,
) -> Result<Image<f32, B, 3>> {
    let (uz, dims) = extract_vec(disp_z)?;
    let (uy, dims_y) = extract_vec(disp_y)?;
    let (ux, dims_x) = extract_vec(disp_x)?;

    if dims_y != dims || dims_x != dims {
        return Err(anyhow!(
            "jacobian_determinant: displacement components must have identical shape; \
             got z={dims:?}, y={dims_y:?}, x={dims_x:?}"
        ));
    }

    let [nz, ny, nx] = dims;
    let total = nz * ny * nx;

    // Physical spacing is stored as f64 in the Vector/Spacing newtype; cast to
    // f32 here because all field data is f32 and the arithmetic stays in f32.
    let sp = disp_z.spacing();
    let sz_inv = (1.0 / sp[0]) as f32;
    let sy_inv = (1.0 / sp[1]) as f32;
    let sx_inv = (1.0 / sp[2]) as f32;

    let mut det_vals = vec![0.0f32; total];

    // Parallelise over Z-slices; slices are independent and do not alias.
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut det_vals,
        ny * nx,
        |z, chunk| {
            for y in 0..ny {
                for x in 0..nx {
                    let gc = GridCoord {
                        z,
                        y,
                        x,
                        nz,
                        ny,
                        nx,
                    };

                    // Row 0 of ∇u: partial derivatives of u_z.
                    let duz_dz = diff_z(&uz, &gc, sz_inv);
                    let duz_dy = diff_y(&uz, &gc, sy_inv);
                    let duz_dx = diff_x(&uz, &gc, sx_inv);

                    // Row 1 of ∇u: partial derivatives of u_y.
                    let duy_dz = diff_z(&uy, &gc, sz_inv);
                    let duy_dy = diff_y(&uy, &gc, sy_inv);
                    let duy_dx = diff_x(&uy, &gc, sx_inv);

                    // Row 2 of ∇u: partial derivatives of u_x.
                    let dux_dz = diff_z(&ux, &gc, sz_inv);
                    let dux_dy = diff_y(&ux, &gc, sy_inv);
                    let dux_dx = diff_x(&ux, &gc, sx_inv);

                    // J = I + ∇u; det via cofactor expansion along first row.
                    chunk[y * nx + x] = det3([
                        1.0 + duz_dz,
                        duz_dy,
                        duz_dx,
                        duy_dz,
                        1.0 + duy_dy,
                        duy_dx,
                        dux_dz,
                        dux_dy,
                        1.0 + dux_dx,
                    ]);
                }
            }
        },
    );

    Ok(rebuild(det_vals, dims, disp_z))
}

/// Analyze a Jacobian determinant field and return per-category voxel counts
/// together with scalar statistics.
///
/// Classification boundaries (disjoint and exhaustive):
///
/// ```text
/// det ≤ 0          → folded     (topological singularity)
/// 0 < det < 1      → compressed (local volume shrinkage)
/// det ≥ 1          → expanded   (volume-preserving or expanding)
/// ```
///
/// `num_valid` = `num_compressed` + `num_expanded`.
///
/// # Errors
/// Returns `Err` when the backend tensor cannot be converted to f32, or the
/// image is empty.
pub fn analyze_jacobian<B: Backend>(jac: &Image<f32, B, 3>) -> Result<JacobianStats> {
    let (vals, _) = extract_vec(jac)?;
    let n = vals.len();
    if n == 0 {
        return Err(anyhow!("analyze_jacobian: image contains no voxels"));
    }

    // Parallel reduction through Moirai: each worker maintains a local accumulator
    // and the results are combined through the policy reduction. This scales with
    // the selected execution policy without coupling statistics to Rayon.
    let (min, max, sum, num_folded, num_compressed, num_expanded) =
        moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
            vals.len(),
            || {
                (
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    0.0f64,
                    0usize,
                    0usize,
                    0usize,
                )
            },
            |(min, max, sum, folded, compressed, expanded), i| {
                let v = vals[i];
                let new_min = if v < min { v } else { min };
                let new_max = if v > max { v } else { max };
                let new_sum = sum + v as f64;
                if v <= 0.0 {
                    (new_min, new_max, new_sum, folded + 1, compressed, expanded)
                } else if v < 1.0 {
                    (new_min, new_max, new_sum, folded, compressed + 1, expanded)
                } else {
                    (new_min, new_max, new_sum, folded, compressed, expanded + 1)
                }
            },
            |(min1, max1, sum1, f1, c1, e1), (min2, max2, sum2, f2, c2, e2)| {
                (
                    min1.min(min2),
                    max1.max(max2),
                    sum1 + sum2,
                    f1 + f2,
                    c1 + c2,
                    e1 + e2,
                )
            },
        );

    Ok(JacobianStats {
        min,
        max,
        mean: sum / n as f64,
        num_folded,
        num_compressed,
        num_expanded,
        num_valid: num_compressed + num_expanded,
        total_voxels: n,
    })
}

#[cfg(test)]
#[path = "tests_jacobian.rs"]
mod tests;
