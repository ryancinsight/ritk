//! Thin-plate-spline inversion of a dense displacement field
//! (`itk::InverseDisplacementFieldImageFilter` / `sitk.InverseDisplacementField`).
//!
//! # Mathematical specification
//!
//! Given a forward field `u` (transform `x ↦ x + u(x)` in world coordinates),
//! the inverse field `v` is built by fitting a kernel (thin-plate-spline)
//! transform to landmark pairs sampled from `u`:
//!
//! 1. **Subsample** the field every `subsampling_factor`-th voxel per axis. The
//!    subsampled grid is a subset of the input grid, so the sample at subsampled
//!    point `k` is the exact field value at input index `k·factor` (no
//!    interpolation). `N = ∏_a ⌊size_a / factor⌋` landmarks.
//! 2. **Landmarks**: for subsampled voxel at world point `p` with displacement
//!    `d`, `source = p + d`, `target = p`; the kernel displacement is
//!    `target − source = −d`.
//! 3. **Fit** the ITK `KernelTransform` (G(r) = r, the thin-plate-spline kernel):
//!    solve `L·W = Y` with `L = [[K, P], [Pᵀ, 0]]`, `K_ij = ‖s_i − s_j‖·I_d`,
//!    `P_i = [s_i[0]·I_d, …, s_i[d−1]·I_d, I_d]`, `Y = [−d_i; 0]`. Reorganise
//!    `W` into the spline matrix `D` (d×N), affine `A` (d×d), and translation
//!    `B` (d): `D[k][i] = W[i·d+k]`, `A[i][j] = W[N·d + j·d + i]` (note the
//!    transpose), `B[k] = W[N·d + d·d + k]`.
//! 4. **Evaluate** per output voxel `q` (world): the inverse displacement is
//!    `A·q + B + Σ_i ‖q − s_i‖·D[:,i]` (`= TransformPoint(q) − q`).
//!
//! The TPS system is unique and well-conditioned, so the result is float-exact
//! to `sitk.InverseDisplacementField` (independent of the linear solver). A
//! `z == 1` field is inverted as a genuine 2-D field (axes `y, x`), matching
//! sitk's 2-D filter. Internal arithmetic is `f64`. Axis-aligned (identity
//! direction) is assumed, as for the sibling inversion filters.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Parameters and entry point for thin-plate-spline displacement-field inversion.
#[derive(Debug, Clone)]
pub struct InverseDisplacementField {
    /// Subsampling factor applied to every axis when building landmarks.
    pub subsampling_factor: usize,
}

impl Default for InverseDisplacementField {
    fn default() -> Self {
        Self {
            subsampling_factor: 16,
        }
    }
}

/// Solve the dense system `a·x = b` by Gaussian elimination with partial
/// pivoting (`a` is `n×n`, consumed). The TPS matrix is non-singular, so the
/// solution is unique.
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                piv = r;
            }
        }
        if piv != col {
            a.swap(piv, col);
            b.swap(piv, col);
        }
        let d = a[col][col];
        for r in (col + 1)..n {
            let f = a[r][col] / d;
            if f != 0.0 {
                for c in col..n {
                    a[r][c] -= f * a[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    // Back-substitution.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for c in (i + 1)..n {
            s -= a[i][c] * x[c];
        }
        x[i] = s / a[i][i];
    }
    x
}

impl InverseDisplacementField {
    /// Invert the field whose world-frame components are `comp_x`, `comp_y`,
    /// `comp_z` (each a scalar `[z, y, x]` image on a shared grid). Returns the
    /// inverse components `(inv_x, inv_y, inv_z)` on the same grid.
    pub fn apply<B: Backend>(
        &self,
        comp_x: &Image<B, 3>,
        comp_y: &Image<B, 3>,
        comp_z: &Image<B, 3>,
    ) -> (Image<B, 3>, Image<B, 3>, Image<B, 3>) {
        let (ux, dims) = extract_vec_infallible(comp_x);
        let (uy, _) = extract_vec_infallible(comp_y);
        let (uz, _) = extract_vec_infallible(comp_z);
        let ux: Vec<f64> = ux.iter().map(|&v| v as f64).collect();
        let uy: Vec<f64> = uy.iter().map(|&v| v as f64).collect();
        let uz: Vec<f64> = uz.iter().map(|&v| v as f64).collect();
        let [nz, ny, nx] = dims;
        let spacing = comp_x.spacing(); // [sz, sy, sx]
        let origin = comp_x.origin(); // [oz, oy, ox]
        let stride = [ny * nx, nx, 1usize];

        // Active axes (tensor-axis indices): a z==1 field is 2-D over (y, x).
        let axes: Vec<usize> = if nz == 1 { vec![1, 2] } else { vec![0, 1, 2] };
        let d = axes.len();
        let comps: [&[f64]; 3] = [&uz, &uy, &ux]; // indexed by tensor axis 0/1/2
        let sp = [spacing[0] as f64, spacing[1] as f64, spacing[2] as f64];
        let og = [origin[0] as f64, origin[1] as f64, origin[2] as f64];

        let f = self.subsampling_factor.max(1);
        // World position along active-axis position `k` of axis `axes[t]`.
        let world = |t: usize, idx: usize| -> f64 { og[axes[t]] + idx as f64 * sp[axes[t]] };

        // ── Build landmarks (source = p + d, target = p; Y = −d) ─────────────
        let counts: Vec<usize> = axes.iter().map(|&a| (dims[a] / f).max(1)).collect();
        let n_land: usize = counts.iter().product();
        if n_land == 0 {
            return (comp_x.clone(), comp_y.clone(), comp_z.clone());
        }
        let mut gstride = vec![1usize; d];
        for t in (0..d - 1).rev() {
            gstride[t] = gstride[t + 1] * counts[t + 1];
        }
        let mut src = vec![vec![0.0_f64; d]; n_land]; // world source points
        let mut ymat = vec![0.0_f64; d * (n_land + d + 1)]; // RHS (−d then zeros)
        for li in 0..n_land {
            // Decode landmark grid index → per-active-axis voxel index (×factor).
            let mut full = [0usize; 3];
            let mut rem = li;
            for t in 0..d {
                let gk = rem / gstride[t];
                rem %= gstride[t];
                full[axes[t]] = gk * f;
            }
            let flat = full[0] * stride[0] + full[1] * stride[1] + full[2] * stride[2];
            for t in 0..d {
                let a = axes[t];
                let p = world(t, full[a]);
                let disp = comps[a][flat];
                src[li][t] = p + disp;
                ymat[li * d + t] = -disp;
            }
        }

        // ── Assemble L = [[K, P], [Pᵀ, 0]] and solve L·W = Y ─────────────────
        let sz = d * (n_land + d + 1);
        let mut l = vec![vec![0.0_f64; sz]; sz];
        for i in 0..n_land {
            for j in 0..n_land {
                let mut r2 = 0.0;
                for t in 0..d {
                    let dd = src[i][t] - src[j][t];
                    r2 += dd * dd;
                }
                let g = r2.sqrt();
                for k in 0..d {
                    l[i * d + k][j * d + k] = g;
                }
            }
            // P block (rows i·d.., cols n_land·d..).
            let pcol = n_land * d;
            for j in 0..d {
                for k in 0..d {
                    l[i * d + k][pcol + j * d + k] = src[i][j];
                }
            }
            for k in 0..d {
                l[i * d + k][pcol + d * d + k] = 1.0;
            }
        }
        // Pᵀ block (lower-left).
        let pcol = n_land * d;
        for i in 0..n_land {
            for j in 0..d {
                for k in 0..d {
                    l[pcol + j * d + k][i * d + k] = src[i][j];
                }
            }
            for k in 0..d {
                l[pcol + d * d + k][i * d + k] = 1.0;
            }
        }
        let w = solve_linear(l, ymat);

        // Reorganise W → spline D (d×N), affine A (d×d), translation B (d).
        let dmat: Vec<Vec<f64>> = (0..d)
            .map(|k| (0..n_land).map(|i| w[i * d + k]).collect())
            .collect();
        let amat: Vec<Vec<f64>> = (0..d)
            .map(|i| (0..d).map(|j| w[n_land * d + j * d + i]).collect())
            .collect();
        let bvec: Vec<f64> = (0..d).map(|k| w[n_land * d + d * d + k]).collect();

        // ── Evaluate inverse displacement at every output voxel ──────────────
        let n = nz * ny * nx;
        let mut out: Vec<Vec<f64>> = vec![vec![0.0; n]; d];
        let mut q = vec![0.0_f64; d];
        for fi in 0..n {
            let iz = fi / stride[0];
            let iy = (fi % stride[0]) / stride[1];
            let ix = fi % stride[1];
            let full = [iz, iy, ix];
            for t in 0..d {
                q[t] = og[axes[t]] + full[axes[t]] as f64 * sp[axes[t]];
            }
            for t in 0..d {
                // Affine part A·q + B.
                let mut acc = bvec[t];
                for j in 0..d {
                    acc += amat[t][j] * q[j];
                }
                out[t][fi] = acc;
            }
            // Spline part Σ_i ‖q − s_i‖ · D[:, i].
            for i in 0..n_land {
                let mut r2 = 0.0;
                for t in 0..d {
                    let dd = q[t] - src[i][t];
                    r2 += dd * dd;
                }
                let g = r2.sqrt();
                if g != 0.0 {
                    for t in 0..d {
                        out[t][fi] += g * dmat[t][i];
                    }
                }
            }
        }

        // Scatter active-axis outputs back to (x, y, z) component buffers.
        let mut ox = vec![0.0_f32; n];
        let mut oy = vec![0.0_f32; n];
        let mut oz = vec![0.0_f32; n];
        for t in 0..d {
            let target: &mut [f32] = match axes[t] {
                0 => &mut oz,
                1 => &mut oy,
                _ => &mut ox,
            };
            for fi in 0..n {
                target[fi] = out[t][fi] as f32;
            }
        }
        (
            rebuild(ox, dims, comp_x),
            rebuild(oy, dims, comp_y),
            rebuild(oz, dims, comp_z),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use ritk_image::test_support as ts;

    type B = NdArray<f32>;

    /// The inverse of a constant translation field `(a, b)` is `(−a, −b)`
    /// everywhere (the TPS reduces to a pure affine translation). z=1 ⇒ 2-D.
    #[test]
    fn translation_inverse_is_negated() {
        let (h, w) = (16usize, 16usize);
        let n = h * w;
        let dx = ts::make_image::<B, 3>(vec![2.0; n], [1, h, w]);
        let dy = ts::make_image::<B, 3>(vec![3.0; n], [1, h, w]);
        let dz = ts::make_image::<B, 3>(vec![0.0; n], [1, h, w]);
        let (ix, iy, _iz) = InverseDisplacementField {
            subsampling_factor: 8,
        }
        .apply(&dx, &dy, &dz);
        let (rx, _) = extract_vec_infallible(&ix);
        let (ry, _) = extract_vec_infallible(&iy);
        for (&vx, &vy) in rx.iter().zip(ry.iter()) {
            assert!((vx - (-2.0)).abs() < 1e-4, "inv x = {vx}, want -2");
            assert!((vy - (-3.0)).abs() < 1e-4, "inv y = {vy}, want -3");
        }
    }
}
