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

use ritk_image::tensor::Backend;
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
/// pivoting. `a` is a **flat row-major** `n×n` matrix (`a[r*n + c]`),
/// consumed along with `b`. The TPS matrix is non-singular, so the solution
/// is unique.
///
/// Flat layout eliminates the `n` per-row heap allocations of a jagged
/// `Vec<Vec<f64>>` and improves cache locality for the row-scan operations
/// in both forward elimination and back-substitution.
fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        // Partial pivot — find the row ≥ col with the largest absolute value
        // in column col.
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if piv != col {
            // piv > col is guaranteed (search starts at col+1); swap the two
            // rows without a temporary Vec using split_at_mut.
            let (lo, hi) = a.split_at_mut(piv * n);
            lo[col * n..(col + 1) * n].swap_with_slice(&mut hi[..n]);
            b.swap(piv, col);
        }
        let diag = a[col * n + col];
        for r in (col + 1)..n {
            let f = a[r * n + col] / diag;
            if f != 0.0 {
                // Borrow row r (hi) and row col (lo) simultaneously via
                // split_at_mut, eliminating the range-loop pattern.
                let (lo, hi) = a.split_at_mut(r * n);
                for k in col..n {
                    hi[k] -= f * lo[col * n + k];
                }
                b[r] -= f * b[col];
            }
        }
    }
    // Back-substitution.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for (c, &xc) in x.iter().enumerate().skip(i + 1) {
            s -= a[i * n + c] * xc;
        }
        x[i] = s / a[i * n + i];
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
                                                  // spacing and origin index as f64; no cast needed.
        let sp = [spacing[0], spacing[1], spacing[2]];
        let og = [origin[0], origin[1], origin[2]];

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
        // Flat row-major layout: src[li * d + t] = world source coordinate of
        // landmark li along active axis t. Eliminates n_land per-landmark heap
        // allocations and gives contiguous access in the O(n_land²) K-block loop
        // and the O(n_voxels × n_land) evaluation loop.
        let mut src = vec![0.0_f64; n_land * d];
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
                src[li * d + t] = p + disp;
                ymat[li * d + t] = -disp;
            }
        }

        // ── Assemble L = [[K, P], [Pᵀ, 0]] and solve L·W = Y ─────────────────
        // Flat row-major layout: l[r * sz + c]. Eliminates sz per-row heap
        // allocations and gives contiguous row access for forward elimination.
        let sz = d * (n_land + d + 1);
        let pcol = n_land * d; // column offset for the P and Pᵀ blocks (constant)
        let mut l = vec![0.0_f64; sz * sz];
        for i in 0..n_land {
            for j in 0..n_land {
                let r2: f64 = src[i * d..(i + 1) * d]
                    .iter()
                    .zip(src[j * d..(j + 1) * d].iter())
                    .map(|(a, b)| {
                        let dd = a - b;
                        dd * dd
                    })
                    .sum();
                let g = r2.sqrt();
                for k in 0..d {
                    l[(i * d + k) * sz + (j * d + k)] = g;
                }
            }
            // P block (rows i·d.., cols n_land·d..).
            for j in 0..d {
                for k in 0..d {
                    l[(i * d + k) * sz + pcol + j * d + k] = src[i * d + j];
                }
            }
            for k in 0..d {
                l[(i * d + k) * sz + pcol + d * d + k] = 1.0;
            }
        }
        // Pᵀ block (lower-left).
        for i in 0..n_land {
            for j in 0..d {
                for k in 0..d {
                    l[(pcol + j * d + k) * sz + i * d + k] = src[i * d + j];
                }
            }
            for k in 0..d {
                l[(pcol + d * d + k) * sz + i * d + k] = 1.0;
            }
        }
        let w = solve_linear(l, ymat);

        // Reorganise W → spline D (d×N), affine A (d×d), translation B (d).
        // Flat row-major coefficient blocks keep the read-heavy Moirai
        // evaluation path contiguous and avoid d + d per-row heap allocations.
        let mut dmat = Vec::with_capacity(d * n_land);
        for k in 0..d {
            for i in 0..n_land {
                dmat.push(w[i * d + k]);
            }
        }
        let mut amat = Vec::with_capacity(d * d);
        for i in 0..d {
            for j in 0..d {
                amat.push(w[n_land * d + j * d + i]);
            }
        }
        let bvec: Vec<f64> = (0..d).map(|k| w[n_land * d + d * d + k]).collect();

        // ── Evaluate inverse displacement at every output voxel ──────────────
        // The per-voxel evaluation (affine part + spline sum) is embarrassingly
        // parallel over fi: each voxel reads shared immutable flat data (src,
        // dmat, amat, bvec) and writes to its own slot. Parallelised via moirai.
        //
        // Output layout: Vec<[f64; 3]> indexed [fi][t] where t in 0..d. Using
        // a stack-allocated [f64; 3] per voxel avoids any per-voxel heap
        // allocation inside the parallel closure (d is 2 or 3 at runtime).
        let n = nz * ny * nx;
        let voxel_out: Vec<[f64; 3]> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |fi| {
                let iz = fi / stride[0];
                let iy = (fi % stride[0]) / stride[1];
                let ix = fi % stride[1];
                let full = [iz, iy, ix];
                let mut q = [0.0_f64; 3];
                for t in 0..d {
                    q[t] = og[axes[t]] + full[axes[t]] as f64 * sp[axes[t]];
                }
                // Affine part A·q + B.
                let mut res = [0.0_f64; 3];
                for t in 0..d {
                    let mut acc = bvec[t];
                    for j in 0..d {
                        acc += amat[t * d + j] * q[j];
                    }
                    res[t] = acc;
                }
                // Spline part Σ_i ‖q − s_i‖ · D[:, i].
                for i in 0..n_land {
                    let r2: f64 = (0..d)
                        .map(|t| {
                            let dd = q[t] - src[i * d + t];
                            dd * dd
                        })
                        .sum();
                    let g = r2.sqrt();
                    if g != 0.0 {
                        for t in 0..d {
                            res[t] += g * dmat[t * n_land + i];
                        }
                    }
                }
                res
            });

        // Scatter active-axis outputs back to (x, y, z) component buffers.
        let mut ox = vec![0.0_f32; n];
        let mut oy = vec![0.0_f32; n];
        let mut oz = vec![0.0_f32; n];
        for (fi, res) in voxel_out.iter().enumerate() {
            for t in 0..d {
                let target = match axes[t] {
                    0 => &mut oz[fi],
                    1 => &mut oy[fi],
                    _ => &mut ox[fi],
                };
                *target = res[t] as f32;
            }
        }
        (
            rebuild(ox, dims, comp_x),
            rebuild(oy, dims, comp_y),
            rebuild(oz, dims, comp_z),
        )
    }
    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B>(
        &self,
        comp_x: &ritk_image::native::Image<f32, B, 3>,
        comp_y: &ritk_image::native::Image<f32, B, 3>,
        comp_z: &ritk_image::native::Image<f32, B, 3>,
        backend: &B::default()) -> anyhow::Result<(
        ritk_image::native::Image<f32, B, 3>,
        ritk_image::native::Image<f32, B, 3>,
        ritk_image::native::Image<f32, B, 3>,
    )>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (ux, dims) = ritk_tensor_ops::native::extract_image_vec(comp_x)?;
        let (uy, _) = ritk_tensor_ops::native::extract_image_vec(comp_y)?;
        let (uz, _) = ritk_tensor_ops::native::extract_image_vec(comp_z)?;
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
                                                  // spacing and origin index as f64; no cast needed.
        let sp = [spacing[0], spacing[1], spacing[2]];
        let og = [origin[0], origin[1], origin[2]];

        let f = self.subsampling_factor.max(1);
        // World position along active-axis position `k` of axis `axes[t]`.
        let world = |t: usize, idx: usize| -> f64 { og[axes[t]] + idx as f64 * sp[axes[t]] };

        // ── Build landmarks (source = p + d, target = p; Y = −d) ─────────────
        let counts: Vec<usize> = axes.iter().map(|&a| (dims[a] / f).max(1)).collect();
        let n_land: usize = counts.iter().product();
        if n_land == 0 {
            return Ok((comp_x.clone(), comp_y.clone(), comp_z.clone()));
        }
        let mut gstride = vec![1usize; d];
        for t in (0..d - 1).rev() {
            gstride[t] = gstride[t + 1] * counts[t + 1];
        }
        // Flat row-major layout: src[li * d + t] = world source coordinate of
        // landmark li along active axis t. Eliminates n_land per-landmark heap
        // allocations and gives contiguous access in the O(n_land²) K-block loop
        // and the O(n_voxels × n_land) evaluation loop.
        let mut src = vec![0.0_f64; n_land * d];
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
                src[li * d + t] = p + disp;
                ymat[li * d + t] = -disp;
            }
        }

        // ── Assemble L = [[K, P], [Pᵀ, 0]] and solve L·W = Y ─────────────────
        // Flat row-major layout: l[r * sz + c]. Eliminates sz per-row heap
        // allocations and gives contiguous row access for forward elimination.
        let sz = d * (n_land + d + 1);
        let pcol = n_land * d; // column offset for the P and Pᵀ blocks (constant)
        let mut l = vec![0.0_f64; sz * sz];
        for i in 0..n_land {
            for j in 0..n_land {
                let r2: f64 = src[i * d..(i + 1) * d]
                    .iter()
                    .zip(src[j * d..(j + 1) * d].iter())
                    .map(|(a, b)| {
                        let dd = a - b;
                        dd * dd
                    })
                    .sum();
                let g = r2.sqrt();
                for k in 0..d {
                    l[(i * d + k) * sz + (j * d + k)] = g;
                }
            }
            // P block (rows i·d.., cols n_land·d..).
            for j in 0..d {
                for k in 0..d {
                    l[(i * d + k) * sz + pcol + j * d + k] = src[i * d + j];
                }
            }
            for k in 0..d {
                l[(i * d + k) * sz + pcol + d * d + k] = 1.0;
            }
        }
        // Pᵀ block (lower-left).
        for i in 0..n_land {
            for j in 0..d {
                for k in 0..d {
                    l[(pcol + j * d + k) * sz + i * d + k] = src[i * d + j];
                }
            }
            for k in 0..d {
                l[(pcol + d * d + k) * sz + i * d + k] = 1.0;
            }
        }
        let w = solve_linear(l, ymat);

        // Reorganise W → spline D (d×N), affine A (d×d), translation B (d).
        // Flat row-major coefficient blocks keep the read-heavy Moirai
        // evaluation path contiguous and avoid d + d per-row heap allocations.
        let mut dmat = Vec::with_capacity(d * n_land);
        for k in 0..d {
            for i in 0..n_land {
                dmat.push(w[i * d + k]);
            }
        }
        let mut amat = Vec::with_capacity(d * d);
        for i in 0..d {
            for j in 0..d {
                amat.push(w[n_land * d + j * d + i]);
            }
        }
        let bvec: Vec<f64> = (0..d).map(|k| w[n_land * d + d * d + k]).collect();

        // ── Evaluate inverse displacement at every output voxel ──────────────
        // The per-voxel evaluation (affine part + spline sum) is embarrassingly
        // parallel over fi: each voxel reads shared immutable flat data (src,
        // dmat, amat, bvec) and writes to its own slot. Parallelised via moirai.
        //
        // Output layout: Vec<[f64; 3]> indexed [fi][t] where t in 0..d. Using
        // a stack-allocated [f64; 3] per voxel avoids any per-voxel heap
        // allocation inside the parallel closure (d is 2 or 3 at runtime).
        let n = nz * ny * nx;
        let voxel_out: Vec<[f64; 3]> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |fi| {
                let iz = fi / stride[0];
                let iy = (fi % stride[0]) / stride[1];
                let ix = fi % stride[1];
                let full = [iz, iy, ix];
                let mut q = [0.0_f64; 3];
                for t in 0..d {
                    q[t] = og[axes[t]] + full[axes[t]] as f64 * sp[axes[t]];
                }
                // Affine part A·q + B.
                let mut res = [0.0_f64; 3];
                for t in 0..d {
                    let mut acc = bvec[t];
                    for j in 0..d {
                        acc += amat[t * d + j] * q[j];
                    }
                    res[t] = acc;
                }
                // Spline part Σ_i ‖q − s_i‖ · D[:, i].
                for i in 0..n_land {
                    let r2: f64 = (0..d)
                        .map(|t| {
                            let dd = q[t] - src[i * d + t];
                            dd * dd
                        })
                        .sum();
                    let g = r2.sqrt();
                    if g != 0.0 {
                        for t in 0..d {
                            res[t] += g * dmat[t * n_land + i];
                        }
                    }
                }
                res
            });

        // Scatter active-axis outputs back to (x, y, z) component buffers.
        let mut ox = vec![0.0_f32; n];
        let mut oy = vec![0.0_f32; n];
        let mut oz = vec![0.0_f32; n];
        for (fi, res) in voxel_out.iter().enumerate() {
            for t in 0..d {
                let target = match axes[t] {
                    0 => &mut oz[fi],
                    1 => &mut oy[fi],
                    _ => &mut ox[fi],
                };
                *target = res[t] as f32;
            }
        }
        Ok((
            crate::native_support::rebuild_image(ox, dims, comp_x, backend)?,
            crate::native_support::rebuild_image(oy, dims, comp_y, backend)?,
            crate::native_support::rebuild_image(oz, dims, comp_z, backend)?,
        ))
    
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
        let dx = ts::burn_compat::make_image::<B, 3>(vec![2.0; n], [1, h, w]);
        let dy = ts::burn_compat::make_image::<B, 3>(vec![3.0; n], [1, h, w]);
        let dz = ts::burn_compat::make_image::<B, 3>(vec![0.0; n], [1, h, w]);
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
