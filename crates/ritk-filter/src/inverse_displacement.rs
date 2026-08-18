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
//!    solve `L·W = Y` with `L = [[K, P], [Páµ€, 0]]`, `K_ij = —–s_i − s_j—–·I_d`,
//!    `P_i = [s_i[0]·I_d, …, s_i[d−1]·I_d, I_d]`, `Y = [−d_i; 0]`. Reorganise
//!    `W` into the spline matrix `D` (d×N), affine `A` (d×d), and translation
//!    `B` (d): `D[k][i] = W[i·d+k]`, `A[i][j] = W[N·d + j·d + i]` (note the
//!    transpose), `B[k] = W[N·d + d·d + k]`.
//! 4. **Evaluate** per output voxel `q` (world): the inverse displacement is
//!    `A·q + B + Σ_i —–q − s_i—–·D[:,i]` (`= TransformPoint(q) − q`).
//!
//! The TPS system is unique and well-conditioned, so the result is float-exact
//! to `sitk.InverseDisplacementField` (independent of the linear solver).
//! Internal arithmetic is `f64`.
//!
//! # Geometry
//!
//! Landmark and evaluation points are the grid's true physical positions,
//! `origin + D S index`, so the direction cosines participate in the fit and an
//! oblique acquisition is inverted about its own axes. The thin-plate-spline
//! kernel `G(r) = r` depends only on Euclidean distance, so for an orthonormal
//! direction the fit is exactly the rotation of the axis-aligned fit.
//!
//! A `z == 1` field is inverted as a genuine 2-D field, matching sitk's 2-D
//! filter. The two solved coordinates are then not the `y` and `x` world axes
//! but an orthonormal basis of the *slab's own plane*, obtained by
//! Gram-Schmidt from the direction columns of the two in-plane index axes; the
//! fitted displacement is mapped back to world components through the same
//! basis. Displacement normal to the slab is outside a 2-D field's
//! representation and is dropped, as it was before.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

use ritk_spatial::CartesianGridGeometry;

/// Euclidean inner product of two physical 3-vectors.
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Orthonormal basis of the subspace the active index axes span in physical
/// space, by Gram-Schmidt over their direction columns.
///
/// For the full 3-D case this returns the world axes themselves, so the solve
/// runs directly in world coordinates; the Gram-Schmidt path exists for the
/// `z == 1` slab, whose plane is arbitrarily oriented under an oblique
/// direction.
fn active_basis(
    geometry: &CartesianGridGeometry<3>,
    axes: &[usize],
) -> anyhow::Result<Vec<[f64; 3]>> {
    if axes.len() == 3 {
        return Ok(vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }
    let mut basis: Vec<[f64; 3]> = Vec::with_capacity(axes.len());
    for &axis in axes {
        let mut candidate = geometry.axis_direction(axis);
        for existing in &basis {
            let projection = dot(candidate, *existing);
            for k in 0..3 {
                candidate[k] -= projection * existing[k];
            }
        }
        let norm = dot(candidate, candidate).sqrt();
        anyhow::ensure!(
            norm > 1e-12,
            "image direction is degenerate over the active axes {axes:?}"
        );
        basis.push([
            candidate[0] / norm,
            candidate[1] / norm,
            candidate[2] / norm,
        ]);
    }
    Ok(basis)
}

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
    /// Solve the inversion on host buffers, independent of how they were
    /// extracted and how the result is rebuilt.
    ///
    /// `Ok(None)` reports the degenerate landmark-free case, where the inverse
    /// is the input field; the caller owns that identity because the two entry
    /// points return different image types.
    ///
    /// # Errors
    ///
    /// Returns an error when the direction is degenerate over the active axes
    /// of a `z == 1` slab.
    fn invert_components(
        &self,
        ux: &[f64],
        uy: &[f64],
        uz: &[f64],
        dims: [usize; 3],
        geometry: &CartesianGridGeometry<3>,
    ) -> anyhow::Result<Option<[Vec<f32>; 3]>> {
        let [nz, ny, nx] = dims;
        let stride = [ny * nx, nx, 1usize];

        // Active axes (tensor-axis indices): a z==1 field is 2-D over (y, x).
        let axes: Vec<usize> = if nz == 1 { vec![1, 2] } else { vec![0, 1, 2] };
        let d = axes.len();
        // Physical basis the solve runs in: the world axes for a 3-D field, the
        // slab's own plane for a z == 1 field.
        let basis = active_basis(geometry, &axes)?;

        let f = self.subsampling_factor.max(1);

        // ── Build landmarks (source = p + d, target = p; Y = −d) ─────────────
        let counts: Vec<usize> = axes.iter().map(|&a| (dims[a] / f).max(1)).collect();
        let n_land: usize = counts.iter().product();
        if n_land == 0 {
            return Ok(None);
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
            // Landmark source is the displaced physical point; both the point
            // and the displacement are resolved in the solve basis.
            let point = geometry.point([full[0] as f64, full[1] as f64, full[2] as f64]);
            let displacement = [ux[flat], uy[flat], uz[flat]];
            for t in 0..d {
                let component = dot(displacement, basis[t]);
                src[li * d + t] = dot(point, basis[t]) + component;
                ymat[li * d + t] = -component;
            }
        }

        // ── Assemble L = [[K, P], [Páµ€, 0]] and solve L·W = Y ─────────────────
        // Flat row-major layout: l[r * sz + c]. Eliminates sz per-row heap
        // allocations and gives contiguous row access for forward elimination.
        let sz = d * (n_land + d + 1);
        let pcol = n_land * d; // column offset for the P and Páµ€ blocks (constant)
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
        // Páµ€ block (lower-left).
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
                let point = geometry.point([iz as f64, iy as f64, ix as f64]);
                let mut q = [0.0_f64; 3];
                for t in 0..d {
                    q[t] = dot(point, basis[t]);
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
                // Spline part Σ_i —–q − s_i—– · D[:, i].
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

        // Recombine the solved basis coordinates into world (x, y, z)
        // components. For a 3-D field the basis is the world axes and this is
        // the identity; for a z == 1 slab it maps the in-plane result back out.
        let mut ox = vec![0.0_f32; n];
        let mut oy = vec![0.0_f32; n];
        let mut oz = vec![0.0_f32; n];
        for (fi, res) in voxel_out.iter().enumerate() {
            let mut world_vector = [0.0_f64; 3];
            for t in 0..d {
                for k in 0..3 {
                    world_vector[k] += res[t] * basis[t][k];
                }
            }
            ox[fi] = world_vector[0] as f32;
            oy[fi] = world_vector[1] as f32;
            oz[fi] = world_vector[2] as f32;
        }
        Ok(Some([ox, oy, oz]))
    }

    /// Invert the field whose world-frame components are `comp_x`, `comp_y`,
    /// `comp_z` (each a scalar `[z, y, x]` image on a shared grid). Returns the
    /// inverse components `(inv_x, inv_y, inv_z)` on the same grid.
    ///
    /// # Errors
    ///
    /// Returns an error when `comp_x`'s coordinate map is not Cartesian, when
    /// its direction matrix is singular, or when the direction is degenerate
    /// over the active axes of a `z == 1` slab.
    pub fn apply<B: Backend>(
        &self,
        comp_x: &Image<f32, B, 3>,
        comp_y: &Image<f32, B, 3>,
        comp_z: &Image<f32, B, 3>,
    ) -> anyhow::Result<crate::DisplacementComponents<B>> {
        let (ux, dims) = extract_vec_infallible(comp_x);
        let (uy, _) = extract_vec_infallible(comp_y);
        let (uz, _) = extract_vec_infallible(comp_z);
        let ux: Vec<f64> = ux.iter().map(|&v| v as f64).collect();
        let uy: Vec<f64> = uy.iter().map(|&v| v as f64).collect();
        let uz: Vec<f64> = uz.iter().map(|&v| v as f64).collect();
        let geometry = comp_x.grid_geometry()?;
        let Some([ox, oy, oz]) = self.invert_components(&ux, &uy, &uz, dims, &geometry)? else {
            return Ok((comp_x.clone(), comp_y.clone(), comp_z.clone()));
        };
        Ok((
            rebuild(ox, dims, comp_x),
            rebuild(oy, dims, comp_y),
            rebuild(oz, dims, comp_z),
        ))
    }

    /// Coeus-native counterpart to the legacy application method.
    ///
    /// # Errors
    ///
    /// Returns an error when `comp_x`'s coordinate map is not Cartesian, when
    /// its direction matrix is singular, when the direction is degenerate over
    /// the active axes of a `z == 1` slab, or when a device buffer cannot be
    /// read back or rebuilt.
    pub fn apply_native<B>(
        &self,
        comp_x: &ritk_image::Image<f32, B, 3>,
        comp_y: &ritk_image::Image<f32, B, 3>,
        comp_z: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<crate::NativeDisplacementField<B>>
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
        let geometry = comp_x.grid_geometry()?;
        let Some([ox, oy, oz]) = self.invert_components(&ux, &uy, &uz, dims, &geometry)? else {
            return Ok(crate::NativeDisplacementField {
                x: comp_x.clone(),
                y: comp_y.clone(),
                z: comp_z.clone(),
            });
        };
        Ok(crate::NativeDisplacementField {
            x: crate::native_support::rebuild_image(ox, dims, comp_x, backend)?,
            y: crate::native_support::rebuild_image(oy, dims, comp_y, backend)?,
            z: crate::native_support::rebuild_image(oz, dims, comp_z, backend)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ritk_image::test_support as ts;

    type B = coeus_core::SequentialBackend;

    /// Direction of a conventional axial acquisition: index axis 0 (depth) runs
    /// along world z, axis 1 (row) along y, axis 2 (column) along x.
    ///
    /// The fixtures below previously used the default identity direction, which
    /// in this crate's convention sends the *depth* axis to world x. The solve
    /// nonetheless produced axial answers because it hard-coded this
    /// permutation instead of reading the matrix; now that the direction is
    /// honoured, the fixture has to state the geometry it always meant.
    fn axial() -> ritk_spatial::Direction<3> {
        ritk_spatial::Direction::from_rows([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    }

    fn axial_image(value: f32, dims: [usize; 3]) -> Image<f32, B, 3> {
        let n: usize = dims.iter().product();
        ts::make_image_with::<f32, B, 3>(vec![value; n], dims, None, None, Some(axial()))
    }

    /// The inverse of a constant translation field `(a, b)` is `(−a, −b)`
    /// everywhere (the TPS reduces to a pure affine translation). z=1 ⇒ 2-D.
    #[test]
    fn translation_inverse_is_negated() {
        let (h, w) = (16usize, 16usize);
        let dx = axial_image(2.0, [1, h, w]);
        let dy = axial_image(3.0, [1, h, w]);
        let dz = axial_image(0.0, [1, h, w]);
        let (ix, iy, _iz) = InverseDisplacementField {
            subsampling_factor: 8,
        }
        .apply(&dx, &dy, &dz)
        .expect("invariant: fixture is Cartesian with an invertible direction");
        let (rx, _) = extract_vec_infallible(&ix);
        let (ry, _) = extract_vec_infallible(&iy);
        for (&vx, &vy) in rx.iter().zip(ry.iter()) {
            assert!((vx - (-2.0)).abs() < 1e-4, "inv x = {vx}, want -2");
            assert!((vy - (-3.0)).abs() < 1e-4, "inv y = {vy}, want -3");
        }
    }

    /// Regression for ATLAS-RITK-TRANSFORM-DIRECTION-081.
    ///
    /// Landmark and evaluation points were built from origin and spacing alone,
    /// so the thin-plate spline was fitted in the index frame no matter how the
    /// volume was oriented.
    ///
    /// The oracle is rigid-motion equivariance, independent of the solver: the
    /// TPS kernel `G(r) = r` depends only on Euclidean distance, so rotating a
    /// fit is the fit of the rotated problem. Inverting on a grid with
    /// direction `R·A` carrying components `R·u` must give exactly `R·v`.
    ///
    /// A direction-blind fit returns the same `v` for both grids, which equals
    /// `R·v` only for `R = I`.
    #[test]
    fn tps_inverse_is_equivariant_under_grid_rotation() {
        let dims = [3usize, 3usize, 3usize];
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;

        // A spatially varying field; a constant translation inverts to its own
        // negation in every frame and so cannot detect a dropped direction.
        let mut ux = Vec::with_capacity(n);
        let mut uy = Vec::with_capacity(n);
        let mut uz = Vec::with_capacity(n);
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let (fz, fy, fx) = (iz as f32, iy as f32, ix as f32);
                    ux.push(0.05 * fx - 0.02 * fy);
                    uy.push(0.03 * fy + 0.01 * fz);
                    uz.push(0.02 * fz - 0.01 * fx);
                }
            }
        }

        let rotation = ritk_spatial::Direction::from_rows([
            [0.6, -0.8, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let filter = InverseDisplacementField {
            subsampling_factor: 1,
        };
        let build = |data: Vec<f32>, direction: ritk_spatial::Direction<3>| {
            ts::make_image_with::<f32, B, 3>(data, dims, None, None, Some(direction))
        };

        let reference = filter
            .apply(
                &build(ux.clone(), axial()),
                &build(uy.clone(), axial()),
                &build(uz.clone(), axial()),
            )
            .expect("invariant: axial fixture is Cartesian and invertible");

        let rotate = |row: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    (rotation[(row, 0)] * f64::from(ux[i])
                        + rotation[(row, 1)] * f64::from(uy[i])
                        + rotation[(row, 2)] * f64::from(uz[i])) as f32
                })
                .collect()
        };
        let rotated_direction = rotation * axial();
        let rotated = filter
            .apply(
                &build(rotate(0), rotated_direction),
                &build(rotate(1), rotated_direction),
                &build(rotate(2), rotated_direction),
            )
            .expect("invariant: rotated fixture is Cartesian and invertible");

        let (rx, _) = extract_vec_infallible(&reference.0);
        let (ry, _) = extract_vec_infallible(&reference.1);
        let (rz, _) = extract_vec_infallible(&reference.2);
        let (gx, _) = extract_vec_infallible(&rotated.0);
        let (gy, _) = extract_vec_infallible(&rotated.1);
        let (gz, _) = extract_vec_infallible(&rotated.2);

        let mut largest = 0.0_f64;
        for i in 0..n {
            let reference_vector = [f64::from(rx[i]), f64::from(ry[i]), f64::from(rz[i])];
            largest = largest.max(
                reference_vector
                    .iter()
                    .fold(0.0_f64, |acc, v| acc.max(v.abs())),
            );
            let got = [f64::from(gx[i]), f64::from(gy[i]), f64::from(gz[i])];
            for row in 0..3 {
                let want: f64 = (0..3)
                    .map(|column| rotation[(row, column)] * reference_vector[column])
                    .sum();
                assert!(
                    (got[row] - want).abs() < 1e-5,
                    "voxel {i} row {row}: rotated run gave {}, rotation of the \
                     reference is {want}",
                    got[row]
                );
            }
        }
        assert!(
            largest > 0.02,
            "fixture must produce a non-trivial inverse field, largest \
             component was {largest}"
        );
    }
}
