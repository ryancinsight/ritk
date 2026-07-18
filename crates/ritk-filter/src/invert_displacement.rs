//! Iterative inversion of a dense displacement field
//! (`itk::InvertDisplacementFieldImageFilter` / `sitk.InvertDisplacementField`).
//!
//! # Mathematical Specification
//!
//! Given a forward displacement field `u` (the transform `x â†¦ x + u(x)` in world
//! coordinates), find the inverse field `v` such that `v(x) + u(x + v(x)) â‰ˆ 0`.
//! ITK uses the fixed-point scheme of Chen et al. (2008):
//!
//! ```text
//! c(x)  = v(x) + u(x + v(x))                 (composed residual; u interpolated)
//! s(x)  = â€–c(x) âŠ˜ spacingâ€–                   (spacing-scaled norm)
//! Îµ     = 0.75 on iteration 1, else 0.5
//! r(x)  = âˆ’c(x), clamped so â€–r âŠ˜ spacingâ€– â‰¤ ÎµÂ·max_x s(x)
//! v(x) â† v(x) + ÎµÂ·r(x)                        (0 on the boundary if enforced)
//! ```
//!
//! iterating until `max_x s(x) â‰¤ max_tol` and `mean_x s(x) â‰¤ mean_tol`, or
//! `max_iter` is reached. `u` is sampled with **vector linear interpolation**
//! matching ITK: a point whose continuous index lies outside `[âˆ’0.5, sizeâˆ’0.5]`
//! in any axis contributes the zero vector; otherwise the eight neighbours are
//! clamped to the buffer. Internal arithmetic is `f64` (ITK's `RealType` for a
//! float field), so the result is float-exact to `sitk.InvertDisplacementField`.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Vector linear interpolation of one displacement component at continuous index
/// `(cz, cy, cx)`, matching ITK's `VectorLinearInterpolateImageFunction` +
/// `IsInsideBuffer`: a point outside `[âˆ’0.5, sizeâˆ’0.5]` in any axis contributes
/// the zero vector; otherwise the eight neighbours are clamped to the buffer.
/// Shared by the displacement-field inversion filters.
pub(crate) fn interp_component(ud: &[f64], dims: [usize; 3], cz: f64, cy: f64, cx: f64) -> f64 {
    let [nz, ny, nx] = dims;
    if cx < -0.5
        || cx > nx as f64 - 0.5
        || cy < -0.5
        || cy > ny as f64 - 0.5
        || cz < -0.5
        || cz > nz as f64 - 0.5
    {
        return 0.0;
    }
    let (x0, y0, z0) = (cx.floor(), cy.floor(), cz.floor());
    let (fx, fy, fz) = (cx - x0, cy - y0, cz - z0);
    let clamp = |v: f64, hi: usize| -> usize { (v.max(0.0) as usize).min(hi) };
    let mut acc = 0.0;
    for dz_ in 0..2 {
        let wz = if dz_ == 1 { fz } else { 1.0 - fz };
        if wz == 0.0 {
            continue;
        }
        let zz = clamp(z0 + dz_ as f64, nz - 1);
        for dy_ in 0..2 {
            let wy = if dy_ == 1 { fy } else { 1.0 - fy };
            if wy == 0.0 {
                continue;
            }
            let yy = clamp(y0 + dy_ as f64, ny - 1);
            for dx_ in 0..2 {
                let wx = if dx_ == 1 { fx } else { 1.0 - fx };
                if wx == 0.0 {
                    continue;
                }
                let xx = clamp(x0 + dx_ as f64, nx - 1);
                acc += wz * wy * wx * ud[(zz * ny + yy) * nx + xx];
            }
        }
    }
    acc
}

/// Parameters and entry point for displacement-field inversion.
#[derive(Debug, Clone)]
pub struct InvertDisplacementField {
    /// Maximum fixed-point iterations (ITK/sitk default 10).
    pub max_iterations: usize,
    /// Stop when the max spacing-scaled residual norm falls below this (default 0.1).
    pub max_error_tolerance: f64,
    /// Stop when the mean spacing-scaled residual norm falls below this (default 0.001).
    pub mean_error_tolerance: f64,
    /// Pin the inverse field to zero on the image boundary (ITK default `true`).
    pub enforce_boundary: bool,
}

impl Default for InvertDisplacementField {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            max_error_tolerance: 0.1,
            mean_error_tolerance: 0.001,
            enforce_boundary: true,
        }
    }
}

impl InvertDisplacementField {
    /// Invert the field given as world components `(dx, dy, dz)` (each a `[z,y,x]`
    /// scalar image). Returns the inverted components `(dx, dy, dz)`.
    pub fn apply<B: Backend>(
        &self,
        dx: &Image<f32, B, 3>,
        dy: &Image<f32, B, 3>,
        dz: &Image<f32, B, 3>,
    ) -> (Image<f32, B, 3>, Image<f32, B, 3>, Image<f32, B, 3>) {
        let (ux, dims) = extract_vec_infallible(dx);
        let (uy, _) = extract_vec_infallible(dy);
        let (uz, _) = extract_vec_infallible(dz);
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        // spacing is tensor-major [sz, sy, sx]; world x â†” axis 2.
        let sp = dx.spacing();
        let (sx, sy, sz) = (sp[2], sp[1], sp[0]);
        let (inv_sx, inv_sy, inv_sz) = (1.0 / sx, 1.0 / sy, 1.0 / sz);

        let uxd: Vec<f64> = ux.iter().map(|&v| v as f64).collect();
        let uyd: Vec<f64> = uy.iter().map(|&v| v as f64).collect();
        let uzd: Vec<f64> = uz.iter().map(|&v| v as f64).collect();

        let mut vx = vec![0.0f64; n];
        let mut vy = vec![0.0f64; n];
        let mut vz = vec![0.0f64; n];

        let interp = |cz: f64, cy: f64, cx: f64, ud: &[f64]| interp_component(ud, dims, cz, cy, cx);

        let mut max_err = f64::MAX;
        let mut mean_err = f64::MAX;
        let mut cx_buf = vec![0.0f64; n];
        let mut cy_buf = vec![0.0f64; n];
        let mut cz_buf = vec![0.0f64; n];
        let mut scaled = vec![0.0f64; n];

        let mut iteration = 0;
        while iteration < self.max_iterations
            && max_err > self.max_error_tolerance
            && mean_err > self.mean_error_tolerance
        {
            iteration += 1;
            // Compose c(x) = v(x) + u(x + v(x)); scaled norm; negate.
            let mut sum = 0.0f64;
            let mut mx = 0.0f64;
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let i = (z * ny + y) * nx + x;
                        let ci_x = x as f64 + vx[i] * inv_sx;
                        let ci_y = y as f64 + vy[i] * inv_sy;
                        let ci_z = z as f64 + vz[i] * inv_sz;
                        let cdx = vx[i] + interp(ci_z, ci_y, ci_x, &uxd);
                        let cdy = vy[i] + interp(ci_z, ci_y, ci_x, &uyd);
                        let cdz = vz[i] + interp(ci_z, ci_y, ci_x, &uzd);
                        let sn = ((cdx * inv_sx).powi(2)
                            + (cdy * inv_sy).powi(2)
                            + (cdz * inv_sz).powi(2))
                        .sqrt();
                        scaled[i] = sn;
                        sum += sn;
                        if sn > mx {
                            mx = sn;
                        }
                        cx_buf[i] = -cdx;
                        cy_buf[i] = -cdy;
                        cz_buf[i] = -cdz;
                    }
                }
            }
            mean_err = sum / n as f64;
            max_err = mx;
            let eps = if iteration == 1 { 0.75 } else { 0.5 };
            let limit = eps * max_err;
            // Estimate inverse update.
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let i = (z * ny + y) * nx + x;
                        let on_border =
                            z == 0 || z == nz - 1 || y == 0 || y == ny - 1 || x == 0 || x == nx - 1;
                        if self.enforce_boundary && on_border {
                            vx[i] = 0.0;
                            vy[i] = 0.0;
                            vz[i] = 0.0;
                            continue;
                        }
                        let sn = scaled[i];
                        let scale = if sn > limit { limit / sn } else { 1.0 };
                        vx[i] += eps * cx_buf[i] * scale;
                        vy[i] += eps * cy_buf[i] * scale;
                        vz[i] += eps * cz_buf[i] * scale;
                    }
                }
            }
        }

        let rx: Vec<f32> = vx.iter().map(|&v| v as f32).collect();
        let ry: Vec<f32> = vy.iter().map(|&v| v as f32).collect();
        let rz: Vec<f32> = vz.iter().map(|&v| v as f32).collect();
        (
            rebuild(rx, dims, dx),
            rebuild(ry, dims, dy),
            rebuild(rz, dims, dz),
        )
    }
    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B>(
        &self,
        dx: &ritk_image::native::Image<f32, B, 3>,
        dy: &ritk_image::native::Image<f32, B, 3>,
        dz: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<crate::NativeDisplacementField<B>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (ux, dims) = ritk_tensor_ops::native::extract_image_vec(dx)?;
        let (uy, _) = ritk_tensor_ops::native::extract_image_vec(dy)?;
        let (uz, _) = ritk_tensor_ops::native::extract_image_vec(dz)?;
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        // spacing is tensor-major [sz, sy, sx]; world x â†” axis 2.
        let sp = dx.spacing();
        let (sx, sy, sz) = (sp[2], sp[1], sp[0]);
        let (inv_sx, inv_sy, inv_sz) = (1.0 / sx, 1.0 / sy, 1.0 / sz);

        let uxd: Vec<f64> = ux.iter().map(|&v| v as f64).collect();
        let uyd: Vec<f64> = uy.iter().map(|&v| v as f64).collect();
        let uzd: Vec<f64> = uz.iter().map(|&v| v as f64).collect();

        let mut vx = vec![0.0f64; n];
        let mut vy = vec![0.0f64; n];
        let mut vz = vec![0.0f64; n];

        let interp = |cz: f64, cy: f64, cx: f64, ud: &[f64]| interp_component(ud, dims, cz, cy, cx);

        let mut max_err = f64::MAX;
        let mut mean_err = f64::MAX;
        let mut cx_buf = vec![0.0f64; n];
        let mut cy_buf = vec![0.0f64; n];
        let mut cz_buf = vec![0.0f64; n];
        let mut scaled = vec![0.0f64; n];

        let mut iteration = 0;
        while iteration < self.max_iterations
            && max_err > self.max_error_tolerance
            && mean_err > self.mean_error_tolerance
        {
            iteration += 1;
            // Compose c(x) = v(x) + u(x + v(x)); scaled norm; negate.
            let mut sum = 0.0f64;
            let mut mx = 0.0f64;
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let i = (z * ny + y) * nx + x;
                        let ci_x = x as f64 + vx[i] * inv_sx;
                        let ci_y = y as f64 + vy[i] * inv_sy;
                        let ci_z = z as f64 + vz[i] * inv_sz;
                        let cdx = vx[i] + interp(ci_z, ci_y, ci_x, &uxd);
                        let cdy = vy[i] + interp(ci_z, ci_y, ci_x, &uyd);
                        let cdz = vz[i] + interp(ci_z, ci_y, ci_x, &uzd);
                        let sn = ((cdx * inv_sx).powi(2)
                            + (cdy * inv_sy).powi(2)
                            + (cdz * inv_sz).powi(2))
                        .sqrt();
                        scaled[i] = sn;
                        sum += sn;
                        if sn > mx {
                            mx = sn;
                        }
                        cx_buf[i] = -cdx;
                        cy_buf[i] = -cdy;
                        cz_buf[i] = -cdz;
                    }
                }
            }
            mean_err = sum / n as f64;
            max_err = mx;
            let eps = if iteration == 1 { 0.75 } else { 0.5 };
            let limit = eps * max_err;
            // Estimate inverse update.
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let i = (z * ny + y) * nx + x;
                        let on_border =
                            z == 0 || z == nz - 1 || y == 0 || y == ny - 1 || x == 0 || x == nx - 1;
                        if self.enforce_boundary && on_border {
                            vx[i] = 0.0;
                            vy[i] = 0.0;
                            vz[i] = 0.0;
                            continue;
                        }
                        let sn = scaled[i];
                        let scale = if sn > limit { limit / sn } else { 1.0 };
                        vx[i] += eps * cx_buf[i] * scale;
                        vy[i] += eps * cy_buf[i] * scale;
                        vz[i] += eps * cz_buf[i] * scale;
                    }
                }
            }
        }

        let rx: Vec<f32> = vx.iter().map(|&v| v as f32).collect();
        let ry: Vec<f32> = vy.iter().map(|&v| v as f32).collect();
        let rz: Vec<f32> = vz.iter().map(|&v| v as f32).collect();
        Ok(crate::NativeDisplacementField {
            x: crate::native_support::rebuild_image(rx, dims, dx, backend)?,
            y: crate::native_support::rebuild_image(ry, dims, dy, backend)?,
            z: crate::native_support::rebuild_image(rz, dims, dz, backend)?,
        })
    }
}

#[cfg(test)]
#[path = "tests_invert_displacement.rs"]
mod tests_invert_displacement;
