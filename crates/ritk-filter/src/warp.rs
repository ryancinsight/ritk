//! Warp filter: resample a moving image through a dense displacement field.
//!
//! # Mathematical Specification
//!
//! Matches `itk::WarpImageFilter`. For every voxel of the output grid (which is
//! the displacement-field grid) at physical point `p`, the output samples the
//! moving image at the *displaced* physical point:
//!
//! ```text
//! out(p) = moving( p + D(p) )
//! ```
//!
//! where `D(p)` is the (physical-space) displacement stored at that grid point.
//! The physical point of grid index `i` is `p = O_f + R_f · (S_f ⊙ i)` (field
//! origin / direction / spacing); the displaced point is mapped back into the
//! moving image's continuous index space by `c = S_m⁻¹ · R_m⁻¹ · (p + D − O_m)`
//! and sampled with trilinear interpolation. Samples whose continuous index
//! falls outside the moving buffer (`c_a ∉ [−0.5, N_a − 0.5)` on any axis) take
//! the edge-padding value (0), reproducing ITK's `IsInsideBuffer` gate; taps
//! that reach one voxel past the border are edge-clamped, matching ITK's linear
//! interpolator.
//!
//! Displacement components are supplied as three scalar images `(D_z, D_y, D_x)`
//! on the field grid, mirroring [`crate`]'s Jacobian-determinant convention.

use anyhow::{anyhow, Result};
use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Trilinear sample of `vals` (shape `[nz, ny, nx]`, row-major) at the
/// axis-major continuous index `c = [cz, cy, cx]`, with taps edge-clamped to the
/// valid range. The caller is responsible for the `IsInsideBuffer` gate.
#[inline]
fn trilinear(vals: &[f32], dims: [usize; 3], c: [f64; 3]) -> f32 {
    let [nz, ny, nx] = dims;
    let base = [c[0].floor(), c[1].floor(), c[2].floor()];
    let frac = [c[0] - base[0], c[1] - base[1], c[2] - base[2]];
    let clamp = |v: f64, n: usize| (v as i64).clamp(0, n as i64 - 1) as usize;
    let z0 = clamp(base[0], nz);
    let y0 = clamp(base[1], ny);
    let x0 = clamp(base[2], nx);
    let z1 = clamp(base[0] + 1.0, nz);
    let y1 = clamp(base[1] + 1.0, ny);
    let x1 = clamp(base[2] + 1.0, nx);
    let (fz, fy, fx) = (frac[0] as f32, frac[1] as f32, frac[2] as f32);
    let at = |z: usize, y: usize, x: usize| vals[z * ny * nx + y * nx + x];
    // Interpolate x, then y, then z.
    let c00 = at(z0, y0, x0) * (1.0 - fx) + at(z0, y0, x1) * fx;
    let c01 = at(z0, y1, x0) * (1.0 - fx) + at(z0, y1, x1) * fx;
    let c10 = at(z1, y0, x0) * (1.0 - fx) + at(z1, y0, x1) * fx;
    let c11 = at(z1, y1, x0) * (1.0 - fx) + at(z1, y1, x1) * fx;
    let c0 = c00 * (1.0 - fy) + c01 * fy;
    let c1 = c10 * (1.0 - fy) + c11 * fy;
    c0 * (1.0 - fz) + c1 * fz
}

/// Warp a moving image through a dense displacement field.
///
/// `moving` is the image to sample; `disp_z`, `disp_y`, `disp_x` are the
/// physical-space displacement components defined on the output (field) grid.
/// All three field components must share the field's shape; the output adopts the
/// field's geometry. Returns an error if the moving image's direction matrix is
/// singular.
pub fn warp_image<B: Backend>(
    moving: &Image<B, 3>,
    disp_z: &Image<B, 3>,
    disp_y: &Image<B, 3>,
    disp_x: &Image<B, 3>,
) -> Result<Image<B, 3>> {
    let field_dims = disp_z.shape();
    if disp_y.shape() != field_dims || disp_x.shape() != field_dims {
        return Err(anyhow!(
            "warp: displacement components must share the field shape {field_dims:?}"
        ));
    }
    let [nz, ny, nx] = field_dims;
    let (mov_vals, mov_dims) = extract_vec_infallible(moving);
    let (dz, _) = extract_vec_infallible(disp_z);
    let (dy, _) = extract_vec_infallible(disp_y);
    let (dx, _) = extract_vec_infallible(disp_x);

    // Field geometry (axis-major [z, y, x]).
    let fo = disp_z.origin();
    let fs = disp_z.spacing();
    let fd = disp_z.direction();
    // Moving geometry; pre-invert the direction so the per-voxel hot loop only
    // does matrix-vector products.
    let mo = moving.origin();
    let ms = moving.spacing();
    let md_inv = moving
        .direction()
        .try_inverse()
        .ok_or_else(|| anyhow!("warp: moving image direction matrix is singular"))?;

    let mut out = vec![0.0f32; nz * ny * nx];
    let lower = -0.5;
    let upper = [
        mov_dims[0] as f64 - 0.5,
        mov_dims[1] as f64 - 0.5,
        mov_dims[2] as f64 - 0.5,
    ];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = iz * ny * nx + iy * nx + ix;
                let idx = [iz as f64, iy as f64, ix as f64];
                // Output physical point p = O_f + R_f · (S_f ⊙ idx), plus the
                // displacement at this grid point (axis-major components).
                let disp = [dz[flat] as f64, dy[flat] as f64, dx[flat] as f64];
                let mut p = [0.0f64; 3];
                for (a, p_a) in p.iter_mut().enumerate() {
                    let mut acc = fo[a];
                    for k in 0..3 {
                        acc += fd[(a, k)] * (idx[k] * fs[k]);
                    }
                    *p_a = acc + disp[a];
                }
                // Moving continuous index c = S_m⁻¹ · R_m⁻¹ · (p − O_m).
                let mut c = [0.0f64; 3];
                let mut inside = true;
                for (a, c_a) in c.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for k in 0..3 {
                        acc += md_inv[(a, k)] * (p[k] - mo[k]);
                    }
                    let ci = acc / ms[a];
                    if ci < lower || ci >= upper[a] {
                        inside = false;
                    }
                    *c_a = ci;
                }
                if inside {
                    out[flat] = trilinear(&mov_vals, mov_dims, c);
                }
            }
        }
    }

    Ok(rebuild(out, field_dims, disp_z))
}

#[cfg(test)]
#[path = "tests_warp.rs"]
mod tests_warp;
