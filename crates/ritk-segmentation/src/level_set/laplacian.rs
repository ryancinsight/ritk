//! Laplacian level set segmentation for 3-D medical images.
//!
//! # Mathematical Specification
//!
//! Segments structures where the image Laplacian is negative (local bright
//! maxima, thin bright structures).  The contour expands toward bright regions
//! driven by the normalised Laplacian speed function.
//!
//! ## Speed Function
//!
//! Let `I` be the Gaussian pre-smoothed input image.  The second-order
//! central-difference Laplacian with clamped boundaries is:
//!
//! ```text
//! L(I)(x) = d2I/dz2 + d2I/dy2 + d2I/dx2
//! ```
//!
//! Normalised to bound values in `(-1, 1)`:
//!
//! ```text
//! F(x) = L(I)(x) / (1.0 + |L(I)(x)|)
//! ```
//!
//! ## PDE
//!
//! Forward-Euler level set evolution:
//!
//! ```text
//! dphi/dt = [w_p * F(x) + w_c * kappa] * |grad phi|
//! ```
//!
//! where:
//! - `kappa = div(grad phi / |grad phi|)` is mean curvature,
//! - `w_p` (`propagation_weight`) controls Laplacian-driven expansion/contraction,
//! - `w_c` (`curvature_weight`) regularises the contour via curvature flow.
//!
//! For a 3-D Gaussian blob `I(r) = exp(-r^2/(2*sigma^2))` the Laplacian at
//! the centre is `L = -3/sigma^2`, giving `F < 0`.  With default `w_p = 1.0`
//! the propagation term `w_p * F < 0`, so the speed is negative and the
//! contour expands toward bright maxima.
//!
//! ## Output
//!
//! Binary mask: `1.0` where `phi < 0`, `0.0` elsewhere.
//!
//! ## Convergence
//!
//! Stops when `max |delta phi| / dt < tolerance` or
//! `iterations >= max_iterations`.
//!
//! # References
//!
//! - Sethian, J. A. (1999). *Level Set Methods and Fast Marching Methods*.
//!   Cambridge University Press.

use std::borrow::Cow;

use super::helpers;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_filter::edge::GaussianSigma;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

/// Laplacian level set segmentation.
///
/// Evolves an initial level set `phi` using a speed function derived from the
/// normalised image Laplacian.  Regions where `L(I) < 0` (local bright
/// maxima) produce negative propagation speed, driving the contour to expand
/// toward bright structures.
///
/// | Parameter            | Default |
/// |----------------------|---------|
/// | `propagation_weight` | 1.0     |
/// | `curvature_weight`   | 0.2     |
/// | `sigma`              | 1.0     |
/// | `dt`                 | 0.05    |
/// | `max_iterations`     | 200     |
/// | `tolerance`          | 1e-3    |
#[derive(Debug, Clone)]
pub struct LaplacianLevelSet {
    /// Weight `w_p` on the Laplacian propagation term.
    pub propagation_weight: f64,
    /// Weight `w_c` on the curvature regularisation term.
    pub curvature_weight: f64,
    /// Standard deviation for Gaussian pre-smoothing of the input image.
    pub sigma: GaussianSigma,
    /// Forward-Euler time step `dt`.
    pub dt: f64,
    /// Maximum number of PDE iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on `max |delta phi| / dt`.
    pub tolerance: f64,
}

impl LaplacianLevelSet {
    /// Construct with default parameters.
    pub fn new() -> Self {
        Self {
            propagation_weight: 1.0,
            curvature_weight: 0.2,
            sigma: GaussianSigma::new_unchecked(1.0),
            dt: 0.05,
            max_iterations: 200,
            tolerance: 1e-3,
        }
    }

    /// Apply Laplacian level set segmentation to a 3-D image.
    ///
    /// # Errors
    ///
    /// Returns `Err` if shapes differ or data cannot be read as `f32`.
    pub fn apply<B: Backend>(
        &self,
        image: &Image<B, 3>,
        initial_phi: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let phi_dims = initial_phi.shape();
        if dims != phi_dims {
            anyhow::bail!(
                "image shape {:?} and initial_phi shape {:?} must match",
                dims,
                phi_dims
            );
        }

        let device = image.data().device();
        let [nz, ny, nx] = dims;
        let n: usize = nz * ny * nx;

        let (img_vals, _) = extract_vec(image)?;
        let (phi_init, _) = extract_vec(initial_phi)?;
        let img_wide: Vec<f64> = img_vals.iter().map(|&v| v as f64).collect();
        let mut phi: Vec<f64> = phi_init.iter().map(|&v| v as f64).collect();

        // Optional Gaussian pre-smoothing of the input image.
        let smoothed: Cow<[f64]> = if self.sigma.get() > 0.0 {
            Cow::Owned(helpers::gaussian_smooth(&img_wide, dims, self.sigma.get()))
        } else {
            Cow::Borrowed(&img_wide)
        };

        // L(I)[i] = d2I/dz2 + d2I/dy2 + d2I/dx2  (central diffs, clamped BC).
        // Parallelised over all elements using moirai.
        let laplacian = moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |i| {
            let iz = i / (ny * nx);
            let rem = i % (ny * nx);
            let iy = rem / nx;
            let ix = rem % nx;
            let zz = iz as isize;
            let yy = iy as isize;
            let xx = ix as isize;

            let d2z = smoothed[helpers::idx_clamped(zz + 1, yy, xx, nz, ny, nx)]
                - 2.0 * smoothed[i]
                + smoothed[helpers::idx_clamped(zz - 1, yy, xx, nz, ny, nx)];
            let d2y = smoothed[helpers::idx_clamped(zz, yy + 1, xx, nz, ny, nx)]
                - 2.0 * smoothed[i]
                + smoothed[helpers::idx_clamped(zz, yy - 1, xx, nz, ny, nx)];
            let d2x = smoothed[helpers::idx_clamped(zz, yy, xx + 1, nz, ny, nx)]
                - 2.0 * smoothed[i]
                + smoothed[helpers::idx_clamped(zz, yy, xx - 1, nz, ny, nx)];

            d2z + d2y + d2x
        });

        // F[i] = L[i] / (1.0 + |L[i]|) maps L into (-1, +1).
        let speed_field: Vec<f64> = laplacian.iter().map(|&l| l / (1.0 + l.abs())).collect();

        // PDE scratch buffers.
        let mut kappa = vec![0.0_f64; n];
        let mut phi_new = phi.clone();
        let mut phi_z = vec![0.0_f64; n];
        let mut phi_y = vec![0.0_f64; n];
        let mut phi_x = vec![0.0_f64; n];

        let slice_len = ny * nx;
        let mut max_changes = vec![0.0_f64; nz];

        // dphi/dt = [w_p * F(x) + w_c * kappa] * |grad phi|
        for _iter in 0..self.max_iterations {
            helpers::compute_curvature_into(&phi, dims, &mut kappa);
            helpers::compute_field_gradient_into(&phi, dims, &mut phi_z, &mut phi_y, &mut phi_x);

            struct SendPtr<T>(*mut T);
            unsafe impl<T> Send for SendPtr<T> {}
            unsafe impl<T> Sync for SendPtr<T> {}
            impl<T> SendPtr<T> {
                unsafe fn write(&self, offset: usize, val: T) {
                    *self.0.add(offset) = val;
                }
            }
            let max_changes_ptr = SendPtr(max_changes.as_mut_ptr());

            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                &mut phi_new,
                slice_len,
                |iz, phi_new_s| {
                    let base = iz * slice_len;
                    let mut local_max = 0.0_f64;
                    for (i, phi_new_val) in phi_new_s.iter_mut().enumerate() {
                        let idx = base + i;
                        let grad_phi_mag = (phi_z[idx] * phi_z[idx]
                            + phi_y[idx] * phi_y[idx]
                            + phi_x[idx] * phi_x[idx])
                            .sqrt();

                        let speed = self.propagation_weight * speed_field[idx]
                            + self.curvature_weight * kappa[idx];
                        let dphi = self.dt * speed * grad_phi_mag;
                        *phi_new_val = phi[idx] + dphi;

                        let change = dphi.abs() / self.dt;
                        if change > local_max {
                            local_max = change;
                        }
                    }
                    unsafe {
                        max_changes_ptr.write(iz, local_max);
                    }
                },
            );

            std::mem::swap(&mut phi, &mut phi_new);

            let max_change = max_changes.iter().copied().fold(0.0_f64, f64::max);
            if max_change < self.tolerance {
                break;
            }
        }

        // Binary mask: phi < 0 => 1.0 (foreground), else => 0.0 (background).
        let mask: Vec<f32> = phi
            .iter()
            .map(|&v| if v < 0.0 { 1.0_f32 } else { 0.0_f32 })
            .collect();

        let tensor = Tensor::<B, 3>::from_data(TensorData::new(mask, Shape::new(dims)), &device);

        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

impl Default for LaplacianLevelSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests_laplacian_level_set.rs"]
mod tests;
