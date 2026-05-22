use rayon::prelude::*;

use super::pde::{compute_divergence_into, gaussian_smooth_1d, make_gaussian_kernel_1d};

// ── Gradient buffer ──────────────────────────────────────────────────────────

/// Gradient buffer: [gz, gy, gx] stored contiguously, each of length n.
pub struct Gradient {
    pub gz: Vec<f64>,
    pub gy: Vec<f64>,
    pub gx: Vec<f64>,
}

// ── Structure tensor products ─────────────────────────────────────────────────

/// 6 independent components of the outer product ∇I·∇Iᵀ at each voxel.
///
/// Layout per voxel: [I_z², I_z·I_y, I_z·I_x, I_y², I_y·I_x, I_x²]
pub struct StructureTensorProducts {
    pub data: Vec<[f64; 6]>,
}

/// Compute the 6 outer-product components at every voxel.
pub fn compute_structure_tensor_products(
    grad: &Gradient,
    dims: [usize; 3],
) -> StructureTensorProducts {
    let n = dims[0] * dims[1] * dims[2];
    let mut st = StructureTensorProducts {
        data: vec![[0.0f64; 6]; n],
    };
    st.data.par_iter_mut().enumerate().for_each(|(i, out)| {
        let gz = grad.gz[i];
        let gy = grad.gy[i];
        let gx = grad.gx[i];
        out[0] = gz * gz; // J_11 = I_z²
        out[1] = gz * gy; // J_12 = I_z·I_y
        out[2] = gz * gx; // J_13 = I_z·I_x
        out[3] = gy * gy; // J_22 = I_y²
        out[4] = gy * gx; // J_23 = I_y·I_x
        out[5] = gx * gx; // J_33 = I_x²
    });
    st
}

// ── Scratch storage ───────────────────────────────────────────────────────────

/// Pre-allocated scratch buffers for the CED iteration loop.
///
/// Reusing a `CedScratch` instance across calls to
/// [`CoherenceEnhancingDiffusionFilter::apply_with_scratch`] avoids
/// repeated per-call heap allocations for the gradient, structure-tensor,
/// smoothed structure-tensor, and divergence buffers.
pub struct CedScratch {
    /// Gradient components per voxel (gz, gy, gx).
    grad_z: Vec<f64>,
    grad_y: Vec<f64>,
    grad_x: Vec<f64>,

    /// 6 structure-tensor outer-product components per voxel.
    st_products: Vec<[f64; 6]>,

    /// Smoothed structure tensor (6 components per voxel).
    st_smooth: Vec<[f64; 6]>,

    /// Divergence buffer.
    divergence: Vec<f64>,

    /// Current image data (mutable copy updated each iteration).
    current: Vec<f64>,

    /// Gaussian kernel (shared across smoothing passes).
    kernel: Vec<f64>,

    /// Previously configured sigma (to detect when kernel must be rebuilt).
    sigma: f64,
}

impl CedScratch {
    /// Ensure all buffers are sized for a volume of `n` voxels.
    pub fn ensure_capacity(&mut self, n: usize, sigma: f64) {
        if self.grad_z.len() != n {
            self.grad_z = vec![0.0; n];
            self.grad_y = vec![0.0; n];
            self.grad_x = vec![0.0; n];
        }
        if self.st_products.len() != n {
            self.st_products = vec![[0.0; 6]; n];
        }
        if self.st_smooth.len() != n {
            self.st_smooth = vec![[0.0; 6]; n];
        }
        if self.divergence.len() != n {
            self.divergence = vec![0.0; n];
        }
        if self.current.len() != n {
            self.current = vec![0.0; n];
        }
        if (self.sigma - sigma).abs() > f64::EPSILON || self.kernel.is_empty() {
            self.kernel = make_gaussian_kernel_1d(sigma);
            self.sigma = sigma;
        }
    }

    /// Run the full CED loop using these scratch buffers.
    pub fn run(
        &mut self,
        data: &[f64],
        dims: [usize; 3],
        config: &super::filter::CoherenceConfig,
    ) -> &[f64] {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        self.ensure_capacity(n, config.sigma);

        // Copy input into current buffer.
        self.current[..n].copy_from_slice(&data[..n]);

        for _ in 0..config.n_iterations {
            // Step 1: gradient into scratch buffers
            compute_gradient_into(
                &self.current,
                dims,
                &mut self.grad_z,
                &mut self.grad_y,
                &mut self.grad_x,
            );

            // Step 2: structure tensor products
            let grad = Gradient {
                gz: self.grad_z.clone(),
                gy: self.grad_y.clone(),
                gx: self.grad_x.clone(),
            };
            let st_products = compute_structure_tensor_products(&grad, dims);
            self.st_products[..n].copy_from_slice(&st_products.data[..n]);

            // Step 3: smooth structure tensor
            smooth_structure_tensor_into(
                &self.st_products,
                dims,
                &self.kernel,
                &mut self.st_smooth,
            );

            // Step 4: divergence
            compute_divergence_into(
                &self.current,
                &self.st_smooth,
                dims,
                config.alpha,
                config.contrast,
                &mut self.divergence,
            );

            // Step 5: explicit Euler update
            let dt = config.time_step;
            for i in 0..n {
                self.current[i] += dt * self.divergence[i];
            }
        }

        &self.current[..n]
    }
}

impl Default for CedScratch {
    fn default() -> Self {
        Self {
            grad_z: Vec::new(),
            grad_y: Vec::new(),
            grad_x: Vec::new(),
            st_products: Vec::new(),
            st_smooth: Vec::new(),
            divergence: Vec::new(),
            current: Vec::new(),
            kernel: Vec::new(),
            sigma: -1.0,
        }
    }
}

// ── Internal helpers (write into pre-allocated buffers) ───────────────────────

/// Central-difference gradient with Neumann (replicate) boundary conditions.
fn compute_gradient_into(
    data: &[f64],
    dims: [usize; 3],
    gz: &mut [f64],
    gy: &mut [f64],
    gx: &mut [f64],
) {
    let [nz, ny, nx] = dims;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let i = iz * ny * nx + iy * nx + ix;
                let iz_p = (iz + 1).min(nz - 1);
                let iz_m = iz.saturating_sub(1);
                let iy_p = (iy + 1).min(ny - 1);
                let iy_m = iy.saturating_sub(1);
                let ix_p = (ix + 1).min(nx - 1);
                let ix_m = ix.saturating_sub(1);

                let dz = (iz_p - iz_m) as f64;
                let dy = (iy_p - iy_m) as f64;
                let dx = (ix_p - ix_m) as f64;

                gz[i] = if dz > 0.0 {
                    (data[iz_p * ny * nx + iy * nx + ix] - data[iz_m * ny * nx + iy * nx + ix]) / dz
                } else {
                    0.0
                };
                gy[i] = if dy > 0.0 {
                    (data[iz * ny * nx + iy_p * nx + ix] - data[iz * ny * nx + iy_m * nx + ix]) / dy
                } else {
                    0.0
                };
                gx[i] = if dx > 0.0 {
                    (data[iz * ny * nx + iy * nx + ix_p] - data[iz * ny * nx + iy * nx + ix_m]) / dx
                } else {
                    0.0
                };
            }
        }
    }
}

/// Smooth structure tensor into a pre-allocated buffer.
fn smooth_structure_tensor_into(
    st_products: &[[f64; 6]],
    dims: [usize; 3],
    kernel: &[f64],
    out: &mut [[f64; 6]],
) {
    let n = dims[0] * dims[1] * dims[2];
    let smoothed_components: Vec<Vec<f64>> = (0..6)
        .map(|c| {
            let mut buf: Vec<f64> = st_products.iter().map(|v| v[c]).collect();
            buf = gaussian_smooth_1d(&buf, dims, 0, kernel);
            buf = gaussian_smooth_1d(&buf, dims, 1, kernel);
            buf = gaussian_smooth_1d(&buf, dims, 2, kernel);
            buf
        })
        .collect();
    for i in 0..n {
        for c in 0..6 {
            out[i][c] = smoothed_components[c][i];
        }
    }
}
