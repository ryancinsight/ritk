use crate::edge::GaussianSigma;
use moirai::prelude::ParallelSliceMut;

use super::pde::compute_divergence_into;

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
    compute_structure_tensor_products_from_slices(&grad.gz, &grad.gy, &grad.gx, dims)
}

/// Compute the 6 outer-product components from gradient slices.
///
/// Accepts `(&[f64], &[f64], &[f64])` instead of `&Gradient` so callers
/// that already hold separate gradient buffers (e.g. `CedScratch`) can
/// pass them directly without cloning into a `Gradient` struct.
pub fn compute_structure_tensor_products_from_slices(
    gz: &[f64],
    gy: &[f64],
    gx: &[f64],
    dims: [usize; 3],
) -> StructureTensorProducts {
    let n = dims[0] * dims[1] * dims[2];
    let mut st = StructureTensorProducts {
        data: vec![[0.0f64; 6]; n],
    };
    st.data.par_mut().enumerate(|i, out| {
        let gz = gz[i];
        let gy = gy[i];
        let gx = gx[i];
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
/// \[`CoherenceEnhancingDiffusionFilter::apply_with_scratch`\] avoids
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

    /// Scratch buffer reused across the 6 component smoothing passes
    /// in `smooth_structure_tensor_into` to avoid per-component allocation.
    smooth_buf: Vec<f64>,

    /// Second scratch buffer for in-place Gaussian smoothing passes.
    smooth_buf2: Vec<f64>,

    /// Pre-allocated diffusion tensors.
    d_tensors: Vec<[f64; 6]>,

    /// Cached sigma from the last kernel build; `None` until first `ensure_capacity` call.
    cached_sigma: Option<GaussianSigma>,
}

impl CedScratch {
    /// Ensure all buffers are sized for a volume of `n` voxels.
    pub fn ensure_capacity(&mut self, n: usize, sigma: GaussianSigma) {
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
        if self.smooth_buf.len() != n {
            self.smooth_buf = vec![0.0; n];
        }
        if self.smooth_buf2.len() != n {
            self.smooth_buf2 = vec![0.0; n];
        }
        if self.d_tensors.len() != n {
            self.d_tensors = vec![[0.0; 6]; n];
        }
        if self.cached_sigma != Some(sigma) || self.kernel.is_empty() {
            self.kernel = crate::gaussian_kernel(sigma.get(), None);
            self.cached_sigma = Some(sigma);
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

            // Step 2: structure tensor products (zero-copy via slices)
            compute_structure_tensor_products_into(
                &self.grad_z,
                &self.grad_y,
                &self.grad_x,
                dims,
                &mut self.st_products,
            );

            // Step 3: smooth structure tensor
            smooth_structure_tensor_into(
                &self.st_products,
                dims,
                &self.kernel,
                &mut self.smooth_buf,
                &mut self.smooth_buf2,
                &mut self.st_smooth,
            );

            // Step 4: divergence
            compute_divergence_into(
                &self.current,
                &self.st_smooth,
                dims,
                config.alpha,
                config.contrast,
                &mut self.d_tensors,
                &mut self.divergence,
            );

            // Step 5: explicit Euler update
            let dt = config.time_step;
            for (cur, &div) in self.current.iter_mut().zip(self.divergence.iter()).take(n) {
                *cur += dt * div;
            }
        }

        &self.current[..n]
    }

    /// Run the full CED loop using these scratch buffers, accepting f32 input data.
    pub fn run_f32(
        &mut self,
        data: &[f32],
        dims: [usize; 3],
        config: &super::filter::CoherenceConfig,
    ) -> &[f64] {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        self.ensure_capacity(n, config.sigma);

        // Copy input into current buffer, casting f32 to f64 directly.
        for (cur, &val) in self.current.iter_mut().zip(data.iter()).take(n) {
            *cur = val as f64;
        }

        for _ in 0..config.n_iterations {
            // Step 1: gradient into scratch buffers
            compute_gradient_into(
                &self.current,
                dims,
                &mut self.grad_z,
                &mut self.grad_y,
                &mut self.grad_x,
            );

            // Step 2: structure tensor products (zero-copy via slices)
            compute_structure_tensor_products_into(
                &self.grad_z,
                &self.grad_y,
                &self.grad_x,
                dims,
                &mut self.st_products,
            );

            // Step 3: smooth structure tensor
            smooth_structure_tensor_into(
                &self.st_products,
                dims,
                &self.kernel,
                &mut self.smooth_buf,
                &mut self.smooth_buf2,
                &mut self.st_smooth,
            );

            // Step 4: divergence
            compute_divergence_into(
                &self.current,
                &self.st_smooth,
                dims,
                config.alpha,
                config.contrast,
                &mut self.d_tensors,
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
        // Delayed-init pattern: ensure_capacity reallocates to exact voxel count
        // on first call. with_capacity(0) avoids a wasted initial grow(0)
        // allocation inside ensure_capacity's vec![0.0; n] replacement path.
        Self {
            grad_z: Vec::with_capacity(0),
            grad_y: Vec::with_capacity(0),
            grad_x: Vec::with_capacity(0),
            st_products: Vec::with_capacity(0),
            st_smooth: Vec::with_capacity(0),
            divergence: Vec::with_capacity(0),
            current: Vec::with_capacity(0),
            smooth_buf: Vec::with_capacity(0),
            smooth_buf2: Vec::with_capacity(0),
            d_tensors: Vec::with_capacity(0),
            kernel: Vec::with_capacity(0),
            cached_sigma: None,
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

/// Compute structure tensor products directly into a pre-allocated buffer.
///
/// Accepts gradient slices so callers with separate buffers (e.g. `CedScratch`)
/// can avoid cloning into a `Gradient` struct.
fn compute_structure_tensor_products_into(
    gz: &[f64],
    gy: &[f64],
    gx: &[f64],
    dims: [usize; 3],
    out: &mut [[f64; 6]],
) {
    let n = dims[0] * dims[1] * dims[2];
    out[..n].par_mut().enumerate(|i, row| {
        let gzi = gz[i];
        let gyi = gy[i];
        let gxi = gx[i];
        row[0] = gzi * gzi;
        row[1] = gzi * gyi;
        row[2] = gzi * gxi;
        row[3] = gyi * gyi;
        row[4] = gyi * gxi;
        row[5] = gxi * gxi;
    });
}

/// Smooth structure tensor into a pre-allocated buffer.
///
/// Processes each of the 6 components sequentially, writing directly into
/// `out` to avoid allocating an outer container (`Vec<Vec<f64>>`). The
/// caller-provided `buf` and `buf2` are reused across components to avoid
/// per-component heap allocation.
fn smooth_structure_tensor_into(
    st_products: &[[f64; 6]],
    dims: [usize; 3],
    kernel: &[f64],
    buf: &mut Vec<f64>,
    buf2: &mut Vec<f64>,
    out: &mut [[f64; 6]],
) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    for c in 0..6 {
        buf.clear();
        buf.extend(st_products.iter().map(|v| v[c]));
        buf2.resize(n, 0.0);

        // Z-axis smoothing
        crate::diffusion::coherence::pde::gaussian_smooth_into(buf, buf2, dims, 0, kernel);
        // Y-axis smoothing
        crate::diffusion::coherence::pde::gaussian_smooth_into(buf2, buf, dims, 1, kernel);
        // X-axis smoothing
        crate::diffusion::coherence::pde::gaussian_smooth_into(buf, buf2, dims, 2, kernel);

        for i in 0..n {
            out[i][c] = buf2[i];
        }
    }
}
