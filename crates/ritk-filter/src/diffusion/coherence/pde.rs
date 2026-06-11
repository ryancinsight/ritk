use moirai::prelude::ParallelSlice;

use super::filter::CoherenceConfig;
use super::scratch::{compute_structure_tensor_products, Gradient, StructureTensorProducts};
use super::tensor::diffusion_tensor;

// ── Core computation ──────────────────────────────────────────────────────────

/// Run explicit Euler CED for the requested number of iterations.
///
/// All arithmetic in f64; caller converts to f32 at output.
pub fn ced_diffuse(data: &[f64], dims: [usize; 3], config: &CoherenceConfig) -> Vec<f64> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut cur = data.to_vec();

    // Pre-build the 1-D Gaussian kernel for structure-tensor smoothing.
    let kernel = crate::gaussian_kernel_1d(config.sigma.get(), None);
    for _ in 0..config.n_iterations {
        // ── Step 1: gradient via central differences ────────────────────
        let grad = compute_gradient(&cur, dims);

        // ── Step 2: structure tensor products ───────────────────────────
        let st_products = compute_structure_tensor_products(&grad, dims);

        // ── Step 3: Gaussian smoothing of structure tensor ──────────────
        let st_smooth = smooth_structure_tensor(&st_products, dims, &kernel);

        // ── Step 4: eigenvalue decomposition + diffusion tensor + divergence
        let div = compute_divergence(&cur, &st_smooth, dims, config.alpha, config.contrast);

        // ── Step 5: explicit Euler update ───────────────────────────────
        let dt = config.time_step;
        for i in 0..n {
            cur[i] += dt * div[i];
        }
    }
    cur
}

// ── Gradient computation ──────────────────────────────────────────────────────

/// Central-difference gradient with Neumann (replicate) boundary conditions.
pub fn compute_gradient(data: &[f64], dims: [usize; 3]) -> Gradient {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut gz = vec![0.0f64; n];
    let mut gy = vec![0.0f64; n];
    let mut gx = vec![0.0f64; n];
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
    Gradient { gz, gy, gx }
}

// ── Gaussian smoothing ────────────────────────────────────────────────────────

/// Separable Gaussian smoothing along a single axis.
///
/// `axis` is 0 (z), 1 (y), or 2 (x). Boundary: replicate (Neumann).
pub fn gaussian_smooth_1d(
    input: &[f64],
    dims: [usize; 3],
    axis: usize,
    kernel: &[f64],
) -> Vec<f64> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let radius = (kernel.len() / 2) as i64;
    let mut output = vec![0.0f64; n];
    match axis {
        0 => {
            // Smooth along z (slowest-varying dimension).
            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let mut val = 0.0f64;
                        for (ki, &kw) in kernel.iter().enumerate() {
                            let k = ki as i64 - radius;
                            let sz = iz as i64 + k;
                            let sz = sz.clamp(0, nz as i64 - 1) as usize;
                            val += kw * input[sz * ny * nx + iy * nx + ix];
                        }
                        output[iz * ny * nx + iy * nx + ix] = val;
                    }
                }
            }
        }
        1 => {
            // Smooth along y.
            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let mut val = 0.0f64;
                        for (ki, &kw) in kernel.iter().enumerate() {
                            let k = ki as i64 - radius;
                            let sy = iy as i64 + k;
                            let sy = sy.clamp(0, ny as i64 - 1) as usize;
                            val += kw * input[iz * ny * nx + sy * nx + ix];
                        }
                        output[iz * ny * nx + iy * nx + ix] = val;
                    }
                }
            }
        }
        2 => {
            // Smooth along x (fastest-varying dimension).
            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let mut val = 0.0f64;
                        for (ki, &kw) in kernel.iter().enumerate() {
                            let k = ki as i64 - radius;
                            let sx = ix as i64 + k;
                            let sx = sx.clamp(0, nx as i64 - 1) as usize;
                            val += kw * input[iz * ny * nx + iy * nx + sx];
                        }
                        output[iz * ny * nx + iy * nx + ix] = val;
                    }
                }
            }
        }
        _ => unreachable!("axis must be 0, 1, or 2"),
    }
    output
}

/// Smooth each of the 6 structure tensor components with a separable 3-D Gaussian.
pub fn smooth_structure_tensor(
    st: &StructureTensorProducts,
    dims: [usize; 3],
    kernel: &[f64],
) -> Vec<[f64; 6]> {
    let n = dims[0] * dims[1] * dims[2];
    // Process each component independently (embarrassingly parallel).
    // Collect into [Vec<f64>; 6] — stack-resident array of 6 independent heap
    // buffers — rather than Vec<Vec<f64>>, which would add an outer heap
    // allocation for the container itself.
    let smoothed_components: [Vec<f64>; 6] =
        moirai::map_collect_index_with::<moirai::Parallel, _, _>(6, |c| {
            // Extract component c as a flat buffer.
            let mut buf: Vec<f64> = st.data.iter().map(|v| v[c]).collect();
            // Separable smoothing along z, y, x.
            buf = gaussian_smooth_1d(&buf, dims, 0, kernel);
            buf = gaussian_smooth_1d(&buf, dims, 1, kernel);
            buf = gaussian_smooth_1d(&buf, dims, 2, kernel);
            buf
        })
        .try_into()
        .expect("map_collect_index_with(6) yields exactly 6 elements");
    // Re-interleave into [f64; 6] per voxel.
    let mut out = vec![[0.0f64; 6]; n];
    for i in 0..n {
        for c in 0..6 {
            out[i][c] = smoothed_components[c][i];
        }
    }
    out
}

// ── Divergence computation ────────────────────────────────────────────────────

/// Compute div(D · ∇I) at every voxel using the face-interpolated flux approach.
///
/// At each face between adjacent voxels, the diffusion tensor is interpolated
/// as D_face = (D_p + D_q) / 2. The flux through that face is
/// F_face = D_face · (forward difference), and the divergence is the sum of
/// flux differences across the voxel.
///
/// This formulation is conservative and guarantees ∫ div(D·∇I) dV = 0 for
/// Neumann boundary conditions, preserving mean intensity.
pub fn compute_divergence(
    data: &[f64],
    st_smooth: &[[f64; 6]],
    dims: [usize; 3],
    alpha: f64,
    contrast: f64,
) -> Vec<f64> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    // Compute diffusion tensor at every voxel (parallel).
    let d_tensors: Vec<[f64; 6]> = st_smooth
        .par()
        .map_collect(|&st| diffusion_tensor(st, alpha, contrast));

    let mut div = vec![0.0f64; n];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let i = iz * ny * nx + iy * nx + ix;
                let mut delta = 0.0f64;

                // ── +z / −z faces ──────────────────────────────────────
                if iz + 1 < nz {
                    let j = (iz + 1) * ny * nx + iy * nx + ix;
                    let d_face = avg_tensor(d_tensors[i], d_tensors[j]);
                    let diff = data[j] - data[i];
                    // Forward gradient = [diff, 0, 0] in z-direction.
                    // Flux = D_face · [diff, 0, 0] = [D_11·diff, D_12·diff, D_13·diff]
                    // z-component of flux = D_11 · diff
                    delta += d_face[0] * diff;
                }
                if iz > 0 {
                    let j = (iz - 1) * ny * nx + iy * nx + ix;
                    let d_face = avg_tensor(d_tensors[i], d_tensors[j]);
                    let diff = data[j] - data[i];
                    // value at iz−1 minus value at iz (negative)
                    // The face between iz−1 and iz: flux_z at face(i−1/2)
                    // = D_face · (I[iz−1] − I[iz]) in the z-component
                    // = D_11 · diff (diff is negative, so this subtracts from delta)
                    delta += d_face[0] * diff;
                }

                // ── +y / −y faces ──────────────────────────────────────
                if iy + 1 < ny {
                    let j = iz * ny * nx + (iy + 1) * nx + ix;
                    let d_face = avg_tensor(d_tensors[i], d_tensors[j]);
                    let diff = data[j] - data[i];
                    // y-component of flux = D_22 · diff (since D_22 = d_face[3])
                    delta += d_face[3] * diff;
                }
                if iy > 0 {
                    let j = iz * ny * nx + (iy - 1) * nx + ix;
                    let d_face = avg_tensor(d_tensors[i], d_tensors[j]);
                    let diff = data[j] - data[i];
                    delta += d_face[3] * diff;
                }

                // ── +x / −x faces ──────────────────────────────────────
                if ix + 1 < nx {
                    let j = iz * ny * nx + iy * nx + (ix + 1);
                    let d_face = avg_tensor(d_tensors[i], d_tensors[j]);
                    let diff = data[j] - data[i];
                    // x-component of flux = D_33 · diff (since D_33 = d_face[5])
                    delta += d_face[5] * diff;
                }
                if ix > 0 {
                    let j = iz * ny * nx + iy * nx + (ix - 1);
                    let d_face = avg_tensor(d_tensors[i], d_tensors[j]);
                    let diff = data[j] - data[i];
                    delta += d_face[5] * diff;
                }

                div[i] = delta;
            }
        }
    }

    div
}

/// Compute divergence into a pre-allocated buffer (used by scratch path).
pub fn compute_divergence_into(
    data: &[f64],
    st_smooth: &[[f64; 6]],
    dims: [usize; 3],
    alpha: f64,
    contrast: f64,
    div: &mut [f64],
) {
    let result = compute_divergence(data, st_smooth, dims, alpha, contrast);
    div.copy_from_slice(&result);
}

/// Element-wise average of two symmetric 3×3 tensor component vectors.
#[inline(always)]
pub(crate) fn avg_tensor(a: [f64; 6], b: [f64; 6]) -> [f64; 6] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
        (a[3] + b[3]) * 0.5,
        (a[4] + b[4]) * 0.5,
        (a[5] + b[5]) * 0.5,
    ]
}
