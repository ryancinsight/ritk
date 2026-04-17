//! Shared computational primitives for deformable image registration algorithms.
//!
//! All functions operate on flat `Vec<f32>` buffers with shape `[nz, ny, nx]`
//! (Z-major / row-major order): flat index = `iz * ny * nx + iy * nx + ix`.
//!
//! # Conventions
//! - `dims = [nz, ny, nx]`  — image dimensions
//! - `spacing = [sz, sy, sx]` — physical voxel size (mm or arbitrary units)
//! - Displacement components are stored in voxel units (not physical units)
//!
//! # Boundary conditions
//! All sampling operations use **clamp-to-border** (replicate boundary):
//! coordinates outside `[0, dim − 1]` are clamped to the nearest valid index.

// ── Indexing ──────────────────────────────────────────────────────────────────

/// Flat voxel index for shape `[nz, ny, nx]`.
#[inline(always)]
pub(crate) fn flat(iz: usize, iy: usize, ix: usize, ny: usize, nx: usize) -> usize {
    iz * ny * nx + iy * nx + ix
}

// ── Trilinear interpolation ───────────────────────────────────────────────────

/// Sample `data` at a continuous position `(z, y, x)` using trilinear
/// interpolation with clamp-to-border boundary condition.
///
/// # Invariants
/// - At integer positions the result equals `data[flat(round(z), round(y), round(x))]`.
/// - Positions outside `[0, nZ−1] × [0, nY−1] × [0, nX−1]` are clamped.
#[inline]
pub(crate) fn trilinear_interpolate(data: &[f32], dims: [usize; 3], z: f32, y: f32, x: f32) -> f32 {
    let [nz, ny, nx] = dims;

    // Clamp to valid range.
    let z = z.max(0.0).min((nz as f32) - 1.0);
    let y = y.max(0.0).min((ny as f32) - 1.0);
    let x = x.max(0.0).min((nx as f32) - 1.0);

    let iz0 = z.floor() as usize;
    let iy0 = y.floor() as usize;
    let ix0 = x.floor() as usize;
    let iz1 = (iz0 + 1).min(nz - 1);
    let iy1 = (iy0 + 1).min(ny - 1);
    let ix1 = (ix0 + 1).min(nx - 1);

    let dz = z - iz0 as f32;
    let dy = y - iy0 as f32;
    let dx = x - ix0 as f32;

    let g = |iz: usize, iy: usize, ix: usize| data[flat(iz, iy, ix, ny, nx)];

    // Trilinear blend: first interpolate along x, then y, then z.
    let c00 = g(iz0, iy0, ix0) * (1.0 - dx) + g(iz0, iy0, ix1) * dx;
    let c01 = g(iz0, iy1, ix0) * (1.0 - dx) + g(iz0, iy1, ix1) * dx;
    let c10 = g(iz1, iy0, ix0) * (1.0 - dx) + g(iz1, iy0, ix1) * dx;
    let c11 = g(iz1, iy1, ix0) * (1.0 - dx) + g(iz1, iy1, ix1) * dx;

    let c0 = c00 * (1.0 - dy) + c01 * dy;
    let c1 = c10 * (1.0 - dy) + c11 * dy;

    c0 * (1.0 - dz) + c1 * dz
}

// ── Gradient ──────────────────────────────────────────────────────────────────

/// Write the gradient of data directly into caller-provided buffers.
///
/// Uses central differences at interior voxels and one-sided first-order
/// differences at boundaries. No allocation occurs; all results are written
/// into gz, gy, gx, each of length dims[0] * dims[1] * dims[2].
pub(crate) fn compute_gradient_into(
    data: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    gz: &mut [f32],
    gy: &mut [f32],
    gx: &mut [f32],
) {
    let [nz, ny, nx] = dims;
    let sz = spacing[0] as f32;
    let sy = spacing[1] as f32;
    let sx = spacing[2] as f32;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);

                gz[fi] = if nz == 1 {
                    0.0
                } else if iz == 0 {
                    (data[flat(1, iy, ix, ny, nx)] - data[fi]) / sz
                } else if iz == nz - 1 {
                    (data[fi] - data[flat(nz - 2, iy, ix, ny, nx)]) / sz
                } else {
                    (data[flat(iz + 1, iy, ix, ny, nx)] - data[flat(iz - 1, iy, ix, ny, nx)])
                        / (2.0 * sz)
                };

                gy[fi] = if ny == 1 {
                    0.0
                } else if iy == 0 {
                    (data[flat(iz, 1, ix, ny, nx)] - data[fi]) / sy
                } else if iy == ny - 1 {
                    (data[fi] - data[flat(iz, ny - 2, ix, ny, nx)]) / sy
                } else {
                    (data[flat(iz, iy + 1, ix, ny, nx)] - data[flat(iz, iy - 1, ix, ny, nx)])
                        / (2.0 * sy)
                };

                gx[fi] = if nx == 1 {
                    0.0
                } else if ix == 0 {
                    (data[flat(iz, iy, 1, ny, nx)] - data[fi]) / sx
                } else if ix == nx - 1 {
                    (data[fi] - data[flat(iz, iy, nx - 2, ny, nx)]) / sx
                } else {
                    (data[flat(iz, iy, ix + 1, ny, nx)] - data[flat(iz, iy, ix - 1, ny, nx)])
                        / (2.0 * sx)
                };
            }
        }
    }
}

/// Compute the gradient of `data` via central differences at interior voxels
/// and one-sided first-order differences at boundaries.
///
/// Each component is divided by the corresponding physical `spacing` so that
/// the result is in (intensity / length) units.
///
/// # Returns
/// `(gz, gy, gx)` — three flat `Vec<f32>` of length `nz * ny * nx`.
pub(crate) fn compute_gradient(
    data: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = dims[0] * dims[1] * dims[2];
    let mut gz = vec![0.0_f32; n];
    let mut gy = vec![0.0_f32; n];
    let mut gx = vec![0.0_f32; n];
    compute_gradient_into(data, dims, spacing, &mut gz, &mut gy, &mut gx);
    (gz, gy, gx)
}

// ── Image warping ─────────────────────────────────────────────────────────────

/// Warp moving by the displacement field into a caller-provided buffer.
///
/// For each voxel p = (iz, iy, ix):
///   output[p] = moving(iz + dz[p], iy + dy[p], ix + dx[p])
/// sampled with trilinear interpolation and clamp-to-border BC.
/// output must have length dims[0] * dims[1] * dims[2].
pub(crate) fn warp_image_into(
    moving: &[f32],
    dims: [usize; 3],
    dz: &[f32],
    dy: &[f32],
    dx: &[f32],
    output: &mut [f32],
) {
    let [nz, ny, nx] = dims;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);
                let wz = iz as f32 + dz[fi];
                let wy = iy as f32 + dy[fi];
                let wx = ix as f32 + dx[fi];
                output[fi] = trilinear_interpolate(moving, dims, wz, wy, wx);
            }
        }
    }
}

/// Warp `moving` by the displacement field `(dz, dy, dx)`.
///
/// For each voxel `p = (iz, iy, ix)`:
///   `warped(p) = moving(iz + dz[p], iy + dy[p], ix + dx[p])`
/// sampled with trilinear interpolation and clamp-to-border BC.
pub(crate) fn warp_image(
    moving: &[f32],
    dims: [usize; 3],
    dz: &[f32],
    dy: &[f32],
    dx: &[f32],
) -> Vec<f32> {
    let n = dims[0] * dims[1] * dims[2];
    let mut warped = vec![0.0_f32; n];
    warp_image_into(moving, dims, dz, dy, dx, &mut warped);
    warped
}

/// Compute mean((fixed - warp(moving, D))^2) without materialising a warped buffer.
///
/// Streams trilinear samples of moving under displacement D = (dz, dy, dx) directly
/// into a squared-error accumulator. No intermediate Vec<f32> is allocated.
///
/// Returns the mean squared error as f64.
pub(crate) fn compute_mse_streaming(
    fixed: &[f32],
    moving: &[f32],
    dims: [usize; 3],
    dz: &[f32],
    dy: &[f32],
    dx: &[f32],
) -> f64 {
    let [nz, ny, nx] = dims;
    let mut sum = 0.0_f64;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);
                let wz = iz as f32 + dz[fi];
                let wy = iy as f32 + dy[fi];
                let wx = ix as f32 + dx[fi];
                let warped = trilinear_interpolate(moving, dims, wz, wy, wx);
                let diff = (fixed[fi] - warped) as f64;
                sum += diff * diff;
            }
        }
    }
    sum / fixed.len() as f64
}

// ── Gaussian smoothing ────────────────────────────────────────────────────────

/// Build a normalised 1-D Gaussian kernel with radius `⌈3σ⌉`.
///
/// The kernel sums to exactly 1.0 (probability-preserving convolution).
fn gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil() as usize;
    let two_sigma2 = 2.0 * sigma * sigma;
    let mut k: Vec<f64> = (0..=(2 * radius))
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-x * x / two_sigma2).exp()
        })
        .collect();
    let sum: f64 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// Convolve `data` along the Z axis with `kernel`; write result into `output`.
/// Uses replicate-border boundary condition.
fn convolve_z(data: &[f32], dims: [usize; 3], kernel: &[f64], output: &mut [f32]) {
    let [nz, ny, nx] = dims;
    let r = kernel.len() / 2;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);
                let mut acc = 0.0_f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let src = (iz as isize + ki as isize - r as isize)
                        .max(0)
                        .min(nz as isize - 1) as usize;
                    acc += kv * data[flat(src, iy, ix, ny, nx)] as f64;
                }
                output[fi] = acc as f32;
            }
        }
    }
}

/// Convolve `data` along the Y axis with `kernel`; write result into `output`.
fn convolve_y(data: &[f32], dims: [usize; 3], kernel: &[f64], output: &mut [f32]) {
    let [nz, ny, nx] = dims;
    let r = kernel.len() / 2;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);
                let mut acc = 0.0_f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let src = (iy as isize + ki as isize - r as isize)
                        .max(0)
                        .min(ny as isize - 1) as usize;
                    acc += kv * data[flat(iz, src, ix, ny, nx)] as f64;
                }
                output[fi] = acc as f32;
            }
        }
    }
}

/// Convolve `data` along the X axis with `kernel`; write result into `output`.
fn convolve_x(data: &[f32], dims: [usize; 3], kernel: &[f64], output: &mut [f32]) {
    let [nz, ny, nx] = dims;
    let r = kernel.len() / 2;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);
                let mut acc = 0.0_f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let src = (ix as isize + ki as isize - r as isize)
                        .max(0)
                        .min(nx as isize - 1) as usize;
                    acc += kv * data[flat(iz, iy, src, ny, nx)] as f64;
                }
                output[fi] = acc as f32;
            }
        }
    }
}

/// Apply separable 3-D Gaussian smoothing to `data` **in place**.
///
/// Convolves sequentially along Z, Y, then X.  Uses a temporary buffer to
/// avoid read-after-write aliasing.  A `sigma ≤ 0` is a no-op.
pub(crate) fn gaussian_smooth_inplace(data: &mut Vec<f32>, dims: [usize; 3], sigma: f64) {
    if sigma <= 0.0 {
        return;
    }
    let kernel = gaussian_kernel_1d(sigma);
    let n = data.len();
    let mut tmp = vec![0.0_f32; n];

    convolve_z(data, dims, &kernel, &mut tmp);
    std::mem::swap(data, &mut tmp);

    convolve_y(data, dims, &kernel, &mut tmp);
    std::mem::swap(data, &mut tmp);

    convolve_x(data, dims, &kernel, &mut tmp);
    std::mem::swap(data, &mut tmp);
}

// ── Displacement field composition ───────────────────────────────────────────

/// Compute the composition `φ_composed = φ₁ ∘ φ₂`.
///
/// `φ_composed(x) = φ₁(x + φ₂(x))` — the combined displacement at each voxel
/// `x` is obtained by displacing `x` by `φ₂(x)` and then sampling `φ₁` at the
/// resulting position via trilinear interpolation.
pub(crate) fn compose_fields(
    phi1_z: &[f32],
    phi1_y: &[f32],
    phi1_x: &[f32],
    phi2_z: &[f32],
    phi2_y: &[f32],
    phi2_x: &[f32],
    dims: [usize; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    let mut cz = vec![0.0_f32; n];
    let mut cy = vec![0.0_f32; n];
    let mut cx = vec![0.0_f32; n];

    compose_fields_into(
        phi1_z, phi1_y, phi1_x, phi2_z, phi2_y, phi2_x, dims, &mut cz, &mut cy, &mut cx,
    );

    (cz, cy, cx)
}

/// Compute the composition `φ_composed = φ₁ ∘ φ₂` into caller-provided buffers.
///
/// Output buffers must have length `dims[0] * dims[1] * dims[2]`.
pub(crate) fn compose_fields_into(
    phi1_z: &[f32],
    phi1_y: &[f32],
    phi1_x: &[f32],
    phi2_z: &[f32],
    phi2_y: &[f32],
    phi2_x: &[f32],
    dims: [usize; 3],
    out_z: &mut [f32],
    out_y: &mut [f32],
    out_x: &mut [f32],
) {
    let [nz, ny, nx] = dims;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);

                // Displaced position x + φ₂(x).
                let wz = iz as f32 + phi2_z[fi];
                let wy = iy as f32 + phi2_y[fi];
                let wx = ix as f32 + phi2_x[fi];

                // Sample φ₁ at the displaced position.
                out_z[fi] = phi2_z[fi] + trilinear_interpolate(phi1_z, dims, wz, wy, wx);
                out_y[fi] = phi2_y[fi] + trilinear_interpolate(phi1_y, dims, wz, wy, wx);
                out_x[fi] = phi2_x[fi] + trilinear_interpolate(phi1_x, dims, wz, wy, wx);
            }
        }
    }
}

// ── Scaling-and-squaring (exponential map) ────────────────────────────────────

/// Compute the exponential map `exp(v)` of a stationary velocity field `v`
/// via the scaling-and-squaring algorithm.
///
/// # Algorithm
/// 1. Scale: `φ ← v / 2^n_steps`
/// 2. Square n_steps times: `φ ← φ ∘ φ`
///
/// Using `n_steps = 6` corresponds to 64 integration steps and is the
/// standard choice for Diffeomorphic Demons (Vercauteren et al. 2009).
///
/// # Invariants
/// - For `v = 0` the result is the identity displacement `(0, 0, 0)`.
/// - For small `v`, `exp(v) ≈ v` (first-order approximation).
pub(crate) fn scaling_and_squaring(
    vz: &[f32],
    vy: &[f32],
    vx: &[f32],
    dims: [usize; 3],
    n_steps: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let scale = 1.0_f32 / (1u32 << n_steps) as f32;

    // Initial scaled displacement.
    let mut phiz: Vec<f32> = vz.iter().map(|&v| v * scale).collect();
    let mut phiy: Vec<f32> = vy.iter().map(|&v| v * scale).collect();
    let mut phix: Vec<f32> = vx.iter().map(|&v| v * scale).collect();

    let n = phiz.len();
    let mut next_z = vec![0.0_f32; n];
    let mut next_y = vec![0.0_f32; n];
    let mut next_x = vec![0.0_f32; n];

    // Squaring steps: φ ← φ ∘ φ.
    for _ in 0..n_steps {
        compose_fields_into(
            &phiz,
            &phiy,
            &phix,
            &phiz,
            &phiy,
            &phix,
            dims,
            &mut next_z,
            &mut next_y,
            &mut next_x,
        );
        std::mem::swap(&mut phiz, &mut next_z);
        std::mem::swap(&mut phiy, &mut next_y);
        std::mem::swap(&mut phix, &mut next_x);
    }

    (phiz, phiy, phix)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ramp(dims: [usize; 3]) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        (0..nz * ny * nx)
            .map(|fi| {
                let ix = fi % nx;
                let iy = (fi / nx) % ny;
                let iz = fi / (ny * nx);
                (iz + iy + ix) as f32
            })
            .collect()
    }

    /// At an integer coordinate the trilinear interpolant must equal the stored value.
    #[test]
    fn trilinear_at_integer_equals_value() {
        let dims = [5usize, 5, 5];
        let data = make_ramp(dims);
        for iz in 0..5 {
            for iy in 0..5 {
                for ix in 0..5 {
                    let fi = flat(iz, iy, ix, 5, 5);
                    let v = trilinear_interpolate(&data, dims, iz as f32, iy as f32, ix as f32);
                    assert!(
                        (v - data[fi]).abs() < 1e-5,
                        "({iz},{iy},{ix}): expected {}, got {v}",
                        data[fi]
                    );
                }
            }
        }
    }

    /// Trilinear interpolation of a constant field returns the constant everywhere.
    #[test]
    fn trilinear_constant_field() {
        let dims = [4usize, 4, 4];
        let data = vec![7.0_f32; 4 * 4 * 4];
        let v = trilinear_interpolate(&data, dims, 1.7, 2.3, 0.8);
        assert!((v - 7.0).abs() < 1e-5, "expected 7.0, got {v}");
    }

    /// Gradient of a linear ramp I[z,y,x] = x should be (0, 0, 1/sx).
    #[test]
    fn gradient_linear_ramp_x() {
        let dims = [4usize, 4, 8];
        let [nz, ny, nx] = dims;
        let data: Vec<f32> = (0..nz * ny * nx).map(|fi| (fi % nx) as f32).collect();
        let spacing = [1.0, 1.0, 1.0];
        let (gz, gy, gx) = compute_gradient(&data, dims, spacing);

        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let fi = flat(iz, iy, ix, ny, nx);
                    assert!((gz[fi]).abs() < 1e-5, "gz should be 0, got {}", gz[fi]);
                    assert!((gy[fi]).abs() < 1e-5, "gy should be 0, got {}", gy[fi]);
                    assert!(
                        (gx[fi] - 1.0).abs() < 1e-5,
                        "gx should be 1, got {}",
                        gx[fi]
                    );
                }
            }
        }
    }

    /// Warp of a constant image is constant regardless of displacement.
    #[test]
    fn warp_constant_image() {
        let dims = [4usize, 4, 4];
        let n = 4 * 4 * 4;
        let data = vec![5.0_f32; n];
        let dz = vec![0.5_f32; n];
        let dy = vec![-0.3_f32; n];
        let dx = vec![1.1_f32; n];
        let out = warp_image(&data, dims, &dz, &dy, &dx);
        for &v in &out {
            assert!((v - 5.0).abs() < 1e-5, "expected 5.0, got {v}");
        }
    }

    /// Warp with zero displacement must return the original image.
    #[test]
    fn warp_identity_displacement() {
        let dims = [6usize, 6, 6];
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dz = vec![0.0_f32; n];
        let dy = vec![0.0_f32; n];
        let dx = vec![0.0_f32; n];
        let out = warp_image(&data, dims, &dz, &dy, &dx);
        for (i, (&orig, &warped)) in data.iter().zip(out.iter()).enumerate() {
            assert!(
                (orig - warped).abs() < 1e-5,
                "voxel {i}: expected {orig}, got {warped}"
            );
        }
    }

    /// Gaussian smoothing of a uniform field leaves the field unchanged.
    #[test]
    fn gaussian_smooth_uniform_unchanged() {
        let dims = [6usize, 6, 6];
        let n = 6 * 6 * 6;
        let mut data = vec![3.0_f32; n];
        gaussian_smooth_inplace(&mut data, dims, 1.5);
        for &v in &data {
            assert!(
                (v - 3.0).abs() < 1e-4,
                "expected 3.0 after smoothing, got {v}"
            );
        }
    }

    /// Gaussian smoothing reduces peak amplitude of a delta-like spike.
    #[test]
    fn gaussian_smooth_reduces_peak() {
        let dims = [9usize, 9, 9];
        let n = 9 * 9 * 9;
        let mut data = vec![0.0_f32; n];
        // Single spike in the centre.
        data[flat(4, 4, 4, 9, 9)] = 1.0;
        let peak_before = 1.0_f32;
        gaussian_smooth_inplace(&mut data, dims, 1.0);
        let peak_after = data.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            peak_after < peak_before,
            "peak should decrease after smoothing: {peak_after} >= {peak_before}"
        );
        // Total mass should be approximately conserved.
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "mass not conserved: sum = {sum}");
    }

    /// Identity composition: φ ∘ 0 = φ.
    #[test]
    fn compose_with_zero_is_identity() {
        let dims = [4usize, 4, 4];
        let n = 4 * 4 * 4;
        let phiz: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let phiy: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.01).collect();
        let phix = vec![0.5_f32; n];
        let zero = vec![0.0_f32; n];

        let (cz, cy, cx) = compose_fields(&phiz, &phiy, &phix, &zero, &zero, &zero, dims);

        for i in 0..n {
            assert!(
                (cz[i] - phiz[i]).abs() < 1e-4,
                "cz[{i}]: expected {}, got {}",
                phiz[i],
                cz[i]
            );
            assert!(
                (cy[i] - phiy[i]).abs() < 1e-4,
                "cy[{i}]: expected {}, got {}",
                phiy[i],
                cy[i]
            );
            assert!(
                (cx[i] - phix[i]).abs() < 1e-4,
                "cx[{i}]: expected {}, got {}",
                phix[i],
                cx[i]
            );
        }
    }

    /// Scaling-and-squaring of the zero field is the zero field.
    #[test]
    fn scaling_and_squaring_zero_field() {
        let dims = [4usize, 4, 4];
        let n = 4 * 4 * 4;
        let vz = vec![0.0_f32; n];
        let vy = vec![0.0_f32; n];
        let vx = vec![0.0_f32; n];
        let (phiz, phiy, phix) = scaling_and_squaring(&vz, &vy, &vx, dims, 6);
        for i in 0..n {
            assert!(
                phiz[i].abs() < 1e-5,
                "phiz[{i}] should be 0, got {}",
                phiz[i]
            );
            assert!(
                phiy[i].abs() < 1e-5,
                "phiy[{i}] should be 0, got {}",
                phiy[i]
            );
            assert!(
                phix[i].abs() < 1e-5,
                "phix[{i}] should be 0, got {}",
                phix[i]
            );
        }
    }

    /// For a small constant velocity field, exp(v) ≈ v (first-order approximation).
    #[test]
    fn scaling_and_squaring_small_velocity_approx_identity() {
        let dims = [4usize, 4, 4];
        let n = 4 * 4 * 4;
        // Small constant velocity (0.01 voxels in x).
        let vz = vec![0.0_f32; n];
        let vy = vec![0.0_f32; n];
        let vx = vec![0.01_f32; n];
        let (phiz, phiy, phix) = scaling_and_squaring(&vz, &vy, &vx, dims, 6);
        for i in 0..n {
            assert!(phiz[i].abs() < 1e-4, "phiz should be ~0, got {}", phiz[i]);
            assert!(phiy[i].abs() < 1e-4, "phiy should be ~0, got {}", phiy[i]);
            // exp(0.01) - 1 ≈ 0.01 for small values; tolerate 10% error.
            assert!(
                (phix[i] - 0.01).abs() < 0.002,
                "phix should be ~0.01, got {}",
                phix[i]
            );
        }
    }
}
