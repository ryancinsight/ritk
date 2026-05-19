//! Coherence-Enhancing Diffusion (CED) filter (Weickert 1999).
//!
//! # Mathematical Specification
//!
//! Coherence-Enhancing Diffusion is an anisotropic diffusion filter that
//! smooths images **along** coherent structures (edges, ridges, fibres) while
//! preserving them **across** their orientation. It uses the structure tensor
//! to steer the diffusion tensor.
//!
//! ## Structure Tensor
//!
//! Given image I(x), the structure tensor J_ρ at integration scale ρ is:
//!
//! J_ρ = G_ρ * (∇I · ∇Iᵀ)
//!
//! where G_ρ is a Gaussian of standard deviation ρ and * denotes convolution.
//!
//! In 3-D the 6 independent components are:
//!
//! J_11 = G_ρ * (I_z²),  J_22 = G_ρ * (I_y²),  J_33 = G_ρ * (I_x²)
//! J_12 = G_ρ * (I_z·I_y),  J_13 = G_ρ * (I_z·I_x),  J_23 = G_ρ * (I_y·I_x)
//!
//! ## Diffusion Tensor
//!
//! Let λ₁ ≤ λ₂ ≤ λ₃ be the eigenvalues of J_ρ with eigenvectors e₁, e₂, e₃.
//! The diffusion tensor D shares the eigenvectors but replaces eigenvalues:
//!
//! α₁ = α + (1 − α) · (1 − exp(−C · (λ₃ − λ₁)² / (λ₃² + ε)))   (coherence dir)
//! α₂ = α + (1 − α) · (1 − exp(−C · (λ₂ − λ₁)² / (λ₃² + ε)))   (intermediate)
//! α₃ = α                                                        (edge dir)
//!
//! α is the flat-region smoothing parameter, C the contrast parameter.
//!
//! Along e₁ (smallest eigenvalue = coherence direction) diffusion is maximal;
//! along e₃ (largest eigenvalue = edge direction) diffusion is minimal.
//!
//! ## PDE
//!
//! ∂I/∂t = div(D · ∇I)
//!
//! Discretised with explicit Euler on the 3-D grid. Stability requires
//! Δt ≤ 1 / (2·D·max(α_i)).
//!
//! # Complexity
//!
//! Per iteration: O(N·k) where N is the number of voxels and k is the
//! Gaussian kernel size (radius ⌈3ρ⌉ along each axis). The eigendecomposition
//! is O(1) per voxel (analytical closed-form).
//!
//! # Invariants
//!
//! - Constant image: ∇I = 0 → J_ρ = 0 → D = α·I → div(D·∇I) = 0 → unchanged.
//! - Linear image: ∇I = const → J_ρ rank-1 → λ₂ = λ₁ = 0 → α₁ = α₂ = α
//!   (no excess diffusion), div(D·∇I) = 0 → unchanged.
//!
//! # References
//!
//! - Weickert, J. (1999). *Coherence-Enhancing Diffusion Filtering*.
//!   Int. J. Comput. Vis. 31(2/3):111–127.
//! - Weickert, J. (1998). *Anisotropic Diffusion in Image Processing*.
//!   Teubner, Stuttgart.

use crate::filter::ops::{extract_vec_infallible, rebuild};
use crate::image::Image;
use burn::tensor::backend::Backend;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::Arc;

// ── Public types ──────────────────────────────────────────────────────────────

/// CED configuration.
#[derive(Debug, Clone)]
pub struct CoherenceConfig {
    /// Gaussian sigma for structure tensor smoothing (integration scale ρ).
    /// Default: 3.0.
    pub sigma: f64,
    /// Contrast parameter C. Default: 1e-10.
    pub contrast: f64,
    /// Smoothing parameter α in flat regions. Default: 0.001.
    pub alpha: f64,
    /// Time step Δt. Default: 0.0625 (1/16).
    pub time_step: f64,
    /// Number of iterations. Default: 10.
    pub n_iterations: usize,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            sigma: 3.0,
            contrast: 1e-10,
            alpha: 0.001,
            time_step: 0.0625,
            n_iterations: 10,
        }
    }
}

/// Coherence-Enhancing Diffusion filter.
///
/// Smooths images along coherent structures while preserving them across the
/// structure orientation, using the structure tensor to drive anisotropic
/// diffusion (Weickert 1999).
#[derive(Debug, Clone)]
pub struct CoherenceEnhancingDiffusionFilter {
    /// Algorithm configuration.
    pub config: CoherenceConfig,
}

impl CoherenceEnhancingDiffusionFilter {
    /// Create a filter with the given configuration.
    #[inline]
    pub fn new(config: CoherenceConfig) -> Self {
        Self { config }
    }

    /// Apply the CED filter to a 3-D image, returning a diffused copy.
    ///
    /// For images of dimension D < 3, the result is identical to the input
    /// (CED is defined only for 3-D volumes).
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let (vals_vec, dims) = extract_vec_infallible(image);

        let result = if D >= 3 && dims.iter().all(|&d| d >= 3) {
            // Extract the leading 3 dimensions.
            let d3 = [dims[0], dims[1], dims[2]];
            let n3 = d3[0] * d3[1] * d3[2];
            let vals3: Vec<f64> = vals_vec[..n3].iter().map(|&v| v as f64).collect();
            let out3 = ced_diffuse(&vals3, d3, &self.config);
            // Write back, converting f64 → f32.
            let mut result = vals_vec;
            for i in 0..n3 {
                result[i] = out3[i] as f32;
            }
            result
        } else {
            // CED undefined for < 3-D or tiny volumes; return input unchanged.
            vals_vec
        };

        rebuild(result, dims, image)
    }
}

// ── Core computation ──────────────────────────────────────────────────────────

/// Numerical floor for the denominator in diffusion eigenvalue construction.
const EPS: f64 = 1e-20;

/// Run explicit Euler CED for the requested number of iterations.
///
/// All arithmetic in f64; caller converts to f32 at output.
fn ced_diffuse(data: &[f64], dims: [usize; 3], config: &CoherenceConfig) -> Vec<f64> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut cur = data.to_vec();

    // Pre-build the 1-D Gaussian kernel for structure-tensor smoothing.
    let kernel = make_gaussian_kernel_1d(config.sigma);
    let kern = Arc::new(kernel);

    for _ in 0..config.n_iterations {
        // ── Step 1: gradient via central differences ────────────────────
        let grad = compute_gradient(&cur, dims);

        // ── Step 2: structure tensor products ───────────────────────────
        let st_products = compute_structure_tensor_products(&grad, dims);

        // ── Step 3: Gaussian smoothing of structure tensor ──────────────
        let st_smooth = smooth_structure_tensor(&st_products, dims, &kern);

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

/// Gradient buffer: [gz, gy, gx] stored contiguously, each of length n.
struct Gradient {
    gz: Vec<f64>,
    gy: Vec<f64>,
    gx: Vec<f64>,
}

/// Central-difference gradient with Neumann (replicate) boundary conditions.
fn compute_gradient(data: &[f64], dims: [usize; 3]) -> Gradient {
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
                    (data[iz_p * ny * nx + iy * nx + ix]
                        - data[iz_m * ny * nx + iy * nx + ix])
                        / dz
                } else {
                    0.0
                };
                gy[i] = if dy > 0.0 {
                    (data[iz * ny * nx + iy_p * nx + ix]
                        - data[iz * ny * nx + iy_m * nx + ix])
                        / dy
                } else {
                    0.0
                };
                gx[i] = if dx > 0.0 {
                    (data[iz * ny * nx + iy * nx + ix_p]
                        - data[iz * ny * nx + iy * nx + ix_m])
                        / dx
                } else {
                    0.0
                };
            }
        }
    }

    Gradient { gz, gy, gx }
}

// ── Structure tensor products ─────────────────────────────────────────────────

/// 6 independent components of the outer product ∇I·∇Iᵀ at each voxel.
///
/// Layout per voxel: [I_z², I_z·I_y, I_z·I_x, I_y², I_y·I_x, I_x²]
struct StructureTensorProducts {
    data: Vec<[f64; 6]>,
}

/// Compute the 6 outer-product components at every voxel.
fn compute_structure_tensor_products(grad: &Gradient, dims: [usize; 3]) -> StructureTensorProducts {
    let n = dims[0] * dims[1] * dims[2];
    let mut st = StructureTensorProducts {
        data: vec![[0.0f64; 6]; n],
    };

    st.data
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out)| {
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

// ── Gaussian smoothing ────────────────────────────────────────────────────────

/// Build a normalised 1-D Gaussian kernel of radius ⌈3·σ⌉.
///
/// The kernel is symmetric and sums to 1.0.
fn make_gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return vec![1.0];
    }
    let radius = (3.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0f64;
    for k in -(radius as i64)..=radius as i64 {
        let x = k as f64;
        let w = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(w);
        sum += w;
    }
    for w in &mut kernel {
        *w /= sum;
    }
    kernel
}

/// Separable Gaussian smoothing along a single axis.
///
/// `axis` is 0 (z), 1 (y), or 2 (x). Boundary: replicate (Neumann).
fn gaussian_smooth_1d(input: &[f64], dims: [usize; 3], axis: usize, kernel: &[f64]) -> Vec<f64> {
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
fn smooth_structure_tensor(
    st: &StructureTensorProducts,
    dims: [usize; 3],
    kernel: &Arc<Vec<f64>>,
) -> Vec<[f64; 6]> {
    let n = dims[0] * dims[1] * dims[2];

    // Process each component independently (embarrassingly parallel).
    let smoothed_components: Vec<Vec<f64>> = (0..6)
        .into_par_iter()
        .map(|c| {
            // Extract component c as a flat buffer.
            let mut buf: Vec<f64> = st.data.iter().map(|v| v[c]).collect();
            // Separable smoothing along z, y, x.
            buf = gaussian_smooth_1d(&buf, dims, 0, kernel);
            buf = gaussian_smooth_1d(&buf, dims, 1, kernel);
            buf = gaussian_smooth_1d(&buf, dims, 2, kernel);
            buf
        })
        .collect();

    // Re-interleave into [f64; 6] per voxel.
    let mut out = vec![[0.0f64; 6]; n];
    for i in 0..n {
        for c in 0..6 {
            out[i][c] = smoothed_components[c][i];
        }
    }
    out
}

// ── Eigendecomposition (3×3 symmetric) ────────────────────────────────────────

/// Eigenvalues and eigenvectors of a 3×3 symmetric matrix.
///
/// Eigenvalues are sorted ascending: λ₁ ≤ λ₂ ≤ λ₃.
/// Columns of `eigenvecs` are the corresponding eigenvectors.
struct EigenDecomp {
    eigenvalues: [f64; 3],
    eigenvecs: [[f64; 3]; 3], // eigenvecs[k] = k-th eigenvector
}

/// Analytical eigenvalue decomposition of a 3×3 symmetric matrix.
///
/// Uses the trigonometric method (Smith 1961; Kopp 2008, arXiv:physics/0610206).
/// For degenerate eigenvalues, eigenvectors are selected by Gram-Schmidt
/// orthogonalisation in the invariant subspace.
///
/// Input layout: `h = [J_11, J_12, J_13, J_22, J_23, J_33]`
/// (upper triangle, row-major).
///
/// ```text
/// M = | h[0] h[1] h[2] |
///     | h[1] h[3] h[4] |
///     | h[2] h[4] h[5] |
/// ```
fn eigen_3x3_symmetric(h: [f64; 6]) -> EigenDecomp {
    let m00 = h[0];
    let m01 = h[1];
    let m02 = h[2];
    let m11 = h[3];
    let m12 = h[4];
    let m22 = h[5];

    // Off-diagonal Frobenius contribution.
    let p1 = m01 * m01 + m02 * m02 + m12 * m12;

    if p1 == 0.0 {
        // Diagonal matrix: eigenvalues are the diagonal entries.
        // Eigenvectors are the standard basis vectors.
        let mut eigs = [m00, m11, m22];
        let mut indices = [0usize, 1, 2];
        indices.sort_unstable_by(|&a, &b| eigs[a].partial_cmp(&eigs[b]).unwrap_or(std::cmp::Ordering::Equal));
        let sorted = [eigs[indices[0]], eigs[indices[1]], eigs[indices[2]]];
        let mut vecs = [[0.0f64; 3]; 3];
        for (k, &idx) in indices.iter().enumerate() {
            vecs[k][idx] = 1.0;
        }
        return EigenDecomp {
            eigenvalues: sorted,
            eigenvecs: vecs,
        };
    }

    let q = (m00 + m11 + m22) / 3.0;
    let p2 = (m00 - q) * (m00 - q)
        + (m11 - q) * (m11 - q)
        + (m22 - q) * (m22 - q)
        + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();

    if p < f64::EPSILON {
        // Numerically a scalar multiple of identity.
        return EigenDecomp {
            eigenvalues: [q, q, q],
            eigenvecs: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
    }

    // B = (M − q·I) / p
    let b00 = (m00 - q) / p;
    let b01 = m01 / p;
    let b02 = m02 / p;
    let b11 = (m11 - q) / p;
    let b12 = m12 / p;
    let b22 = (m22 - q) / p;

    // det(B) via cofactor expansion along the first row.
    let det_b = b00 * (b11 * b22 - b12 * b12)
        - b01 * (b01 * b22 - b12 * b02)
        + b02 * (b01 * b12 - b11 * b02);

    // r = det(B)/2, clamped to [−1, 1] for numerical safety.
    let r = (det_b / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;

    // Three eigenvalues before sorting.
    let eig_a = q + 2.0 * p * phi.cos();
    let eig_c = q + 2.0 * p * (phi + 2.0 * PI / 3.0).cos();
    // Trace identity: eig_b = 3q − eig_a − eig_c.
    let eig_b = 3.0 * q - eig_a - eig_c;

    let eigs_unsorted = [eig_a, eig_b, eig_c];
    let mut idx = [0usize, 1, 2];
    idx.sort_unstable_by(|&a, &b| {
        eigs_unsorted[a]
            .partial_cmp(&eigs_unsorted[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let eigenvalues = [
        eigs_unsorted[idx[0]],
        eigs_unsorted[idx[1]],
        eigs_unsorted[idx[2]],
    ];

    // ── Eigenvectors ──────────────────────────────────────────────────
    // (M − λI) v = 0  →  solve via cross products of rows.
    let eigenvecs = eigenvectors_3x3_symmetric(h, eigenvalues);

    EigenDecomp {
        eigenvalues,
        eigenvecs,
    }
}

/// Compute eigenvectors of a 3×3 symmetric matrix given its eigenvalues.
///
/// For each eigenvalue λ, the eigenvector is the cross product of two
/// rows of (M − λI). For degenerate eigenvalues, Gram-Schmidt is applied
/// within the invariant subspace.
fn eigenvectors_3x3_symmetric(h: [f64; 6], eigenvalues: [f64; 3]) -> [[f64; 3]; 3] {
    let m00 = h[0];
    let m01 = h[1];
    let m02 = h[2];
    let m11 = h[3];
    let m12 = h[4];
    let m22 = h[5];

    let mut vecs = [[0.0f64; 3]; 3];

    for (k, &lam) in eigenvalues.iter().enumerate() {
        // (M − λI) rows:
        let r0 = [m00 - lam, m01, m02];
        let r1 = [m01, m11 - lam, m12];
        let r2 = [m02, m12, m22 - lam];

        // Cross products of row pairs.
        let c01 = cross3(r0, r1);
        let c02 = cross3(r0, r2);
        let c12 = cross3(r1, r2);

        let n01 = norm3(c01);
        let n02 = norm3(c02);
        let n12 = norm3(c12);

        // Select the cross product with the largest norm (best conditioning).
        let v = if n01 >= n02 && n01 >= n12 {
            if n01 > 0.0 {
                scale3(c01, 1.0 / n01)
            } else {
                // Fallback: try cross product with canonical axis.
                let c = cross3(r0, [1.0, 0.0, 0.0]);
                let nc = norm3(c);
                if nc > 0.0 {
                    scale3(c, 1.0 / nc)
                } else {
                    [0.0, 1.0, 0.0]
                }
            }
        } else if n02 >= n12 {
            scale3(c02, 1.0 / n02)
        } else {
            scale3(c12, 1.0 / n12)
        };

        vecs[k] = v;
    }

    // Handle degenerate eigenvalues: orthogonalise within each subspace.
    // Since eigenvalues are sorted ascending, we process in order.
    let degen_tol = f64::max(1e-12, f64::abs(eigenvalues[2]) * 1e-10);

    // Check λ₁ ≈ λ₂.
    if (eigenvalues[1] - eigenvalues[0]).abs() < degen_tol {
        // Orthogonalise v₂ against v₁.
        vecs[1] = orthogonalise_against(vecs[1], vecs[0]);
    }
    // Check λ₂ ≈ λ₃.
    if (eigenvalues[2] - eigenvalues[1]).abs() < degen_tol {
        // Orthogonalise v₃ against v₂ (and transitively v₁).
        vecs[2] = orthogonalise_against(vecs[2], vecs[1]);
        vecs[2] = orthogonalise_against(vecs[2], vecs[0]);
    }
    // If λ₁ ≈ λ₃ (implies all equal), orthogonalise v₃ against v₁ too.
    if (eigenvalues[2] - eigenvalues[0]).abs() < degen_tol
        && (eigenvalues[2] - eigenvalues[1]).abs() >= degen_tol
    {
        vecs[2] = orthogonalise_against(vecs[2], vecs[0]);
    }

    vecs
}

// ── Small vector helpers ──────────────────────────────────────────────────────

#[inline(always)]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
fn norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

#[inline(always)]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline(always)]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Orthogonalise `v` against `u` (assumed unit). Returns a unit vector.
#[inline(always)]
fn orthogonalise_against(v: [f64; 3], u: [f64; 3]) -> [f64; 3] {
    let proj = dot3(v, u);
    let mut w = [v[0] - proj * u[0], v[1] - proj * u[1], v[2] - proj * u[2]];
    let n = norm3(w);
    if n > 1e-15 {
        w = scale3(w, 1.0 / n);
    } else {
        // v was parallel to u; pick an arbitrary perpendicular.
        w = if u[0].abs() < 0.9 {
            let c = cross3(u, [1.0, 0.0, 0.0]);
            scale3(c, 1.0 / norm3(c))
        } else {
            let c = cross3(u, [0.0, 1.0, 0.0]);
            scale3(c, 1.0 / norm3(c))
        };
    }
    w
}

// ── Diffusion tensor ──────────────────────────────────────────────────────────

/// Construct the diffusion tensor D at a single voxel from the structure tensor.
///
/// Returns the 6 independent components of D in the same layout as the
/// structure tensor: [D_11, D_12, D_13, D_22, D_23, D_33].
///
/// Eigenvalue assignment (Weickert 1999):
///   α₁ = α + (1 − α) · (1 − exp(−C · (λ₃ − λ₁)² / (λ₃² + ε)))   [coherence dir]
///   α₂ = α + (1 − α) · (1 − exp(−C · (λ₂ − λ₁)² / (λ₃² + ε)))   [intermediate]
///   α₃ = α                                                         [edge dir]
fn diffusion_tensor(st: [f64; 6], alpha: f64, contrast: f64) -> [f64; 6] {
    let decomp = eigen_3x3_symmetric(st);
    let [lam1, lam2, lam3] = decomp.eigenvalues;
    let [e1, e2, e3] = decomp.eigenvecs;

    let lam3_sq = lam3 * lam3;

    // Coherence measure for the primary coherence direction.
    let diff_31 = lam3 - lam1;
    let exponent1 = -contrast * diff_31 * diff_31 / (lam3_sq + EPS);
    let alpha1 = alpha + (1.0 - alpha) * (1.0 - exponent1.exp());

    // Intermediate direction.
    let diff_21 = lam2 - lam1;
    let exponent2 = -contrast * diff_21 * diff_21 / (lam3_sq + EPS);
    let alpha2 = alpha + (1.0 - alpha) * (1.0 - exponent2.exp());

    // Edge direction: minimal diffusion.
    let alpha3 = alpha;

    // Reconstruct D = α₁·e₁·e₁ᵀ + α₂·e₂·e₂ᵀ + α₃·e₃·e₃ᵀ
    let mut d = [0.0f64; 6];
    for (alpha_i, e) in [(alpha1, e1), (alpha2, e2), (alpha3, e3)] {
        // Outer product: e · eᵀ, upper triangle.
        d[0] += alpha_i * e[0] * e[0]; // D_11
        d[1] += alpha_i * e[0] * e[1]; // D_12
        d[2] += alpha_i * e[0] * e[2]; // D_13
        d[3] += alpha_i * e[1] * e[1]; // D_22
        d[4] += alpha_i * e[1] * e[2]; // D_23
        d[5] += alpha_i * e[2] * e[2]; // D_33
    }

    d
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
fn compute_divergence(
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
        .par_iter()
        .map(|&st| diffusion_tensor(st, alpha, contrast))
        .collect();

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
                    let diff = data[j] - data[i]; // value at iz−1 minus value at iz (negative)
                    // The face between iz−1 and iz: flux_z at face(i−1/2)
                    // = D_face · (I[iz−1] − I[iz]) in the z-component
                    // = D_11 · diff  (diff is negative, so this subtracts from delta)
                    delta += d_face[0] * diff;
                }

                // ── +y / −y faces ──────────────────────────────────────
                if iy + 1 < ny {
                    let j = iz * ny * nx + (iy + 1) * nx + ix;
                    let d_face = avg_tensor(d_tensors[i], d_tensors[j]);
                    let diff = data[j] - data[i];
                    // y-component of flux = D_22 · diff  (since D_22 = d_face[3])
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
                    // x-component of flux = D_33 · diff  (since D_33 = d_face[5])
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

/// Element-wise average of two symmetric 3×3 tensor component vectors.
#[inline(always)]
fn avg_tensor(a: [f64; 6], b: [f64; 6]) -> [f64; 6] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
        (a[3] + b[3]) * 0.5,
        (a[4] + b[4]) * 0.5,
        (a[5] + b[5]) * 0.5,
    ]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_coherence.rs"]
mod tests;
