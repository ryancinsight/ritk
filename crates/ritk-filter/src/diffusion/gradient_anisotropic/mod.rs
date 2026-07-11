//! Gradient anisotropic diffusion filter (ITK `GradientNDAnisotropicDiffusionFunction`).
//!
//! # Mathematical Specification
//!
//! Perona-Malik anisotropic diffusion (Gerig et al. 1992), discretised exactly as
//! ITK's `GradientAnisotropicDiffusionImageFilter`. Each explicit-Euler step:
//!
//! 1. Recompute `m_K = avgGradMagSq · K² · −2`, where `avgGradMagSq` is the mean
//!    over all voxels of `Σ_d (central_d)²` (spacing-scaled central differences,
//!    ZeroFluxNeumann boundary). `m_K` is negative; conductance `c = exp(g²/m_K)`.
//! 2. For each voxel and dimension `i`, the conductance is evaluated on the
//!    **full gradient magnitude at the ±i half-pixel faces** — the face-normal
//!    forward/backward difference plus the averaged tangential central
//!    differences in the orthogonal dimensions:
//!
//!    ```text
//!    gms± = dx_face[i]² + Σ_{j≠i} ¼·(central_j + central_j^{±i})²
//!    c±   = exp(gms± / m_K)
//!    δ   += c⁺·dx_fwd[i] − c⁻·dx_bwd[i]
//!    ```
//! 3. `I_new = I + Δt·δ`.
//!
//! Derivatives are scaled by image spacing (ITK `UseImageSpacing = true`);
//! boundary conditions are ZeroFluxNeumann (index-clamp). Verified against
//! SimpleITK `GradientAnisotropicDiffusion` on cthead1 to ≈ 6 × 10⁻⁸ relative
//! (f64-vs-f32 round-off).
//!
//! # Distinction from `AnisotropicDiffusionFilter` (Perona-Malik)
//!
//! `perona_malik.rs` is a simpler scheme (single-direction-difference conductance,
//! optional quadratic conductance) with no per-iteration `K` rescaling; it is the
//! `"quadratic"` kind of the Python `anisotropic_diffusion` binding, while the
//! `"exponential"` default dispatches to this ITK-exact filter.
//!
//! # Stability
//!
//! Explicit Euler stability for the 6-neighbour Laplacian in 3-D requires
//! `Δt ≤ 1/(2·D)` (in the image coordinate frame); ITK warns when exceeded.
//!
//! # Invariants
//!
//! - Constant image: `avgGradMagSq = 0 → m_K = 0 → c = 0 → δ = 0`; unchanged.
//!
//! # References
//! - Gerig, G., Kübler, O., Kikinis, R. & Jolesz, F. A. (1992). Nonlinear
//!   anisotropic filtering of MRI data. *IEEE Trans. Med. Imag.* 11(2):221–232.
//! - ITK `itkGradientNDAnisotropicDiffusionFunction.hxx`,
//!   `itkScalarAnisotropicDiffusionFunction.hxx`.

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};

// ── Public types ──────────────────────────────────────────────────────────────

/// Configuration for [`GradientAnisotropicDiffusionFilter`].
#[derive(Debug, Clone)]
pub struct GradientDiffusionConfig {
    /// Number of explicit Euler time steps.
    ///
    /// ITK default: 5.
    pub num_iterations: usize,
    /// Time step Δt.
    ///
    /// Must satisfy `Δt ≤ 1/6` for stability in 3-D.
    /// ITK default: 0.125.
    pub time_step: f32,
    /// Conductance K.
    ///
    /// Controls the intensity-difference threshold below which diffusion is
    /// strong.  Larger K → more isotropic smoothing.
    /// ITK default: 1.0.
    pub conductance: f32,
}

impl Default for GradientDiffusionConfig {
    fn default() -> Self {
        Self {
            num_iterations: 5,
            time_step: 0.125,
            conductance: 1.0,
        }
    }
}

/// Gradient anisotropic diffusion filter — ITK `GradientAnisotropicDiffusionImageFilter`.
///
/// Reduces noise while preserving edges. Reproduces ITK's output (face-gradient
/// conductance + per-iteration average-gradient-magnitude `K` rescaling); see the
/// module docs for the full discretisation.
#[derive(Debug, Clone)]
pub struct GradientAnisotropicDiffusionFilter {
    /// Algorithm configuration.
    pub config: GradientDiffusionConfig,
}

impl GradientAnisotropicDiffusionFilter {
    /// Create a filter with the given configuration.
    #[inline]
    pub fn new(config: GradientDiffusionConfig) -> Self {
        Self { config }
    }

    /// Apply the filter to `image`, returning a diffused copy.
    ///
    /// # Errors
    /// Returns an error if the image tensor cannot be interpreted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec(image)?;
        let vals = &vals_vec;

        let sp = image.spacing();
        let spacing = [sp[0], sp[1], sp[2]];
        let result = diffuse(vals, dims, spacing, &self.config);

        Ok(rebuild(result, dims, image))
    }

    /// Apply the diffusion kernel to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let spacing = image.spacing();
        ritk_image::native::Image::from_flat_on(
            diffuse(
                image.data_slice()?,
                image.shape(),
                [spacing[0], spacing[1], spacing[2]],
                &self.config,
            ),
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }
}

// ── Core computation (ITK GradientNDAnisotropicDiffusionFunction) ─────────────

use super::{central_diff as central, clamp_at as at};

/// Perform `config.num_iterations` explicit Euler steps of the ITK
/// `GradientNDAnisotropicDiffusionFunction` on `data`.
///
/// Each step:
/// 1. Recompute `m_K = avgGradMagSq · K² · −2`, where `avgGradMagSq` is the mean
///    over all voxels of `Σ_d (central_d)²` (spacing-scaled central differences).
/// 2. For every voxel and dimension `i`, evaluate the conductance on the *full*
///    gradient magnitude at the `±i` half-pixel faces — the face-normal
///    forward/backward difference plus the averaged tangential central
///    differences in the orthogonal dimensions:
///    `c± = exp((dx_face² + Σ_{j≠i} ¼(central_j + central_j^{±i})²) / m_K)` —
///    then accumulate the divergence `δ += c⁺·dx_fwd − c⁻·dx_bwd`.
/// 3. `I_new = I + Δt·δ`.
///
/// Boundary conditions are ZeroFluxNeumann (index-clamp). Derivatives are scaled
/// by the image spacing, matching ITK's default `UseImageSpacing = true`.
#[allow(clippy::needless_range_loop)]
fn diffuse(
    data: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    config: &GradientDiffusionConfig,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut cur: Vec<f32> = data.to_vec();
    let mut nxt: Vec<f32> = vec![0.0_f32; n];

    let dt = config.time_step as f64;
    let cond = config.conductance as f64;
    let inv_sp = [1.0 / spacing[0], 1.0 / spacing[1], 1.0 / spacing[2]];
    let inv_2sp = [0.5 / spacing[0], 0.5 / spacing[1], 0.5 / spacing[2]];

    for _iter in 0..config.num_iterations {
        // 1. Average gradient magnitude squared over the whole image.
        let mut sum_gms = 0.0_f64;
        for z in 0..nz as isize {
            for y in 0..ny as isize {
                for x in 0..nx as isize {
                    let g0 = central(&cur, dims, inv_2sp, 0, z, y, x);
                    let g1 = central(&cur, dims, inv_2sp, 1, z, y, x);
                    let g2 = central(&cur, dims, inv_2sp, 2, z, y, x);
                    sum_gms += g0 * g0 + g1 * g1 + g2 * g2;
                }
            }
        }
        let avg_gms = sum_gms / n as f64;
        // m_K is negative; the conductance is exp(gradMag² / m_K) ∈ (0, 1].
        let m_k = avg_gms * cond * cond * -2.0;

        // 2. Diffusion update.
        for z in 0..nz as isize {
            for y in 0..ny as isize {
                for x in 0..nx as isize {
                    let center = at(&cur, dims, z, y, x);
                    let dc = [
                        central(&cur, dims, inv_2sp, 0, z, y, x),
                        central(&cur, dims, inv_2sp, 1, z, y, x),
                        central(&cur, dims, inv_2sp, 2, z, y, x),
                    ];

                    let mut delta = 0.0_f64;
                    for i in 0..3 {
                        // Face-normal forward/backward differences in dimension i.
                        let (fp, fm) = match i {
                            0 => (at(&cur, dims, z + 1, y, x), at(&cur, dims, z - 1, y, x)),
                            1 => (at(&cur, dims, z, y + 1, x), at(&cur, dims, z, y - 1, x)),
                            _ => (at(&cur, dims, z, y, x + 1), at(&cur, dims, z, y, x - 1)),
                        };
                        let fwd = (fp - center) * inv_sp[i];
                        let bwd = (center - fm) * inv_sp[i];

                        // Tangential gradient contributions at the ± i faces:
                        // the central difference in each orthogonal dim j averaged
                        // between the centre and the ± i neighbour.
                        let mut accum_f = 0.0_f64;
                        let mut accum_b = 0.0_f64;
                        for j in 0..3 {
                            if j == i {
                                continue;
                            }
                            let (aug, dim_) = match i {
                                0 => (
                                    central(&cur, dims, inv_2sp, j, z + 1, y, x),
                                    central(&cur, dims, inv_2sp, j, z - 1, y, x),
                                ),
                                1 => (
                                    central(&cur, dims, inv_2sp, j, z, y + 1, x),
                                    central(&cur, dims, inv_2sp, j, z, y - 1, x),
                                ),
                                _ => (
                                    central(&cur, dims, inv_2sp, j, z, y, x + 1),
                                    central(&cur, dims, inv_2sp, j, z, y, x - 1),
                                ),
                            };
                            let sf = dc[j] + aug;
                            let sb = dc[j] + dim_;
                            accum_f += 0.25 * sf * sf;
                            accum_b += 0.25 * sb * sb;
                        }

                        let (cx, cxd) = if m_k == 0.0 {
                            (0.0, 0.0)
                        } else {
                            (
                                ((fwd * fwd + accum_f) / m_k).exp(),
                                ((bwd * bwd + accum_b) / m_k).exp(),
                            )
                        };
                        delta += cx * fwd - cxd * bwd;
                    }

                    let p = (z as usize) * ny * nx + (y as usize) * nx + (x as usize);
                    nxt[p] = (center + dt * delta) as f32;
                }
            }
        }
        // `nxt` holds every voxel's updated value this sweep; swap the buffers
        // to commit it in O(1) and reuse the old `cur` as next sweep's scratch,
        // avoiding an N-element memcpy per iteration. Bit-identical.
        std::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests;
