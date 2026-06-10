//! Perona-Malik anisotropic diffusion filter.
//!
//! # Mathematical Specification
//!
//! The Perona-Malik anisotropic diffusion PDE (Perona & Malik 1990):
//!
//! ∂I/∂t = div(c(|∇I|) · ∇I)
//!
//! where the conductance function c controls the amount of diffusion at each
//! location:
//!
//! - Exponential: c(s) = exp(−(s/K)²)
//! - Quadratic: c(s) = 1 / (1 + (s/K)²)
//!
//! Both functions satisfy c(0) = 1 (maximum diffusion where gradient is zero)
//! and c(s) → 0 as s → ∞ (no diffusion across strong edges).
//!
//! # Discretisation
//!
//! Explicit Euler finite differences on a 3-D regular grid with spacing
//! (sz, sy, sx). For each voxel (iz, iy, ix), six nearest-neighbour fluxes
//! are computed:
//!
//! Δ±z I = I[iz±1, iy, ix] − I[iz, iy, ix] (zero at boundaries → Neumann BC)
//! flux±z = c(|Δ±z I| / sz) · Δ±z I / sz²
//!
//! Update:
//! I_new = I + Δt · (flux+z + flux−z + flux+y + flux−y + flux+x + flux−x)
//!
//! Stability condition for explicit Euler in 3-D: Δt ≤ 1/6 (unit spacing).
//! The default time-step 1/16 provides a safety factor of ~2.67.
//!
//! # Reference
//! Perona, P. & Malik, J. (1990). Scale-space and edge detection using
//! anisotropic diffusion. *IEEE Trans. Pattern Anal. Mach. Intell.*
//! 12(7):629–639. doi:10.1109/34.56205

use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use burn::tensor::backend::Backend;

// ── ZST conductance strategy ─────────────────────────────────────────────────

/// Trait for diffusion conductance (edge-stopping) functions.
///
/// Each implementation is a zero-sized type so that the compiler monomorphises
/// the diffusion loop with the conductance call fully inlined and the match
/// branch eliminated — zero runtime overhead versus a hand-written variant.
pub trait ConductanceKernel: Default {
    /// Evaluate the conductance function at gradient magnitude `s` with
    /// conductance parameter `k`.
    fn conductance(s: f32, k: f32) -> f32;
}

/// c(s) = exp(−(s/K)²) — Perona-Malik option 1.
///
/// Favours high-contrast edges over low-contrast ones.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExponentialConductance;

impl ConductanceKernel for ExponentialConductance {
    #[inline(always)]
    fn conductance(s: f32, k: f32) -> f32 {
        (-(s / k) * (s / k)).exp()
    }
}

/// c(s) = 1 / (1 + (s/K)²) — Perona-Malik option 2.
///
/// Favours wide regions over smaller ones.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuadraticConductance;

impl ConductanceKernel for QuadraticConductance {
    #[inline(always)]
    fn conductance(s: f32, k: f32) -> f32 {
        let r = s / k;
        1.0 / (1.0 + r * r)
    }
}

// ── Backward-compatible enum ─────────────────────────────────────────────────

/// Choice of conductance (edge-stopping) function.
///
/// Preserved for API compatibility. Internally converted to the
/// corresponding ZST strategy type before computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConductanceFunction {
    /// c(s) = exp(−(s/K)²) — Perona-Malik option 1.
    ///
    /// Favours high-contrast edges over low-contrast ones.
    #[default]
    Exponential,
    /// c(s) = 1 / (1 + (s/K)²) — Perona-Malik option 2.
    ///
    /// Favours wide regions over smaller ones.
    Quadratic,
}

/// Configuration for the Perona-Malik anisotropic diffusion filter.
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Number of explicit Euler time steps to perform.
    pub num_iterations: usize,
    /// Time step Δt. Must satisfy Δt ≤ 1/(2·D) where D is the number of
    /// spatial dimensions. Default: 0.0625 = 1/16 (safe for 3-D).
    pub time_step: f32,
    /// Conductance parameter K. Controls the gradient threshold below which
    /// diffusion is strong. Larger K → more smoothing across edges.
    pub conductance: f32,
    /// Which conductance function to use.
    pub function: ConductanceFunction,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            num_iterations: 20,
            time_step: 0.0625,
            conductance: 3.0,
            function: ConductanceFunction::Exponential,
        }
    }
}

// ── Generic filter struct ────────────────────────────────────────────────────

/// Anisotropic diffusion filter (Perona & Malik 1990).
///
/// Reduces noise while preserving edges by using a conductance function that
/// suppresses diffusion in regions of high gradient magnitude. The conductance
/// function is selected at compile time via the type parameter `K`, ensuring
/// zero-cost monomorphisation.
#[derive(Debug, Clone)]
pub struct AnisotropicDiffusionFilter<K: ConductanceKernel> {
    /// Algorithm configuration.
    pub config: DiffusionConfig,
    /// Phantom for the compile-time conductance strategy.
    _kernel: core::marker::PhantomData<K>,
}

impl<K: ConductanceKernel> AnisotropicDiffusionFilter<K> {
    /// Create a filter with the given configuration.
    pub fn new(config: DiffusionConfig) -> Self {
        Self {
            config,
            _kernel: core::marker::PhantomData,
        }
    }

    /// Apply the filter to `image`, returning a smoothed copy.
    ///
    /// # Errors
    /// Returns an error if the image tensor cannot be converted to `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec(image)?;
        let vals = &vals_vec;

        // Spacing from the image metadata (physical units); used to normalise
        // gradient differences so that conductance responds to physical gradient
        // magnitude regardless of voxel size.
        let spacing = [
            image.spacing()[0] as f32,
            image.spacing()[1] as f32,
            image.spacing()[2] as f32,
        ];

        let result = diffuse::<K>(vals, dims, spacing, &self.config);

        Ok(rebuild(result, dims, image))
    }
}

// ── Backward-compatible non-generic entry point ──────────────────────────────

impl DiffusionConfig {
    /// Apply anisotropic diffusion using the conductance function selected in
    /// `self.function`, dispatching to the appropriate monomorphised filter.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec(image)?;
        let spacing = [
            image.spacing()[0] as f32,
            image.spacing()[1] as f32,
            image.spacing()[2] as f32,
        ];
        let result = match self.function {
            ConductanceFunction::Exponential => {
                diffuse::<ExponentialConductance>(&vals_vec, dims, spacing, self)
            }
            ConductanceFunction::Quadratic => {
                diffuse::<QuadraticConductance>(&vals_vec, dims, spacing, self)
            }
        };
        Ok(rebuild(result, dims, image))
    }
}

// ── Core computation ─────────────────────────────────────────────────────────

/// Run the anisotropic diffusion PDE for the requested number of iterations.
///
/// Neumann (zero-flux) boundary conditions: fluxes across the image boundary
/// are set to zero by clamping neighbour indices to valid range and treating
/// the difference as zero when the neighbour index equals the current index.
fn diffuse<K: ConductanceKernel>(
    data: &[f32],
    dims: [usize; 3],
    spacing: [f32; 3],
    config: &DiffusionConfig,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut cur = data.to_vec();
    let mut nxt = vec![0.0_f32; n];

    let sz2 = spacing[0] * spacing[0];
    let sy2 = spacing[1] * spacing[1];
    let sx2 = spacing[2] * spacing[2];

    let sz = spacing[0];
    let sy = spacing[1];
    let sx = spacing[2];

    let dt = config.time_step;
    let k = config.conductance;

    let idx = |iz: usize, iy: usize, ix: usize| -> usize { iz * ny * nx + iy * nx + ix };

    for _iter in 0..config.num_iterations {
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let flat = idx(iz, iy, ix);
                    let v = cur[flat];

                    // ── z-axis fluxes ────────────────────────────────────────
                    // Neumann BC: flux is zero at the boundary (clamped neighbour
                    // equals current voxel → Δ = 0).
                    let fluxz_pos = if iz + 1 < nz {
                        let delta = (cur[idx(iz + 1, iy, ix)] - v) / sz;
                        K::conductance(delta.abs(), k) * delta / sz
                    } else {
                        0.0
                    };
                    let fluxz_neg = if iz > 0 {
                        let delta = (cur[idx(iz - 1, iy, ix)] - v) / sz;
                        K::conductance(delta.abs(), k) * delta / sz
                    } else {
                        0.0
                    };

                    // ── y-axis fluxes ────────────────────────────────────────
                    let fluxy_pos = if iy + 1 < ny {
                        let delta = (cur[idx(iz, iy + 1, ix)] - v) / sy;
                        K::conductance(delta.abs(), k) * delta / sy
                    } else {
                        0.0
                    };
                    let fluxy_neg = if iy > 0 {
                        let delta = (cur[idx(iz, iy - 1, ix)] - v) / sy;
                        K::conductance(delta.abs(), k) * delta / sy
                    } else {
                        0.0
                    };

                    // ── x-axis fluxes ────────────────────────────────────────
                    let fluxx_pos = if ix + 1 < nx {
                        let delta = (cur[idx(iz, iy, ix + 1)] - v) / sx;
                        K::conductance(delta.abs(), k) * delta / sx
                    } else {
                        0.0
                    };
                    let fluxx_neg = if ix > 0 {
                        let delta = (cur[idx(iz, iy, ix - 1)] - v) / sx;
                        K::conductance(delta.abs(), k) * delta / sx
                    } else {
                        0.0
                    };

                    // Unused: sz2/sy2/sx2 are the squared spacings; the flux
                    // already contains 1/s in the conductance denominator and
                    // another 1/s in the delta normalisation, yielding 1/s².
                    // The explicit references below silence any dead-code lint.
                    let _ = (sz2, sy2, sx2);

                    nxt[flat] = v + dt
                        * (fluxz_pos + fluxz_neg + fluxy_pos + fluxy_neg + fluxx_pos + fluxx_neg);
                }
            }
        }
        std::mem::swap(&mut cur, &mut nxt);
    }

    cur
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_perona_malik.rs"]
mod tests;
