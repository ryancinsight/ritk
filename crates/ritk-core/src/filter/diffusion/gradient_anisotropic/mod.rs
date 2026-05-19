//! Gradient anisotropic diffusion filter.
//!
//! # Mathematical Specification
//!
//! Implements the ITK `GradientAnisotropicDiffusionImageFilter` as described in
//! Gerig et al. (1992) and ITK Software Guide §6.4.1.
//!
//! The discrete update rule at each voxel `p` over one time step is:
//!
//! ```text
//! I_new(p) = I(p) + Δt · Σ_{q ∈ N₆(p)} c(|I(q) − I(p)|) · (I(q) − I(p))
//! ```
//!
//! where N₆(p) is the 6-neighbourhood (±z, ±y, ±x), and the conductance
//! function is:
//!
//! ```text
//! c(s) = exp(−(s / K)²)
//! ```
//!
//! # Distinction from `AnisotropicDiffusionFilter` (Perona-Malik)
//!
//! The Perona-Malik filter in this crate (`perona_malik.rs`) uses spacing-
//! normalised gradients `delta/spacing` inside the conductance evaluation and
//! `flux = c(|delta/s|) · delta/s²` in the divergence form.  This filter
//! uses **raw intensity differences** (no spacing normalisation) both in
//! conductance and in the direct-flux summation, matching the ITK
//! `GradientAnisotropicDiffusionImageFilter` implementation exactly.
//!
//! # Stability
//!
//! Explicit Euler stability for the 6-neighbour Laplacian in 3-D requires
//! `Δt ≤ 1 / (2 · D) = 1/6`.  The ITK default `Δt = 0.125 < 1/6` satisfies
//! this bound with a safety factor of ~1.33.
//!
//! # Boundary Conditions
//!
//! Neumann (zero-flux): neighbour terms that fall outside the image are
//! omitted (i.e. contribute 0 to the sum).
//!
//! # Invariants
//!
//! - Constant image: all differences = 0 → update = 0 → image unchanged.
//! - Conductance K → ∞: c(s) → 1 for all s; the update becomes isotropic
//!   (scaled 6-neighbour Laplacian smoothing).
//! - Conductance K → 0: c(s) → 0 for all s ≠ 0; the update → 0 (no
//!   diffusion across any edge).
//!
//! # References
//! - Gerig, G., Kübler, O., Kikinis, R. & Jolesz, F. A. (1992). Nonlinear
//!   anisotropic filtering of MRI data. *IEEE Trans. Med. Imag.* 11(2):221–232.
//!   doi:10.1109/42.141646
//! - Ibanez, L. et al. (2005). *The ITK Software Guide*, 2nd ed. §6.4.1.

use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use burn::tensor::backend::Backend;

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

/// Gradient anisotropic diffusion filter (ITK `GradientAnisotropicDiffusionImageFilter`).
///
/// Reduces noise while preserving edges.  Distinct from
/// [`crate::filter::diffusion::AnisotropicDiffusionFilter`] in that conductance
/// is evaluated on raw intensity differences (not spacing-normalised gradients)
/// and the update uses direct-flux summation, matching the ITK reference
/// implementation exactly.
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

        let result = diffuse(vals, dims, &self.config);

        Ok(rebuild(result, dims, image))
    }
}

// ── Core computation ──────────────────────────────────────────────────────────

/// Evaluate the exponential conductance function c(s) = exp(−(s/K)²).
///
/// - `s` — raw unsigned intensity difference |I(q) − I(p)|
/// - `k` — conductance parameter K
///
/// # Invariants
/// - c(0)   = 1  (maximum diffusion in flat regions)
/// - c(∞)   = 0  (no diffusion across strong edges)
/// - c(s)   ∈ (0, 1] for all finite s ≥ 0
#[inline(always)]
fn conductance_exp(s: f32, k: f32) -> f32 {
    let r = s / k;
    (-(r * r)).exp()
}

/// Perform `config.num_iterations` explicit Euler steps on `data`.
///
/// Returns the diffused voxel buffer with the same length as `data`.
///
/// # Boundary conditions
/// Neumann (zero-flux): neighbour terms that fall outside `dims` are omitted.
fn diffuse(data: &[f32], dims: [usize; 3], config: &GradientDiffusionConfig) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut cur: Vec<f32> = data.to_vec();
    let mut nxt: Vec<f32> = vec![0.0_f32; n];

    let dt = config.time_step;
    let k = config.conductance;

    // Row-major linear index.
    let idx = |iz: usize, iy: usize, ix: usize| -> usize { iz * ny * nx + iy * nx + ix };

    for _iter in 0..config.num_iterations {
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let p = idx(iz, iy, ix);
                    let v = cur[p];
                    let mut update = 0.0_f32;

                    // +z neighbour
                    if iz + 1 < nz {
                        let delta = cur[idx(iz + 1, iy, ix)] - v;
                        update += conductance_exp(delta.abs(), k) * delta;
                    }
                    // -z neighbour
                    if iz > 0 {
                        let delta = cur[idx(iz - 1, iy, ix)] - v;
                        update += conductance_exp(delta.abs(), k) * delta;
                    }
                    // +y neighbour
                    if iy + 1 < ny {
                        let delta = cur[idx(iz, iy + 1, ix)] - v;
                        update += conductance_exp(delta.abs(), k) * delta;
                    }
                    // -y neighbour
                    if iy > 0 {
                        let delta = cur[idx(iz, iy - 1, ix)] - v;
                        update += conductance_exp(delta.abs(), k) * delta;
                    }
                    // +x neighbour
                    if ix + 1 < nx {
                        let delta = cur[idx(iz, iy, ix + 1)] - v;
                        update += conductance_exp(delta.abs(), k) * delta;
                    }
                    // -x neighbour
                    if ix > 0 {
                        let delta = cur[idx(iz, iy, ix - 1)] - v;
                        update += conductance_exp(delta.abs(), k) * delta;
                    }

                    nxt[p] = v + dt * update;
                }
            }
        }
        cur.copy_from_slice(&nxt);
    }
    cur
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests;
