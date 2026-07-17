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

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

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

    /// Coeus-native sister of [`AnisotropicDiffusionFilter::apply`].
    ///
    /// Runs the identical explicit-Euler Perona–Malik PDE (double-buffered on a
    /// flat host array) via the shared `diffuse` host core, so the result is
    /// bitwise-identical to the Burn path. No Burn tensor is constructed.
    /// Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B::default()) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let spacing = [
            image.spacing()[0] as f32,
            image.spacing()[1] as f32,
            image.spacing()[2] as f32,
        ];
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            diffuse::<K>(vals, dims, spacing, &self.config)
        })
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

    /// Coeus-native sister of [`DiffusionConfig::apply`].
    ///
    /// Dispatches to the conductance kernel selected in `self.function` and runs
    /// the shared `diffuse` host core, bitwise-identical to the Burn path. No
    /// Burn tensor is constructed.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B::default()) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let spacing = [
            image.spacing()[0] as f32,
            image.spacing()[1] as f32,
            image.spacing()[2] as f32,
        ];
        crate::native_support::map_flat_image(image, backend, |vals, dims| match self.function {
            ConductanceFunction::Exponential => {
                diffuse::<ExponentialConductance>(vals, dims, spacing, self)
            }
            ConductanceFunction::Quadratic => {
                diffuse::<QuadraticConductance>(vals, dims, spacing, self)
            }
        })
    }
}

// ── Core computation ─────────────────────────────────────────────────────────

/// Run the anisotropic diffusion PDE for the requested number of iterations.
///
/// Neumann (zero-flux) boundary conditions: at the image edge the neighbour
/// index is clamped to the same voxel, so the difference Δ is zero and the
/// matched flux term contributes nothing.
fn diffuse<K: ConductanceKernel>(
    data: &[f32],
    dims: [usize; 3],
    spacing: [f32; 3],
    config: &DiffusionConfig,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let slab = ny * nx;
    let mut cur = data.to_vec();
    let mut next = vec![0.0_f32; n];

    let sz = spacing[0];
    let sy = spacing[1];
    let sx = spacing[2];

    let dt = config.time_step;
    let k = config.conductance;

    let idx = |iz: usize, iy: usize, ix: usize| -> usize { iz * slab + iy * nx + ix };

    for iter in 0..config.num_iterations {
        let (src, dst) = if iter % 2 == 0 {
            (&cur, &mut next)
        } else {
            (&next, &mut cur)
        };

        moirai::enumerate_mut_with::<moirai::Adaptive, _, _>(&mut dst[..n], |flat, val| {
            let iz = flat / slab;
            let rem = flat - iz * slab;
            let iy = rem / nx;
            let ix = rem - iy * nx;

            let v = src[flat];

            let iz_p = if iz + 1 < nz { iz + 1 } else { iz };
            let iz_m = if iz > 0 { iz - 1 } else { iz };
            let iy_p = if iy + 1 < ny { iy + 1 } else { iy };
            let iy_m = if iy > 0 { iy - 1 } else { iy };
            let ix_p = if ix + 1 < nx { ix + 1 } else { ix };
            let ix_m = if ix > 0 { ix - 1 } else { ix };

            // ── z-axis fluxes ────────────────────────────────────────
            let delta_zp = (src[idx(iz_p, iy, ix)] - v) / sz;
            let fluxz_pos = K::conductance(delta_zp.abs(), k) * delta_zp / sz;
            let delta_zn = (src[idx(iz_m, iy, ix)] - v) / sz;
            let fluxz_neg = K::conductance(delta_zn.abs(), k) * delta_zn / sz;

            // ── y-axis fluxes ────────────────────────────────────────
            let delta_yp = (src[idx(iz, iy_p, ix)] - v) / sy;
            let fluxy_pos = K::conductance(delta_yp.abs(), k) * delta_yp / sy;
            let delta_yn = (src[idx(iz, iy_m, ix)] - v) / sy;
            let fluxy_neg = K::conductance(delta_yn.abs(), k) * delta_yn / sy;

            // ── x-axis fluxes ────────────────────────────────────────
            let delta_xp = (src[idx(iz, iy, ix_p)] - v) / sx;
            let fluxx_pos = K::conductance(delta_xp.abs(), k) * delta_xp / sx;
            let delta_xn = (src[idx(iz, iy, ix_m)] - v) / sx;
            let fluxx_neg = K::conductance(delta_xn.abs(), k) * delta_xn / sx;

            *val = v + dt * (fluxz_pos + fluxz_neg + fluxy_pos + fluxy_neg + fluxx_pos + fluxx_neg);
        });
    }

    if config.num_iterations.is_multiple_of(2) {
        cur
    } else {
        next
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_perona_malik.rs"]
mod tests;

#[cfg(test)]
mod tests_native {
    use super::DiffusionConfig;
    use crate::native_support::{assert_native_matches_burn, make_native_image, native_vals};
    use coeus_core::SequentialBackend;

    #[test]
    fn matches_burn() {
        let vals: Vec<f32> = (0..60).map(|i| ((i * 7) % 13) as f32).collect();
        assert_native_matches_burn(
            vals,
            [3, 4, 5],
            |img| {
                DiffusionConfig::default()
                    .apply(img)
                    .expect("burn diffusion")
            },
            |img, backend| DiffusionConfig::default().apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_constant_field_preserved() {
        // Zero gradients everywhere → zero flux → the field is a fixed point.
        let img = make_native_image(vec![5.0f32; 27], [3, 3, 3]);
        let out = DiffusionConfig::default()
            .apply_native(&img, &SequentialBackend)
            .expect("native diffusion");
        for &v in &native_vals(&out) {
            assert!(
                (v - 5.0).abs() < 1e-5,
                "constant field must be preserved, got {v}"
            );
        }
    }
}
