//! Perona-Malik anisotropic diffusion filter.
//!
//! # Mathematical Specification
//!
//! The Perona-Malik anisotropic diffusion PDE (Perona & Malik 1990):
//!
//!   ∂I/∂t = div(c(|∇I|) · ∇I)
//!
//! where the conductance function c controls the amount of diffusion at each
//! location:
//!
//! - Exponential: c(s) = exp(−(s/K)²)
//! - Quadratic:   c(s) = 1 / (1 + (s/K)²)
//!
//! Both functions satisfy c(0) = 1 (maximum diffusion where gradient is zero)
//! and c(s) → 0 as s → ∞ (no diffusion across strong edges).
//!
//! # Discretisation
//!
//! Explicit Euler finite differences on a 3-D regular grid with spacing
//! (sz, sy, sx).  For each voxel (iz, iy, ix), six nearest-neighbour fluxes
//! are computed:
//!
//!   Δ±z I = I[iz±1, iy, ix] − I[iz, iy, ix]   (zero at boundaries → Neumann BC)
//!   flux±z = c(|Δ±z I| / sz) · Δ±z I / sz²
//!
//! Update:
//!   I_new = I + Δt · (flux+z + flux−z + flux+y + flux−y + flux+x + flux−x)
//!
//! Stability condition for explicit Euler in 3-D: Δt ≤ 1/6 (unit spacing).
//! The default time-step 1/16 provides a safety factor of ~2.67.
//!
//! # Reference
//! Perona, P. & Malik, J. (1990). Scale-space and edge detection using
//! anisotropic diffusion. *IEEE Trans. Pattern Anal. Mach. Intell.*
//! 12(7):629–639. doi:10.1109/34.56205

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

// ── Public types ──────────────────────────────────────────────────────────────

/// Choice of conductance (edge-stopping) function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConductanceFunction {
    /// c(s) = exp(−(s/K)²) — Perona-Malik option 1.
    ///
    /// Favours high-contrast edges over low-contrast ones.
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
    /// Time step Δt.  Must satisfy Δt ≤ 1/(2·D) where D is the number of
    /// spatial dimensions.  Default: 0.0625 = 1/16 (safe for 3-D).
    pub time_step: f32,
    /// Conductance parameter K.  Controls the gradient threshold below which
    /// diffusion is strong.  Larger K → more smoothing across edges.
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

/// Anisotropic diffusion filter (Perona & Malik 1990).
///
/// Reduces noise while preserving edges by using a conductance function that
/// suppresses diffusion in regions of high gradient magnitude.
#[derive(Debug, Clone)]
pub struct AnisotropicDiffusionFilter {
    /// Algorithm configuration.
    pub config: DiffusionConfig,
}

impl AnisotropicDiffusionFilter {
    /// Create a filter with the given configuration.
    pub fn new(config: DiffusionConfig) -> Self {
        Self { config }
    }

    /// Apply the filter to `image`, returning a smoothed copy.
    ///
    /// # Errors
    /// Returns an error if the image tensor cannot be converted to `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let td = image.data().clone().into_data();
        let vals: Vec<f32> = td
            .as_slice::<f32>()
            .map_err(|e| {
                anyhow::anyhow!(
                    "AnisotropicDiffusionFilter requires f32 image data: {:?}",
                    e
                )
            })?
            .to_vec();

        let dims = image.shape();
        // Spacing from the image metadata (physical units); used to normalise
        // gradient differences so that conductance responds to physical gradient
        // magnitude regardless of voxel size.
        let spacing = [
            image.spacing()[0] as f32,
            image.spacing()[1] as f32,
            image.spacing()[2] as f32,
        ];

        let result = diffuse(&vals, dims, spacing, &self.config);

        let device = image.data().device();
        let td2 = TensorData::new(result, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td2, &device);
        Ok(Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        ))
    }
}

// ── Core computation ──────────────────────────────────────────────────────────

/// Evaluate the conductance function at gradient magnitude `s`.
#[inline(always)]
fn conductance(s: f32, k: f32, func: ConductanceFunction) -> f32 {
    match func {
        ConductanceFunction::Exponential => (-(s / k) * (s / k)).exp(),
        ConductanceFunction::Quadratic => {
            let r = s / k;
            1.0 / (1.0 + r * r)
        }
    }
}

/// Run the anisotropic diffusion PDE for the requested number of iterations.
///
/// Neumann (zero-flux) boundary conditions: fluxes across the image boundary
/// are set to zero by clamping neighbour indices to valid range and treating
/// the difference as zero when the neighbour index equals the current index.
fn diffuse(
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
    let func = config.function;

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
                        conductance(delta.abs(), k, func) * delta / sz
                    } else {
                        0.0
                    };
                    let fluxz_neg = if iz > 0 {
                        let delta = (cur[idx(iz - 1, iy, ix)] - v) / sz;
                        conductance(delta.abs(), k, func) * delta / sz
                    } else {
                        0.0
                    };

                    // ── y-axis fluxes ────────────────────────────────────────
                    let fluxy_pos = if iy + 1 < ny {
                        let delta = (cur[idx(iz, iy + 1, ix)] - v) / sy;
                        conductance(delta.abs(), k, func) * delta / sy
                    } else {
                        0.0
                    };
                    let fluxy_neg = if iy > 0 {
                        let delta = (cur[idx(iz, iy - 1, ix)] - v) / sy;
                        conductance(delta.abs(), k, func) * delta / sy
                    } else {
                        0.0
                    };

                    // ── x-axis fluxes ────────────────────────────────────────
                    let fluxx_pos = if ix + 1 < nx {
                        let delta = (cur[idx(iz, iy, ix + 1)] - v) / sx;
                        conductance(delta.abs(), k, func) * delta / sx
                    } else {
                        0.0
                    };
                    let fluxx_neg = if ix > 0 {
                        let delta = (cur[idx(iz, iy, ix - 1)] - v) / sx;
                        conductance(delta.abs(), k, func) * delta / sx
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn image_stats(vals: &[f32]) -> (f32, f32) {
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / vals.len() as f32;
        (mean, var.sqrt())
    }

    /// Uniform image → after N iterations the image is still uniform (all
    /// fluxes are zero since all differences are zero).
    #[test]
    fn test_uniform_image_unchanged() {
        let dims = [8, 8, 8];
        let val = 42.0_f32;
        let vals = vec![val; 8 * 8 * 8];
        let img = make_image(vals, dims);
        let filter = AnisotropicDiffusionFilter::new(DiffusionConfig {
            num_iterations: 20,
            ..Default::default()
        });
        let out = filter.apply(&img).unwrap();

        let td = out.data().clone().into_data();
        let result = td.as_slice::<f32>().unwrap();
        for &v in result {
            assert!(
                (v - val).abs() < 1e-4,
                "expected {val} for uniform image after diffusion, got {v}"
            );
        }
    }

    /// Step-edge test: left half = 50, right half = 200.
    ///
    /// After anisotropic diffusion with K large enough to inhibit diffusion
    /// across the edge:
    /// 1. Mean intensity is conserved (Neumann BC → no mass leaving domain).
    /// 2. The sign of (mean_right − mean_left) remains positive.
    /// 3. The mean of the left homogeneous region stays close to 50.
    #[test]
    fn test_step_edge_preservation() {
        let [nz, ny, nx] = [10usize, 10, 20];
        let n = nz * ny * nx;

        let mut vals = vec![0.0_f32; n];
        let initial_mean;
        {
            let mut sum = 0.0_f32;
            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let v = if ix < nx / 2 { 50.0_f32 } else { 200.0_f32 };
                        vals[iz * ny * nx + iy * nx + ix] = v;
                        sum += v;
                    }
                }
            }
            initial_mean = sum / n as f32;
        }

        let img = make_image(vals.clone(), [nz, ny, nx]);
        let filter = AnisotropicDiffusionFilter::new(DiffusionConfig {
            num_iterations: 10,
            conductance: 30.0, // large K → moderate edge inhibition
            ..Default::default()
        });
        let out = filter.apply(&img).unwrap();

        let td = out.data().clone().into_data();
        let result = td.as_slice::<f32>().unwrap();

        // 1. Mean conservation (Neumann BC → total mass is invariant).
        let final_mean: f32 = result.iter().sum::<f32>() / n as f32;
        assert!(
            (final_mean - initial_mean).abs() < 0.5,
            "mean should be conserved: initial={initial_mean:.4} final={final_mean:.4}"
        );

        // 2. Sign of (mean_right − mean_left) is preserved.
        let mut mean_left = 0.0_f32;
        let mut mean_right = 0.0_f32;
        let half = n / 2;
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let flat = iz * ny * nx + iy * nx + ix;
                    if ix < nx / 2 {
                        mean_left += result[flat];
                    } else {
                        mean_right += result[flat];
                    }
                }
            }
        }
        mean_left /= half as f32;
        mean_right /= half as f32;
        assert!(
            mean_right > mean_left,
            "edge sign should be preserved: left={mean_left:.2} right={mean_right:.2}"
        );

        // 3. Left region mean stays within 5.0 of original 50.0 after 10 iters.
        assert!(
            (mean_left - 50.0).abs() < 5.0,
            "left region mean too far from 50: {mean_left:.4}"
        );
    }

    /// Mean conservation with any image: total intensity should be approximately
    /// invariant under anisotropic diffusion with Neumann BCs.
    #[test]
    fn test_mean_conservation() {
        let [nz, ny, nx] = [8usize, 8, 8];
        let n = nz * ny * nx;
        // Simple ramp image
        let vals: Vec<f32> = (0..n).map(|i| i as f32 / n as f32 * 100.0).collect();
        let img = make_image(vals.clone(), [nz, ny, nx]);

        let (initial_mean, _) = image_stats(&vals);

        let filter = AnisotropicDiffusionFilter::new(DiffusionConfig {
            num_iterations: 30,
            ..Default::default()
        });
        let out = filter.apply(&img).unwrap();

        let td = out.data().clone().into_data();
        let result = td.as_slice::<f32>().unwrap();
        let (final_mean, _) = image_stats(result);

        let rel_error = (final_mean - initial_mean).abs() / (initial_mean.abs() + 1e-6);
        assert!(
            rel_error < 0.005,
            "mean not conserved: initial={initial_mean:.4} final={final_mean:.4} rel={rel_error:.6}"
        );
    }

    /// Quadratic conductance function also converges without blowing up.
    #[test]
    fn test_quadratic_conductance_stable() {
        let [nz, ny, nx] = [6usize, 6, 6];
        let n = nz * ny * nx;
        let vals: Vec<f32> = (0..n).map(|i| (i % 10) as f32 * 10.0).collect();
        let img = make_image(vals.clone(), [nz, ny, nx]);

        let filter = AnisotropicDiffusionFilter::new(DiffusionConfig {
            num_iterations: 20,
            function: ConductanceFunction::Quadratic,
            ..Default::default()
        });
        let out = filter.apply(&img).unwrap();

        let td = out.data().clone().into_data();
        let result = td.as_slice::<f32>().unwrap();

        // All values should remain finite and in a reasonable range.
        let initial_max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let initial_min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        for &v in result {
            assert!(
                v.is_finite(),
                "quadratic conductance produced non-finite value"
            );
            assert!(
                v >= initial_min - 1.0 && v <= initial_max + 1.0,
                "value {v} outside initial range [{initial_min}, {initial_max}]"
            );
        }
    }
}
