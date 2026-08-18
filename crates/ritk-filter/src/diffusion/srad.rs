//! Speckle-reducing anisotropic diffusion (SRAD), Yu & Acton 2002.
//!
//! Perona–Malik keys its conductance on gradient magnitude, which is the right
//! edge detector for *additive* noise. Ultrasound speckle is **multiplicative**:
//! its magnitude scales with local intensity, so a gradient threshold that
//! preserves edges in a bright region over-smooths a dim one, and vice versa.
//!
//! SRAD instead keys on the **instantaneous coefficient of variation** (ICOV),
//! a normalized ratio of local gradient and Laplacian to intensity. Because it
//! is a ratio, it is scale-free: a fully developed speckle region has the same
//! ICOV whether it is bright or dim, so one parameter governs the whole image.
//!
//! # Mathematical specification
//!
//! Per Yu & Acton, with `I` the intensity, `|∇I|` the gradient magnitude and
//! `∇²I` the Laplacian:
//!
//! ```text
//! q² = [ ½(|∇I|/I)² − (¼·∇²I/I)² ] / [ 1 + ¼·∇²I/I ]²        (35)
//! c  = 1 / ( 1 + (q² − q₀²) / (q₀²·(1 + q₀²)) )               (33)
//! q₀(t) = q₀·exp(−ρ·Δt·t)                                     (speckle scale)
//! ```
//!
//! `q₀` is the coefficient of variation of a *fully developed speckle* region
//! in the image being filtered — the reference against which a neighbourhood is
//! judged homogeneous. Its exponential decay tightens that reference as the
//! iteration proceeds, so late iterations diffuse only what still looks like
//! pure speckle.
//!
//! The numerator of `q²` is clamped at zero: where the Laplacian term dominates
//! it goes negative, which would make `q` complex. Yu & Acton take the real
//! part, and ITK does the same.
//!
//! The update is the divergence of `c·∇I` (58), discretized over the
//! ZeroFluxNeumann neighbourhood this module shares, then an explicit Euler
//! step `I ← I + Δt·div`.
//!
//! # Dimensionality
//!
//! Yu & Acton and `itkSpeckleReducingAnisotropicDiffusionFunction` are stated
//! for 2-D. This implementation runs the same operator over all three axes of
//! ritk's 3-D image, matching the rest of this module; a single-slice image
//! (`nz == 1`) reduces to the 2-D case exactly, because the clamped boundary
//! makes both z-neighbours the centre voxel and their contributions vanish.
//!
//! # References
//! - Yu, Y., & Acton, S. T. (2002). "Speckle reducing anisotropic diffusion."
//!   *IEEE Trans. Image Processing* 11(11), 1260–1270. Equations 33, 35, 58.
//! - `itkSpeckleReducingAnisotropicDiffusionFunction.hxx`,
//!   KitwareMedical/ITKUltrasound.

use anyhow::{bail, Result};
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

use super::clamp_at;

/// Configuration for [`SpeckleReducingDiffusionFilter`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SradConfig {
    /// Number of explicit Euler steps.
    pub num_iterations: usize,
    /// Time step `Δt`. Explicit diffusion is stable for `Δt ≤ 1/(2·D)`; the
    /// default is safe for 3-D.
    pub time_step: f64,
    /// Coefficient of variation of a fully developed speckle region, `q₀`.
    /// For fully formed speckle in envelope-detected B-mode this is near
    /// `0.5227`; on log-compressed data it is smaller.
    pub q0: f64,
    /// Decay rate `ρ` of the speckle scale function `q₀·exp(−ρ·Δt·t)`.
    pub rho: f64,
}

impl Default for SradConfig {
    fn default() -> Self {
        Self {
            num_iterations: 20,
            time_step: 0.0625,
            // Rayleigh-distributed envelope: CV = sqrt(4/pi - 1) ~ 0.5227.
            q0: 0.5227,
            rho: 1.0,
        }
    }
}

impl SradConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when any parameter is non-finite, when `time_step` or
    /// `q0` is not strictly positive (a zero `q0` makes the diffusion
    /// coefficient undefined — every neighbourhood would be infinitely far from
    /// the speckle reference), or when `rho` is negative.
    pub fn validate(&self) -> Result<()> {
        if !self.time_step.is_finite() || !self.q0.is_finite() || !self.rho.is_finite() {
            bail!("SRAD parameters must be finite");
        }
        if self.time_step <= 0.0 {
            bail!("SRAD time_step must be > 0, got {}", self.time_step);
        }
        if self.q0 <= 0.0 {
            bail!("SRAD q0 must be > 0, got {}", self.q0);
        }
        if self.rho < 0.0 {
            bail!("SRAD rho must be >= 0, got {}", self.rho);
        }
        Ok(())
    }
}

/// Speckle-reducing anisotropic diffusion filter (Yu & Acton 2002).
#[derive(Debug, Clone, Copy)]
pub struct SpeckleReducingDiffusionFilter {
    config: SradConfig,
}

impl SpeckleReducingDiffusionFilter {
    /// Create a filter.
    #[must_use]
    pub fn new(config: SradConfig) -> Self {
        Self { config }
    }

    /// Apply the filter.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration is invalid or the image tensor
    /// cannot be read as a contiguous host buffer.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
        self.config.validate()?;
        let (values, dims) = extract_vec(image)?;
        let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];
        let result = diffuse(&values, dims, spacing, &self.config);
        Ok(rebuild(result, dims, image))
    }
}

/// Local gradient magnitude and Laplacian at one voxel, spacing-scaled.
///
/// Both are evaluated on the same ZeroFluxNeumann-clamped neighbourhood the
/// rest of this module uses, so a boundary voxel sees a replicated neighbour
/// rather than a fabricated value.
#[inline]
fn gradient_and_laplacian(
    buf: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    z: isize,
    y: isize,
    x: isize,
) -> (f64, f64) {
    let centre = clamp_at(buf, dims, z, y, x);
    let neighbours = [
        (
            clamp_at(buf, dims, z + 1, y, x),
            clamp_at(buf, dims, z - 1, y, x),
            spacing[0],
        ),
        (
            clamp_at(buf, dims, z, y + 1, x),
            clamp_at(buf, dims, z, y - 1, x),
            spacing[1],
        ),
        (
            clamp_at(buf, dims, z, y, x + 1),
            clamp_at(buf, dims, z, y, x - 1),
            spacing[2],
        ),
    ];

    let mut gradient_sq = 0.0;
    let mut laplacian = 0.0;
    for (plus, minus, step) in neighbours {
        let first = (plus - minus) / (2.0 * step);
        gradient_sq += first * first;
        laplacian += (plus - 2.0 * centre + minus) / (step * step);
    }
    (gradient_sq.sqrt(), laplacian)
}

/// Diffusion coefficient `c` at one voxel, Yu & Acton (35) then (33).
#[inline]
fn diffusion_coefficient(intensity: f64, gradient: f64, laplacian: f64, q0_t: f64) -> f64 {
    // Guards the ratios where the image is dark; f64::EPSILON matches the role
    // of ITK's Math::eps in the same expressions.
    let eps = f64::EPSILON;
    let scaled_laplacian = 0.25 * laplacian / (intensity + eps);
    let scaled_gradient = gradient / (intensity + eps);

    // (35). The numerator goes negative where the Laplacian term dominates,
    // which would make q complex; the real part is taken, as in Yu & Acton.
    let numerator =
        (0.5 * scaled_gradient * scaled_gradient - scaled_laplacian * scaled_laplacian).max(0.0);
    let denominator = (1.0 + scaled_laplacian).powi(2);
    let q_sq = numerator / (denominator + eps);

    // (33): c → 1 where the neighbourhood is as variable as pure speckle,
    // and → 0 where it is far more variable, i.e. at an edge.
    let q0_sq = q0_t * q0_t;
    1.0 / (1.0 + (q_sq - q0_sq) / (q0_sq * (1.0 + q0_sq) + eps))
}

/// Explicit-Euler SRAD over a flat row-major `[nz, ny, nx]` buffer.
fn diffuse(values: &[f32], dims: [usize; 3], spacing: [f64; 3], config: &SradConfig) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let mut current = values.to_vec();
    let mut next = current.clone();

    for iteration in 0..config.num_iterations {
        // Speckle scale function: the homogeneity reference tightens with time.
        let q0_t = config.q0 * (-config.rho * config.time_step * (iteration + 1) as f64).exp();

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let (zi, yi, xi) = (z as isize, y as isize, x as isize);
                    let centre = clamp_at(&current, dims, zi, yi, xi);

                    // c at the centre serves the backward differences; each
                    // forward neighbour carries its own c, per (58).
                    let (g, l) = gradient_and_laplacian(&current, dims, spacing, zi, yi, xi);
                    let c_centre = diffusion_coefficient(centre, g, l, q0_t);

                    let mut divergence = 0.0;
                    for (dz, dy, dx) in [(1_isize, 0_isize, 0_isize), (0, 1, 0), (0, 0, 1)] {
                        let (fz, fy, fx) = (zi + dz, yi + dy, xi + dx);
                        let forward = clamp_at(&current, dims, fz, fy, fx);
                        let (fg, fl) = gradient_and_laplacian(&current, dims, spacing, fz, fy, fx);
                        let c_forward = diffusion_coefficient(forward, fg, fl, q0_t);
                        divergence += c_forward * (forward - centre);

                        let backward = clamp_at(&current, dims, zi - dz, yi - dy, xi - dx);
                        divergence += c_centre * (backward - centre);
                    }
                    // Six neighbour contributions; ITK's 2-D form divides by its
                    // four for the same reason.
                    divergence /= 6.0;

                    next[z * ny * nx + y * nx + x] =
                        (centre + config.time_step * divergence) as f32;
                }
            }
        }
        std::mem::swap(&mut current, &mut next);
    }

    current
}

#[cfg(test)]
#[path = "tests_srad.rs"]
mod tests;
