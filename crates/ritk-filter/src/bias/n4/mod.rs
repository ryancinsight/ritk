//! N4 bias field correction filter.
//!
//! # Reference
//! Tustison, N.J., et al. (2010). N4ITK: Improved N3 Bias Correction.
//! *IEEE Trans. Med. Imaging*, 29(6):1310–1320. doi:10.1109/TMI.2010.2046908
//!
//! # Algorithm
//! Models multiplicative bias: I(x) = S(x)·B(x). In log-space this becomes
//! v(x) = ln(S(x)) + ln(B(x)). The accumulated log-bias field b is estimated
//! via a multi-resolution B-spline fitting loop.
//!
//! Per-level iteration:
//! 1. w = v − b (current debiased log-intensity)
//! 2. w̃ = histogram_sharpen(w) (Wiener deconvolution sharpens tissue peaks)
//! 3. r = w − w̃ (residual ≈ remaining low-frequency bias)
//! 4. Δb = bspline_smooth(r) (smooth B-spline fit to residual)
//! 5. b ← b + Δb (additive accumulation)
//! 6. Converge when ‖Δb‖_RMS < threshold.
//!
//! Corrected image: exp(v − b).
//!
//! # Histogram Sharpening
//! Faithful to ITK `N4BiasFieldCorrectionImageFilter::SharpenImage`. Models
//! H_observed = H_true ∗ G_noise, recovers the deconvolved density U via Wiener
//! deconvolution Û\[k\] = Ĥ\[k\]·Ĝ*\[k\] / (|Ĝ\[k\]|² + wiener_noise), then maps
//! each intensity through the conditional expectation
//! E\[i\] = (U·c ⋆ G)\[i\] / (U ⋆ G)\[i\] (c = bin centre). This pulls intensities
//! toward the sharpened tissue peaks — the actual N4 sharpening, in place of the
//! earlier rank-preserving CDF/quantile transfer that left N4 behaving like N3.

mod dft;
mod histogram_sharpen;

use super::bspline_bias::{bspline_evaluate, bspline_fit};
use anyhow::{anyhow, bail};
use ritk_image::tensor::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use histogram_sharpen::{histogram_sharpen, HistogramSharpenScratch};
use ritk_core::image::Image;
use ritk_spatial::VolumeDims;
use ritk_tensor_ops::extract_vec;

// Re-export the DFT helpers so the `tests_n4` module can reach them
// via `use super::*;` (the tests were failing with E0425 "cannot find
// function" because the dft submodule is private and the functions
// were not re-exported).
#[cfg(test)]
pub(crate) use dft::{dft_real_into, idft_real_into, next_pow2};

// ── Public types ───────────────────────────────────────────────────────────────

/// Configuration for the N4 bias field correction filter.
#[derive(Debug, Clone)]
pub struct N4Config {
    /// Number of multi-resolution B-spline fitting levels.
    pub num_fitting_levels: usize,
    /// Maximum iterations per fitting level.
    pub num_iterations: usize,
    /// Convergence threshold: ‖Δb‖_RMS < threshold triggers early exit.
    pub convergence_threshold: f64,
    /// Number of histogram bins for Wiener-based sharpening.
    pub num_histogram_bins: usize,
    /// Histogram-sharpening full-width-at-half-maximum in log-intensity units
    /// (ITK/ANTs `m_BiasFieldFullWidthAtHalfMaximum`, default 0.15). Converted to
    /// bins via `fwhm / bin_width` (ITK's `scaledFWHM`) to set the Gaussian width
    /// in the Wiener deconvolution.
    pub bias_field_fwhm: f64,
    /// Initial B-spline mesh resolution (number of mesh *elements* per dimension)
    /// at level 0. Control points per dimension at level `L` are
    /// `mesh\[d\]·2^L + spline_order(3)`, doubling the element count each level
    /// (ITK/ANTs control-lattice refinement). ANTs default: one element per dim.
    pub bspline_mesh: VolumeDims,
    /// Wiener filter noise term in the histogram-sharpening deconvolution
    /// (ITK/ANTs default 0.01): Û = Ĥ·conj(Ĝ) / (|Ĝ|² + wiener_noise).
    pub noise_estimate: f64,
    /// Isotropic shrink factor: the EM bias estimation runs on the input
    /// downsampled by this factor (block averaging), then the fitted log-bias
    /// control lattice is evaluated at full resolution (ITK/ANTs `shrinkFactor`,
    /// default 4). Adapted down so the smallest shrunk dimension stays ≥ 4.
    pub shrink_factor: usize,
}

impl Default for N4Config {
    fn default() -> Self {
        Self {
            num_fitting_levels: 4,
            num_iterations: 50,
            convergence_threshold: 0.001,
            num_histogram_bins: 200,
            bias_field_fwhm: 0.15,
            bspline_mesh: VolumeDims::new([1, 1, 1]),
            noise_estimate: 0.01,
            shrink_factor: 4,
        }
    }
}

/// N4 bias field correction filter.
///
/// Corrects spatially varying multiplicative intensity non-uniformity
/// (bias field) in 3-D volumetric images using the N4ITK algorithm.
pub struct N4BiasFieldCorrectionFilter {
    pub config: N4Config,
}

impl N4BiasFieldCorrectionFilter {
    /// Construct with the provided configuration.
    pub fn new(config: N4Config) -> Self {
        Self { config }
    }

    /// Apply N4 bias field correction to a 3-D image with f32 element type.
    ///
    /// The EM bias estimation runs on the input downsampled by `shrink_factor`
    /// (ITK/ANTs strategy): histogram sharpening and B-spline fitting operate on
    /// the shrunk grid, and each iteration's fitted control lattice is also
    /// evaluated at full resolution to accumulate the full-resolution log-bias
    /// field used for the final correction.
    ///
    /// # Errors
    /// Returns `Err` if the tensor data cannot be read as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, shape) = extract_vec(image)?;
        let out = apply_n4_bias_correction_values(&vals, shape, &self.config)?;

        let td2 = TensorData::new(out, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td2, &image.data().device());

        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

/// Apply N4 bias-field correction to flat z-major 3-D image values.
///
/// This is the backend-neutral N4 SSOT used by both the legacy Burn image
/// filter and the Coeus-backed registration preprocessing executor.
///
/// # Errors
/// Returns an error when `dims` overflows, `vals.len()` does not match `dims`,
/// or histogram sharpening or B-spline fitting fails.
#[must_use = "N4 correction computes a replacement image buffer"]
pub fn apply_n4_bias_correction_values(
    vals: &[f32],
    dims: [usize; 3],
    config: &N4Config,
) -> anyhow::Result<Vec<f32>> {
    // Cubic B-spline order (control points per element span = order + 1).
    const SPLINE_ORDER: usize = 3;
    const EPS: f32 = 1e-4;

    // ── 1. Extract CPU data; full-resolution log-intensity ─────────────
    let [nz, ny, nx] = dims;
    let n_full = checked_voxel_count(dims)?;
    if vals.len() != n_full {
        bail!(
            "N4 input value count {} does not match voxel count {} for dims {:?}",
            vals.len(),
            n_full,
            dims
        );
    }
    let dims = [nz, ny, nx];
    let v_full: Vec<f32> = vals.iter().map(|&x| x.max(EPS).ln()).collect();

    // ── 2. Shrink the log-intensity grid (block averaging) ─────────────
    // The shrink factor is adapted down per session so the smallest shrunk
    // dimension stays ≥ 4 (cubic B-spline needs ≥ 1 span = 4 control points).
    let min_dim = dims.iter().copied().min().unwrap_or(1);
    let shrink = config.shrink_factor.clamp(1, (min_dim / 4).max(1));
    let sdims: [usize; 3] = std::array::from_fn(|d| (dims[d] / shrink).max(1));
    let [sz, sy, sx] = sdims;
    let n_s = sz * sy * sx;
    let v_s = block_average(&v_full, dims, sdims, shrink);

    // ── 3. Multi-resolution bias estimation on the shrunk grid ─────────
    // b_s accumulates the log-bias field at shrunk resolution (drives the EM);
    // b_full accumulates it at full resolution (drives the final correction).
    let mut b_s = vec![0.0f32; n_s];
    let mut b_full = vec![0.0f32; n_full];
    let mut w = vec![0.0f32; n_s];
    let mut r = vec![0.0f32; n_s];
    let mut hs_scratch = HistogramSharpenScratch::new(config.num_histogram_bins, n_s);

    for level in 0..config.num_fitting_levels {
        // Control-lattice refinement: element count doubles each level.
        // cg[d] = mesh[d]·2^level + order, capped to the shrunk extent.
        let shift = 1usize << level;
        let cg: [usize; 3] = std::array::from_fn(|d| {
            (config.bspline_mesh.0[d] * shift + SPLINE_ORDER)
                .min(sdims[d])
                .max(4)
        });

        for _ in 0..config.num_iterations {
            // w = v_s − b_s
            for i in 0..n_s {
                w[i] = v_s[i] - b_s[i];
            }

            // Wiener-sharpened estimate of the "true" log-intensity
            // distribution (removes Rician noise blurring of tissue peaks).
            histogram_sharpen(
                &w,
                config.num_histogram_bins,
                config.bias_field_fwhm,
                config.noise_estimate,
                &mut hs_scratch,
            )?;

            // r = w − w_sharp (residual ≈ remaining low-frequency bias)
            for i in 0..n_s {
                r[i] = w[i] - hs_scratch.w_sharp[i];
            }

            // Fit a smooth B-spline to the residual at shrunk resolution.
            let ctrl = bspline_fit(&r, sdims, cg)?;

            // Evaluate the same control lattice at both resolutions.
            let delta_s = bspline_evaluate(&ctrl, cg, sdims);
            let delta_full = bspline_evaluate(&ctrl, cg, dims);

            // Convergence criterion: ‖Δb‖_RMS (shrunk grid) < threshold.
            let change: f64 = {
                let ss: f64 = delta_s.iter().map(|&x| (x as f64).powi(2)).sum();
                (ss / n_s as f64).sqrt()
            };

            for i in 0..n_s {
                b_s[i] += delta_s[i];
            }
            for i in 0..n_full {
                b_full[i] += delta_full[i];
            }

            if change < config.convergence_threshold {
                break;
            }
        }
    }

    // ── 4. Corrected image at full resolution: exp(v − b) ──────────────
    let mut out = vec![0.0f32; n_full];
    for i in 0..n_full {
        out[i] = (v_full[i] - b_full[i]).exp();
    }

    Ok(out)
}

/// Downsample a z-major `dims` volume to `sdims` by averaging each
/// `shrink × shrink × shrink` block (ITK `ShrinkImageFilter` block mean).
///
/// Block `(sz, sy, sx)` averages source voxels in
/// `[sz·f, sz·f + f) × …`, clamped to the source extent. `sdims[d]` is assumed
/// to equal `max(1, dims[d] / shrink)`.
fn block_average(src: &[f32], dims: [usize; 3], sdims: [usize; 3], shrink: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let [sz, sy, sx] = sdims;
    let f = shrink.max(1);
    let mut out = vec![0.0f32; sz * sy * sx];

    for oz in 0..sz {
        let z0 = oz * f;
        let z1 = (z0 + f).min(nz);
        for oy in 0..sy {
            let y0 = oy * f;
            let y1 = (y0 + f).min(ny);
            for ox in 0..sx {
                let x0 = ox * f;
                let x1 = (x0 + f).min(nx);

                let mut sum = 0.0f64;
                let mut count = 0u32;
                for z in z0..z1 {
                    for y in y0..y1 {
                        let row = (z * ny + y) * nx;
                        for x in x0..x1 {
                            sum += src[row + x] as f64;
                            count += 1;
                        }
                    }
                }
                // count ≥ 1: oz < sz ≤ nz/f ⇒ z0 < nz ⇒ z1 > z0 (likewise y, x).
                out[(oz * sy + oy) * sx + ox] = (sum / count as f64) as f32;
            }
        }
    }
    out
}

fn checked_voxel_count(dims: [usize; 3]) -> anyhow::Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| anyhow!("N4 image dims {:?} product overflows usize", dims))
    })
}

#[cfg(test)]
#[path = "tests_n4.rs"]
mod tests_n4;
