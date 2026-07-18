//! Chan-Vese level set segmentation (Active Contours Without Edges).
//!
//! # Mathematical Specification
//!
//! Implements the Chan & Vese (2001) model for segmenting an image into two
//! regions without relying on gradient (edge) information. The energy functional:
//!
//! ```text
//! E(Ï†, câ‚, câ‚‚) = Î¼Â·Length(C) + Î½Â·Area(inside(C))
//!               + Î»â‚ âˆ« |uâ‚€ - câ‚|Â² H(Ï†) dx
//!               + Î»â‚‚ âˆ« |uâ‚€ - câ‚‚|Â² (1 - H(Ï†)) dx
//! ```
//!
//! where:
//! - `Ï†` is the level set function (C = {Ï† = 0} is the contour)
//! - `câ‚` = mean intensity inside C (where Ï† > 0 after Heaviside)
//! - `câ‚‚` = mean intensity outside C
//! - `Î¼` = curvature (length) penalty weight
//! - `Î½` = area penalty weight
//! - `Î»â‚`, `Î»â‚‚` = data fidelity weights for inside/outside regions
//!
//! ## PDE Evolution (Euler-Lagrange)
//!
//! ```text
//! âˆ‚Ï†/âˆ‚t = Î´_Îµ(Ï†) [ Î¼ Â· div(âˆ‡Ï†/|âˆ‡Ï†|) - Î½ - Î»â‚(uâ‚€ - câ‚)Â² + Î»â‚‚(uâ‚€ - câ‚‚)Â² ]
//! ```
//!
//! ## Regularised Heaviside and Dirac
//!
//! ```text
//! H_Îµ(z) = 0.5 Â· (1 + (2/Ï€) Â· arctan(z/Îµ))
//! Î´_Îµ(z) = (Îµ/Ï€) / (ÎµÂ² + zÂ²)
//! ```
//!
//! ## Mean Intensity Updates
//!
//! ```text
//! câ‚ = âˆ« uâ‚€ Â· H_Îµ(Ï†) dx  /  âˆ« H_Îµ(Ï†) dx
//! câ‚‚ = âˆ« uâ‚€ Â· (1 - H_Îµ(Ï†)) dx  /  âˆ« (1 - H_Îµ(Ï†)) dx
//! ```
//!
//! ## Curvature
//!
//! ```text
//! Îº = div(âˆ‡Ï†/|âˆ‡Ï†|)
//! ```
//!
//! computed via second-order central finite differences with clamped boundaries.
//!
//! ## Initialization
//!
//! Checkerboard signed distance function:
//!
//! ```text
//! Ï†â‚€(x,y,z) = -cos(Ï€x/5) Â· cos(Ï€y/5) Â· cos(Ï€z/5)
//! ```
//!
//! where x, y, z are voxel indices. Negative regions seed the interior.
//!
//! ## Convergence
//!
//! Iteration stops when `max|Ï†^{n+1} - Ï†^n| / dt < tolerance` or
//! `max_iterations` is reached.
//!
//! # Complexity
//!
//! - Per iteration: O(N) where N = total voxels (two passes: mean update + PDE step)
//! - Total: O(max_iterations Â· N)
//! - Memory: O(N) for Ï†, curvature buffer, and scratch arrays
//!
//! # References
//!
//! - Chan, T. F. & Vese, L. A. (2001). "Active Contours Without Edges."
//!   *IEEE Transactions on Image Processing*, 10(2), 266â€“277.

use super::helpers::{self, compute_curvature_into, regularised_dirac, regularised_heaviside};
use ritk_image::tensor::{Backend, Tensor};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

// â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Chan-Vese level set segmentation filter.
///
/// Segments a 3D image into foreground and background by evolving a level set
/// function under the Chan-Vese energy functional. No edge information is
/// required; the model is driven purely by region statistics.
#[derive(Debug, Clone)]
pub struct ChanVeseSegmentation {
    /// Length (curvature) penalty weight Î¼. Controls boundary smoothness.
    pub mu: f64,
    /// Area penalty weight Î½. Positive values penalise large interior regions.
    pub nu: f64,
    /// Data fidelity weight for the inside region.
    pub lambda1: f64,
    /// Data fidelity weight for the outside region.
    pub lambda2: f64,
    /// Regularisation width Îµ for Heaviside and Dirac approximations.
    pub epsilon: f64,
    /// Euler forward time step Î”t.
    pub dt: f64,
    /// Maximum number of PDE evolution iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on max|Î”Ï†|/dt.
    pub tolerance: f64,
}

impl ChanVeseSegmentation {
    /// Construct with default parameters.
    ///
    /// | Parameter       | Default |
    /// |-----------------|---------|
    /// | `mu`            | 0.25    |
    /// | `nu`            | 0.0     |
    /// | `lambda1`       | 1.0     |
    /// | `lambda2`       | 1.0     |
    /// | `epsilon`       | 1.0     |
    /// | `dt`            | 0.1     |
    /// | `max_iterations`| 200     |
    /// | `tolerance`     | 1e-3    |
    pub fn new() -> Self {
        Self {
            mu: 0.25,
            nu: 0.0,
            lambda1: 1.0,
            lambda2: 1.0,
            epsilon: 1.0,
            dt: 0.1,
            max_iterations: 200,
            tolerance: 1e-3,
        }
    }

    /// Segment `image` into a binary mask via Chan-Vese level set evolution.
    ///
    /// Returns an `Image<f32, B, 3>` with values 1.0 (inside, where Ï† â‰¥ 0 at
    /// convergence) and 0.0 (outside). Spatial metadata (origin, spacing,
    /// direction) is preserved from `image`.
    ///
    /// Initialisation: the level set Ï†â‚€ is set to `I(x) âˆ’ t*` where t* is the
    /// Otsu between-class-variance-maximising threshold of the image histogram.
    /// This immediately separates the bright and dark classes so câ‚ â‰ˆ mean_bright
    /// and câ‚‚ â‰ˆ mean_dark from iteration 1, enabling fast convergence on bimodal
    /// images.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tensor data cannot be read as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let dims = image.shape();
        let device = B::default();
        let (img_vals, _dims) = extract_vec(image)?;
        let img: &[f32] = &img_vals;
        let mask = self.evolve(img, dims);

        let tensor = Tensor::<f32, B>::from_slice_on(dims, &mask, &device);

        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }

    /// Segment a Coeus-native image into a binary mask via Chan-Vese evolution.
    ///
    /// # Errors
    ///
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the native output image cannot be constructed.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |img, dims| self.evolve(img, dims))
    }
}

impl Default for ChanVeseSegmentation {
    fn default() -> Self {
        Self::new()
    }
}

// â”€â”€ Core algorithm â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl ChanVeseSegmentation {
    /// Run the PDE evolution on a flat f32 slice with shape `[nz, ny, nx]`.
    /// Returns a binary `Vec<f32>` (1.0 inside, 0.0 outside).
    fn evolve(&self, img: &[f32], dims: [usize; 3]) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        debug_assert_eq!(img.len(), n);

        if n == 0 {
            return Vec::new();
        }

        let img_f64: Vec<f64> = img.iter().map(|&v| v as f64).collect();

        // Initialise Ï† via Otsu-threshold bipartition.
        //
        // The checkerboard heuristic (Ï† = -cos(Ï€x/5)Â·cos(Ï€y/5)Â·cos(Ï€z/5)) fails for
        // objects that occupy a small fraction of the image: câ‚ â‰ˆ câ‚‚ â‰ˆ background_mean
        // initially, so the data-fidelity terms cancel and only curvature drives the
        // evolution, which typically converges to the wrong partition.
        //
        // Otsu-based initialisation: compute the between-class-variance-maximising
        // threshold t* in O(n + 256) time; set Ï†(x) = I(x) âˆ’ t*. This ensures
        // câ‚ â‰ˆ mean of bright class and câ‚‚ â‰ˆ mean of dark class from iteration 1,
        // so the data-fidelity terms immediately drive the contour toward the correct
        // bimodal boundary.
        let otsu_t = local_otsu_threshold(&img_f64);
        let mut phi: Vec<f64> = img_f64.iter().map(|&v| v - otsu_t).collect();
        let eps = self.epsilon;

        // Scratch buffers (reused across iterations).
        let mut kappa = vec![0.0_f64; n];
        let mut max_dphis = vec![0.0_f64; nz];

        for _iter in 0..self.max_iterations {
            // â”€â”€ 1. Compute region means c1, c2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            let (c1, c2) = compute_region_means(&img_f64, &phi, eps);

            // â”€â”€ 2. Compute curvature Îº = div(âˆ‡Ï†/|âˆ‡Ï†|) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            compute_curvature_into(&phi, dims, &mut kappa);

            // â”€â”€ 3. Evolve Ï† â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            let slice_len = ny * nx;

            helpers::evolve_slices_with_metric(&mut phi, &mut max_dphis, slice_len, |iz, phi_s| {
                let base = iz * slice_len;
                let mut local_max = 0.0_f64;
                for (i, phi_val) in phi_s.iter_mut().enumerate() {
                    let idx = base + i;
                    let dirac = regularised_dirac(*phi_val, eps);

                    let diff1 = img_f64[idx] - c1;
                    let diff2 = img_f64[idx] - c2;

                    let force = self.mu * kappa[idx] - self.nu - self.lambda1 * diff1 * diff1
                        + self.lambda2 * diff2 * diff2;

                    let dphi = self.dt * dirac * force;
                    *phi_val += dphi;

                    let abs_dphi = dphi.abs();
                    if abs_dphi > local_max {
                        local_max = abs_dphi;
                    }
                }
                local_max
            });

            let max_dphi = max_dphis.iter().copied().fold(0.0_f64, f64::max);

            // â”€â”€ 4. Convergence check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if max_dphi / self.dt < self.tolerance {
                break;
            }
        }

        // â”€â”€ Threshold Ï† â†’ binary mask â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        phi.iter()
            .map(|&v| if v >= 0.0 { 1.0_f32 } else { 0.0_f32 })
            .collect()
    }
}

// â”€â”€ Region mean computation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Compute câ‚ (mean intensity inside) and câ‚‚ (mean intensity outside).
///
/// ```text
/// câ‚ = Î£ uâ‚€Â·H_Îµ(Ï†) / Î£ H_Îµ(Ï†)
/// câ‚‚ = Î£ uâ‚€Â·(1 - H_Îµ(Ï†)) / Î£ (1 - H_Îµ(Ï†))
/// ```
///
/// If either denominator is zero (degenerate partition), the corresponding
/// mean is set to 0.0 to avoid division by zero.
fn compute_region_means(img: &[f64], phi: &[f64], eps: f64) -> (f64, f64) {
    let n = img.len();
    let (sum_h, sum_uh, sum_1mh, sum_u1mh) = moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
        n,
        || (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64),
        |(sh, suh, s1mh, su1mh), i| {
            let h = regularised_heaviside(phi[i], eps);
            let omh = 1.0 - h;
            (sh + h, suh + img[i] * h, s1mh + omh, su1mh + img[i] * omh)
        },
        |(ah, auh, a1mh, au1mh), (bh, buh, b1mh, bu1mh)| {
            (ah + bh, auh + buh, a1mh + b1mh, au1mh + bu1mh)
        },
    );

    let c1 = if sum_h > 1e-15 { sum_uh / sum_h } else { 0.0 };
    let c2 = if sum_1mh > 1e-15 {
        sum_u1mh / sum_1mh
    } else {
        0.0
    };

    (c1, c2)
}

// â”€â”€ Otsu threshold (f64 slice) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Near-zero class-weight guard used inside the local Otsu computation: threshold
/// candidates where either class weight falls below this value are skipped to
/// avoid division by zero when computing class means.
const WEIGHT_ZERO_GUARD: f64 = 1e-12;

/// Compute the Otsu between-class-variance-maximising threshold on a `f64` slice.
///
/// Uses a 256-bin histogram. Returns the intensity value t* that maximises
/// between-class variance:
///
/// ```text
/// ÏƒÂ²_B(t) = Pâ‚(t) Â· Pâ‚‚(t) Â· (Î¼â‚(t) âˆ’ Î¼â‚‚(t))Â²
/// ```
///
/// For a constant image returns the uniform intensity. Complexity: O(n + 256).
fn local_otsu_threshold(img: &[f64]) -> f64 {
    const NUM_BINS: usize = 256;
    let n = img.len();
    if n == 0 {
        return 0.0;
    }
    let (x_min, x_max) = img
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    if (x_max - x_min).abs() < f64::EPSILON {
        return x_min;
    }
    let range = x_max - x_min;
    let num_bins_f = (NUM_BINS - 1) as f64;
    let mut counts = [0u64; NUM_BINS];
    for &v in img {
        let bin = ((v - x_min) / range * num_bins_f).floor() as usize;
        counts[bin.min(NUM_BINS - 1)] += 1;
    }
    // SEG-05: inline normalization over `counts` directly, eliminating the
    // 256-element `Vec<f64>` allocation for `h`.
    let n_f = n as f64;
    let total_mu: f64 = counts
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64 / n_f)
        .sum();
    let mut best_sigma2 = 0.0_f64;
    let mut best_t = 0_usize;
    let mut w1 = 0.0_f64;
    let mut mu1_partial = 0.0_f64;
    for t in 1..NUM_BINS {
        w1 += counts[t - 1] as f64 / n_f;
        mu1_partial += (t - 1) as f64 * counts[t - 1] as f64 / n_f;
        let w2 = 1.0 - w1;
        if w1 < WEIGHT_ZERO_GUARD || w2 < WEIGHT_ZERO_GUARD {
            continue;
        }
        let mu1 = mu1_partial / w1;
        let mu2 = (total_mu - mu1_partial) / w2;
        let sigma2 = w1 * w2 * (mu1 - mu2) * (mu1 - mu2);
        if sigma2 > best_sigma2 {
            best_sigma2 = sigma2;
            best_t = t;
        }
    }
    x_min + best_t as f64 * range / num_bins_f
}

#[cfg(test)]
#[path = "tests_chan_vese.rs"]
mod tests_chan_vese;
