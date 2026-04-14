//! White Stripe intensity normalization for brain MRI.
//!
//! # Algorithm
//! Shinohara, R.T., Sweeney, E.M., Goldsmith, J., et al. (2014).
//! "Statistical normalization techniques for magnetic resonance imaging."
//! *NeuroImage: Clinical* 6:9–19.
//!
//! Fortin, J.-P., Sweeney, E.M., Muschelli, J., Crainiceanu, C.M., Shinohara, R.T. (2018).
//! "Removing inter-subject technical variability in magnetic resonance imaging studies."
//! *NeuroImage* 132:198–212.
//!
//! # Mathematical Specification
//!
//! Given a brain MRI volume I and an optional brain mask M:
//!
//! 1. **Foreground extraction**: Collect voxels V = {I(p) : M(p) > 0} (or I(p) > 0 if no mask).
//!
//! 2. **Kernel Density Estimation (KDE)**: Estimate the smoothed density f̂(x) of V on a
//!    uniform grid of `num_bins` points spanning [min(V), max(V)] using a Gaussian kernel:
//!
//!      f̂(x) = (1 / (n · h)) · Σᵢ K((x − Vᵢ) / h)
//!
//!    where K(u) = (1/√(2π)) · exp(−u²/2) and h is the bandwidth.
//!
//!    If bandwidth is not specified, Silverman's rule of thumb is used:
//!      h = 0.9 · min(σ̂, IQR/1.34) · n^(−1/5)
//!
//! 3. **White matter peak detection**:
//!    - T1-weighted: WM is the brightest tissue class. Search for the mode of f̂ in the
//!      upper half of the intensity range [median(V), max(V)].
//!    - T2-weighted / FLAIR: WM peak is in the lower portion. Search in [min(V), median(V)].
//!
//! 4. **White stripe definition**:
//!    - Compute p_wm = empirical quantile rank of the WM peak intensity in V.
//!    - Lower bound = quantile(V, p_wm − width)
//!    - Upper bound = quantile(V, p_wm + width)
//!    - White stripe S = {Vᵢ : lower ≤ Vᵢ ≤ upper}
//!
//! 5. **Normalization**:
//!    - μ_ws = mean(S), σ_ws = std(S)   (population std)
//!    - I_norm(p) = (I(p) − μ_ws) / (σ_ws + ε),  ε = 1e-10
//!
//! # Invariants
//! - Output image preserves spatial metadata (origin, spacing, direction).
//! - If σ_ws ≈ 0 (degenerate white stripe, e.g., uniform image), division uses ε
//!   to avoid infinity. The caller should inspect `sigma` in the result.
//! - `stripe_size` in the result is always > 0 for valid inputs with at least one
//!   foreground voxel.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// MRI contrast type for white stripe peak detection.
///
/// Determines the intensity range in which to search for the white matter peak.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MriContrast {
    /// T1-weighted: WM peak is the brightest tissue class.
    /// Search range: [median(V), max(V)].
    T1,
    /// T2-weighted or FLAIR: WM peak is darker than CSF.
    /// Search range: [min(V), median(V)].
    T2,
}

/// Configuration for white stripe normalization.
///
/// # Fields
/// - `contrast`: MRI contrast type determining WM peak search direction.
/// - `width`: Half-width of the white stripe in quantile units. Default 0.05 (±5 percentile points).
/// - `num_bins`: Number of evaluation points for the KDE grid. Default 2048.
/// - `bandwidth`: KDE bandwidth in intensity units. If `None`, Silverman's rule is used.
#[derive(Debug, Clone)]
pub struct WhiteStripeConfig {
    /// MRI contrast type (determines where to search for WM peak).
    pub contrast: MriContrast,
    /// Half-width of the white stripe in quantile units (default 0.05).
    pub width: f64,
    /// Number of histogram bins for KDE (default 2048).
    pub num_bins: usize,
    /// KDE bandwidth in intensity units. If None, use Silverman's rule.
    pub bandwidth: Option<f64>,
}

impl Default for WhiteStripeConfig {
    fn default() -> Self {
        Self {
            contrast: MriContrast::T1,
            width: 0.05,
            num_bins: 2048,
            bandwidth: None,
        }
    }
}

/// White stripe normalization result.
///
/// Contains the normalized image and all intermediate quantities needed
/// for reproducibility and diagnostics.
#[derive(Debug, Clone)]
pub struct WhiteStripeResult<B: Backend> {
    /// Normalized image: I_norm = (I − μ_ws) / (σ_ws + ε).
    pub normalized: Image<B, 3>,
    /// White stripe mean (μ_ws).
    pub mu: f64,
    /// White stripe standard deviation (σ_ws), population std.
    pub sigma: f64,
    /// Detected white matter peak intensity.
    pub wm_peak: f64,
    /// Number of voxels in the white stripe.
    pub stripe_size: usize,
}

/// White stripe normalizer.
///
/// Implements the Shinohara et al. (2014) white stripe method for inter-subject
/// intensity normalization of brain MRI.
pub struct WhiteStripeNormalizer;

impl WhiteStripeNormalizer {
    /// Normalize a brain MRI using the white stripe method.
    ///
    /// # Arguments
    /// - `image`: 3D brain MRI volume.
    /// - `mask`: Optional brain mask. Voxels with mask > 0.5 are foreground.
    ///   If `None`, all voxels with intensity > 0 are used.
    /// - `config`: White stripe configuration parameters.
    ///
    /// # Panics
    /// - If no foreground voxels exist.
    /// - If the white stripe contains zero voxels (degenerate quantile configuration).
    ///
    /// # Returns
    /// [`WhiteStripeResult`] containing the normalized image and diagnostic quantities.
    pub fn normalize<B: Backend>(
        image: &Image<B, 3>,
        mask: Option<&Image<B, 3>>,
        config: &WhiteStripeConfig,
    ) -> WhiteStripeResult<B> {
        // Step 1: Extract foreground voxel intensities.
        let all_data = image.data().clone().into_data();
        let all_slice = all_data.as_slice::<f32>().expect("f32 tensor data");

        let foreground: Vec<f64> = match mask {
            Some(m) => {
                let mask_data = m.data().clone().into_data();
                let mask_slice = mask_data.as_slice::<f32>().expect("f32 mask tensor data");
                assert_eq!(
                    all_slice.len(),
                    mask_slice.len(),
                    "image and mask must have identical element count"
                );
                all_slice
                    .iter()
                    .zip(mask_slice.iter())
                    .filter(|(_, &mv)| mv > 0.5)
                    .map(|(&v, _)| v as f64)
                    .collect()
            }
            None => all_slice
                .iter()
                .filter(|&&v| v > 0.0)
                .map(|&v| v as f64)
                .collect(),
        };

        assert!(
            !foreground.is_empty(),
            "no foreground voxels found for white stripe normalization"
        );

        let mut sorted_fg = foreground.clone();
        sorted_fg.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_fg.len();
        let fg_min = sorted_fg[0];
        let fg_max = sorted_fg[n - 1];
        let fg_median = quantile_sorted(&sorted_fg, 0.5);

        // Step 2: Kernel Density Estimation.
        let bandwidth = match config.bandwidth {
            Some(bw) => bw,
            None => silverman_bandwidth(&sorted_fg),
        };

        let (grid, density) = kde_gaussian(&sorted_fg, fg_min, fg_max, config.num_bins, bandwidth);

        // Step 3: Detect white matter peak.
        // For T1: WM is the brightest tissue class → find the rightmost local maximum
        //   in [median, max]. This avoids selecting the GM peak which may have higher
        //   density but sits at a lower intensity.
        // For T2: WM is darker than CSF → find the leftmost local maximum in [min, median].
        let (search_lo, search_hi) = match config.contrast {
            MriContrast::T1 => (fg_median, fg_max),
            MriContrast::T2 => (fg_min, fg_median),
        };

        let rightmost = matches!(config.contrast, MriContrast::T1);
        let wm_peak = find_extreme_local_mode(&grid, &density, search_lo, search_hi, rightmost);

        // Step 4: Define the white stripe.
        // Compute the empirical quantile rank of the WM peak.
        let p_wm = empirical_cdf_rank(&sorted_fg, wm_peak);

        let p_lo = (p_wm - config.width).clamp(0.0, 1.0);
        let p_hi = (p_wm + config.width).clamp(0.0, 1.0);

        let intensity_lo = quantile_sorted(&sorted_fg, p_lo);
        let intensity_hi = quantile_sorted(&sorted_fg, p_hi);

        let stripe: Vec<f64> = sorted_fg
            .iter()
            .copied()
            .filter(|&v| v >= intensity_lo && v <= intensity_hi)
            .collect();

        assert!(
            !stripe.is_empty(),
            "white stripe contains zero voxels; adjust width or inspect the input"
        );

        let stripe_size = stripe.len();

        // Step 5: Compute white stripe statistics.
        let mu_ws: f64 = stripe.iter().sum::<f64>() / stripe_size as f64;
        let var_ws: f64 = stripe
            .iter()
            .map(|&v| (v - mu_ws) * (v - mu_ws))
            .sum::<f64>()
            / stripe_size as f64;
        let sigma_ws = var_ws.sqrt();

        // Step 6: Normalize.
        let eps = 1e-10_f64;
        let denom = sigma_ws + eps;

        let normalized_data: Vec<f32> = all_slice
            .iter()
            .map(|&v| ((v as f64 - mu_ws) / denom) as f32)
            .collect();

        let device = image.data().device();
        let shape = image.shape();
        let tensor =
            Tensor::<B, 3>::from_data(TensorData::new(normalized_data, Shape::new(shape)), &device);

        let normalized = Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        );

        WhiteStripeResult {
            normalized,
            mu: mu_ws,
            sigma: sigma_ws,
            wm_peak,
            stripe_size,
        }
    }
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Compute the quantile of a sorted slice using linear interpolation.
///
/// `p` ∈ [0, 1]. Uses the "linear interpolation between closest ranks" method (type 7 in R).
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }

    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    let frac = idx - lo as f64;

    if hi >= n {
        sorted[n - 1]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Compute the empirical CDF rank of a value in a sorted array.
///
/// Returns the fraction of values ≤ `value`, in [0, 1].
fn empirical_cdf_rank(sorted: &[f64], value: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.5;
    }
    // Count values ≤ value using binary search.
    let count = match sorted.partition_point(|&v| v <= value) {
        0 => 0,
        c => c,
    };
    // Map to [0, 1] range with continuity correction.
    (count as f64 - 0.5) / n as f64
}

/// Silverman's rule of thumb for Gaussian KDE bandwidth.
///
/// h = 0.9 · min(σ̂, IQR / 1.34) · n^(−1/5)
///
/// σ̂ is the sample standard deviation, IQR = Q3 − Q1.
fn silverman_bandwidth(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n < 2 {
        return 1.0;
    }

    let mean: f64 = sorted.iter().sum::<f64>() / n as f64;
    let var: f64 = sorted.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    let sigma = var.sqrt();

    let q1 = quantile_sorted(sorted, 0.25);
    let q3 = quantile_sorted(sorted, 0.75);
    let iqr = q3 - q1;

    let spread = if iqr > 0.0 {
        sigma.min(iqr / 1.34)
    } else {
        sigma
    };

    let h = 0.9 * spread * (n as f64).powf(-0.2);

    // Guard against zero bandwidth.
    if h < 1e-15 {
        let range = sorted[n - 1] - sorted[0];
        if range > 0.0 {
            range / n as f64
        } else {
            1.0
        }
    } else {
        h
    }
}

/// Gaussian KDE on a uniform grid.
///
/// Evaluates the kernel density estimate at `num_bins` equally spaced points
/// in [lo, hi]. Returns (grid_points, density_values).
///
/// Uses the direct O(n · num_bins) evaluation. For the typical use case
/// (n ~ 10⁵–10⁶ voxels, num_bins = 2048), this is acceptable.
///
/// Each density value:
///   f̂(x_j) = (1 / (n · h)) · Σᵢ K((x_j − vᵢ) / h)
/// where K(u) = (1/√(2π)) exp(−u²/2).
fn kde_gaussian(
    sorted: &[f64],
    lo: f64,
    hi: f64,
    num_bins: usize,
    bandwidth: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert!(num_bins >= 2, "num_bins must be >= 2");
    let n = sorted.len();
    let step = (hi - lo) / (num_bins - 1) as f64;

    let grid: Vec<f64> = (0..num_bins).map(|i| lo + i as f64 * step).collect();
    let mut density = vec![0.0f64; num_bins];

    let inv_h = 1.0 / bandwidth;
    let norm = 1.0 / (n as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt());

    for &v in sorted {
        // For each data point, find the range of grid bins within ~4σ.
        let lo_idx = {
            let x = ((v - 4.0 * bandwidth - lo) / step).floor() as isize;
            x.max(0) as usize
        };
        let hi_idx = {
            let x = ((v + 4.0 * bandwidth - lo) / step).ceil() as isize;
            (x as usize).min(num_bins - 1)
        };

        for j in lo_idx..=hi_idx {
            let u = (grid[j] - v) * inv_h;
            density[j] += (-0.5 * u * u).exp();
        }
    }

    for d in density.iter_mut() {
        *d *= norm;
    }

    (grid, density)
}

/// Find the mode (maximum density) within a specified intensity range on the KDE grid.
///
/// Returns the grid value with the highest density in [range_lo, range_hi].
/// Used as a fallback when no local maxima are found by [`find_extreme_local_mode`].
fn find_mode_in_range(grid: &[f64], density: &[f64], range_lo: f64, range_hi: f64) -> f64 {
    let mut best_val = grid[0];
    let mut best_density = f64::NEG_INFINITY;

    for (&x, &d) in grid.iter().zip(density.iter()) {
        if x >= range_lo && x <= range_hi && d > best_density {
            best_density = d;
            best_val = x;
        }
    }

    best_val
}

/// Find the rightmost (or leftmost) local maximum within a density range.
///
/// A local maximum at grid index `i` satisfies `density[i] > density[i-1]` and
/// `density[i] > density[i+1]` (boundary indices are handled as one-sided).
///
/// - `rightmost = true` (T1): return the local maximum with the **highest intensity**
///   (rightmost in the grid). This selects the WM peak over the GM peak.
/// - `rightmost = false` (T2): return the local maximum with the **lowest intensity**
///   (leftmost in the grid). This selects the WM peak over the CSF peak.
///
/// Falls back to [`find_mode_in_range`] (global max density) if no local maximum exists.
fn find_extreme_local_mode(
    grid: &[f64],
    density: &[f64],
    range_lo: f64,
    range_hi: f64,
    rightmost: bool,
) -> f64 {
    let n = grid.len();
    if n < 2 {
        return find_mode_in_range(grid, density, range_lo, range_hi);
    }

    // Collect all local maxima within the range.
    let mut local_maxima: Vec<(f64, f64)> = Vec::new(); // (grid_val, density_val)

    for i in 0..n {
        let x = grid[i];
        if x < range_lo || x > range_hi {
            continue;
        }

        let left_ok = i == 0 || density[i] > density[i - 1];
        let right_ok = i == n - 1 || density[i] > density[i + 1];

        if left_ok && right_ok {
            local_maxima.push((x, density[i]));
        }
    }

    if local_maxima.is_empty() {
        return find_mode_in_range(grid, density, range_lo, range_hi);
    }

    if rightmost {
        // T1: pick the local maximum with the highest intensity (rightmost).
        local_maxima
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(x, _)| x)
            .unwrap()
    } else {
        // T2: pick the local maximum with the lowest intensity (leftmost).
        local_maxima
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(x, _)| x)
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn get_values(image: &Image<TestBackend, 3>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    /// Generate a synthetic tri-modal brain MRI volume.
    ///
    /// Three tissue classes with Gaussian-distributed intensities:
    /// - CSF:  mean = 0.2, std = 0.02, count = n_csf
    /// - GM:   mean = 0.5, std = 0.03, count = n_gm
    /// - WM:   mean = 0.8, std = 0.02, count = n_wm
    ///
    /// Uses a deterministic pseudo-random sequence (LCG) for reproducibility.
    fn make_trimodal_volume(n_csf: usize, n_gm: usize, n_wm: usize) -> (Vec<f32>, usize) {
        let total = n_csf + n_gm + n_wm;
        let mut data = Vec::with_capacity(total);

        // Deterministic LCG for reproducible pseudo-normal samples via Box-Muller.
        let mut seed: u64 = 42;
        let mut next_uniform = || -> f64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to (0, 1) excluding exact 0.
            ((seed >> 11) as f64 + 1.0) / (1u64 << 53) as f64
        };

        let mut next_normal = |mean: f64, std: f64| -> f64 {
            let u1 = next_uniform();
            let u2 = next_uniform();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            mean + std * z
        };

        for _ in 0..n_csf {
            data.push(next_normal(0.2, 0.02).clamp(0.01, 0.99) as f32);
        }
        for _ in 0..n_gm {
            data.push(next_normal(0.5, 0.03).clamp(0.01, 0.99) as f32);
        }
        for _ in 0..n_wm {
            data.push(next_normal(0.8, 0.02).clamp(0.01, 0.99) as f32);
        }

        (data, total)
    }

    // ── Test 1: Synthetic tri-modal T1 → WM peak detection ────────────────

    #[test]
    fn test_trimodal_t1_wm_peak_detection() {
        // CSF: 500 voxels at ~0.2, GM: 1000 at ~0.5, WM: 800 at ~0.8.
        let (data, total) = make_trimodal_volume(500, 1000, 800);
        // Arrange into a 3D shape.
        let nz = 1;
        let ny = 1;
        let nx = total;
        let image = make_image_3d(data, [nz, ny, nx]);

        let config = WhiteStripeConfig {
            contrast: MriContrast::T1,
            width: 0.05,
            num_bins: 2048,
            bandwidth: None,
        };

        let result = WhiteStripeNormalizer::normalize(&image, None, &config);

        // WM peak must be detected near 0.8 (within ±0.05).
        assert!(
            (result.wm_peak - 0.8).abs() < 0.05,
            "T1 WM peak must be near 0.8, got {}",
            result.wm_peak
        );

        // Stripe size must be nonzero.
        assert!(
            result.stripe_size > 0,
            "stripe_size must be > 0, got {}",
            result.stripe_size
        );

        // mu must be near 0.8 (the WM mean).
        assert!(
            (result.mu - 0.8).abs() < 0.05,
            "mu_ws must be near 0.8, got {}",
            result.mu
        );

        // sigma must be small (WM std is ~0.02).
        assert!(
            result.sigma < 0.1,
            "sigma_ws must be small, got {}",
            result.sigma
        );
    }

    // ── Test 2: After normalization, white stripe voxels ≈ mean 0, std 1 ──

    #[test]
    fn test_normalized_white_stripe_mean_zero_std_one() {
        let (data, total) = make_trimodal_volume(500, 1000, 800);
        let image = make_image_3d(data.clone(), [1, 1, total]);

        let config = WhiteStripeConfig::default(); // T1, width=0.05

        let result = WhiteStripeNormalizer::normalize(&image, None, &config);
        let norm_vals = get_values(&result.normalized);

        // Extract the white stripe voxels from the normalized image.
        // The white stripe is defined on the ORIGINAL intensities,
        // so we identify them from the original data.
        let mu_ws = result.mu;
        let sigma_ws = result.sigma;

        // Re-derive the white stripe bounds.
        let mut sorted_fg: Vec<f64> = data
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| v as f64)
            .collect();
        sorted_fg.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p_wm = empirical_cdf_rank(&sorted_fg, result.wm_peak);
        let p_lo = (p_wm - 0.05).clamp(0.0, 1.0);
        let p_hi = (p_wm + 0.05).clamp(0.0, 1.0);
        let lo_int = quantile_sorted(&sorted_fg, p_lo);
        let hi_int = quantile_sorted(&sorted_fg, p_hi);

        // Collect normalized values of white stripe voxels.
        let stripe_norm: Vec<f64> = data
            .iter()
            .zip(norm_vals.iter())
            .filter(|(&orig, _)| {
                let o = orig as f64;
                o >= lo_int && o <= hi_int
            })
            .map(|(_, &nv)| nv as f64)
            .collect();

        assert!(!stripe_norm.is_empty(), "stripe must have voxels");

        let stripe_mean: f64 = stripe_norm.iter().sum::<f64>() / stripe_norm.len() as f64;
        let stripe_var: f64 = stripe_norm
            .iter()
            .map(|&v| (v - stripe_mean) * (v - stripe_mean))
            .sum::<f64>()
            / stripe_norm.len() as f64;
        let stripe_std = stripe_var.sqrt();

        assert!(
            stripe_mean.abs() < 0.1,
            "white stripe normalized mean must be ≈ 0, got {stripe_mean}"
        );

        // sigma_ws > 0 case: std of normalized stripe ≈ 1.
        if sigma_ws > 1e-8 {
            assert!(
                (stripe_std - 1.0).abs() < 0.1,
                "white stripe normalized std must be ≈ 1, got {stripe_std}"
            );
        }
    }

    // ── Test 3: T2 contrast — WM peak in lower range ──────────────────────

    #[test]
    fn test_t2_contrast_wm_peak_lower_range() {
        // For T2/FLAIR, WM is darker. Simulate: WM at 0.3, GM at 0.5, CSF at 0.8.
        let mut seed: u64 = 123;
        let mut next_uniform = || -> f64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 + 1.0) / (1u64 << 53) as f64
        };
        let mut next_normal = |mean: f64, std: f64| -> f64 {
            let u1 = next_uniform();
            let u2 = next_uniform();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            mean + std * z
        };

        let mut data = Vec::new();
        for _ in 0..800 {
            data.push(next_normal(0.3, 0.02).clamp(0.01, 0.99) as f32); // WM (dark in T2)
        }
        for _ in 0..1000 {
            data.push(next_normal(0.5, 0.03).clamp(0.01, 0.99) as f32); // GM
        }
        for _ in 0..500 {
            data.push(next_normal(0.8, 0.02).clamp(0.01, 0.99) as f32); // CSF (bright in T2)
        }

        let total = data.len();
        let image = make_image_3d(data, [1, 1, total]);

        let config = WhiteStripeConfig {
            contrast: MriContrast::T2,
            width: 0.05,
            num_bins: 2048,
            bandwidth: None,
        };

        let result = WhiteStripeNormalizer::normalize(&image, None, &config);

        // WM peak must be near 0.3 for T2 contrast.
        assert!(
            (result.wm_peak - 0.3).abs() < 0.1,
            "T2 WM peak must be near 0.3, got {}",
            result.wm_peak
        );

        assert!(
            result.stripe_size > 0,
            "stripe_size must be > 0, got {}",
            result.stripe_size
        );
    }

    // ── Test 4: Default config produces non-degenerate result ─────────────

    #[test]
    fn test_default_config_non_degenerate() {
        let (data, total) = make_trimodal_volume(300, 600, 500);
        let image = make_image_3d(data, [1, 1, total]);

        let config = WhiteStripeConfig::default();
        let result = WhiteStripeNormalizer::normalize(&image, None, &config);

        assert!(result.stripe_size > 0, "stripe_size must be > 0");
        assert!(result.sigma > 0.0, "sigma must be > 0 for multi-modal data");
        assert!(result.mu > 0.0, "mu must be > 0 for positive intensities");

        // Normalized image must have same shape.
        assert_eq!(result.normalized.shape(), [1, 1, total]);
    }

    // ── Test 5: Uniform image — graceful handling ─────────────────────────

    #[test]
    fn test_uniform_image_sigma_near_zero() {
        // All voxels have the same positive intensity → sigma ≈ 0.
        let val = 0.5f32;
        let data = vec![val; 1000];
        let image = make_image_3d(data, [10, 10, 10]);

        let config = WhiteStripeConfig::default();
        let result = WhiteStripeNormalizer::normalize(&image, None, &config);

        // Sigma must be near 0 for uniform input.
        assert!(
            result.sigma < 1e-6,
            "uniform image sigma must be ≈ 0, got {}",
            result.sigma
        );

        // All normalized values must be ≈ 0 (since every voxel = mu_ws).
        let vals = get_values(&result.normalized);
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v.abs() < 1e-3,
                "uniform image normalized voxel {i} must be ≈ 0, got {v}"
            );
        }

        // mu must equal the constant value.
        assert!(
            (result.mu - val as f64).abs() < 1e-4,
            "mu must equal the constant value {val}, got {}",
            result.mu
        );

        // stripe_size must be > 0.
        assert!(result.stripe_size > 0, "stripe_size must be > 0");
    }

    // ── Test 6: Mask restricts foreground ─────────────────────────────────

    #[test]
    fn test_mask_restricts_foreground() {
        // Image: background region at intensity 0.1, brain region at 0.5–0.9.
        // Mask selects only the brain region.
        let mut data = Vec::new();
        let mut mask_data = Vec::new();

        // 500 "non-brain" voxels (low intensity, mask = 0).
        for _ in 0..500 {
            data.push(0.1f32);
            mask_data.push(0.0f32);
        }

        // 500 "brain" voxels (higher intensity, mask = 1).
        // Use a simple ramp within brain for variety.
        for i in 0..500 {
            data.push(0.5 + 0.4 * (i as f32 / 499.0)); // 0.5 to 0.9
            mask_data.push(1.0f32);
        }

        let total = data.len();
        let image = make_image_3d(data.clone(), [1, 1, total]);
        let mask = make_image_3d(mask_data, [1, 1, total]);

        let config = WhiteStripeConfig {
            contrast: MriContrast::T1,
            width: 0.05,
            num_bins: 512,
            bandwidth: None,
        };

        let result = WhiteStripeNormalizer::normalize(&image, Some(&mask), &config);

        // WM peak must be in the brain intensity range [0.5, 0.9], not at 0.1.
        assert!(
            result.wm_peak >= 0.5 && result.wm_peak <= 0.95,
            "masked WM peak must be in [0.5, 0.95], got {}",
            result.wm_peak
        );

        // mu must be in the brain range.
        assert!(
            result.mu >= 0.5 && result.mu <= 0.95,
            "masked mu must be in brain range, got {}",
            result.mu
        );
    }

    // ── Test 7: Spatial metadata preserved ────────────────────────────────

    #[test]
    fn test_preserves_spatial_metadata() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.5f32; 27], Shape::new([3, 3, 3])),
            &device,
        );
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 0.5, 0.5]);
        let direction = Direction::identity();
        let image: Image<TestBackend, 3> = Image::new(tensor, origin, spacing, direction);

        let config = WhiteStripeConfig::default();
        let result = WhiteStripeNormalizer::normalize(&image, None, &config);

        assert_eq!(result.normalized.origin(), &origin, "origin preserved");
        assert_eq!(result.normalized.spacing(), &spacing, "spacing preserved");
        assert_eq!(
            result.normalized.direction(),
            &direction,
            "direction preserved"
        );
        assert_eq!(result.normalized.shape(), [3, 3, 3], "shape preserved");
    }

    // ── Test 8: Explicit bandwidth overrides Silverman ────────────────────

    #[test]
    fn test_explicit_bandwidth() {
        let (data, total) = make_trimodal_volume(500, 1000, 800);
        let image = make_image_3d(data, [1, 1, total]);

        let config_auto = WhiteStripeConfig {
            contrast: MriContrast::T1,
            width: 0.05,
            num_bins: 2048,
            bandwidth: None,
        };
        let config_explicit = WhiteStripeConfig {
            contrast: MriContrast::T1,
            width: 0.05,
            num_bins: 2048,
            bandwidth: Some(0.01),
        };

        let result_auto = WhiteStripeNormalizer::normalize(&image, None, &config_auto);
        let result_explicit = WhiteStripeNormalizer::normalize(&image, None, &config_explicit);

        // Both must detect WM peak near 0.8, but they need not be identical.
        assert!(
            (result_auto.wm_peak - 0.8).abs() < 0.05,
            "auto bandwidth WM peak near 0.8: {}",
            result_auto.wm_peak
        );
        assert!(
            (result_explicit.wm_peak - 0.8).abs() < 0.05,
            "explicit bandwidth WM peak near 0.8: {}",
            result_explicit.wm_peak
        );
    }

    // ── Internal: quantile_sorted ─────────────────────────────────────────

    #[test]
    fn test_quantile_sorted_basic() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile_sorted(&sorted, 0.0) - 1.0).abs() < 1e-10);
        assert!((quantile_sorted(&sorted, 1.0) - 5.0).abs() < 1e-10);
        assert!((quantile_sorted(&sorted, 0.5) - 3.0).abs() < 1e-10);
        assert!((quantile_sorted(&sorted, 0.25) - 2.0).abs() < 1e-10);
        assert!((quantile_sorted(&sorted, 0.75) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_sorted_single() {
        let sorted = vec![42.0];
        assert!((quantile_sorted(&sorted, 0.0) - 42.0).abs() < 1e-10);
        assert!((quantile_sorted(&sorted, 0.5) - 42.0).abs() < 1e-10);
        assert!((quantile_sorted(&sorted, 1.0) - 42.0).abs() < 1e-10);
    }

    // ── Internal: silverman_bandwidth ─────────────────────────────────────

    #[test]
    fn test_silverman_bandwidth_positive() {
        let sorted: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let bw = silverman_bandwidth(&sorted);
        assert!(bw > 0.0, "Silverman bandwidth must be > 0, got {bw}");
    }

    #[test]
    fn test_silverman_bandwidth_constant_data() {
        let sorted = vec![5.0; 100];
        let bw = silverman_bandwidth(&sorted);
        // Constant data: sigma=0, IQR=0. Fallback must produce a finite positive value.
        assert!(
            bw > 0.0 && bw.is_finite(),
            "Silverman bandwidth for constant data must be finite positive, got {bw}"
        );
    }

    // ── Internal: empirical_cdf_rank ──────────────────────────────────────

    #[test]
    fn test_empirical_cdf_rank_basic() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Value 3.0: 3 values ≤ 3.0, so rank = (3 - 0.5) / 5 = 0.5.
        let rank = empirical_cdf_rank(&sorted, 3.0);
        assert!(
            (rank - 0.5).abs() < 1e-10,
            "CDF rank of median must be 0.5, got {rank}"
        );

        // Value below all: 0 values ≤ 0.0 → rank = 0 (clamped).
        let rank_lo = empirical_cdf_rank(&sorted, 0.0);
        assert!(
            rank_lo < 0.1,
            "CDF rank below all values must be near 0, got {rank_lo}"
        );

        // Value above all: 5 values ≤ 6.0 → rank = (5 - 0.5) / 5 = 0.9.
        let rank_hi = empirical_cdf_rank(&sorted, 6.0);
        assert!(
            rank_hi > 0.8,
            "CDF rank above all values must be near 1, got {rank_hi}"
        );
    }

    // ── Internal: find_mode_in_range ──────────────────────────────────────

    #[test]
    fn test_find_mode_in_range_basic() {
        let grid = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let density = vec![0.1, 0.2, 0.8, 0.5, 0.3];

        // Full range: mode at 0.5.
        let mode = find_mode_in_range(&grid, &density, 0.0, 1.0);
        assert!(
            (mode - 0.5).abs() < 1e-10,
            "full range mode must be 0.5, got {mode}"
        );

        // Restricted to [0.6, 1.0]: mode at 0.75.
        let mode_upper = find_mode_in_range(&grid, &density, 0.6, 1.0);
        assert!(
            (mode_upper - 0.75).abs() < 1e-10,
            "upper range mode must be 0.75, got {mode_upper}"
        );
    }
}
