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
//! 2. **Kernel Density Estimation (KDE)**: Estimate the smoothed density fÌ‚(x) of V on a
//!    uniform grid of `num_bins` points spanning [min(V), max(V)] using a Gaussian kernel:
//!
//!      f̂(x) = (1 / (n · h)) · Σᵢ K((x − Vᵢ) / h)
//!
//!    where K(u) = (1/√(2π)) · exp(−u²/2) and h is the bandwidth.
//!
//!    If bandwidth is not specified, Silverman's rule of thumb is used:
//!    h = 0.9 · min(σ̂, IQR/1.34) · n^(−1/5)
//!
//! 3. **White matter peak detection**:
//!    - T1-weighted: WM is the brightest tissue class. Search for the mode of fÌ‚ in the
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

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::tensor::Backend;
use ritk_image::Image as NativeImage;
use ritk_image::Image;
use ritk_tensor_ops::native as tensor_ops;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Selects which extreme of the local-maxima set to return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtremeSide {
    /// Pick the local maximum with the highest intensity (T1 WM peak).
    Rightmost,
    /// Pick the local maximum with the lowest intensity (T2 WM peak).
    Leftmost,
}

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
    pub normalized: Image<f32, B, 3>,
    /// White stripe mean (μ_ws).
    pub mu: f64,
    /// White stripe standard deviation (σ_ws), population std.
    ///
    /// Guaranteed > 0 by construction (var_ws + ε under sqrt).
    pub sigma: f64,
    /// Detected white matter peak intensity.
    pub wm_peak: f64,
    /// Number of voxels in the white stripe.
    pub stripe_size: usize,
}

/// Coeus-native white stripe result (sister of [`WhiteStripeResult`]).
///
/// Field-shape identical to the Coeus-keyed result except the `normalized` image
/// is the Coeus-backed [`ritk_image::Image`].
#[derive(Debug, Clone)]
pub struct NativeWhiteStripeResult<B: ComputeBackend> {
    /// Normalized image: `I_norm = (I − μ_ws) / (σ_ws + ε)`.
    pub normalized: NativeImage<f32, B, 3>,
    /// White stripe mean (μ_ws).
    pub mu: f64,
    /// White stripe standard deviation (σ_ws), population std (guaranteed > 0).
    pub sigma: f64,
    /// Detected white matter peak intensity.
    pub wm_peak: f64,
    /// Number of voxels in the white stripe.
    pub stripe_size: usize,
}

/// Backend-independent white stripe computation output over host buffers.
struct WhiteStripeComputed {
    normalized: Vec<f32>,
    mu: f64,
    sigma: f64,
    wm_peak: f64,
    stripe_size: usize,
}

/// Shared host core: compute white stripe statistics and the normalized flat
/// buffer from an image slice and an optional mask slice.
///
/// Both the Coeus-backed [`WhiteStripeNormalizer::normalize`] and the Coeus-native
/// [`WhiteStripeNormalizer::normalize_native`] delegate here, so the KDE, peak
/// detection, and normalization math have exactly one home.
///
/// # Panics
/// Panics when no foreground voxels exist, when image and mask element counts
/// differ, or when the white stripe is empty — matching the Coeus contract.
fn compute_white_stripe(
    all_slice: &[f32],
    mask_slice: Option<&[f32]>,
    config: &WhiteStripeConfig,
) -> WhiteStripeComputed {
    let foreground: Vec<f64> = match mask_slice {
        Some(mask_slice) => {
            assert_eq!(
                all_slice.len(),
                mask_slice.len(),
                "image and mask must have identical element count"
            );
            all_slice
                .iter()
                .zip(mask_slice.iter())
                .filter(|(_, &mv)| mv > crate::FOREGROUND_THRESHOLD)
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
    sorted_fg.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("foreground values must be comparable")
    });

    let n = sorted_fg.len();
    let fg_min = sorted_fg[0];
    let fg_max = sorted_fg[n - 1];
    let fg_median = quantile_sorted(&sorted_fg, 0.5);

    let bandwidth = match config.bandwidth {
        Some(bw) => bw,
        None => silverman_bandwidth(&sorted_fg),
    };

    let (grid, density) = kde_gaussian(&sorted_fg, fg_min, fg_max, config.num_bins, bandwidth);

    let (search_lo, search_hi) = match config.contrast {
        MriContrast::T1 => (fg_median, fg_max),
        MriContrast::T2 => (fg_min, fg_median),
    };

    let side = if matches!(config.contrast, MriContrast::T1) {
        ExtremeSide::Rightmost
    } else {
        ExtremeSide::Leftmost
    };
    let wm_peak = find_extreme_local_mode(&grid, &density, search_lo, search_hi, side);

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

    let mu_ws: f64 = stripe.iter().sum::<f64>() / stripe_size as f64;
    let var_ws: f64 = stripe
        .iter()
        .map(|&v| (v - mu_ws) * (v - mu_ws))
        .sum::<f64>()
        / stripe_size as f64;
    let sigma_ws = var_ws.sqrt();

    let eps = 1e-10_f64;
    let denom = sigma_ws + eps;

    let normalized: Vec<f32> = all_slice
        .iter()
        .map(|&v| ((v as f64 - mu_ws) / denom) as f32)
        .collect();

    WhiteStripeComputed {
        normalized,
        mu: mu_ws,
        sigma: sigma_ws.max(1e-9),
        wm_peak,
        stripe_size,
    }
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
    pub fn normalize<B>(
        image: &Image<f32, B, 3>,
        mask: Option<&Image<f32, B, 3>>,
        config: &WhiteStripeConfig,
    ) -> WhiteStripeResult<B>
    where
        B: Backend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let (all_vec, dims) = extract_vec_infallible(image);
        let mask_vec = mask.map(|m| extract_vec_infallible(m).0);
        let computed = compute_white_stripe(&all_vec, mask_vec.as_deref(), config);

        WhiteStripeResult {
            normalized: rebuild(computed.normalized, dims, image),
            mu: computed.mu,
            sigma: computed.sigma,
            wm_peak: computed.wm_peak,
            stripe_size: computed.stripe_size,
        }
    }

    /// Coeus-native sister of [`WhiteStripeNormalizer::normalize`].
    ///
    /// Identical Shinohara et al. (2014) algorithm over host-resident Coeus
    /// tensors, returning a [`NativeWhiteStripeResult`].
    ///
    /// # Errors
    /// Returns an error when the image or mask tensor is not host-addressable or
    /// contiguous, or the rebuilt tensor fails shape validation.
    ///
    /// # Panics
    /// Panics under the same conditions as
    /// [`WhiteStripeNormalizer::normalize`] (no foreground voxels, image/mask
    /// element-count mismatch, or empty white stripe).
    pub fn normalize_native<B>(
        image: &NativeImage<f32, B, 3>,
        mask: Option<&NativeImage<f32, B, 3>>,
        config: &WhiteStripeConfig,
    ) -> anyhow::Result<NativeWhiteStripeResult<B>>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let (all_vec, dims) = tensor_ops::extract_image_vec(image)?;
        let mask_vec = match mask {
            Some(m) => Some(tensor_ops::extract_image_vec(m)?.0),
            None => None,
        };
        let computed = compute_white_stripe(&all_vec, mask_vec.as_deref(), config);

        Ok(NativeWhiteStripeResult {
            normalized: tensor_ops::rebuild_image(computed.normalized, dims, image, &B::default())?,
            mu: computed.mu,
            sigma: computed.sigma,
            wm_peak: computed.wm_peak,
            stripe_size: computed.stripe_size,
        })
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
    let count = sorted.partition_point(|&v| v <= value);
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
/// - `side = ExtremeSide::Rightmost` (T1): return the local maximum with the **highest intensity**
///   (rightmost in the grid). This selects the WM peak over the GM peak.
/// - `side = ExtremeSide::Leftmost` (T2): return the local maximum with the **lowest intensity**
///   (leftmost in the grid). This selects the WM peak over the CSF peak.
///
/// Falls back to [`find_mode_in_range`] (global max density) if no local maximum exists.
fn find_extreme_local_mode(
    grid: &[f64],
    density: &[f64],
    range_lo: f64,
    range_hi: f64,
    side: ExtremeSide,
) -> f64 {
    let n = grid.len();
    if n < 2 {
        return find_mode_in_range(grid, density, range_lo, range_hi);
    }

    // Collect all local maxima within the range.
    let mut local_maxima: Vec<(f64, f64)> = Vec::with_capacity(n / 4); // (grid_val, density_val)

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

    if side == ExtremeSide::Rightmost {
        // T1: pick the local maximum with the highest intensity (rightmost).
        local_maxima
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(x, _)| x)
            .expect("at least one local maximum must exist after non-empty check")
    } else {
        // T2: pick the local maximum with the lowest intensity (leftmost).
        local_maxima
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(x, _)| x)
            .expect("at least one local maximum must exist after non-empty check")
    }
}

#[cfg(test)]
#[path = "tests_white_stripe/mod.rs"]
mod tests_white_stripe;
