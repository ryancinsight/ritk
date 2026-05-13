//! Tests for white_stripe
//! Extracted to keep the 500-line structural limit.
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
    let _mu_ws = result.mu;
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
