use super::super::{
    empirical_cdf_rank, quantile_sorted, MriContrast, WhiteStripeConfig, WhiteStripeNormalizer,
};
use super::*;
use coeus_core::SequentialBackend;
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::Tensor;
use ritk_image::test_support::make_image;
use ritk_image::Image;

// ── Test 1: Synthetic tri-modal T1 → WM peak detection ────────────────

#[test]
fn test_trimodal_t1_wm_peak_detection() {
    let (data, total) = make_trimodal_volume(500, 1000, 800);
    let image: Image<f32, TestBackend, 3> = make_image(data, [1, 1, total]);

    let config = WhiteStripeConfig {
        contrast: MriContrast::T1,
        width: 0.05,
        num_bins: 2048,
        bandwidth: None,
    };

    let result = WhiteStripeNormalizer::normalize(&image, None, &config);

    assert!(
        (result.wm_peak - 0.8).abs() < 0.05,
        "T1 WM peak must be near 0.8, got {}",
        result.wm_peak
    );
    assert!(
        result.stripe_size > 0,
        "stripe_size must be > 0, got {}",
        result.stripe_size
    );
    assert!(
        (result.mu - 0.8).abs() < 0.05,
        "mu_ws must be near 0.8, got {}",
        result.mu
    );
    assert!(
        result.sigma < 0.1,
        "sigma_ws must be small, got {}",
        result.sigma
    );
}

#[test]
fn native_white_stripe_preserves_geometry_and_reports_diagnostics() {
    let (data, total) = make_trimodal_volume(64, 128, 96);
    let image = NativeImage::from_flat_on(
        data,
        [1, 1, total],
        ritk_spatial::Point::new([1.0, 2.0, 3.0]),
        ritk_spatial::Spacing::new([0.5, 1.0, 2.0]),
        ritk_spatial::Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let result = WhiteStripeNormalizer::normalize_native(
        &image,
        None,
        &WhiteStripeConfig {
            num_bins: 128,
            ..Default::default()
        },
    )
    .expect("native white stripe succeeds");

    assert!(result.stripe_size > 0);
    assert!((result.wm_peak - 0.8).abs() < 0.08);
    assert_eq!(result.normalized.origin(), image.origin());
    assert_eq!(result.normalized.spacing(), image.spacing());
    assert_eq!(result.normalized.direction(), image.direction());
}

// ── Test 2: After normalization, white stripe voxels ≈ mean 0, std 1 ──

#[test]
fn test_normalized_white_stripe_mean_zero_std_one() {
    let (data, total) = make_trimodal_volume(500, 1000, 800);
    let image: Image<f32, TestBackend, 3> = make_image(data.clone(), [1, 1, total]);

    let config = WhiteStripeConfig::default();

    let result = WhiteStripeNormalizer::normalize(&image, None, &config);
    let norm_vals = get_values(&result.normalized);

    let _mu_ws = result.mu;
    let sigma_ws = result.sigma;

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

    if sigma_ws > crate::normalization::NORMALIZER_EPSILON as f64 {
        assert!(
            (stripe_std - 1.0).abs() < 0.1,
            "white stripe normalized std must be ≈ 1, got {stripe_std}"
        );
    }
}

// ── Test 3: T2 contrast — WM peak in lower range ──────────────────────

#[test]
fn test_t2_contrast_wm_peak_lower_range() {
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
        data.push(next_normal(0.3, 0.02).clamp(0.01, 0.99) as f32);
    }
    for _ in 0..1000 {
        data.push(next_normal(0.5, 0.03).clamp(0.01, 0.99) as f32);
    }
    for _ in 0..500 {
        data.push(next_normal(0.8, 0.02).clamp(0.01, 0.99) as f32);
    }

    let total = data.len();
    let image: Image<f32, TestBackend, 3> = make_image(data, [1, 1, total]);

    let config = WhiteStripeConfig {
        contrast: MriContrast::T2,
        width: 0.05,
        num_bins: 2048,
        bandwidth: None,
    };

    let result = WhiteStripeNormalizer::normalize(&image, None, &config);

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
    let image: Image<f32, TestBackend, 3> = make_image(data, [1, 1, total]);

    let config = WhiteStripeConfig::default();
    let result = WhiteStripeNormalizer::normalize(&image, None, &config);

    assert!(result.stripe_size > 0, "stripe_size must be > 0");
    assert!(result.sigma > 0.0, "sigma must be > 0 for multi-modal data");
    assert!(result.mu > 0.0, "mu must be > 0 for positive intensities");
    assert_eq!(result.normalized.shape(), [1, 1, total]);
}

// ── Test 5: Uniform image — graceful handling ─────────────────────────

#[test]
fn test_uniform_image_sigma_near_zero() {
    let val = 0.5f32;
    let data = vec![val; 1000];
    let image: Image<f32, TestBackend, 3> = make_image(data, [10, 10, 10]);

    let config = WhiteStripeConfig::default();
    let result = WhiteStripeNormalizer::normalize(&image, None, &config);

    assert!(
        result.sigma < 1e-6,
        "uniform image sigma must be ≈ 0, got {}",
        result.sigma
    );

    let vals = get_values(&result.normalized);
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.abs() < 1e-3,
            "uniform image normalized voxel {i} must be ≈ 0, got {v}"
        );
    }

    assert!(
        (result.mu - val as f64).abs() < 1e-4,
        "mu must equal the constant value {val}, got {}",
        result.mu
    );
    assert!(result.stripe_size > 0, "stripe_size must be > 0");
}

// ── Test 6: Mask restricts foreground ─────────────────────────────────

#[test]
fn test_mask_restricts_foreground() {
    let mut data = Vec::new();
    let mut mask_data = Vec::new();

    for _ in 0..500 {
        data.push(0.1f32);
        mask_data.push(0.0f32);
    }
    for i in 0..500 {
        data.push(0.5 + 0.4 * (i as f32 / 499.0));
        mask_data.push(1.0f32);
    }

    let total = data.len();
    let image: Image<f32, TestBackend, 3> = make_image(data.clone(), [1, 1, total]);
    let mask: Image<f32, TestBackend, 3> = make_image(mask_data, [1, 1, total]);

    let config = WhiteStripeConfig {
        contrast: MriContrast::T1,
        width: 0.05,
        num_bins: 512,
        bandwidth: None,
    };

    let result = WhiteStripeNormalizer::normalize(&image, Some(&mask), &config);

    assert!(
        result.wm_peak >= 0.5 && result.wm_peak <= 0.95,
        "masked WM peak must be in [0.5, 0.95], got {}",
        result.wm_peak
    );
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
    let tensor = Tensor::<f32, TestBackend>::from_slice_on([3, 3, 3], &[0.5f32; 27], &device);
    let origin = ritk_spatial::Point::new([1.0, 2.0, 3.0]);
    let spacing = ritk_spatial::Spacing::new([0.5, 0.5, 0.5]);
    let direction = ritk_spatial::Direction::identity();
    let image: Image<f32, TestBackend, 3> = Image::new(tensor, origin, spacing, direction);

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
    let image: Image<f32, TestBackend, 3> = make_image(data, [1, 1, total]);

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
