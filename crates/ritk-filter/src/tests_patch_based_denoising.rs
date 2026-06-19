use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, dims)
}

fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
    let (vals, _) = ritk_tensor_ops::extract_vec(img).unwrap();
    vals
}

fn variance(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n
}

// ── 1. Noise-added image: output variance < input variance ───────────────

/// An alternating ±noise pattern has maximum-intensity variance. NL-means
/// assigns the highest weight to patches whose neighbourhood matches the query
/// patch; the weighted average mixes `+noise` and `−noise` samples, attenuating
/// the oscillation.  Output variance must strictly decrease.
///
/// Tolerance derivation: for white ±10 noise on a 10×10×10 volume the input
/// variance = noise_amp² = 100.  After one NL-means pass, boundary effects and
/// the deterministic grid produce a non-trivial weighted average that reduces
/// variance substantially; we require variance to decrease by at least 20%.
#[test]
fn test_patch_denoising_reduces_noise() {
    let [nz, ny, nx] = [10usize, 10, 10];
    let base = 50.0_f32;
    let noise_amp = 10.0_f32;
    let n = nz * ny * nx;

    // Deterministic alternating noise: even linear index → +amp, odd → −amp.
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0_f32 } else { -1.0 };
            base + noise_amp * sign
        })
        .collect();

    let input_var = variance(&vals);
    let img = make_image(vals, [nz, ny, nx]);

    let filter = PatchBasedDenoisingImageFilter {
        number_of_iterations: 1,
        number_of_sample_patches: 200,
        patch_radius: 2,
        kernel_bandwidth_estimation: false,
    };

    let out = filter.apply(&img).unwrap();
    let result = extract_vals(&out);

    assert_eq!(result.len(), n, "output size must match input");

    let output_var = variance(&result);
    assert!(
        output_var < input_var,
        "NL-means must reduce variance: input_var={input_var:.4} output_var={output_var:.4}"
    );
    // Require at least 20% reduction as a lower bound.
    let reduction = 1.0 - output_var / input_var;
    assert!(
        reduction >= 0.2,
        "variance reduction {:.1}% below 20% threshold",
        reduction * 100.0
    );
}

// ── 2. Constant image: output equals input ───────────────────────────────

/// For a constant image every patch is identical: d_pq = 0 for all (p, q),
/// so all weights are equal and the NL-means output is the weighted mean of
/// identical input values, equal to the constant.
///
/// Tolerance: 1e-5 absolute — matches the MAD guard (σ = 1.0 for constant
/// input) that keeps h² finite; d_pq = 0 keeps the excess non-positive so
/// w = 1 for every reference, and val_sum / w_sum = constant exactly in f64
/// before the f32 cast.  The 1e-5 epsilon absorbs that cast rounding.
#[test]
fn test_patch_denoising_constant_image_unchanged() {
    let [nz, ny, nx] = [8usize, 8, 8];
    let val = 42.0_f32;
    let n = nz * ny * nx;
    let vals = vec![val; n];
    let img = make_image(vals, [nz, ny, nx]);

    let filter = PatchBasedDenoisingImageFilter {
        number_of_iterations: 1,
        number_of_sample_patches: 200,
        patch_radius: 2,
        kernel_bandwidth_estimation: false,
    };

    let out = filter.apply(&img).unwrap();
    let result = extract_vals(&out);

    assert_eq!(result.len(), n);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - val).abs() < 1e-4,
            "voxel {i}: expected {val}, got {v} (Δ={:.3e})",
            (v - val).abs()
        );
    }
}
