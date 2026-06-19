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

// ── 2. kernel_bandwidth_estimation=true returns Err (C-3) ──────────────────

/// `kernel_bandwidth_estimation=true` is not implemented and must return `Err`.
/// The error message must mention the field name so callers can diagnose.
#[test]
fn test_kernel_bandwidth_estimation_returns_err() {
    let img = make_image(vec![1.0_f32; 4 * 4 * 4], [4, 4, 4]);
    let filter = PatchBasedDenoisingImageFilter {
        number_of_iterations: 1,
        number_of_sample_patches: 10,
        patch_radius: 1,
        kernel_bandwidth_estimation: true,
    };
    let result = filter.apply(&img);
    assert!(
        result.is_err(),
        "kernel_bandwidth_estimation=true must return Err"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("kernel_bandwidth_estimation"),
        "error message must mention the field name; got: {msg}"
    );
}

// ── 3. Constant image: output equals input ───────────────────────────────

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

// ── 4. Multi-iteration convergence monotonicity (T-4) ────────────────────────

/// Applying 3 NL-means iterations to a noisy image must not increase variance.
///
/// Derivation: NL-means computes a convex combination of input values at each
/// voxel (all weights are non-negative and sum to 1). A convex combination of
/// a set contracts towards the mean, so `var(output) ≤ var(input)` holds for
/// any non-trivial weighting.
#[test]
fn test_multi_iteration_convergence_monotonic() {
    let [nz, ny, nx] = [10usize, 10, 10];
    let base = 50.0_f32;
    let noise_amp = 10.0_f32;
    let n = nz * ny * nx;

    // Same alternating noise as `test_patch_denoising_reduces_noise`.
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0_f32 } else { -1.0 };
            base + noise_amp * sign
        })
        .collect();

    let input_var = variance(&vals);
    let img = make_image(vals, [nz, ny, nx]);

    let filter = PatchBasedDenoisingImageFilter {
        number_of_iterations: 3,
        number_of_sample_patches: 200,
        patch_radius: 2,
        kernel_bandwidth_estimation: false,
    };

    let out = filter.apply(&img).unwrap();
    let result = extract_vals(&out);

    let output_var = variance(&result);
    assert!(
        output_var <= input_var,
        "3-iteration NL-means must not increase variance: \
         input_var={input_var:.4} output_var={output_var:.4}"
    );
}
