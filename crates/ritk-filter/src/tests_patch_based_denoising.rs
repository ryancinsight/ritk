use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_tensor_ops::extract_vec;

type B = NdArray<f32>;

/// The ITK MersenneTwister port must reproduce the canonical MT19937 sequence:
/// seed 0 → first output 2357136044; seed 5489 → 3499211612.
#[test]
fn test_itk_mt19937_canonical() {
    let mut mt0 = ItkMt::new(0);
    assert_eq!(mt0.next_u32(), 2_357_136_044, "seed 0 first output");
    let mut mt5489 = ItkMt::new(5489);
    assert_eq!(mt5489.next_u32(), 3_499_211_612, "seed 5489 first output");
}

/// Determinism: the same input yields the same output (seeded RNG, seed 0).
#[test]
fn test_patch_based_denoising_deterministic() {
    let (ny, nx) = (12usize, 12);
    let vals: Vec<f32> = (0..ny * nx).map(|i| ((i * 37) % 90) as f32 + 5.0).collect();
    let img = ts::make_image::<B, 3>(vals, [1, ny, nx]);
    let filt = PatchBasedDenoisingImageFilter {
        patch_radius: 1,
        number_of_iterations: 1,
        ..Default::default()
    };
    let a = extract_vec(&filt.apply(&img).unwrap()).unwrap().0;
    let b = extract_vec(&filt.apply(&img).unwrap()).unwrap().0;
    assert_eq!(a, b, "seeded denoising must be deterministic");
    assert!(a.iter().all(|v| v.is_finite()));
}

/// Batching is an execution strategy only: changing its memory partition must
/// preserve the seeded sample stream and every pixel's reduction order exactly.
#[test]
fn test_patch_based_denoising_batch_partition_invariant() {
    let (ny, nx) = (11usize, 13);
    let data: Vec<f32> = (0..ny * nx).map(|i| ((i * 37) % 90) as f32 + 5.0).collect();
    let filter = PatchBasedDenoisingImageFilter {
        patch_radius: 1,
        number_of_iterations: 1,
        number_of_sample_patches: 32,
        ..Default::default()
    };
    let sample_bytes = size_of::<usize>() * filter.number_of_sample_patches;
    let reference = filter.pass_with_sample_budget(&data, [1, ny, nx], sample_bytes);

    for pixel_capacity in [2, 7, 17, ny * nx - 1, ny * nx, ny * nx + 1] {
        let partitioned =
            filter.pass_with_sample_budget(&data, [1, ny, nx], sample_bytes * pixel_capacity);
        assert_eq!(partitioned, reference, "pixel capacity {pixel_capacity}");
    }
}

#[test]
fn test_sampling_interval_intersects_patch_and_sampler_regions() {
    let size = 64;
    let patch_radius = 2;
    let sample_radius = 50;
    assert_eq!(
        sampling_interval(0, size, patch_radius, sample_radius),
        (0, 50)
    );
    assert_eq!(
        sampling_interval(32, size, patch_radius, sample_radius),
        (2, 61)
    );
    assert_eq!(
        sampling_interval(63, size, patch_radius, sample_radius),
        (13, 63)
    );
    assert_eq!(sampling_interval(0, size, patch_radius, i64::MAX), (0, 61));
}

#[test]
fn test_patch_based_denoising_rejects_unbounded_sample_storage() {
    let image = ts::make_image::<B, 3>(vec![1.0f32; 9], [1, 3, 3]);
    let max_samples = SAMPLE_BATCH_BYTES / size_of::<usize>();
    let error = PatchBasedDenoisingImageFilter {
        number_of_sample_patches: max_samples + 1,
        ..Default::default()
    }
    .apply(&image)
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        format!(
            "number_of_sample_patches {} exceeds bounded capacity {max_samples}",
            max_samples + 1
        )
    );
}

/// A constant image is a fixed point (every patch distance is 0 → all weights 1
/// → the gradient of (c − c) is exactly 0).
#[test]
fn test_patch_based_denoising_constant_is_fixed_point() {
    let (ny, nx) = (10usize, 10);
    let img = ts::make_image::<B, 3>(vec![42.0f32; ny * nx], [1, ny, nx]);
    let out = PatchBasedDenoisingImageFilter {
        patch_radius: 1,
        ..Default::default()
    }
    .apply(&img)
    .unwrap();
    let r = extract_vec(&out).unwrap().0;
    assert!(
        r.iter().all(|&v| (v - 42.0).abs() < 1e-4),
        "constant image must be preserved"
    );
}

/// Smooth-disc weights: centre weight is 1 (squared), and the edge weight
/// exceeds the corner weight (disc decays with distance).
#[test]
fn test_smooth_disc_weights() {
    let w = smooth_disc_weights_sq(1, 2); // 3x3
    assert_eq!(w.len(), 9);
    assert!((w[4] - 1.0).abs() < 1e-12, "centre weight² = 1");
    assert!(w[1] > w[0], "edge weight > corner weight");
}
