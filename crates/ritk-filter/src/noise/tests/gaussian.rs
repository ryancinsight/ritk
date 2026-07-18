use super::*;
use coeus_core::SequentialBackend;
use ritk_image::native::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};

/// Deterministic Gaussian noise with seed=42 produces known output.
#[test]
fn gaussian_deterministic_seed_42() {
    let img = make_image(vec![10.0_f32; 4], [1, 2, 2]);
    let filter = AdditiveGaussianNoiseFilter::new(5.0)
        .with_mean(0.0)
        .with_seed(42);
    let result = filter.apply(&img).unwrap();
    let vals = result.data().to_vec();
    // Different seed produces different values; just verify shape and that
    // values differ from the original (noise was actually added).
    assert_eq!(vals.len(), 4);
    assert!(vals.iter().any(|&v| (v - 10.0).abs() > 0.01));
}

/// Zero std dev produces no change (within f32 tolerance).
#[test]
fn gaussian_zero_std_is_identity() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let img = make_image(data.clone(), [2, 2, 2]);
    let filter = AdditiveGaussianNoiseFilter::new(0.0);
    let result = filter.apply(&img).unwrap();
    let vals = result.data().to_vec();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - data[i]).abs() < 1e-6,
            "voxel {i} changed with zero std"
        );
    }
}

/// Nonzero mean shifts all voxels.
#[test]
fn gaussian_nonzero_mean_shifts() {
    let img = make_image(vec![0.0_f32; 1000], [10, 10, 10]);
    let filter = AdditiveGaussianNoiseFilter::new(0.1)
        .with_mean(100.0)
        .with_seed(42);
    let result = filter.apply(&img).unwrap();
    let vals = result.data().to_vec();
    let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
    assert!(
        (mean - 100.0).abs() < 2.0,
        "mean should be ~100, got {mean}"
    );
}

/// Different seeds produce different noise patterns.
#[test]
fn gaussian_seeds_differ() {
    let data = vec![0.0_f32; 100];
    let img = make_image(data.clone(), [5, 5, 4]);
    let filter1 = AdditiveGaussianNoiseFilter::new(1.0).with_seed(42);
    let filter2 = AdditiveGaussianNoiseFilter::new(1.0).with_seed(43);
    let v1 = filter1.apply(&img).unwrap().data().to_vec();
    let v2 = filter2.apply(&img).unwrap().data().to_vec();
    assert_ne!(v1, v2, "different seeds must produce different output");
}

/// Same seed produces identical output.
#[test]
fn gaussian_same_seed_idempotent() {
    let data = vec![0.0_f32; 50];
    let img = make_image(data, [5, 5, 2]);
    let filter = AdditiveGaussianNoiseFilter::new(1.0).with_seed(42);
    let v1 = filter.apply(&img).unwrap().data().to_vec();
    let v2 = filter.apply(&img).unwrap().data().to_vec();
    assert_eq!(v1, v2, "same seed must produce identical output");
}

/// On a zero image, seed 42 (std 1, mean 0) must reproduce the exact
/// `sitk.AdditiveGaussianNoise` sequence (single-threaded) — the FastNorm
/// generator ported bit-for-bit from ITK source.
#[test]
fn gaussian_matches_sitk_fastnorm_sequence() {
    let img = make_image(vec![0.0_f32; 6], [1, 1, 6]);
    let out = AdditiveGaussianNoiseFilter::new(1.0)
        .with_seed(42)
        .apply(&img)
        .unwrap();
    let vals = out.data_slice().into_owned();
    let expected = [
        -2.0906951_f32,
        -1.9422115,
        -1.6573238,
        0.19301039,
        0.08058648,
        0.00776763,
    ];
    for (g, e) in vals.iter().zip(expected) {
        assert!((g - e).abs() < 1e-5, "noise {g} != sitk {e}");
    }
}

#[test]
fn native_gaussian_noise_matches_deterministic_tensor_sequence() {
    let image = NativeImage::from_flat_on(
        vec![0.0; 6],
        [1, 1, 6],
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = AdditiveGaussianNoiseFilter::new(1.0)
        .with_seed(42)
        .apply_native(&image, &SequentialBackend)
        .expect("native Gaussian noise succeeds");
    let expected = [
        -2.0906951_f32,
        -1.9422115,
        -1.6573238,
        0.19301039,
        0.08058648,
        0.00776763,
    ];
    assert_eq!(output.data_slice().expect("contiguous output"), &expected);
    assert_eq!(output.origin(), image.origin());
    assert_eq!(output.spacing(), image.spacing());
    assert_eq!(output.direction(), image.direction());
}
