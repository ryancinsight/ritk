use burn_ndarray::NdArray;
use ritk_image::test_support::make_image;
use ritk_image::Image;

pub(super) type TestBackend = NdArray<f32>;

pub(super) fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
    make_image(data, dims)
}

pub(super) fn get_values(image: &Image<TestBackend, 3>) -> Vec<f32> {
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
pub(super) fn make_trimodal_volume(n_csf: usize, n_gm: usize, n_wm: usize) -> (Vec<f32>, usize) {
    let total = n_csf + n_gm + n_wm;
    let mut data = Vec::with_capacity(total);

    let mut seed: u64 = 42;
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

mod behavior;
mod internals;
