use super::*;
use crate::edge::GaussianSigma;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::image::Image;

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    use ritk_spatial::{Direction, Point, Spacing};
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

fn make_image_with_metadata(
    vals: Vec<f32>,
    dims: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
) -> Image<B, 3> {
    use ritk_spatial::{Direction, Point, Spacing};
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new(origin),
        Spacing::new(spacing),
        Direction::identity(),
    )
}

fn image_vals(img: &Image<B, 3>) -> Vec<f32> {
    img.data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

// ── Test 1: constant image must be unchanged ───────────────────────────────
// ∇I = 0 → J_ρ = 0 → D = α·I → div(D·∇I) = 0 → I_new = I
#[test]
fn test_constant_image_unchanged() {
    let dims = [8, 8, 8];
    let n = 8 * 8 * 8;
    let vals = vec![5.0f32; n];
    let img = make_image(vals.clone(), dims);

    let filter = CoherenceEnhancingDiffusionFilter::new(CoherenceConfig {
        sigma: GaussianSigma::new_unchecked(1.0),
        contrast: 1e-10,
        alpha: 0.001,
        time_step: 0.0625,
        n_iterations: 5,
    });
    let out = filter.apply(&img);
    let out_vals = image_vals(&out);

    let max_diff = out_vals
        .iter()
        .zip(vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "constant image must be unchanged; max diff = {max_diff}"
    );
}

// ── Test 2: linear image must be unchanged ─────────────────────────────────
// ∇I = const → J_ρ rank-1 → λ₁ = λ₂ = 0, λ₃ > 0
// α₁ = α₂ = α (no excess diffusion), α₃ = α → D = α·I
// div(α·∇I) = α·ΔI = 0 for linear I → unchanged.
#[test]
fn test_linear_image_unchanged() {
    let [nz, ny, nx] = [10usize, 10, 10];
    let vals: Vec<f32> = (0..nz * ny * nx)
        .map(|i| {
            let ix = (i % nx) as f32;
            let iy = ((i / nx) % ny) as f32;
            let iz = (i / (ny * nx)) as f32;
            0.3 * ix + 0.5 * iy + 0.7 * iz
        })
        .collect();

    let img = make_image(vals.clone(), [nz, ny, nx]);

    let filter = CoherenceEnhancingDiffusionFilter::new(CoherenceConfig {
        sigma: GaussianSigma::new_unchecked(1.0),
        contrast: 1e-10,
        alpha: 0.001,
        time_step: 0.0625,
        n_iterations: 5,
    });
    let out = filter.apply(&img);
    let out_vals = image_vals(&out);

    // Interior voxels only (boundary gradient with Neumann BC introduces
    // small artefacts from the face-averaged diffusion tensor).
    let mut max_interior_diff = 0.0f32;
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let i = iz * ny * nx + iy * nx + ix;
                let diff = (out_vals[i] - vals[i]).abs();
                max_interior_diff = max_interior_diff.max(diff);
            }
        }
    }

    assert!(
        max_interior_diff < 1e-3,
        "linear image interior should be unchanged; max diff = {max_interior_diff}"
    );
}

// ── Test 3: noise reduction ────────────────────────────────────────────────
// Adding noise to a smooth image: CED should reduce variance while
// preserving the underlying structure.
#[test]
fn test_noise_reduction() {
    let dims = [16, 16, 16];
    let n = 16 * 16 * 16;

    // Base smooth image: planar ramp.
    let base: Vec<f32> = (0..n)
        .map(|i| {
            let ix = (i % 16) as f32;
            let iy = ((i / 16) % 16) as f32;
            ix * 0.1 + iy * 0.2
        })
        .collect();

    // Add deterministic noise.
    let noisy: Vec<f32> = base
        .iter()
        .enumerate()
        .map(|(i, &b)| b + (((i * 17 + 3) % 97) as f32 - 48.0) * 0.01)
        .collect();

    let img = make_image(noisy.clone(), dims);

    let filter = CoherenceEnhancingDiffusionFilter::new(CoherenceConfig {
        sigma: GaussianSigma::new_unchecked(1.5),
        contrast: 1e-10,
        alpha: 0.001,
        time_step: 0.0625,
        n_iterations: 10,
    });
    let out = filter.apply(&img);
    let out_vals = image_vals(&out);

    // Variance of the diffused image should be closer to the base than the noisy.
    let var_noisy: f32 = noisy
        .iter()
        .zip(base.iter())
        .map(|(n, b)| (n - b).powi(2))
        .sum::<f32>()
        / n as f32;

    let var_out: f32 = out_vals
        .iter()
        .zip(base.iter())
        .map(|(o, b)| (o - b).powi(2))
        .sum::<f32>()
        / n as f32;

    assert!(
        var_out < var_noisy,
        "CED should reduce noise; var_noisy={var_noisy:.6} var_out={var_out:.6}"
    );
}

// ── Test 4: zero iterations → identical output ──────────────────────────────
#[test]
fn test_zero_iterations() {
    let dims = [8, 8, 8];
    let n = 8 * 8 * 8;
    let vals: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) % 100) as f32 * 0.1).collect();
    let img = make_image(vals.clone(), dims);

    let filter = CoherenceEnhancingDiffusionFilter::new(CoherenceConfig {
        n_iterations: 0,
        ..CoherenceConfig::default()
    });
    let out = filter.apply(&img);
    let out_vals = image_vals(&out);

    for (i, (&a, &b)) in out_vals.iter().zip(vals.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "zero iterations should produce identical output; voxel {i}: {a} vs {b}"
        );
    }
}

// ── Test 5: small image with step edge ─────────────────────────────────────
// An 8×8×8 image with a step edge at z=4. CED should smooth along the
// edge (z-plane) but not across it (z-direction).
#[test]
fn test_step_edge() {
    let [nz, ny, nx] = [8usize, 8, 8];
    let n = nz * ny * nx;
    // Step edge: first half = 0, second half = 1.
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            let iz = i / (ny * nx);
            if iz < 4 {
                0.0
            } else {
                1.0
            }
        })
        .collect();

    let img = make_image(vals.clone(), [nz, ny, nx]);

    let filter = CoherenceEnhancingDiffusionFilter::new(CoherenceConfig {
        sigma: GaussianSigma::new_unchecked(1.0),
        contrast: 1e-8,
        alpha: 0.001,
        time_step: 0.0625,
        n_iterations: 5,
    });
    let out = filter.apply(&img);
    let out_vals = image_vals(&out);

    // Verify the output is finite and no NaN/Inf.
    for (i, &v) in out_vals.iter().enumerate() {
        assert!(v.is_finite(), "output voxel {i} is not finite: {v}");
    }

    // The edge should still be present: mean intensity in the first half
    // should be lower than in the second half.
    let mean_first: f32 = out_vals[..4 * ny * nx].iter().sum::<f32>() / (4 * ny * nx) as f32;
    let mean_second: f32 = out_vals[4 * ny * nx..].iter().sum::<f32>() / (4 * ny * nx) as f32;
    assert!(
        mean_first < mean_second,
        "step edge should be preserved; mean_first={mean_first:.4} mean_second={mean_second:.4}"
    );
}

// ── Test 6: stability — output finite after many iterations ────────────────
#[test]
fn test_stability() {
    let dims = [10, 10, 10];
    let n = 10 * 10 * 10;
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            let ix = i % 10;
            let iy = (i / 10) % 10;
            let iz = i / 100;
            ((ix + iy * 3 + iz * 7) % 17) as f32 * 10.0
        })
        .collect();

    let img = make_image(vals, dims);

    let filter = CoherenceEnhancingDiffusionFilter::new(CoherenceConfig {
        sigma: GaussianSigma::new_unchecked(1.0),
        contrast: 1e-10,
        alpha: 0.001,
        time_step: 0.0625,
        n_iterations: 20,
    });
    let out = filter.apply(&img);
    let out_vals = image_vals(&out);

    for (i, &v) in out_vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "output voxel {i} is not finite after 20 iterations: {v}"
        );
    }
}

// ── Test 7: determinism ────────────────────────────────────────────────────
// Same input + same config → same output.
#[test]
fn test_determinism() {
    let dims = [8, 8, 8];
    let n = 8 * 8 * 8;
    let vals: Vec<f32> = (0..n).map(|i| ((i * 7 + 13) % 50) as f32 * 0.1).collect();
    let img1 = make_image(vals.clone(), dims);
    let img2 = make_image(vals, dims);

    let config = CoherenceConfig {
        sigma: GaussianSigma::new_unchecked(1.0),
        contrast: 1e-10,
        alpha: 0.001,
        time_step: 0.0625,
        n_iterations: 5,
    };
    let filter = CoherenceEnhancingDiffusionFilter::new(config);

    let out1 = filter.apply(&img1);
    let out2 = filter.apply(&img2);

    let v1 = image_vals(&out1);
    let v2 = image_vals(&out2);

    for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "determinism violation at voxel {i}: {a} vs {b}"
        );
    }
}

// ── Test 8: spatial metadata preservation ──────────────────────────────────
#[test]
fn test_metadata_preservation() {
    let dims = [8, 8, 8];
    let n = 8 * 8 * 8;
    let vals = vec![1.0f32; n];
    let origin = [10.0f64, 20.0, 30.0];
    let spacing = [0.5f64, 0.5, 2.0];
    let img = make_image_with_metadata(vals, dims, origin, spacing);

    let filter = CoherenceEnhancingDiffusionFilter::new(CoherenceConfig {
        sigma: GaussianSigma::new_unchecked(1.0),
        contrast: 1e-10,
        alpha: 0.001,
        time_step: 0.0625,
        n_iterations: 3,
    });
    let out = filter.apply(&img);

    for d in 0..3 {
        assert!(
            (out.origin()[d] - origin[d]).abs() < 1e-10,
            "origin mismatch at dim {d}: got {} expected {}",
            out.origin()[d],
            origin[d]
        );
        assert!(
            (out.spacing()[d] - spacing[d]).abs() < 1e-10,
            "spacing mismatch at dim {d}: got {} expected {}",
            out.spacing()[d],
            spacing[d]
        );
    }
}

// ── Test 9: default config values ──────────────────────────────────────────
#[test]
fn test_default_config() {
    let cfg = CoherenceConfig::default();
    assert!(
        (cfg.sigma.get() - 3.0).abs() < 1e-10,
        "default sigma: expected 3.0, got {}",
        cfg.sigma.get()
    );
    assert!(
        (cfg.contrast - 1e-10).abs() < 1e-20,
        "default contrast: expected 1e-10, got {}",
        cfg.contrast
    );
    assert!(
        (cfg.alpha - 0.001).abs() < 1e-10,
        "default alpha: expected 0.001, got {}",
        cfg.alpha
    );
    assert!(
        (cfg.time_step - 0.0625).abs() < 1e-10,
        "default time_step: expected 0.0625, got {}",
        cfg.time_step
    );
    assert_eq!(
        cfg.n_iterations, 10,
        "default n_iterations: expected 10, got {}",
        cfg.n_iterations
    );
}
