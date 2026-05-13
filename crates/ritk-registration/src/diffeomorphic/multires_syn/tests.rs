use super::cc::{cc_forces, mean_local_cc};
use super::pyramid::{downsample, upsample_field};
use super::{MultiResSyNConfig, MultiResSyNRegistration};

/// `I[z,y,x] = sin(π·z/nz) · cos(π·y/ny) · (x + 1)`.
/// Analytically non-trivial gradients in all three axes.
fn make_test_image(dims: [usize; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    (0..nz * ny * nx)
        .map(|fi| {
            let ix = fi % nx;
            let iy = (fi / nx) % ny;
            let iz = fi / (ny * nx);
            let sz = std::f32::consts::PI * iz as f32 / nz as f32;
            let sy = std::f32::consts::PI * iy as f32 / ny as f32;
            sz.sin() * sy.cos() * (ix as f32 + 1.0)
        })
        .collect()
}

fn translate_x(data: &[f32], dims: [usize; 3], shift: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let mut out = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in shift..nx {
                out[iz * ny * nx + iy * nx + ix] = data[iz * ny * nx + iy * nx + (ix - shift)];
            }
        }
    }
    out
}

fn make_config(num_levels: usize, iters: Vec<usize>, ic: bool) -> MultiResSyNConfig {
    MultiResSyNConfig {
        num_levels,
        iterations_per_level: iters,
        sigma_smooth: 2.0,
        convergence_threshold: 1e-7,
        convergence_window: 10,
        n_squarings: 6,
        cc_window_radius: 2,
        gradient_step: 0.25,
        enforce_inverse_consistency: ic,
    }
}

// ── Downsample / upsample ─────────────────────────────────────────────────────

/// Average-pool of constant field preserves value (analytical: mean(c) = c).
#[test]
fn downsample_constant_preserves_value() {
    let dims = [8, 8, 8];
    let image = vec![7.0_f32; 8 * 8 * 8];
    let ds = downsample(&image, dims, 2);
    assert_eq!(ds.len(), 4 * 4 * 4);
    for &v in &ds {
        assert!((v - 7.0).abs() < 1e-6, "expected 7.0, got {v}");
    }
}

/// Upsample constant field with component=0: output = value × (new_nz / old_nz).
/// Analytical: 3.0 × (8/4) = 6.0.
#[test]
fn upsample_constant_field_scales_correctly() {
    let old = [4, 4, 4];
    let new = [8, 8, 8];
    let field = vec![3.0_f32; 4 * 4 * 4];
    let up = upsample_field(&field, old, new, 0);
    assert_eq!(up.len(), 8 * 8 * 8);
    for &v in &up {
        assert!((v - 6.0).abs() < 1e-4, "expected 6.0, got {v}");
    }
}

// ── Registration ──────────────────────────────────────────────────────────────

/// Identical images → CC > 0.9 (analytical: perfect correlation).
#[test]
fn identity_registration_high_cc() {
    let dims = [10, 10, 10];
    let image = make_test_image(dims);
    let reg = MultiResSyNRegistration::new(make_config(2, vec![10, 10], false));
    let result = reg.register(&image, &image, dims, [1.0, 1.0, 1.0]).unwrap();
    assert!(
        result.final_cc > 0.9,
        "identity CC should be > 0.9, got {}",
        result.final_cc
    );
}

/// Single-level (num_levels=1) equivalent to standard SyN.
#[test]
fn single_level_equivalent_to_syn() {
    let dims = [8, 8, 8];
    let image = make_test_image(dims);
    let reg = MultiResSyNRegistration::new(make_config(1, vec![15], false));
    let result = reg.register(&image, &image, dims, [1.0, 1.0, 1.0]).unwrap();
    assert!(
        result.final_cc > 0.9,
        "single-level CC should be > 0.9, got {}",
        result.final_cc
    );
}

/// Multi-res on translated pair: non-divergence, non-trivial fields.
#[test]
fn multires_registration_non_divergence() {
    let dims = [12, 12, 16];
    let n = dims[0] * dims[1] * dims[2];
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 2);
    let reg = MultiResSyNRegistration::new(make_config(2, vec![10, 15], false));
    let result = reg
        .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
        .unwrap();

    let rms = |f: &[f32]| -> f64 {
        (f.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n as f64).sqrt()
    };
    let fwd_x = rms(&result.forward_field.2);
    let inv_x = rms(&result.inverse_field.2);
    assert!(
        fwd_x > 0.001 || inv_x > 0.001,
        "x-field must be non-trivial: fwd={fwd_x:.6} inv={inv_x:.6}"
    );
    assert!(
        result.final_cc > 0.8,
        "CC must be > 0.8, got {}",
        result.final_cc
    );
    for &v in result
        .forward_field
        .0
        .iter()
        .chain(result.forward_field.1.iter())
        .chain(result.forward_field.2.iter())
        .chain(result.inverse_field.0.iter())
        .chain(result.inverse_field.1.iter())
        .chain(result.inverse_field.2.iter())
    {
        assert!(v.is_finite(), "non-finite value: {v}");
    }
}

/// Inverse consistency produces finite fields with high CC.
#[test]
fn inverse_consistency_produces_finite_fields() {
    let dims = [10, 10, 12];
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 2);
    let reg = MultiResSyNRegistration::new(make_config(2, vec![8, 12], true));
    let result = reg
        .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
        .unwrap();
    for &v in result
        .forward_field
        .0
        .iter()
        .chain(result.forward_field.1.iter())
        .chain(result.forward_field.2.iter())
        .chain(result.inverse_field.0.iter())
        .chain(result.inverse_field.1.iter())
        .chain(result.inverse_field.2.iter())
    {
        assert!(v.is_finite(), "IC field non-finite: {v}");
    }
    assert!(
        result.final_cc > 0.8,
        "IC CC should be > 0.8, got {}",
        result.final_cc
    );
}

// ── Error cases ───────────────────────────────────────────────────────────────

#[test]
fn mismatched_fixed_length_returns_error() {
    let dims = [4, 4, 4];
    let reg = MultiResSyNRegistration::new(make_config(1, vec![5], false));
    let err = reg.register(&vec![0.0_f32; 80], &vec![0.0_f32; 64], dims, [1.0; 3]);
    assert!(err.is_err());
    assert!(format!("{}", err.unwrap_err()).contains("fixed length"));
}

#[test]
fn mismatched_moving_length_returns_error() {
    let dims = [4, 4, 4];
    let reg = MultiResSyNRegistration::new(make_config(1, vec![5], false));
    let err = reg.register(&vec![0.0_f32; 64], &vec![0.0_f32; 80], dims, [1.0; 3]);
    assert!(err.is_err());
    assert!(format!("{}", err.unwrap_err()).contains("moving length"));
}

#[test]
fn invalid_iterations_per_level_returns_error() {
    let dims = [4, 4, 4];
    let img = vec![0.0_f32; 64];
    let reg = MultiResSyNRegistration::new(make_config(3, vec![5, 5], false));
    let err = reg.register(&img, &img, dims, [1.0; 3]);
    assert!(err.is_err());
    assert!(format!("{}", err.unwrap_err()).contains("iterations_per_level"));
}

#[test]
fn zero_levels_returns_error() {
    let dims = [4, 4, 4];
    let img = vec![0.0_f32; 64];
    let reg = MultiResSyNRegistration::new(make_config(0, vec![], false));
    assert!(reg.register(&img, &img, dims, [1.0; 3]).is_err());
}

// ── CC primitive tests ────────────────────────────────────────────────────────

/// Identical non-constant images → CC ≈ 1.0 (analytical: perfect correlation).
#[test]
fn mean_local_cc_identical_images() {
    let dims = [6, 6, 6];
    let image = make_test_image(dims);
    let cc = mean_local_cc(&image, &image, dims, 1);
    assert!(
        cc > 0.99,
        "CC of identical images should be ≈ 1.0, got {cc}"
    );
}

/// Constant images → CC = 0 (zero variance, degenerate).
#[test]
fn mean_local_cc_constant_images_is_zero() {
    let dims = [5, 5, 5];
    let a = vec![3.0_f32; 125];
    let cc = mean_local_cc(&a, &a, dims, 1);
    assert!(cc.is_finite(), "CC must be finite, got {cc}");
    assert!(
        cc.abs() < 1e-6,
        "CC of constant images should be 0, got {cc}"
    );
}

/// CC forces on identical images are bounded (at optimum, gradient is small).
#[test]
fn cc_forces_identical_images_bounded() {
    let dims = [6, 6, 6];
    let n = 216;
    let image = make_test_image(dims);
    let (gz, gy, gx) =
        crate::deformable_field_ops::compute_gradient(&image, dims, [1.0, 1.0, 1.0]);
    let (fz, fy, fx) = cc_forces(&image, &image, &gz, &gy, &gx, dims, 1);
    let rms = |f: &[f32]| -> f64 {
        (f.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n as f64).sqrt()
    };
    assert!(
        rms(&fz) < 10.0 && rms(&fy) < 10.0 && rms(&fx) < 10.0,
        "CC forces on identical images should be bounded"
    );
}
