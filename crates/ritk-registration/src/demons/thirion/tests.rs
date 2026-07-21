//! Unit tests for Thirion Demons registration.

use super::super::config::DemonsConfig;
use super::registration::ThirionDemonsRegistration;

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
            for ix in 0..nx {
                if ix >= shift {
                    let src = iz * ny * nx + iy * nx + (ix - shift);
                    out[iz * ny * nx + iy * nx + ix] = data[src];
                }
            }
        }
    }
    out
}

/// Registering identical images must produce near-zero MSE.
#[test]
fn identity_registration_near_zero_mse() {
    let dims = [8usize, 8, 8];
    let image = make_test_image(dims);
    let reg = ThirionDemonsRegistration::new(DemonsConfig {
        max_iterations: 20,
        ..Default::default()
    });
    let result = reg
        .register(&image, &image, dims, [1.0, 1.0, 1.0])
        .expect("infallible: validated precondition");
    assert!(
        result.final_mse < 1e-3,
        "identity MSE should be < 1e-3, got {}",
        result.final_mse
    );
}

/// MSE must be lower after registration than before.
#[test]
fn registration_reduces_mse() {
    let dims = [10usize, 10, 14];
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 2);

    let initial_mse: f64 = fixed
        .iter()
        .zip(moving.iter())
        .map(|(&f, &m)| ((f - m) as f64).powi(2))
        .sum::<f64>()
        / n as f64;

    let reg = ThirionDemonsRegistration::new(DemonsConfig {
        max_iterations: 50,
        ..Default::default()
    });
    let result = reg
        .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
        .expect("infallible: validated precondition");

    assert!(
        result.final_mse < initial_mse,
        "MSE should decrease: initial={initial_mse:.6} final={:.6}",
        result.final_mse
    );
    assert!(
        result.final_mse < initial_mse * 0.5,
        "MSE should decrease by at least 50%: initial={initial_mse:.6} final={:.6}",
        result.final_mse
    );
}

/// Translation recovery: moving = translate(fixed, +2 voxels in x).
///
/// Under the forward-warp convention `warped(p) = moving(p + D(p))`,
/// aligning `moving[ix] = fixed[ix−2]` requires `D_x = +2` so that
/// `warped(ix) = moving(ix + 2) = fixed(ix + 2 − 2) = fixed(ix)`.
/// Therefore the mean interior disp_x must be **positive**.
#[test]
fn translation_recovery_direction() {
    let dims = [8usize, 8, 12];
    let [nz, ny, nx] = dims;
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 2);

    let reg = ThirionDemonsRegistration::new(DemonsConfig {
        max_iterations: 50,
        ..Default::default()
    });
    let result = reg
        .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
        .expect("infallible: validated precondition");

    let mut sum_dx = 0.0_f64;
    let mut count = 0usize;
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 2..nx - 2 {
                let fi = iz * ny * nx + iy * nx + ix;
                sum_dx += result.disp_x[fi] as f64;
                count += 1;
            }
        }
    }
    let mean_dx = sum_dx / count as f64;

    assert!(
        mean_dx > 0.0,
        "mean interior disp_x should be positive (≈+2) for forward-warp convention, got {mean_dx:.4}"
    );
}

/// Registering a constant image against itself produces zero displacement.
#[test]
fn constant_image_zero_forces() {
    let dims = [6usize, 6, 6];
    let n = 6 * 6 * 6;
    let image = vec![42.0_f32; n];
    let reg = ThirionDemonsRegistration::new(DemonsConfig {
        max_iterations: 10,
        ..Default::default()
    });
    let result = reg
        .register(&image, &image, dims, [1.0, 1.0, 1.0])
        .expect("infallible: validated precondition");

    let max_disp = result
        .disp_x
        .iter()
        .chain(result.disp_y.iter())
        .chain(result.disp_z.iter())
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_disp < 1e-4,
        "constant image: max displacement should be ~0, got {max_disp}"
    );
}

/// Error is returned when fixed and moving have different lengths.
#[test]
fn mismatched_lengths_returns_error() {
    let dims = [4usize, 4, 4];
    let fixed = vec![0.0_f32; 4 * 4 * 4];
    let moving = vec![0.0_f32; 4 * 4 * 5];
    let reg = ThirionDemonsRegistration::new(DemonsConfig::default());
    assert!(
        reg.register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .is_err(),
        "should return error for mismatched lengths"
    );
}

/// Mean displacement magnitude is finite after registration.
#[test]
fn displacement_field_finite() {
    let dims = [6usize, 6, 8];
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 1);
    let reg = ThirionDemonsRegistration::new(DemonsConfig {
        max_iterations: 20,
        ..Default::default()
    });
    let result = reg
        .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
        .expect("infallible: validated precondition");
    for (&dz, (&dy, &dx)) in result
        .disp_z
        .iter()
        .zip(result.disp_y.iter().zip(result.disp_x.iter()))
    {
        assert!(dz.is_finite(), "disp_z contains non-finite value: {dz}");
        assert!(dy.is_finite(), "disp_y contains non-finite value: {dy}");
        assert!(dx.is_finite(), "disp_x contains non-finite value: {dx}");
    }
}
