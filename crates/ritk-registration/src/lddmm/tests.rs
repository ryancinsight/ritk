//! Unit and integration tests for LDDMM.

use super::{
    adjoint::{epdiff_adjoint, epdiff_adjoint_into},
    geodesic::{integrate_geodesic, integrate_geodesic_into},
    LddmmConfig, LddmmRegistration,
};
use crate::error::RegistrationError;

fn make_test_image(dims: [usize; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    (0..n).map(|i| (i as f32) / (n as f32)).collect()
}

fn gaussian_blob(dims: [usize; 3], center: [f32; 3], sigma: f32) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    (0..n)
        .map(|i| {
            let iz = (i / (ny * nx)) as f32;
            let iy = ((i % (ny * nx)) / nx) as f32;
            let ix = (i % nx) as f32;
            let r2 = (iz - center[0]).powi(2) + (iy - center[1]).powi(2) + (ix - center[2]).powi(2);
            (-r2 * inv_2s2).exp()
        })
        .collect()
}

#[test]
fn identity_registration_low_mse() {
    let dims = [6, 6, 6];
    let img = make_test_image(dims);
    let reg = LddmmRegistration::new(LddmmConfig {
        max_iterations: 5,
        num_time_steps: 2,
        kernel_sigma: 1.0,
        learning_rate: 0.01,
        regularization_weight: 1.0,
        convergence_threshold: 1e-12,
    });
    let result = reg.register(&img, &img, dims, [1.0; 3]).unwrap();

    assert!(
        result.final_metric < 1e-10,
        "final_metric = {} exceeds 1e-10",
        result.final_metric
    );
    let max_disp = result
        .displacement_field
        .0
        .iter()
        .chain(result.displacement_field.1.iter())
        .chain(result.displacement_field.2.iter())
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_disp < 1e-6,
        "max displacement = {} exceeds 1e-6",
        max_disp
    );
}

#[test]
fn metric_improves_over_iterations() {
    let dims = [8, 8, 8];
    let center_f = [3.5_f32, 3.5, 3.5];
    let center_m = [3.5_f32, 3.5, 4.5];
    let fixed = gaussian_blob(dims, center_f, 2.0);
    let moving = gaussian_blob(dims, center_m, 2.0);

    let n = dims[0] * dims[1] * dims[2];
    let initial_mse: f64 = fixed
        .iter()
        .zip(moving.iter())
        .map(|(&f, &m)| {
            let d = (m - f) as f64;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    assert!(
        initial_mse > 1e-6,
        "initial_mse {} too small for meaningful test",
        initial_mse
    );

    let reg = LddmmRegistration::new(LddmmConfig {
        max_iterations: 30,
        num_time_steps: 2,
        kernel_sigma: 1.0,
        learning_rate: 0.1,
        regularization_weight: 0.01,
        convergence_threshold: 1e-12,
    });
    let result = reg.register(&fixed, &moving, dims, [1.0; 3]).unwrap();

    assert!(
        result.final_metric < initial_mse,
        "final_metric {} >= initial_mse {}",
        result.final_metric,
        initial_mse
    );
}

#[test]
fn displacement_field_is_finite() {
    let dims = [4, 4, 4];
    let img = make_test_image(dims);
    let reg = LddmmRegistration::new(LddmmConfig {
        max_iterations: 3,
        num_time_steps: 2,
        kernel_sigma: 1.0,
        ..LddmmConfig::default()
    });
    let result = reg.register(&img, &img, dims, [1.0; 3]).unwrap();

    for &v in result
        .displacement_field
        .0
        .iter()
        .chain(result.displacement_field.1.iter())
        .chain(result.displacement_field.2.iter())
    {
        assert!(v.is_finite(), "non-finite displacement value: {}", v);
    }
}

#[test]
fn mismatched_dims_returns_error() {
    let dims = [4, 4, 4];
    let n = 4 * 4 * 4;
    let img = vec![0.0_f32; n];
    let short = vec![0.0_f32; n - 1];
    let reg = LddmmRegistration::new(LddmmConfig::default());

    let err = reg.register(&img, &short, dims, [1.0; 3]);
    assert!(
        matches!(err, Err(RegistrationError::DimensionMismatch(_))),
        "expected DimensionMismatch for short moving, got {:?}",
        err
    );

    let err2 = reg.register(&short, &img, dims, [1.0; 3]);
    assert!(
        matches!(err2, Err(RegistrationError::DimensionMismatch(_))),
        "expected DimensionMismatch for short fixed, got {:?}",
        err2
    );
}

#[test]
fn geodesic_shooting_zero_velocity_produces_identity() {
    let dims = [4, 4, 4];
    let n = 4 * 4 * 4;
    let zeros = vec![0.0_f32; n];
    let (dz, dy, dx) = integrate_geodesic(&zeros, &zeros, &zeros, dims, [1.0; 3], 5, 1.0);

    for i in 0..n {
        assert_eq!(dz[i], 0.0, "dz[{}] = {} != 0", i, dz[i]);
        assert_eq!(dy[i], 0.0, "dy[{}] = {} != 0", i, dy[i]);
        assert_eq!(dx[i], 0.0, "dx[{}] = {} != 0", i, dx[i]);
    }
}

#[test]
fn epdiff_adjoint_zero_momentum_is_zero() {
    let dims = [4, 4, 4];
    let n = 4 * 4 * 4;
    let v: Vec<f32> = (0..n).map(|i| 0.01 * i as f32).collect();
    let zeros = vec![0.0_f32; n];
    let (az, ay, ax) = epdiff_adjoint(&v, &v, &v, &zeros, &zeros, &zeros, dims, [1.0; 3]);

    for i in 0..n {
        assert_eq!(az[i], 0.0, "ad_z[{}] = {} != 0", i, az[i]);
        assert_eq!(ay[i], 0.0, "ad_y[{}] = {} != 0", i, ay[i]);
        assert_eq!(ax[i], 0.0, "ad_x[{}] = {} != 0", i, ax[i]);
    }
}

#[test]
fn epdiff_adjoint_into_matches_allocating() {
    let dims = [5, 5, 5];
    let n = 5 * 5 * 5;
    let vz: Vec<f32> = (0..n).map(|i| 0.01 * i as f32).collect();
    let vy: Vec<f32> = (0..n).map(|i| 0.005 * i as f32).collect();
    let vx: Vec<f32> = (0..n).map(|i| -0.003 * i as f32).collect();
    let mz: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let my: Vec<f32> = (0..n).map(|i| (i % 11) as f32).collect();
    let mx: Vec<f32> = (0..n).map(|i| (i % 13) as f32).collect();
    let spacing = [1.2, 1.0, 0.8];

    let (ref_z, ref_y, ref_x) = epdiff_adjoint(&vz, &vy, &vx, &mz, &my, &mx, dims, spacing);

    let mut into_z = vec![0.0_f32; n];
    let mut into_y = vec![0.0_f32; n];
    let mut into_x = vec![0.0_f32; n];
    epdiff_adjoint_into(
        &vz,
        &vy,
        &vx,
        &mz,
        &my,
        &mx,
        dims,
        spacing,
        &mut into_z,
        &mut into_y,
        &mut into_x,
    );

    for i in 0..n {
        assert!(
            (ref_z[i] - into_z[i]).abs() < 1e-6,
            "ad_z[{}]: ref={} into={}",
            i,
            ref_z[i],
            into_z[i]
        );
        assert!(
            (ref_y[i] - into_y[i]).abs() < 1e-6,
            "ad_y[{}]: ref={} into={}",
            i,
            ref_y[i],
            into_y[i]
        );
        assert!(
            (ref_x[i] - into_x[i]).abs() < 1e-6,
            "ad_x[{}]: ref={} into={}",
            i,
            ref_x[i],
            into_x[i]
        );
    }
}

#[test]
fn integrate_geodesic_into_matches_allocating() {
    let dims = [4, 4, 4];
    let n = 4 * 4 * 4;
    let v0z: Vec<f32> = (0..n).map(|i| 0.005 * (i % 5) as f32).collect();
    let v0y: Vec<f32> = (0..n).map(|i| -0.003 * (i % 7) as f32).collect();
    let v0x: Vec<f32> = (0..n).map(|i| 0.002 * (i % 11) as f32).collect();
    let spacing = [1.0, 1.0, 1.0];
    let num_steps = 4;
    let kernel_sigma = 1.0;

    let (ref_z, ref_y, ref_x) =
        integrate_geodesic(&v0z, &v0y, &v0x, dims, spacing, num_steps, kernel_sigma);

    let mut dz = vec![0.0_f32; n];
    let mut dy = vec![0.0_f32; n];
    let mut dx = vec![0.0_f32; n];
    let mut smooth_tmp = vec![0.0_f32; n];
    let mut mz = vec![0.0_f32; n];
    let mut my = vec![0.0_f32; n];
    let mut mx = vec![0.0_f32; n];
    let mut adz = vec![0.0_f32; n];
    let mut ady = vec![0.0_f32; n];
    let mut adx = vec![0.0_f32; n];
    let mut step_z = vec![0.0_f32; n];
    let mut step_y = vec![0.0_f32; n];
    let mut step_x = vec![0.0_f32; n];
    let mut comp_z = vec![0.0_f32; n];
    let mut comp_y = vec![0.0_f32; n];
    let mut comp_x = vec![0.0_f32; n];
    integrate_geodesic_into(
        &v0z,
        &v0y,
        &v0x,
        dims,
        spacing,
        num_steps,
        kernel_sigma,
        &mut dz,
        &mut dy,
        &mut dx,
        &mut smooth_tmp,
        &mut mz,
        &mut my,
        &mut mx,
        &mut adz,
        &mut ady,
        &mut adx,
        &mut step_z,
        &mut step_y,
        &mut step_x,
        &mut comp_z,
        &mut comp_y,
        &mut comp_x,
    );

    for i in 0..n {
        assert!(
            (ref_z[i] - dz[i]).abs() < 1e-5,
            "disp_z[{}]: ref={} into={}",
            i,
            ref_z[i],
            dz[i]
        );
        assert!(
            (ref_y[i] - dy[i]).abs() < 1e-5,
            "disp_y[{}]: ref={} into={}",
            i,
            ref_y[i],
            dy[i]
        );
        assert!(
            (ref_x[i] - dx[i]).abs() < 1e-5,
            "disp_x[{}]: ref={} into={}",
            i,
            ref_x[i],
            dx[i]
        );
    }
}
