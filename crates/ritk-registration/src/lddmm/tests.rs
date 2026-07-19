//! Unit and integration tests for LDDMM.

use super::{
    adjoint::epdiff_adjoint, geodesic::integrate_geodesic, LddmmConfig, LddmmRegistration,
};
use crate::deformable_field_ops::VectorField;
use crate::error::RegistrationError;
use ritk_filter::edge::GaussianSigma;

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
        kernel_sigma: GaussianSigma::new_unchecked(1.0),
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
        kernel_sigma: GaussianSigma::new_unchecked(1.0),
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
        kernel_sigma: GaussianSigma::new_unchecked(1.0),
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
    let d = integrate_geodesic(&zeros, &zeros, &zeros, dims, [1.0; 3], 5, 1.0);

    for i in 0..n {
        assert_eq!(d.z[i], 0.0, "dz[{}] = {} != 0", i, d.z[i]);
        assert_eq!(d.y[i], 0.0, "dy[{}] = {} != 0", i, d.y[i]);
        assert_eq!(d.x[i], 0.0, "dx[{}] = {} != 0", i, d.x[i]);
    }
}

#[test]
fn epdiff_adjoint_zero_momentum_is_zero() {
    let dims = [4, 4, 4];
    let n = 4 * 4 * 4;
    let v: Vec<f32> = (0..n).map(|i| 0.01 * i as f32).collect();
    let zeros = vec![0.0_f32; n];
    let ad = epdiff_adjoint(
        VectorField {
            z: &v,
            y: &v,
            x: &v,
        },
        VectorField {
            z: &zeros,
            y: &zeros,
            x: &zeros,
        },
        dims,
        [1.0; 3],
    );

    for i in 0..n {
        assert_eq!(ad.z[i], 0.0, "ad_z[{}] = {} != 0", i, ad.z[i]);
        assert_eq!(ad.y[i], 0.0, "ad_y[{}] = {} != 0", i, ad.y[i]);
        assert_eq!(ad.x[i], 0.0, "ad_x[{}] = {} != 0", i, ad.x[i]);
    }
}
