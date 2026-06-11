//! Unit tests for inverse-consistent diffeomorphic Demons registration.

use super::engine::InverseConsistentDiffeomorphicDemonsRegistration;
use super::types::InverseConsistentDemonsConfig;
use crate::demons::config::DemonsConfig;
use crate::demons::diffeomorphic::DiffeomorphicDemonsRegistration;
use ritk_filter::GaussianSigma;

fn default_config() -> InverseConsistentDemonsConfig {
    InverseConsistentDemonsConfig {
        demons: DemonsConfig {
            max_iterations: 20,
            sigma_diffusion: Some(GaussianSigma::new_unchecked(1.0)),
            sigma_fluid: None,
            max_step_length: 2.0,
        },
        inverse_consistency_weight: 0.5,
        n_squarings: 6,
    }
}

fn make_image(nz: usize, ny: usize, nx: usize) -> Vec<f32> {
    let n = nz * ny * nx;
    (0..n)
        .map(|i| {
            let (z, ry) = (i / (ny * nx), i % (ny * nx));
            let (y, x) = (ry / nx, ry % nx);
            let cz = nz as f32 / 2.0;
            let cy = ny as f32 / 2.0;
            let cx = nx as f32 / 2.0;
            let r2 = (z as f32 - cz).powi(2) + (y as f32 - cy).powi(2) + (x as f32 - cx).powi(2);
            (-(r2 / (2.0 * 4.0_f32.powi(2)))).exp()
        })
        .collect()
}

#[test]
fn test_identity_registration_has_near_zero_mse() {
    let dims = [16usize, 16, 16];
    let img = make_image(16, 16, 16);
    let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(default_config());
    let result = reg.register(&img, &img, dims, [1.0, 1.0, 1.0]).unwrap();
    assert!(
        result.final_mse < 1e-4,
        "identity MSE must be < 1e-4; got {}",
        result.final_mse
    );
}

#[test]
fn test_ic_residual_near_zero_for_identity_registration() {
    let dims = [16usize, 16, 16];
    let img = make_image(16, 16, 16);
    let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(default_config());
    let result = reg.register(&img, &img, dims, [1.0, 1.0, 1.0]).unwrap();
    assert!(
        result.inverse_consistency_residual < 1e-3,
        "IC residual must be < 1e-3; got {}",
        result.inverse_consistency_residual
    );
}

#[test]
fn test_registration_reduces_mse() {
    let (nz, ny, nx) = (16, 16, 16);
    let n = nz * ny * nx;
    let flat = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;
    let fixed = make_image(nz, ny, nx);
    let mut moving = vec![0.0_f32; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 3..nx {
                moving[flat(z, y, x - 3)] = fixed[flat(z, y, x)];
            }
        }
    }
    let initial_mse: f64 = fixed
        .iter()
        .zip(moving.iter())
        .map(|(&fi, &mi)| {
            let d = (fi - mi) as f64;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(default_config());
    let result = reg
        .register(&fixed, &moving, [nz, ny, nx], [1.0, 1.0, 1.0])
        .unwrap();
    assert!(
        result.final_mse < initial_mse,
        "registration must reduce MSE: initial={initial_mse:.6} final={:.6}",
        result.final_mse
    );
}

#[test]
fn test_forward_and_inverse_fields_have_same_length() {
    let dims = [12usize, 12, 12];
    let n = dims[0] * dims[1] * dims[2];
    let img = make_image(12, 12, 12);
    let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(default_config());
    let result = reg.register(&img, &img, dims, [1.0, 1.0, 1.0]).unwrap();
    assert_eq!(result.disp_z.len(), n);
    assert_eq!(result.disp_y.len(), n);
    assert_eq!(result.disp_x.len(), n);
    assert_eq!(result.inv_disp_z.len(), n);
    assert_eq!(result.inv_disp_y.len(), n);
    assert_eq!(result.inv_disp_x.len(), n);
}

#[test]
fn test_all_displacement_values_finite() {
    let dims = [12usize, 12, 12];
    let img = make_image(12, 12, 12);
    let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(default_config());
    let result = reg.register(&img, &img, dims, [1.0, 1.0, 1.0]).unwrap();
    for (&dz, (&dy, &dx)) in result
        .disp_z
        .iter()
        .zip(result.disp_y.iter().zip(result.disp_x.iter()))
    {
        assert!(
            dz.is_finite() && dy.is_finite() && dx.is_finite(),
            "forward disp must be finite: ({dz},{dy},{dx})"
        );
    }
    for (&dz, (&dy, &dx)) in result
        .inv_disp_z
        .iter()
        .zip(result.inv_disp_y.iter().zip(result.inv_disp_x.iter()))
    {
        assert!(
            dz.is_finite() && dy.is_finite() && dx.is_finite(),
            "inverse disp must be finite: ({dz},{dy},{dx})"
        );
    }
}

#[test]
fn test_weight_zero_matches_standard_diffeomorphic() {
    let (nz, ny, nx) = (12, 12, 12);
    let img = make_image(nz, ny, nx);
    let dims = [nz, ny, nx];
    let spacing = [1.0, 1.0, 1.0];

    let config_ic = InverseConsistentDemonsConfig {
        demons: DemonsConfig {
            max_iterations: 10,
            sigma_diffusion: Some(GaussianSigma::new_unchecked(1.0)),
            sigma_fluid: None,
            max_step_length: 2.0,
        },
        inverse_consistency_weight: 0.0,
        n_squarings: 6,
    };
    let reg_ic = InverseConsistentDiffeomorphicDemonsRegistration::new(config_ic);
    let result_ic = reg_ic.register(&img, &img, dims, spacing).unwrap();

    let config_std = DemonsConfig {
        max_iterations: 10,
        sigma_diffusion: Some(GaussianSigma::new_unchecked(1.0)),
        sigma_fluid: None,
        max_step_length: 2.0,
    };
    let reg_std = DiffeomorphicDemonsRegistration::with_squarings(config_std, 6);
    let result_std = reg_std.register(&img, &img, dims, spacing).unwrap();

    assert!(
        (result_ic.final_mse - result_std.final_mse).abs() < 1e-8,
        "w=0 IC must match standard: ic={:.9} std={:.9}",
        result_ic.final_mse,
        result_std.final_mse
    );
}

#[test]
fn test_ic_residual_decreases_with_symmetric_weight() {
    let dims = [12usize, 12, 12];
    let img = make_image(12, 12, 12);

    let reg_fwd =
        InverseConsistentDiffeomorphicDemonsRegistration::new(InverseConsistentDemonsConfig {
            demons: DemonsConfig {
                max_iterations: 15,
                sigma_diffusion: Some(GaussianSigma::new_unchecked(1.0)),
                sigma_fluid: None,
                max_step_length: 2.0,
            },
            inverse_consistency_weight: 0.0,
            n_squarings: 6,
        });
    let reg_sym =
        InverseConsistentDiffeomorphicDemonsRegistration::new(InverseConsistentDemonsConfig {
            demons: DemonsConfig {
                max_iterations: 15,
                sigma_diffusion: Some(GaussianSigma::new_unchecked(1.0)),
                sigma_fluid: None,
                max_step_length: 2.0,
            },
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
        });
    let result_fwd = reg_fwd.register(&img, &img, dims, [1.0; 3]).unwrap();
    let result_sym = reg_sym.register(&img, &img, dims, [1.0; 3]).unwrap();

    assert!(
        result_fwd.inverse_consistency_residual < 1e-3,
        "forward IC residual must be < 1e-3: {}",
        result_fwd.inverse_consistency_residual
    );
    assert!(
        result_sym.inverse_consistency_residual < 1e-3,
        "symmetric IC residual must be < 1e-3: {}",
        result_sym.inverse_consistency_residual
    );
}

#[test]
fn test_shape_mismatch_returns_error() {
    let fixed = vec![0.0_f32; 100];
    let moving = vec![0.0_f32; 200];
    let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(default_config());
    let result = reg.register(&fixed, &moving, [4, 5, 5], [1.0, 1.0, 1.0]);
    assert!(result.is_err(), "shape mismatch must return Err");
}

#[test]
fn test_fixed_mismatch_returns_error() {
    let fixed = vec![0.0_f32; 50];
    let moving = vec![0.0_f32; 125];
    let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(default_config());
    let result = reg.register(&fixed, &moving, [5, 5, 5], [1.0, 1.0, 1.0]);
    assert!(result.is_err(), "fixed length mismatch must return Err");
}
