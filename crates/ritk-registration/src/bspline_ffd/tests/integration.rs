use crate::deformable_field_ops::{flat, trilinear_interpolate};
use crate::error::RegistrationError;
use super::super::config::BSplineFFDConfig;
use super::super::metric::compute_ncc;
use super::super::registration::BSplineFFDRegistration;
use super::make_test_image;

#[test]
fn metric_improves_after_iterations() {
    let dims = [8, 10, 12];
    let spacing = [1.0, 1.0, 1.0];
    let fixed = make_test_image(dims);

    // Create a translated version of the fixed image (shift +1 in x).
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut moving = vec![0.0_f32; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let src_x = (ix as f32 + 1.0).min((nx - 1) as f32);
                moving[flat(iz, iy, ix, ny, nx)] =
                    trilinear_interpolate(&fixed, dims, iz as f32, iy as f32, src_x);
            }
        }
    }

    let config = BSplineFFDConfig {
        initial_control_spacing: [4, 4, 4],
        num_levels: 1,
        max_iterations_per_level: 10,
        learning_rate: 0.5,
        regularization_weight: 0.0,
        convergence_threshold: 1e-8,
    };

    let initial_ncc = compute_ncc(&fixed, &moving);

    let result =
        BSplineFFDRegistration::register(&fixed, &moving, dims, spacing, &config).unwrap();

    assert!(
        result.final_metric >= initial_ncc - 1e-6,
        "metric should not degrade: initial={}, final={}",
        initial_ncc,
        result.final_metric
    );
    assert_eq!(result.warped_moving.len(), n);
    assert_eq!(result.control_grid_dims[0] * result.control_grid_dims[1] * result.control_grid_dims[2],
               result.control_points.0.len());
}

#[test]
fn mismatched_fixed_length_returns_error() {
    let config = BSplineFFDConfig::default();
    let fixed = vec![0.0_f32; 100];
    let moving = vec![0.0_f32; 8];
    let result =
        BSplineFFDRegistration::register(&fixed, &moving, [2, 2, 2], [1.0; 3], &config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, RegistrationError::DimensionMismatch(_)),
        "expected DimensionMismatch, got {:?}",
        err
    );
}

#[test]
fn mismatched_moving_length_returns_error() {
    let config = BSplineFFDConfig::default();
    let fixed = vec![0.0_f32; 8];
    let moving = vec![0.0_f32; 100];
    let result =
        BSplineFFDRegistration::register(&fixed, &moving, [2, 2, 2], [1.0; 3], &config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, RegistrationError::DimensionMismatch(_)),
        "expected DimensionMismatch, got {:?}",
        err
    );
}

#[test]
fn zero_levels_returns_invalid_configuration() {
    let config = BSplineFFDConfig { num_levels: 0, ..Default::default() };
    let img = vec![0.0_f32; 8];
    let result = BSplineFFDRegistration::register(&img, &img, [2, 2, 2], [1.0; 3], &config);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        RegistrationError::InvalidConfiguration(_)
    ));
}

#[test]
fn zero_spacing_returns_invalid_configuration() {
    let config = BSplineFFDConfig {
        initial_control_spacing: [0, 4, 4],
        ..Default::default()
    };
    let img = vec![0.0_f32; 8];
    let result = BSplineFFDRegistration::register(&img, &img, [2, 2, 2], [1.0; 3], &config);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        RegistrationError::InvalidConfiguration(_)
    ));
}
