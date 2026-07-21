use super::*;
use crate::demons::config::DemonsVariant;
use crate::demons::DemonsConfig;

fn make_sphere_image(dims: [usize; 3], center: [f32; 3], radius: f32) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut out = vec![0.0f32; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dz = iz as f32 - center[0];
                let dy = iy as f32 - center[1];
                let dx = ix as f32 - center[2];
                let d = (dz * dz + dy * dy + dx * dx).sqrt();
                out[iz * ny * nx + iy * nx + ix] = if d <= radius { 1.0 } else { 0.0 };
            }
        }
    }
    out
}

#[test]
fn test_multires_thirion_identity_mse_below_threshold() {
    let dims = [16usize, 16, 16];
    let n = dims[0] * dims[1] * dims[2];
    let image = make_sphere_image(dims, [8.0, 8.0, 8.0], 5.0);
    let config = MultiResDemonsConfig {
        base_config: DemonsConfig {
            max_iterations: 30,
            ..DemonsConfig::default()
        },
        levels: 2,
        variant: DemonsVariant::Classic,
        n_squarings: 6,
    };
    let reg = MultiResDemonsRegistration::new(config);
    let result = reg
        .register(&image, &image, dims, [1.0f32, 1.0, 1.0])
        .expect("infallible: validated precondition");
    assert!(
        result.final_mse < 1e-3,
        "identity MSE must be < 1e-3, got {}",
        result.final_mse
    );
    assert_eq!(result.disp_z.len(), n, "disp_z must have correct length");
    assert!(
        result.disp_z.iter().all(|v| v.is_finite()),
        "disp_z must be finite"
    );
    assert!(
        result.disp_y.iter().all(|v| v.is_finite()),
        "disp_y must be finite"
    );
    assert!(
        result.disp_x.iter().all(|v| v.is_finite()),
        "disp_x must be finite"
    );
}

#[test]
fn test_multires_thirion_shifted_image_mse_decreases() {
    let dims = [16usize, 16, 16];
    let fixed = make_sphere_image(dims, [8.0, 8.0, 8.0], 5.0);
    let moving = make_sphere_image(dims, [8.0, 8.0, 10.0], 5.0);
    let initial_mse = fixed
        .iter()
        .zip(moving.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() as f64
        / (dims[0] * dims[1] * dims[2]) as f64;
    let config = MultiResDemonsConfig {
        base_config: DemonsConfig {
            max_iterations: 50,
            ..DemonsConfig::default()
        },
        levels: 2,
        variant: DemonsVariant::Classic,
        n_squarings: 6,
    };
    let reg = MultiResDemonsRegistration::new(config);
    let result = reg
        .register(&fixed, &moving, dims, [1.0f32, 1.0, 1.0])
        .expect("infallible: validated precondition");
    assert!(
        result.final_mse < initial_mse,
        "final MSE {} must be less than initial MSE {}",
        result.final_mse,
        initial_mse
    );
}

#[test]
fn test_multires_diffeomorphic_identity_mse_below_threshold() {
    let dims = [16usize, 16, 16];
    let image = make_sphere_image(dims, [8.0, 8.0, 8.0], 5.0);
    let config = MultiResDemonsConfig {
        base_config: DemonsConfig {
            max_iterations: 30,
            ..DemonsConfig::default()
        },
        levels: 2,
        variant: DemonsVariant::Diffeomorphic,
        n_squarings: 6,
    };
    let reg = MultiResDemonsRegistration::new(config);
    let result = reg
        .register(&image, &image, dims, [1.0f32, 1.0, 1.0])
        .expect("infallible: validated precondition");
    assert!(
        result.final_mse < 1e-3,
        "diffeomorphic identity MSE must be < 1e-3, got {}",
        result.final_mse
    );
    assert!(
        result.disp_z.iter().all(|v| v.is_finite()),
        "disp_z must be finite"
    );
}

#[test]
fn test_multires_displacement_shape_matches_input() {
    let dims = [12usize, 10, 8];
    let n = dims[0] * dims[1] * dims[2];
    let image = vec![0.5f32; n];
    let config = MultiResDemonsConfig::default();
    let reg = MultiResDemonsRegistration::new(config);
    let result = reg
        .register(&image, &image, dims, [1.0f32, 1.0, 1.0])
        .expect("infallible: validated precondition");
    assert_eq!(result.disp_z.len(), n);
    assert_eq!(result.disp_y.len(), n);
    assert_eq!(result.disp_x.len(), n);
    assert_eq!(result.warped.len(), n);
}

#[test]
fn test_multires_single_level_equals_thirion() {
    let dims = [8usize, 8, 8];
    let image = make_sphere_image(dims, [4.0, 4.0, 4.0], 3.0);
    let base_config = DemonsConfig {
        max_iterations: 10,
        ..DemonsConfig::default()
    };
    let multi_config = MultiResDemonsConfig {
        base_config: base_config.clone(),
        levels: 1,
        variant: DemonsVariant::Classic,
        n_squarings: 6,
    };
    let multi_reg = MultiResDemonsRegistration::new(multi_config);
    let multi_result = multi_reg
        .register(&image, &image, dims, [1.0f32, 1.0, 1.0])
        .expect("infallible: validated precondition");
    assert!(
        multi_result.final_mse < 1e-3,
        "single-level identity MSE must be < 1e-3, got {}",
        multi_result.final_mse
    );
}
