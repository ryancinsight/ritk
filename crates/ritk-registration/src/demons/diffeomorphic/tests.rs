//! Unit tests for Diffeomorphic Demons registration.

use super::registration::DiffeomorphicDemonsRegistration;
use crate::deformable_field_ops::compose_fields;
use crate::demons::config::DemonsConfig;

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

/// Registering identical images must yield near-zero MSE.
#[test]
fn identity_registration_near_zero_mse() {
    let dims = [8usize, 8, 8];
    let image = make_test_image(dims);
    let reg = DiffeomorphicDemonsRegistration::new(DemonsConfig {
        max_iterations: 20,
        ..Default::default()
    });
    let result = reg.register(&image, &image, dims, [1.0, 1.0, 1.0]).expect("infallible: validated precondition");
    assert!(
        result.final_mse < 1e-3,
        "identity MSE should be < 1e-3, got {}",
        result.final_mse
    );
}

/// MSE must decrease after registration of translated images.
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

    let reg = DiffeomorphicDemonsRegistration::new(DemonsConfig {
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
        "MSE should decrease by ≥50 %: initial={initial_mse:.6} final={:.6}",
        result.final_mse
    );
}

/// All components of the final displacement field must be finite.
#[test]
fn displacement_field_finite() {
    let dims = [6usize, 6, 8];
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 1);
    let reg = DiffeomorphicDemonsRegistration::new(DemonsConfig {
        max_iterations: 15,
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
        assert!(dz.is_finite(), "disp_z non-finite: {dz}");
        assert!(dy.is_finite(), "disp_y non-finite: {dy}");
        assert!(dx.is_finite(), "disp_x non-finite: {dx}");
    }
}

/// Error is returned for length-mismatched inputs.
#[test]
fn mismatched_lengths_returns_error() {
    let dims = [4usize, 4, 4];
    let fixed = vec![0.0_f32; 4 * 4 * 4];
    let moving = vec![0.0_f32; 4 * 4 * 5];
    let reg = DiffeomorphicDemonsRegistration::new(DemonsConfig::default());
    assert!(
        reg.register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .is_err(),
        "should return error for mismatched lengths"
    );
}

/// The scaling-and-squaring exponential map of a zero velocity field is zero.
#[test]
fn zero_velocity_zero_displacement() {
    use crate::deformable_field_ops::compute_mse_streaming;
    use crate::deformable_field_ops::scaling_and_squaring;
    let dims = [4usize, 4, 4];
    let n = 4 * 4 * 4;
    let image = make_test_image(dims);
    let zero = vec![0.0_f32; n];
    let phi = scaling_and_squaring(&zero, &zero, &zero, dims.into(), 6);
    let mse = compute_mse_streaming(&image, &image, dims.into(), &phi.z, &phi.y, &phi.x);
    assert!(mse < 1e-10, "zero velocity should give zero MSE, got {mse}");
}

/// Diffeomorphic Demons results must retain the stationary velocity field.
#[test]
fn result_retains_stationary_velocity_field() {
    let dims = [6usize, 6, 8];
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 1);
    let reg = DiffeomorphicDemonsRegistration::with_squarings(
        DemonsConfig {
            max_iterations: 10,
            ..Default::default()
        },
        4,
    );

    let result = reg
        .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
        .expect("infallible: validated precondition");

    let n = dims[0] * dims[1] * dims[2];
    assert_eq!(
        result.vel_z.as_ref().map(Vec::len),
        Some(n),
        "vel_z must be present and match voxel count"
    );
    assert_eq!(
        result.vel_y.as_ref().map(Vec::len),
        Some(n),
        "vel_y must be present and match voxel count"
    );
    assert_eq!(
        result.vel_x.as_ref().map(Vec::len),
        Some(n),
        "vel_x must be present and match voxel count"
    );
}

/// The exact inverse computed from the retained SVF composes near identity.
#[test]
fn exact_inverse_composes_to_near_identity() {
    let dims = [8usize, 8, 10];
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 1);
    let reg = DiffeomorphicDemonsRegistration::with_squarings(
        DemonsConfig {
            max_iterations: 20,
            ..Default::default()
        },
        4,
    );

    let result = reg
        .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
        .expect("infallible: validated precondition");
    let inv = reg.invert_result(&result, dims);
    let comp = compose_fields(
        &inv.z,
        &inv.y,
        &inv.x,
        &result.disp_z,
        &result.disp_y,
        &result.disp_x,
        dims.into(),
    );

    let n = dims[0] * dims[1] * dims[2];
    let mut max_err = 0.0_f32;
    let mut mean_err = 0.0_f64;

    for i in 0..n {
        let err = (comp.z[i] * comp.z[i] + comp.y[i] * comp.y[i] + comp.x[i] * comp.x[i]).sqrt();
        max_err = max_err.max(err);
        mean_err += err as f64;
    }
    mean_err /= n as f64;

    assert!(
        max_err < 0.35,
        "forward/inverse composition max error {max_err:.6} exceeds 0.35 voxels"
    );
    assert!(
        mean_err < 0.08,
        "forward/inverse composition mean error {mean_err:.6} exceeds 0.08 voxels"
    );
}

/// The inverse returned by `invert_result` must equal `exp(-v)`.
#[test]
fn invert_result_matches_negated_velocity_exponential() {
    use crate::deformable_field_ops::scaling_and_squaring;
    let dims = [6usize, 6, 8];
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 1);
    let reg = DiffeomorphicDemonsRegistration::with_squarings(
        DemonsConfig {
            max_iterations: 12,
            ..Default::default()
        },
        5,
    );

    let result = reg
        .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
        .expect("infallible: validated precondition");
    let inv = reg.invert_result(&result, dims);

    let vel_z = result.vel_z.as_ref().expect("vel_z must be present");
    let vel_y = result.vel_y.as_ref().expect("vel_y must be present");
    let vel_x = result.vel_x.as_ref().expect("vel_x must be present");

    let neg_vel_z: Vec<f32> = vel_z.iter().map(|&v| -v).collect();
    let neg_vel_y: Vec<f32> = vel_y.iter().map(|&v| -v).collect();
    let neg_vel_x: Vec<f32> = vel_x.iter().map(|&v| -v).collect();

    let expected = scaling_and_squaring(
        &neg_vel_z,
        &neg_vel_y,
        &neg_vel_x,
        dims.into(),
        reg.n_squarings,
    );

    for i in 0..expected.z.len() {
        assert!(
            (inv.z[i] - expected.z[i]).abs() < 1e-5,
            "inv_z[{i}] mismatch: got {}, expected {}",
            inv.z[i],
            expected.z[i]
        );
        assert!(
            (inv.y[i] - expected.y[i]).abs() < 1e-5,
            "inv_y[{i}] mismatch: got {}, expected {}",
            inv.y[i],
            expected.y[i]
        );
        assert!(
            (inv.x[i] - expected.x[i]).abs() < 1e-5,
            "inv_x[{i}] mismatch: got {}, expected {}",
            inv.x[i],
            expected.x[i]
        );
    }
}
