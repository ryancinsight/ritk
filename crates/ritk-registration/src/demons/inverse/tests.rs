//! Unit tests for displacement field inversion.

use super::displacement::{invert_displacement_field, warp_displacement_into, InverseFieldConfig};
use super::svf::invert_velocity_field;

fn warp_displacement(
    disp_z: &[f32],
    disp_y: &[f32],
    disp_x: &[f32],
    query_z: &[f32],
    query_y: &[f32],
    query_x: &[f32],
    dims: [usize; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = dims[0] * dims[1] * dims[2];
    let mut out_z = vec![0.0_f32; n];
    let mut out_y = vec![0.0_f32; n];
    let mut out_x = vec![0.0_f32; n];
    warp_displacement_into(
        disp_z, disp_y, disp_x, query_z, query_y, query_x, dims, &mut out_z, &mut out_y, &mut out_x,
    );
    for v in &mut out_z {
        *v = -*v;
    }
    for v in &mut out_y {
        *v = -*v;
    }
    for v in &mut out_x {
        *v = -*v;
    }
    (out_z, out_y, out_x)
}

/// The exact inverse of the zero velocity field is the zero velocity field.
///
/// Base case: `exp(0) ∘ exp(−0) = id ∘ id = id`.
#[test]
fn test_velocity_field_negation_is_exact_inverse() {
    let n = 4 * 4 * 4;
    let vel_z = vec![0.0_f32; n];
    let vel_y = vec![0.0_f32; n];
    let vel_x = vec![0.0_f32; n];

    let (inv_z, inv_y, inv_x) = invert_velocity_field(&vel_z, &vel_y, &vel_x);

    assert_eq!(inv_z.len(), n, "inv_z length mismatch");
    assert_eq!(inv_y.len(), n, "inv_y length mismatch");
    assert_eq!(inv_x.len(), n, "inv_x length mismatch");

    assert!(
        inv_z.iter().all(|&v| v == 0.0),
        "inv_z must be identically zero for the zero velocity field"
    );
    assert!(
        inv_y.iter().all(|&v| v == 0.0),
        "inv_y must be identically zero for the zero velocity field"
    );
    assert!(
        inv_x.iter().all(|&v| v == 0.0),
        "inv_x must be identically zero for the zero velocity field"
    );
}

/// The iterative inverse of a zero displacement field (identity map) is zero.
#[test]
fn test_invert_displacement_identity_field_is_zero() {
    let dims = [8usize, 8, 8];
    let n = dims[0] * dims[1] * dims[2];

    let disp_z = vec![0.0_f32; n];
    let disp_y = vec![0.0_f32; n];
    let disp_x = vec![0.0_f32; n];

    let config = InverseFieldConfig::default();
    let (inv_z, inv_y, inv_x, iters) =
        invert_displacement_field(&disp_z, &disp_y, &disp_x, dims, &config);

    assert!(iters >= 1, "at least one iteration must be performed");
    assert!(
        iters <= config.max_iterations,
        "iters {iters} exceeds max_iterations {}",
        config.max_iterations
    );

    let bound = 1e-5_f32;
    for i in 0..n {
        assert!(
            inv_z[i].abs() <= bound,
            "inv_z[{i}] = {} exceeds {bound}",
            inv_z[i]
        );
        assert!(
            inv_y[i].abs() <= bound,
            "inv_y[{i}] = {} exceeds {bound}",
            inv_y[i]
        );
        assert!(
            inv_x[i].abs() <= bound,
            "inv_x[{i}] = {} exceeds {bound}",
            inv_x[i]
        );
    }
}

/// The inverse of a uniform x-translation by +2.0 voxels has mean
/// x-displacement in [−2.1, −1.9].
#[test]
fn test_invert_small_translation() {
    let dims = [16usize, 16, 16];
    let n = dims[0] * dims[1] * dims[2];

    let disp_z = vec![0.0_f32; n];
    let disp_y = vec![0.0_f32; n];
    let disp_x = vec![2.0_f32; n];

    let config = InverseFieldConfig::default();
    let (inv_z, inv_y, inv_x, _iters) =
        invert_displacement_field(&disp_z, &disp_y, &disp_x, dims, &config);

    let mean_inv_x: f64 = inv_x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    assert!(
        (-2.1..=-1.9).contains(&mean_inv_x),
        "mean(inv_x) = {mean_inv_x:.6}, expected in [−2.1, −1.9]"
    );

    let mean_inv_z: f64 = inv_z.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mean_inv_y: f64 = inv_y.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    assert!(
        mean_inv_z.abs() < 1e-5,
        "mean(inv_z) = {mean_inv_z:.6}, expected near 0"
    );
    assert!(
        mean_inv_y.abs() < 1e-5,
        "mean(inv_y) = {mean_inv_y:.6}, expected near 0"
    );
}

/// Composing a non-trivial sinusoidal displacement field with its computed
/// inverse must yield a composition displacement below 0.5 voxels everywhere.
#[test]
fn test_invert_result_composition_near_identity() {
    use std::f32::consts::PI;

    let dims = [12usize, 12, 12];
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    let mut disp_z = vec![0.0_f32; n];
    let mut disp_y = vec![0.0_f32; n];
    let mut disp_x = vec![0.0_f32; n];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let i = iz * ny * nx + iy * nx + ix;
                disp_x[i] = 2.0 * (PI * ix as f32 / (nx as f32 - 1.0)).sin();
                disp_y[i] = 1.5 * (PI * iy as f32 / (ny as f32 - 1.0)).sin();
                disp_z[i] = 1.0 * (PI * iz as f32 / (nz as f32 - 1.0)).sin();
            }
        }
    }

    let config = InverseFieldConfig {
        max_iterations: 20,
        tolerance: 1e-6,
    };
    let (inv_z, inv_y, inv_x, _iters) =
        invert_displacement_field(&disp_z, &disp_y, &disp_x, dims, &config);

    let (comp_z, comp_y, comp_x) =
        warp_displacement(&inv_z, &inv_y, &inv_x, &disp_z, &disp_y, &disp_x, dims);

    let mut max_err = 0.0_f32;
    for i in 0..n {
        let ez = disp_z[i] + comp_z[i];
        let ey = disp_y[i] + comp_y[i];
        let ex = disp_x[i] + comp_x[i];
        let err = (ez * ez + ey * ey + ex * ex).sqrt();
        if err > max_err {
            max_err = err;
        }
    }

    assert!(
        max_err < 0.5,
        "max composition error {max_err:.6} voxels exceeds 0.5-voxel bound"
    );
}

/// `invert_displacement_field` returns at most `config.max_iterations` iterations.
#[test]
fn test_max_iterations_bound() {
    use std::f32::consts::PI;

    let dims = [8usize, 8, 8];
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    let disp_x: Vec<f32> = (0..n)
        .map(|i| {
            let ix = i % nx;
            2.0 * (PI * ix as f32 / (nx as f32 - 1.0)).sin()
        })
        .collect();
    let disp_z = vec![0.0_f32; n];
    let disp_y = vec![0.0_f32; n];

    debug_assert_eq!(n, nz * ny * nx);

    let max_iter = 5usize;
    let config = InverseFieldConfig {
        max_iterations: max_iter,
        tolerance: 1e-30,
    };

    let (_, _, _, iters) = invert_displacement_field(&disp_z, &disp_y, &disp_x, dims, &config);

    assert!(
        iters <= max_iter,
        "returned {iters} iterations exceeds max_iterations {max_iter}"
    );
    assert_eq!(
        iters, max_iter,
        "expected exactly {max_iter} iterations with sub-epsilon tolerance, got {iters}"
    );
}
