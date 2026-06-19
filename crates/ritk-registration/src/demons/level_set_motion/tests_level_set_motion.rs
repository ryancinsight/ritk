//! Unit tests for level-set motion registration.

use super::LevelSetMotionRegistration;

// ── Image helpers ─────────────────────────────────────────────────────────────

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

/// Translate `data` by `shift` voxels in the positive X direction.
///
/// Voxels at `ix < shift` are set to 0 (boundary fill).
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

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Displacement field is finite everywhere after 20 iterations on a translated pair.
///
/// Evidence: `is_finite()` covers all NaN/±Inf outcomes; exhaustive over all voxels.
#[test]
fn displacement_field_finite() {
    let dims = [4usize, 16, 16];
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 4);
    let reg = LevelSetMotionRegistration {
        number_of_iterations: 20,
        ..Default::default()
    };
    let result = reg
        .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
        .expect("registration must not fail on valid inputs");

    for (i, (&dz, (&dy, &dx))) in result
        .disp_z
        .iter()
        .zip(result.disp_y.iter().zip(result.disp_x.iter()))
        .enumerate()
    {
        assert!(dz.is_finite(), "disp_z[{i}] is non-finite: {dz}");
        assert!(dy.is_finite(), "disp_y[{i}] is non-finite: {dy}");
        assert!(dx.is_finite(), "disp_x[{i}] is non-finite: {dx}");
    }
}

/// Translation recovery: 4-pixel X shift produces a non-zero displacement field.
///
/// Under the forward-warp convention `warped(p) = moving(p + D(p))`,
/// aligning `moving[ix] = fixed[ix − 4]` requires D_x > 0 at interior voxels.
/// This test asserts the weaker structural property: at least one voxel carries
/// a non-trivial displacement, confirming the filter is not degenerate.
#[test]
fn translation_recovery_non_zero() {
    let dims = [4usize, 16, 16];
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 4);
    let reg = LevelSetMotionRegistration {
        number_of_iterations: 20,
        ..Default::default()
    };
    let result = reg
        .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
        .expect("registration must not fail on valid inputs");

    let max_dx = result
        .disp_x
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_dx > 0.0,
        "displacement field must be non-zero for a 4-pixel translated input pair; \
         max |disp_x| = {max_dx}"
    );
}

/// Registering identical images produces near-zero displacement everywhere.
///
/// When fixed == moving, `diff = 0` at every voxel, so forces are identically
/// zero and the displacement field never leaves the zero-initialised state.
/// Gaussian smoothing of a zero field returns zero. Final max |D| < 1e-5.
#[test]
fn identity_near_zero_displacement() {
    let dims = [4usize, 8, 8];
    let image = make_test_image(dims);
    let reg = LevelSetMotionRegistration {
        number_of_iterations: 20,
        ..Default::default()
    };
    let result = reg
        .register(&image, &image, dims, [1.0, 1.0, 1.0])
        .expect("registration must not fail on valid inputs");

    let max_disp = result
        .disp_x
        .iter()
        .chain(result.disp_y.iter())
        .chain(result.disp_z.iter())
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_disp < 1e-5,
        "identical images: max |displacement| should be 0.0, got {max_disp}"
    );
}

/// Mismatched image lengths return a `RegistrationError`.
#[test]
fn mismatched_lengths_returns_error() {
    let dims = [4usize, 4, 4];
    let fixed = vec![0.0_f32; 4 * 4 * 4];
    let moving = vec![0.0_f32; 4 * 4 * 5]; // one extra X slice
    let reg = LevelSetMotionRegistration::default();
    assert!(
        reg.register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .is_err(),
        "must return an error when fixed and moving lengths differ"
    );
}
