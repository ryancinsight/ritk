use super::*;

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
    let reg = SymmetricDemonsRegistration::new(DemonsConfig {
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

    let reg = SymmetricDemonsRegistration::new(DemonsConfig {
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
        "MSE should decrease by ≥50%: initial={initial_mse:.6} final={:.6}",
        result.final_mse
    );
}

/// Approximate symmetry: register(F, M) and register(M, F) produce
/// displacements that are approximately negatives of each other.
///
/// # Invariant verified
/// For the x-displacement in the interior region:
///   |mean(disp_x_FM) + mean(disp_x_MF)| < |mean(disp_x_FM)| × 0.5
///
/// This is a loose symmetry check appropriate for discrete finite-difference
/// implementations with boundary effects.
#[test]
fn approximate_symmetry_fm_vs_mf() {
    let dims = [8usize, 8, 12];
    let [nz, ny, nx] = dims;
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 2);

    let reg = SymmetricDemonsRegistration::new(DemonsConfig {
        max_iterations: 30,
        ..Default::default()
    });

    let res_fm = reg
        .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
        .expect("infallible: validated precondition");
    let res_mf = reg
        .register(&moving, &fixed, dims, [1.0, 1.0, 1.0])
        .expect("infallible: validated precondition");

    // Compute mean interior disp_x for FM and MF.
    let mut sum_fm = 0.0_f64;
    let mut sum_mf = 0.0_f64;
    let mut count = 0usize;
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 2..nx - 2 {
                let fi = iz * ny * nx + iy * nx + ix;
                sum_fm += res_fm.disp_x[fi] as f64;
                sum_mf += res_mf.disp_x[fi] as f64;
                count += 1;
            }
        }
    }
    let mean_fm = sum_fm / count as f64;
    let mean_mf = sum_mf / count as f64;

    // FM and MF displacements should have opposite signs.
    assert!(
        mean_fm * mean_mf < 0.0,
        "FM and MF mean disp_x should have opposite signs: \
         mean_fm={mean_fm:.4} mean_mf={mean_mf:.4}"
    );

    // Magnitude of (FM + MF) should be less than 50% of FM magnitude.
    let asymmetry = (mean_fm + mean_mf).abs();
    let scale = mean_fm.abs().max(1e-6);
    assert!(
        asymmetry < scale * 0.5,
        "Asymmetry too large: |FM+MF|={asymmetry:.4} vs |FM|={:.4}",
        mean_fm.abs()
    );
}

/// All displacement field components must be finite.
#[test]
fn displacement_field_finite() {
    let dims = [6usize, 6, 8];
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 1);
    let reg = SymmetricDemonsRegistration::new(DemonsConfig {
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

/// Error is returned for mismatched image lengths.
#[test]
fn mismatched_lengths_returns_error() {
    let dims = [4usize, 4, 4];
    let fixed = vec![0.0_f32; 4 * 4 * 4];
    let moving = vec![0.0_f32; 4 * 4 * 5];
    let reg = SymmetricDemonsRegistration::new(DemonsConfig::default());
    assert!(
        reg.register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .is_err(),
        "should return error for mismatched lengths"
    );
}
