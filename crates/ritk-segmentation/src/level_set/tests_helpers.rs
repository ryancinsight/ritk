use super::*;
use std::f64::consts::PI;

// ── idx_clamped ────────────────────────────────────────────────────────────

#[test]
fn test_idx_clamped_interior() {
    assert_eq!(idx_clamped(1, 2, 3, 4, 5, 6), 5 * 6 + 2 * 6 + 3);
}

#[test]
fn test_idx_clamped_negative() {
    assert_eq!(idx_clamped(-1, -2, -3, 4, 5, 6), 0);
}

#[test]
fn test_idx_clamped_overflow() {
    assert_eq!(idx_clamped(10, 10, 10, 4, 5, 6), 3 * 5 * 6 + 4 * 6 + 5);
}

// ── Gaussian kernel ────────────────────────────────────────────────────────

#[test]
fn test_gaussian_kernel_normalised() {
    let kernel = ritk_filter::gaussian_kernel_1d(1.5_f64, Some(5));
    let sum: f64 = kernel.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "kernel sum = {sum}");
}

#[test]
fn test_gaussian_kernel_length() {
    let kernel = ritk_filter::gaussian_kernel_1d(1.0_f64, Some(5));
    assert_eq!(kernel.len(), 11);
}

#[test]
fn test_gaussian_kernel_symmetric() {
    let kernel = ritk_filter::gaussian_kernel_1d(2.0_f64, Some(6));
    let n = kernel.len();
    for i in 0..n {
        assert!(
            (kernel[i] - kernel[n - 1 - i]).abs() < 1e-15,
            "asymmetry at i={i}: {} vs {}",
            kernel[i],
            kernel[n - 1 - i]
        );
    }
}

#[test]
fn test_gaussian_kernel_peak_at_center() {
    let kernel = ritk_filter::gaussian_kernel_1d(1.0_f64, Some(3));
    let center = kernel.len() / 2;
    for (i, &w) in kernel.iter().enumerate() {
        if i != center {
            assert!(
                kernel[center] >= w,
                "center {} < kernel[{i}] = {w}",
                kernel[center]
            );
        }
    }
}

// ── Regularised Heaviside ──────────────────────────────────────────────────

#[test]
fn test_heaviside_at_zero() {
    assert!((regularised_heaviside(0.0, 1.0) - 0.5).abs() < 1e-15);
}

#[test]
fn test_heaviside_positive_large() {
    let h = regularised_heaviside(1e6, 1.0);
    assert!((h - 1.0).abs() < 1e-6, "H(1e6) = {h}");
}

#[test]
fn test_heaviside_negative_large() {
    let h = regularised_heaviside(-1e6, 1.0);
    assert!(h.abs() < 1e-6, "H(-1e6) = {h}");
}

#[test]
fn test_heaviside_monotone() {
    let vals: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
    for w in vals.windows(2) {
        assert!(
            regularised_heaviside(w[1], 1.0) >= regularised_heaviside(w[0], 1.0),
            "monotonicity violated at z = {}",
            w[0]
        );
    }
}

// ── Regularised Dirac ──────────────────────────────────────────────────────

#[test]
fn test_dirac_peak_at_zero() {
    let eps = 1.0;
    let expected = 1.0 / (PI * eps);
    let got = regularised_dirac(0.0, eps);
    assert!(
        (got - expected).abs() < 1e-15,
        "delta(0,1) = {got}, expected {expected}"
    );
}

#[test]
fn test_dirac_positive_everywhere() {
    for i in -100..=100 {
        let z = i as f64 * 0.3;
        assert!(regularised_dirac(z, 1.0) > 0.0, "delta({z}) <= 0");
    }
}

#[test]
fn test_dirac_symmetric() {
    for i in 1..=50 {
        let z = i as f64 * 0.2;
        let d_pos = regularised_dirac(z, 1.0);
        let d_neg = regularised_dirac(-z, 1.0);
        assert!(
            (d_pos - d_neg).abs() < 1e-15,
            "asymmetry at z = {z}: {d_pos} vs {d_neg}"
        );
    }
}

// ── Curvature ──────────────────────────────────────────────────────────────

#[test]
fn test_curvature_flat_phi_is_zero() {
    let dims = [5, 5, 5];
    let n = 125;
    let phi = vec![7.0_f64; n];
    let mut kappa = vec![0.0_f64; n];
    compute_curvature_into(&phi, dims, &mut kappa);
    for (i, &k) in kappa.iter().enumerate() {
        assert!(k.abs() < 1e-6, "kappa[{i}] = {k} for constant phi");
    }
}

#[test]
fn test_curvature_sphere_positive() {
    // Signed distance function for a sphere of radius R=3 centred in
    // an 11^3 grid.  Analytical mean curvature on the surface = 2/R.
    let side = 11_usize;
    let dims = [side, side, side];
    let n = side * side * side;
    let center = 5.0_f64;
    let r = 3.0_f64;

    let mut phi = vec![0.0_f64; n];
    for iz in 0..side {
        for iy in 0..side {
            for ix in 0..side {
                let dz = iz as f64 - center;
                let dy = iy as f64 - center;
                let dx = ix as f64 - center;
                phi[iz * side * side + iy * side + ix] = (dz * dz + dy * dy + dx * dx).sqrt() - r;
            }
        }
    }

    let mut kappa = vec![0.0_f64; n];
    compute_curvature_into(&phi, dims, &mut kappa);

    let expected_kappa = 2.0 / r; // 0.6667
                                  // Check voxels on the surface (|phi| < 0.6).
    let mut checked = 0_usize;
    for iz in 1..side - 1 {
        for iy in 1..side - 1 {
            for ix in 1..side - 1 {
                let idx = iz * side * side + iy * side + ix;
                if phi[idx].abs() < 0.6 {
                    let err = (kappa[idx] - expected_kappa).abs();
                    assert!(
                        err < 0.15,
                        "kappa[{iz},{iy},{ix}] = {}, expected ~{expected_kappa}, err = {err}",
                        kappa[idx]
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0, "no surface voxels found");
}

// ── Edge stopping ──────────────────────────────────────────────────────────

#[test]
fn test_edge_stopping_at_zero() {
    let g = compute_edge_stopping(&[0.0], 1.0);
    assert!((g[0] - 1.0).abs() < 1e-15);
}

#[test]
fn test_edge_stopping_large_gradient() {
    let g = compute_edge_stopping(&[1e6], 1.0);
    assert!(g[0] < 1e-10, "g(1e6) = {}", g[0]);
}

#[test]
fn test_edge_stopping_bounded() {
    let vals: Vec<f64> = (0..200).map(|i| i as f64 * 0.5).collect();
    let g = compute_edge_stopping(&vals, 5.0);
    for (i, &gi) in g.iter().enumerate() {
        assert!(gi > 0.0, "g[{i}] = {gi} <= 0");
        assert!(gi <= 1.0, "g[{i}] = {gi} > 1");
    }
}

// ── Gaussian smoothing ─────────────────────────────────────────────────────

#[test]
fn test_gaussian_smooth_identity_for_constant() {
    let dims = [4, 5, 6];
    let n = 4 * 5 * 6;
    let data = vec![42.0_f64; n];
    let smoothed = gaussian_smooth_3d(&data, dims, 1.5);
    for (i, &v) in smoothed.iter().enumerate() {
        assert!(
            (v - 42.0).abs() < 1e-10,
            "smoothed[{i}] = {v}, expected 42.0"
        );
    }
}

#[test]
fn test_gaussian_smooth_sigma_zero_is_identity() {
    let dims = [3, 3, 3];
    let data: Vec<f64> = (0..27).map(|i| i as f64).collect();
    let smoothed = gaussian_smooth_3d(&data, dims, 0.0);
    assert_eq!(smoothed, data);
}

// ── Gradient magnitude ─────────────────────────────────────────────────────

#[test]
fn test_gradient_magnitude_constant_is_zero() {
    let dims = [4, 5, 6];
    let n = 4 * 5 * 6;
    let data = vec![std::f64::consts::PI; n];
    let grad = compute_gradient_magnitude(&data, dims);
    for (i, &g) in grad.iter().enumerate() {
        assert!(g.abs() < 1e-15, "grad[{i}] = {g} for constant field");
    }
}

#[test]
fn test_gradient_magnitude_linear_ramp() {
    // f(z, y, x) = x  ->  interior |grad f| = 1.0
    let (nz, ny, nx) = (5, 5, 8);
    let dims = [nz, ny, nx];
    let n = nz * ny * nx;
    let mut data = vec![0.0_f64; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                data[iz * ny * nx + iy * nx + ix] = ix as f64;
            }
        }
    }
    let grad = compute_gradient_magnitude(&data, dims);
    // Interior voxels (1 <= ix <= nx-2) have exact central diff = 1.0.
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 1..nx - 1 {
                let idx = iz * ny * nx + iy * nx + ix;
                assert!(
                    (grad[idx] - 1.0).abs() < 1e-12,
                    "grad[{iz},{iy},{ix}] = {}, expected 1.0",
                    grad[idx]
                );
            }
        }
    }
}

// ── Field gradient ─────────────────────────────────────────────────────────

#[test]
fn test_field_gradient_constant() {
    let dims = [4, 5, 6];
    let n = 4 * 5 * 6;
    let data = vec![99.0_f64; n];
    let (gz, gy, gx) = compute_field_gradient(&data, dims);
    for i in 0..n {
        assert!(gz[i].abs() < 1e-15, "gz[{i}] = {}", gz[i]);
        assert!(gy[i].abs() < 1e-15, "gy[{i}] = {}", gy[i]);
        assert!(gx[i].abs() < 1e-15, "gx[{i}] = {}", gx[i]);
    }
}
