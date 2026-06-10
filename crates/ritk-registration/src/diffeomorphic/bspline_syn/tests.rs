use super::super::local_cc::mean_local_cc;
use super::primitives::{accumulate_to_cp, bspline_basis, cp_count, cp_laplacian, evaluate_dense};
use super::{BSplineSyNConfig, BSplineSyNRegistration};
use crate::deformable_field_ops::flat;

fn make_default_config() -> BSplineSyNConfig {
    BSplineSyNConfig {
        max_iterations: 15,
        control_spacing: [3, 3, 3],
        sigma_smooth: 1.5,
        convergence_threshold: 1e-7,
        convergence_window: 10,
        n_squarings: 6,
        cc_window_radius: 2,
        gradient_step: 0.25,
        regularization_weight: 0.01,
    }
}

/// Smooth test image: `I[z,y,x] = sin(π·z/nz) · cos(π·y/ny) · (x + 1)`.
/// Analytically derived to produce non-trivial gradients in all three axes.
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

/// Shift image +`shift` voxels in x with zero-padding at the left boundary.
fn translate_x(data: &[f32], dims: [usize; 3], shift: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let mut out = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in shift..nx {
                out[iz * ny * nx + iy * nx + ix] = data[iz * ny * nx + iy * nx + (ix - shift)];
            }
        }
    }
    out
}

// ── B-spline basis tests ──────────────────────────────────────────────────────

/// Partition of unity: `Σ_{k=0}^{3} Bₖ(u) = 1` for all `u ∈ [0, 1]`.
/// Verified at 101 uniformly-spaced parameter values (analytically exact
/// for uniform cubic B-splines).
#[test]
fn bspline_basis_partition_of_unity() {
    for i in 0..=100 {
        let u = i as f64 / 100.0;
        let sum: f64 = (0..4).map(|k| bspline_basis(k, u)).sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "B-spline basis at u={u} sums to {sum}, expected 1.0"
        );
    }
}

/// Boundary values: `B₀(0) = 1/6`, `B₁(0) = 4/6`, `B₂(0) = 1/6`, `B₃(0) = 0`.
#[test]
fn bspline_basis_boundary_values() {
    let tol = 1e-14;
    assert!((bspline_basis(0, 0.0) - 1.0 / 6.0).abs() < tol);
    assert!((bspline_basis(1, 0.0) - 4.0 / 6.0).abs() < tol);
    assert!((bspline_basis(2, 0.0) - 1.0 / 6.0).abs() < tol);
    assert!(bspline_basis(3, 0.0).abs() < tol);

    assert!(bspline_basis(0, 1.0).abs() < tol);
    assert!((bspline_basis(1, 1.0) - 1.0 / 6.0).abs() < tol);
    assert!((bspline_basis(2, 1.0) - 4.0 / 6.0).abs() < tol);
    assert!((bspline_basis(3, 1.0) - 1.0 / 6.0).abs() < tol);
}

/// All basis values are non-negative for `u ∈ [0, 1]`.
#[test]
fn bspline_basis_non_negative() {
    for i in 0..=1000 {
        let u = i as f64 / 1000.0;
        for k in 0..4 {
            let v = bspline_basis(k, u);
            assert!(v >= -1e-15, "B_{k}({u}) = {v} is negative");
        }
    }
}

// ── CP lattice tests ──────────────────────────────────────────────────────────

/// `cp_count` formula: `(dim - 1) / spacing + 4` for `dim > 1`.
#[test]
fn cp_count_formula() {
    assert_eq!(cp_count(10, 3), 3 + 4); // (9/3) + 4 = 7
    assert_eq!(cp_count(12, 4), 2 + 4); // (11/4) + 4 = 6
    assert_eq!(cp_count(1, 5), 4); // edge case: single voxel
    assert_eq!(cp_count(13, 3), 4 + 4); // (12/3) + 4 = 8
}

/// Evaluating a constant CP lattice produces a constant dense field equal
/// to the CP value (by partition of unity).
#[test]
fn constant_cp_produces_constant_field() {
    let dims = [8, 8, 8];
    let cs = [3, 3, 3];
    let cp_dims = [cp_count(8, 3), cp_count(8, 3), cp_count(8, 3)];
    let cp_n = cp_dims[0] * cp_dims[1] * cp_dims[2];
    let cp = vec![5.0_f32; cp_n];
    let dense = evaluate_dense(&cp, cp_dims, dims, cs);
    assert_eq!(dense.len(), 8 * 8 * 8);
    for (i, &v) in dense.iter().enumerate() {
        assert!((v - 5.0).abs() < 1e-5, "voxel {i}: expected 5.0, got {v}");
    }
}

/// Laplacian of a constant CP lattice is zero (no curvature).
#[test]
fn laplacian_constant_cp_is_zero() {
    let cp_dims = [5, 5, 5];
    let cp_n = 5 * 5 * 5;
    let cp = vec![3.0_f32; cp_n];
    let lap = cp_laplacian(&cp, cp_dims);
    for (i, &v) in lap.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "Laplacian of constant field at {i} should be 0, got {v}"
        );
    }
}

/// Laplacian at the centre of a CP lattice with a single non-zero point
/// matches the analytical 6-connected discrete Laplacian.
#[test]
fn laplacian_single_spike() {
    let cp_dims = [5, 5, 5];
    let cp_n = 5 * 5 * 5;
    let mut cp = vec![0.0_f32; cp_n];
    let centre = flat(2, 2, 2, 5, 5);
    cp[centre] = 1.0;
    let lap = cp_laplacian(&cp, cp_dims);
    // At centre: Δ = 0 + 0 + 0 + 0 + 0 + 0 − 6 · 1 = −6
    assert!(
        (lap[centre] - (-6.0)).abs() < 1e-6,
        "centre Laplacian should be -6.0, got {}",
        lap[centre]
    );
    // At each face neighbour: Δ = 1 − 6·0 = 1 (if interior with 6 neighbours)
    // but if the neighbour is not on the boundary of the lattice and only has
    // one non-zero neighbour (the centre), Δ = 1 - count*0 = 1.
    let nb = flat(2, 2, 3, 5, 5);
    assert!(
        (lap[nb] - 1.0).abs() < 1e-6,
        "neighbour Laplacian should be 1.0, got {}",
        lap[nb]
    );
}

/// Accumulation of a constant force field to CPs yields the constant value
/// (by the weighted-average normalisation).
#[test]
fn accumulate_constant_force() {
    let dims = [8, 8, 8];
    let cs = [3, 3, 3];
    let cp_dims = [cp_count(8, 3), cp_count(8, 3), cp_count(8, 3)];
    let force = vec![2.0_f32; 8 * 8 * 8];
    let acc = accumulate_to_cp(&force, dims, cp_dims, cs);
    for (i, &v) in acc.iter().enumerate() {
        assert!(
            (v - 2.0).abs() < 1e-4,
            "CP {i}: accumulated constant force should be 2.0, got {v}"
        );
    }
}

// ── Registration tests ────────────────────────────────────────────────────────

/// Registering identical images produces CC > 0.9.
#[test]
fn identity_registration_high_cc() {
    let dims = [10, 10, 10];
    let image = make_test_image(dims);
    let cfg = make_default_config();
    let reg = BSplineSyNRegistration::new(cfg);
    let result = reg.register(&image, &image, dims, [1.0, 1.0, 1.0]).unwrap();
    assert!(
        result.final_cc > 0.9,
        "identity registration CC should be > 0.9, got {}",
        result.final_cc
    );
}

/// BSplineSyN on a translated pair: non-divergence and non-trivial fields.
#[test]
fn bspline_registration_non_divergence() {
    let dims = [12, 12, 16];
    let n = dims[0] * dims[1] * dims[2];
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 2);

    let cfg = BSplineSyNConfig {
        max_iterations: 20,
        control_spacing: [4, 4, 4],
        sigma_smooth: 1.5,
        convergence_threshold: 1e-7,
        convergence_window: 10,
        n_squarings: 6,
        cc_window_radius: 2,
        gradient_step: 0.25,
        regularization_weight: 0.01,
    };
    let reg = BSplineSyNRegistration::new(cfg);
    let result = reg
        .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
        .unwrap();

    // Velocity fields must have non-trivial x-magnitude.
    let fwd_rms_x: f64 = (result
        .forward_field
        .x
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    let inv_rms_x: f64 = (result
        .inverse_field
        .x
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    assert!(
        fwd_rms_x > 0.001 || inv_rms_x > 0.001,
        "at least one x-field must be non-trivial: fwd={fwd_rms_x:.6} inv={inv_rms_x:.6}"
    );

    // CC must remain high.
    assert!(
        result.final_cc > 0.8,
        "final CC must be > 0.8, got {}",
        result.final_cc
    );

    // All field values must be finite.
    for &v in result
        .forward_field
        .z
        .iter()
        .chain(result.forward_field.y.iter())
        .chain(result.forward_field.x.iter())
        .chain(result.inverse_field.z.iter())
        .chain(result.inverse_field.y.iter())
        .chain(result.inverse_field.x.iter())
    {
        assert!(v.is_finite(), "field contains non-finite value: {v}");
    }
}

/// B-spline velocity fields are intrinsically smooth: the dense field
/// Laplacian RMS must be bounded. This is a structural property of the
/// cubic B-spline representation.
#[test]
fn bspline_field_smoothness() {
    let dims = [10, 10, 10];
    let n = dims[0] * dims[1] * dims[2];
    let fixed = make_test_image(dims);
    let moving = translate_x(&fixed, dims, 1);

    let cfg = make_default_config();
    let reg = BSplineSyNRegistration::new(cfg);
    let result = reg
        .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
        .unwrap();

    // Compute discrete Laplacian of the dense forward x-field.
    let vx = &result.forward_field.x;
    let [nz, ny, nx] = dims;
    let mut lap_ss = 0.0_f64;
    let mut field_ss = 0.0_f64;
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let fi = flat(iz, iy, ix, ny, nx);
                let c = vx[fi] as f64;
                let lap = vx[flat(iz - 1, iy, ix, ny, nx)] as f64
                    + vx[flat(iz + 1, iy, ix, ny, nx)] as f64
                    + vx[flat(iz, iy - 1, ix, ny, nx)] as f64
                    + vx[flat(iz, iy + 1, ix, ny, nx)] as f64
                    + vx[flat(iz, iy, ix - 1, ny, nx)] as f64
                    + vx[flat(iz, iy, ix + 1, ny, nx)] as f64
                    - 6.0 * c;
                lap_ss += lap * lap;
                field_ss += c * c;
            }
        }
    }
    let field_rms = (field_ss / n as f64).sqrt();
    let lap_rms = (lap_ss / n as f64).sqrt();
    // For a zero field (identity registration), both are near zero.
    // For non-trivial fields, the Laplacian RMS should be small relative
    // to the field RMS, or both should be small.
    if field_rms > 1e-6 {
        let ratio = lap_rms / field_rms;
        assert!(
            ratio < 50.0,
            "Laplacian/field RMS ratio {ratio:.2} too large; field not smooth"
        );
    }
    // In any case, the Laplacian should not be enormous.
    assert!(
        lap_rms < 100.0,
        "Laplacian RMS {lap_rms:.4} too large for B-spline field"
    );
}

// ── Error-case tests ──────────────────────────────────────────────────────────

/// Mismatched fixed-image length returns DimensionMismatch.
#[test]
fn mismatched_fixed_length_returns_error() {
    let dims = [4, 4, 4];
    let fixed = vec![0.0_f32; 4 * 4 * 5]; // wrong length
    let moving = vec![0.0_f32; 4 * 4 * 4];
    let cfg = make_default_config();
    let reg = BSplineSyNRegistration::new(cfg);
    let err = reg.register(&fixed, &moving, dims, [1.0, 1.0, 1.0]);
    assert!(err.is_err(), "should error for mismatched fixed length");
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("fixed length"),
        "error should mention fixed: {msg}"
    );
}

/// Mismatched moving-image length returns DimensionMismatch.
#[test]
fn mismatched_moving_length_returns_error() {
    let dims = [4, 4, 4];
    let fixed = vec![0.0_f32; 4 * 4 * 4];
    let moving = vec![0.0_f32; 4 * 4 * 5]; // wrong length
    let cfg = make_default_config();
    let reg = BSplineSyNRegistration::new(cfg);
    let err = reg.register(&fixed, &moving, dims, [1.0, 1.0, 1.0]);
    assert!(err.is_err(), "should error for mismatched moving length");
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("moving length"),
        "error should mention moving: {msg}"
    );
}

/// Zero control spacing returns InvalidConfiguration.
#[test]
fn zero_control_spacing_returns_error() {
    let dims = [4, 4, 4];
    let image = vec![0.0_f32; 4 * 4 * 4];
    let mut cfg = make_default_config();
    cfg.control_spacing = [0, 3, 3];
    let reg = BSplineSyNRegistration::new(cfg);
    let err = reg.register(&image, &image, dims, [1.0, 1.0, 1.0]);
    assert!(err.is_err(), "should error for zero control spacing");
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("control_spacing"),
        "error should mention control_spacing: {msg}"
    );
}

// ── CC primitive tests ────────────────────────────────────────────────────────

/// mean_local_cc of identical non-constant images is close to 1.0.
#[test]
fn mean_local_cc_identical_images() {
    let dims = [6, 6, 6];
    let image = make_test_image(dims);
    let cc = mean_local_cc(&image, &image, dims, 1);
    assert!(
        cc > 0.99,
        "CC of identical images should be ≈ 1.0, got {cc}"
    );
}

/// mean_local_cc of constant images is 0 (zero variance → degenerate).
#[test]
fn mean_local_cc_constant_images_is_zero() {
    let dims = [5, 5, 5];
    let a = vec![3.0_f32; 5 * 5 * 5];
    let cc = mean_local_cc(&a, &a, dims, 1);
    assert!(
        cc.is_finite(),
        "CC of constant images must be finite, got {cc}"
    );
    assert!(
        cc.abs() < 1e-6,
        "CC of constant images should be 0, got {cc}"
    );
}
