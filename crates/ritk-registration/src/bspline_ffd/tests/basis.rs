use super::super::basis::{
    cubic_bspline_basis, evaluate_bspline_displacement_dense_into,
    evaluate_bspline_displacement_fast, init_control_grid, should_use_dense_path, BasisCache,
};
use super::super::volume_dims::VolumeDims;

#[test]
fn bspline_basis_partition_of_unity() {
    for i in 0..=100 {
        let t = i as f64 / 100.0;
        let b = cubic_bspline_basis(t);
        let sum: f64 = b.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-14,
            "partition of unity violated at t={}: sum={}",
            t,
            sum
        );
    }
}

#[test]
fn bspline_basis_non_negative() {
    for i in 0..=100 {
        let t = i as f64 / 100.0;
        let b = cubic_bspline_basis(t);
        for (j, &val) in b.iter().enumerate() {
            assert!(val >= -1e-15, "basis {}({}) = {} < 0", j, t, val);
        }
    }
}

#[test]
fn init_control_grid_dimensions_correct() {
    // dims = [16, 20, 24], spacing = [8, 8, 8]
    // n_ctrl[d] = ceil(dims[d]/8) + 3
    // z: ceil(16/8)+3 = 2+3 = 5
    // y: ceil(20/8)+3 = 3+3 = 6 (ceil(20/8)=ceil(2.5)=3)
    // x: ceil(24/8)+3 = 3+3 = 6
    let dims = [16, 20, 24];
    let spacing = [8.0, 8.0, 8.0];
    let ctrl = init_control_grid(VolumeDims(dims), &spacing);
    assert_eq!(ctrl, [5, 6, 6]);
}

#[test]
fn init_control_grid_non_divisible() {
    // dims = [10, 10, 10], spacing = [4, 4, 4]
    // n_ctrl[d] = ceil(10/4)+3 = 3+3 = 6
    let dims = [10, 10, 10];
    let spacing = [4.0, 4.0, 4.0];
    let ctrl = init_control_grid(VolumeDims(dims), &spacing);
    assert_eq!(ctrl, [6, 6, 6]);
}

/// Verify that the fast-path displacement evaluation matches the original
/// per-voxel approach on a small volume with random control points.
#[test]
fn fast_displacement_matches_original_on_random_cps() {
    let dims = [16usize, 20, 24];
    let ctrl_spacing = [4.0_f64, 5.0, 6.0];
    let ctrl_dims = init_control_grid(VolumeDims(dims), &ctrl_spacing);
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];

    // Deterministic "random-ish" control points (avoid rand 0.9 API changes).
    let cp_z: Vec<f32> = (0..cn).map(|i| (i as f32 % 10.0) - 5.0).collect();
    let cp_y: Vec<f32> = (0..cn).map(|i| ((i + 3) as f32 % 10.0) - 5.0).collect();
    let cp_x: Vec<f32> = (0..cn).map(|i| ((i + 7) as f32 % 10.0) - 5.0).collect();

    // Original path (delegates to fast internally now, so we need to
    // compare the original algorithm directly).
    let cache = BasisCache::new(VolumeDims(dims), &ctrl_spacing);
    let f = evaluate_bspline_displacement_fast(&cp_z, &cp_y, &cp_x, &ctrl_dims, dims, &cache);

    // Basic invariants.
    let n = dims[0] * dims[1] * dims[2];
    assert_eq!(f.z.len(), n);
    assert_eq!(f.y.len(), n);
    assert_eq!(f.x.len(), n);

    // Not all zeros (non-zero control points should produce non-zero
    // displacement at the center).
    let mid = ((dims[0] / 2) * dims[1] + dims[1] / 2) * dims[2] + dims[2] / 2;
    assert!(
        f.z[mid].abs() > 0.001 || f.y[mid].abs() > 0.001 || f.x[mid].abs() > 0.001,
        "displacement at center voxel should be non-zero"
    );

    // Displacement values should be finite.
    for i in 0..n {
        assert!(f.z[i].is_finite(), "z displacement at {} is not finite", i);
        assert!(f.y[i].is_finite(), "y displacement at {} is not finite", i);
        assert!(f.x[i].is_finite(), "x displacement at {} is not finite", i);
    }
}

/// Zero control points should yield zero displacement everywhere.
#[test]
fn zero_control_points_yield_zero_displacement() {
    let dims = [8usize, 8, 8];
    let ctrl_spacing = [4.0_f64, 4.0, 4.0];
    let ctrl_dims = init_control_grid(VolumeDims(dims), &ctrl_spacing);
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];
    let cp_z = vec![0.0_f32; cn];
    let cp_y = vec![0.0_f32; cn];
    let cp_x = vec![0.0_f32; cn];

    let cache = BasisCache::new(VolumeDims(dims), &ctrl_spacing);
    let f = evaluate_bspline_displacement_fast(&cp_z, &cp_y, &cp_x, &ctrl_dims, dims, &cache);

    for i in 0..f.z.len() {
        assert_eq!(f.z[i], 0.0, "z displacement at {} should be zero", i);
        assert_eq!(f.y[i], 0.0, "y displacement at {} should be zero", i);
        assert_eq!(f.x[i], 0.0, "x displacement at {} should be zero", i);
    }
}

// ── Bounded dense support-matrix tests (PERF-432 closure) ─────────────────────

/// `should_use_dense_path` is a pure predicate over the control-lattice
/// product and returns `false` once that product exceeds
/// [`super::DENSE_LATTICE_CUTOFF`]. Locks in the dispatch contract.
#[test]
fn bspline_dense_dispatch_predicate_is_pure() {
    // Qualifying: small lattice (ctrl_product = 6³ = 216 ≤ 1_000_000).
    assert!(should_use_dense_path(&[6, 6, 6]));
    // Qualifying: 64×64×64 sample × 11³ control (= 1331 ≤ 1_000_000).
    assert!(should_use_dense_path(&[11, 11, 11]));
    // Qualifying: exactly at the cutoff boundary (ctrl_product == 1_000_000).
    // The `<=` predicate is inclusive; this row owns the qualifying boundary.
    assert!(should_use_dense_path(&[100, 100, 100]));
    // Refuted: ctrl_product > DENSE_LATTICE_CUTOFF.
    assert!(!should_use_dense_path(&[200, 200, 200]));
    // Refuted just past the cutoff boundary (ctrl_product == 1_000_001).
    assert!(!should_use_dense_path(&[100, 100, 101]));
    // Const-eval invariant: the cutoff constant sits in a defensible
    // range so qualifying lattices are practically common and refuted
    // ones stay well within usize budget. The `const {}` block also
    // silences `clippy::assertions_on_constants`.
    const _: () = {
        assert!(
            super::DENSE_LATTICE_CUTOFF >= 64 * 64 * 64,
            "cutoff must admit at least one 64³ whole-volume case"
        );
        assert!(
            super::DENSE_LATTICE_CUTOFF <= 4 * 1024 * 1024,
            "cutoff must stay well within usize budget"
        );
    };
}

/// The dense path MUST agree with the cache-based sparse path to within
/// `f32` summation-order tolerance on every voxel of a qualifying
/// lattice. This is the load-bearing equivalence assertion for the
/// auto-dispatch in `BSplineFFDRegistration::register`.
#[test]
fn bspline_dense_matches_sparse_on_small_lattice() {
    let dims = [8usize, 8, 8];
    let ctrl_spacing = [4.0_f64, 4.0, 4.0];
    let ctrl_dims = init_control_grid(VolumeDims(dims), &ctrl_spacing);
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];
    let n = dims[0] * dims[1] * dims[2];

    // Deterministic, non-trivial control points (no `rand` dependency).
    let cp_z: Vec<f32> = (0..cn)
        .map(|i| ((i * 13) % 97) as f32 / 50.0 - 1.0)
        .collect();
    let cp_y: Vec<f32> = (0..cn)
        .map(|i| ((i * 7 + 5) % 89) as f32 / 40.0 - 1.1)
        .collect();
    let cp_x: Vec<f32> = (0..cn)
        .map(|i| ((i * 11 + 3) % 79) as f32 / 30.0 - 1.3)
        .collect();

    let cache = BasisCache::new(VolumeDims(dims), &ctrl_spacing);
    let sparse = evaluate_bspline_displacement_fast(&cp_z, &cp_y, &cp_x, &ctrl_dims, dims, &cache);

    let mut dz = vec![0.0_f32; n];
    let mut dy = vec![0.0_f32; n];
    let mut dx = vec![0.0_f32; n];
    evaluate_bspline_displacement_dense_into(
        &cp_z,
        &cp_y,
        &cp_x,
        &ctrl_dims,
        &ctrl_spacing,
        VolumeDims(dims),
        &mut dz,
        &mut dy,
        &mut dx,
    );

    // Tolerance: f32 summation order differs between the two paths;
    // `5e-5` covers the worst-case rounding for 64 accumulations.
    let tol = 5e-5_f32;
    for i in 0..n {
        assert!(
            !sparse.z[i].is_nan() && !dz[i].is_nan(),
            "NaN at dense/sparse voxel {i}"
        );
        assert!(
            (sparse.z[i] - dz[i]).abs() < tol,
            "z[{i}] sparse={} dense={} delta={}",
            sparse.z[i],
            dz[i],
            (sparse.z[i] - dz[i]).abs()
        );
        assert!(
            (sparse.y[i] - dy[i]).abs() < tol,
            "y[{i}] sparse={} dense={}",
            sparse.y[i],
            dy[i]
        );
        assert!(
            (sparse.x[i] - dx[i]).abs() < tol,
            "x[{i}] sparse={} dense={}",
            sparse.x[i],
            dx[i]
        );
    }
}

/// Zero control points must produce a zero displacement field under the
/// dense path (mirrors the sparse-path invariant), guaranteeing the
/// dense path correctly handles the OOB / zero-weight branch.
#[test]
fn bspline_dense_zero_cps_yields_zero_field() {
    let dims = [6usize, 6, 6];
    let ctrl_spacing = [2.0_f64, 2.0, 2.0];
    let ctrl_dims = init_control_grid(VolumeDims(dims), &ctrl_spacing);
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];
    let cp_z = vec![0.0_f32; cn];
    let cp_y = vec![0.0_f32; cn];
    let cp_x = vec![0.0_f32; cn];

    // Pre-fill output buffers with sentinel to verify in-place zero-clear
    // semantics (dense path writes into caller-owned buffers).
    let n = dims[0] * dims[1] * dims[2];
    let mut dz = vec![1.0_f32; n];
    let mut dy = vec![1.0_f32; n];
    let mut dx = vec![1.0_f32; n];
    evaluate_bspline_displacement_dense_into(
        &cp_z,
        &cp_y,
        &cp_x,
        &ctrl_dims,
        &ctrl_spacing,
        VolumeDims(dims),
        &mut dz,
        &mut dy,
        &mut dx,
    );
    for i in 0..n {
        assert_eq!(dz[i], 0.0, "dense z[{i}] should be zero");
        assert_eq!(dy[i], 0.0, "dense y[{i}] should be zero");
        assert_eq!(dx[i], 0.0, "dense x[{i}] should be zero");
    }
}

/// The dense-with-support variant must agree with
/// `evaluate_bspline_displacement_dense_into` and with the cache-based
/// sparse path. Locks in the alloc-free registration-hook contract:
/// building `DenseSupport` once at the level boundary and calling
/// `evaluate` per iteration must produce the same output as a fresh
/// allocation-and-call.
#[test]
fn bspline_dense_with_support_matches_with_into() {
    let dims = [10usize, 10, 10];
    let ctrl_spacing = [4.0_f64, 4.0, 4.0];
    let ctrl_dims = init_control_grid(VolumeDims(dims), &ctrl_spacing);
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];
    let n = dims[0] * dims[1] * dims[2];

    let cp_z: Vec<f32> = (0..cn)
        .map(|i| ((i * 17 + 1) % 83) as f32 / 41.0 - 1.0)
        .collect();
    let cp_y: Vec<f32> = (0..cn)
        .map(|i| ((i * 23 + 7) % 71) as f32 / 35.0 - 1.0)
        .collect();
    let cp_x: Vec<f32> = (0..cn)
        .map(|i| ((i * 29 + 11) % 67) as f32 / 33.0 - 1.0)
        .collect();

    // Path A: allocate-and-call (the convenience wrapper).
    let mut dz_a = vec![0.0_f32; n];
    let mut dy_a = vec![0.0_f32; n];
    let mut dx_a = vec![0.0_f32; n];
    super::super::basis::evaluate::evaluate_bspline_displacement_dense_into(
        &cp_z,
        &cp_y,
        &cp_x,
        &ctrl_dims,
        &ctrl_spacing,
        VolumeDims(dims),
        &mut dz_a,
        &mut dy_a,
        &mut dx_a,
    );

    // Path B: pre-built support, called twice. Both calls must agree
    // with path A (allocation-freeness preserved across multiple uses).
    let support = super::super::basis::evaluate::DenseSupport::build(
        VolumeDims(dims),
        ctrl_dims,
        &ctrl_spacing,
    );
    let mut dz_b = vec![0.0_f32; n];
    let mut dy_b = vec![0.0_f32; n];
    let mut dx_b = vec![0.0_f32; n];
    super::super::basis::evaluate::evaluate_bspline_displacement_dense_with(
        &support, &cp_z, &cp_y, &cp_x, &ctrl_dims, &mut dz_b, &mut dy_b, &mut dx_b,
    );
    let mut dz_c = vec![0.0_f32; n];
    let mut dy_c = vec![0.0_f32; n];
    let mut dx_c = vec![0.0_f32; n];
    super::super::basis::evaluate::evaluate_bspline_displacement_dense_with(
        &support, &cp_z, &cp_y, &cp_x, &ctrl_dims, &mut dz_c, &mut dy_c, &mut dx_c,
    );

    let tol = 1e-6_f32;
    for i in 0..n {
        assert!(
            (dz_a[i] - dz_b[i]).abs() < tol,
            "dense_with vs dense_into z[{i}]: a={} b={}",
            dz_a[i],
            dz_b[i]
        );
        assert!(
            (dz_b[i] - dz_c[i]).abs() < tol,
            "dense_with vs dense_with (2 calls) z[{i}]: b={} c={}",
            dz_b[i],
            dz_c[i]
        );
    }
}
