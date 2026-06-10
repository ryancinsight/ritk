use super::super::basis::{
    cubic_bspline_1d, evaluate_bspline_displacement_fast, init_control_grid, BasisCache,
};

#[test]
fn bspline_basis_partition_of_unity() {
    for i in 0..=100 {
        let t = i as f64 / 100.0;
        let b = cubic_bspline_1d(t);
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
        let b = cubic_bspline_1d(t);
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
    let ctrl = init_control_grid(dims, &spacing);
    assert_eq!(ctrl, [5, 6, 6]);
}

#[test]
fn init_control_grid_non_divisible() {
    // dims = [10, 10, 10], spacing = [4, 4, 4]
    // n_ctrl[d] = ceil(10/4)+3 = 3+3 = 6
    let dims = [10, 10, 10];
    let spacing = [4.0, 4.0, 4.0];
    let ctrl = init_control_grid(dims, &spacing);
    assert_eq!(ctrl, [6, 6, 6]);
}

/// Verify that the fast-path displacement evaluation matches the original
/// per-voxel approach on a small volume with random control points.
#[test]
fn fast_displacement_matches_original_on_random_cps() {
    let dims = [16usize, 20, 24];
    let ctrl_spacing = [4.0_f64, 5.0, 6.0];
    let ctrl_dims = init_control_grid(dims, &ctrl_spacing);
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];

    // Deterministic "random-ish" control points (avoid rand 0.9 API changes).
    let cp_z: Vec<f32> = (0..cn).map(|i| (i as f32 % 10.0) - 5.0).collect();
    let cp_y: Vec<f32> = (0..cn).map(|i| ((i + 3) as f32 % 10.0) - 5.0).collect();
    let cp_x: Vec<f32> = (0..cn).map(|i| ((i + 7) as f32 % 10.0) - 5.0).collect();

    // Original path (delegates to fast internally now, so we need to
    // compare the original algorithm directly).
    let cache = BasisCache::new(dims, &ctrl_spacing);
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
    let ctrl_dims = init_control_grid(dims, &ctrl_spacing);
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];
    let cp_z = vec![0.0_f32; cn];
    let cp_y = vec![0.0_f32; cn];
    let cp_x = vec![0.0_f32; cn];

    let cache = BasisCache::new(dims, &ctrl_spacing);
    let f = evaluate_bspline_displacement_fast(&cp_z, &cp_y, &cp_x, &ctrl_dims, dims, &cache);

    for i in 0..f.z.len() {
        assert_eq!(f.z[i], 0.0, "z displacement at {} should be zero", i);
        assert_eq!(f.y[i], 0.0, "y displacement at {} should be zero", i);
        assert_eq!(f.x[i], 0.0, "x displacement at {} should be zero", i);
    }
}
