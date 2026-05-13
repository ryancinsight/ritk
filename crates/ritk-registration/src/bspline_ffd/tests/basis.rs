use super::super::basis::{cubic_bspline_1d, init_control_grid};

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
