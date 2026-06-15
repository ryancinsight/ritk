use super::try_invert_small;

/// Multiply two flat row-major D×D matrices and return the flat result.
fn mat_mul<const D: usize>(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = D;
    let mut out = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[r * n + k] * b[k * n + c];
            }
            out[r * n + c] = sum;
        }
    }
    out
}

/// Compute max absolute difference between two flat matrices.
fn max_abs_diff<const D: usize>(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ── D = 1 ─────────────────────────────────────────────────────────

#[test]
fn invert_1x1_identity() {
    let a = [1.0f32];
    let inv = try_invert_small::<1>(&a).expect("identity should be invertible");
    assert!((inv[0] - 1.0).abs() < 1e-6);
}

#[test]
fn invert_1x1_non_trivial() {
    let a = [4.0f32];
    let inv = try_invert_small::<1>(&a).expect("4×1 should be invertible");
    assert!((inv[0] - 0.25).abs() < 1e-6);
}

#[test]
fn invert_1x1_singular() {
    assert!(try_invert_small::<1>(&[0.0f32]).is_none());
    assert!(try_invert_small::<1>(&[1e-12f32]).is_none());
}

#[test]
fn invert_1x1_round_trip() {
    let a = [7.5f32];
    let inv = try_invert_small::<1>(&a).unwrap();
    let prod = mat_mul::<1>(&a, &inv);
    let ident = [1.0f32];
    assert!(max_abs_diff::<1>(&prod, &ident) < 1e-5);
}

// ── D = 2 ─────────────────────────────────────────────────────────

#[test]
fn invert_2x2_identity() {
    let a = [1.0, 0.0, 0.0, 1.0];
    let inv = try_invert_small::<2>(&a).expect("identity should be invertible");
    assert!(max_abs_diff::<2>(&inv, &[1.0, 0.0, 0.0, 1.0]) < 1e-6);
}

#[test]
fn invert_2x2_known_inverse() {
    // A = [[2, 1], [5, 3]]  →  A⁻¹ = [[3, -1], [-5, 2]]
    let a = [2.0, 1.0, 5.0, 3.0];
    let expected = [3.0, -1.0, -5.0, 2.0];
    let inv = try_invert_small::<2>(&a).expect("should be invertible");
    assert!(max_abs_diff::<2>(&inv, &expected) < 1e-5);
}

#[test]
fn invert_2x2_round_trip() {
    let a = [2.0, 1.0, 5.0, 3.0];
    let inv = try_invert_small::<2>(&a).unwrap();
    let prod = mat_mul::<2>(&a, &inv);
    let ident: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    assert!(max_abs_diff::<2>(&prod, &ident) < 1e-5);
}

#[test]
fn invert_2x2_singular() {
    // Rows are linearly dependent: [1,2] and [2,4]
    let a = [1.0, 2.0, 2.0, 4.0];
    assert!(try_invert_small::<2>(&a).is_none());
}

#[test]
fn invert_2x2_diagonal() {
    let a = [3.0, 0.0, 0.0, 5.0];
    let expected = [1.0 / 3.0, 0.0, 0.0, 1.0 / 5.0];
    let inv = try_invert_small::<2>(&a).expect("diagonal should be invertible");
    assert!(max_abs_diff::<2>(&inv, &expected) < 1e-5);
}

// ── D = 3 ─────────────────────────────────────────────────────────

#[test]
fn invert_3x3_identity() {
    let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let inv = try_invert_small::<3>(&a).expect("identity should be invertible");
    assert!(max_abs_diff::<3>(&inv, &a) < 1e-6);
}

#[test]
fn invert_3x3_known_inverse() {
    // A = [[3, 0, 2], [2, 0, -2], [0, 1, 1]]
    // A⁻¹ = [[0.2, 0.2, 0.0], [-0.2, 0.3, 1.0], [0.2, -0.3, 0.0]]
    let a = [3.0, 0.0, 2.0, 2.0, 0.0, -2.0, 0.0, 1.0, 1.0];
    let expected = [0.2, 0.2, 0.0, -0.2, 0.3, 1.0, 0.2, -0.3, 0.0];
    let inv = try_invert_small::<3>(&a).expect("should be invertible");
    assert!(max_abs_diff::<3>(&inv, &expected) < 1e-4);
}

#[test]
fn invert_3x3_round_trip() {
    let a = [3.0, 0.0, 2.0, 2.0, 0.0, -2.0, 0.0, 1.0, 1.0];
    let inv = try_invert_small::<3>(&a).unwrap();
    let prod = mat_mul::<3>(&a, &inv);
    let ident: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert!(max_abs_diff::<3>(&prod, &ident) < 1e-5);
}

#[test]
fn invert_3x3_singular() {
    // Row 2 = 2 * Row 0
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 4.0, 6.0];
    assert!(try_invert_small::<3>(&a).is_none());
}

#[test]
fn invert_3x3_diagonal() {
    let a = [2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0];
    let expected = [0.5, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.125];
    let inv = try_invert_small::<3>(&a).expect("diagonal should be invertible");
    assert!(max_abs_diff::<3>(&inv, &expected) < 1e-5);
}

#[test]
fn invert_3x3_rotation_round_trip() {
    // 90° rotation around Z axis
    let a = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let inv = try_invert_small::<3>(&a).unwrap();
    let prod = mat_mul::<3>(&a, &inv);
    let ident: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert!(max_abs_diff::<3>(&prod, &ident) < 1e-5);
}

// ── D = 4 ─────────────────────────────────────────────────────────

#[test]
fn invert_4x4_identity() {
    let mut a = vec![0.0f32; 16];
    a[0] = 1.0;
    a[5] = 1.0;
    a[10] = 1.0;
    a[15] = 1.0;
    let inv = try_invert_small::<4>(&a).expect("identity should be invertible");
    assert!(max_abs_diff::<4>(&inv, &a) < 1e-6);
}

#[test]
fn invert_4x4_diagonal() {
    let mut a = vec![0.0f32; 16];
    a[0] = 2.0;
    a[5] = 3.0;
    a[10] = 5.0;
    a[15] = 7.0;
    let inv = try_invert_small::<4>(&a).expect("diagonal should be invertible");
    let mut expected = vec![0.0f32; 16];
    expected[0] = 0.5;
    expected[5] = 1.0 / 3.0;
    expected[10] = 0.2;
    expected[15] = 1.0 / 7.0;
    assert!(max_abs_diff::<4>(&inv, &expected) < 1e-5);
}

#[test]
fn invert_4x4_round_trip() {
    let a = [
        2.0, 1.0, 0.0, 3.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 3.0, 0.0, 1.0, 2.0,
    ];
    let inv = try_invert_small::<4>(&a).unwrap();
    let prod = mat_mul::<4>(&a, &inv);
    let mut ident = vec![0.0f32; 16];
    ident[0] = 1.0;
    ident[5] = 1.0;
    ident[10] = 1.0;
    ident[15] = 1.0;
    assert!(max_abs_diff::<4>(&prod, &ident) < 1e-5);
}

#[test]
fn invert_4x4_singular() {
    // Row 2 is a zero row — clearly rank-deficient.
    let a = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 13.0, 14.0, 15.0, 16.0,
    ];
    assert!(try_invert_small::<4>(&a).is_none());
}

// ── Near-singular (pivoting stress) ───────────────────────────────

#[test]
fn invert_near_singular_pivots() {
    // Matrix that is invertible but requires pivoting because the
    // (0,0) entry is small relative to (1,0).
    let a = [1e-7, 2.0, 2.0, 3.0];
    let inv = try_invert_small::<2>(&a).expect("should be invertible with pivoting");
    let prod = mat_mul::<2>(&a, &inv);
    let ident = [1.0, 0.0, 0.0, 1.0];
    assert!(max_abs_diff::<2>(&prod, &ident) < 1e-5);
}
