//! solve_linear_system unit tests.

use crate::atlas::label_fusion::jlf::solve_linear_system;

// ── solve_linear_system unit tests ───────────────────────────────────

/// 2×2 identity system: Ix = [3, 7] → x = [3, 7].
#[test]
fn solve_identity_2x2() {
    let mut a = vec![1.0, 0.0, 0.0, 1.0];
    let mut b = vec![3.0, 7.0];
    let x = solve_linear_system(&mut a, 2, &mut b).expect("infallible: validated precondition");
    assert!((x[0] - 3.0).abs() < 1e-12, "x[0] = {}", x[0]);
    assert!((x[1] - 7.0).abs() < 1e-12, "x[1] = {}", x[1]);
}

/// 2×2 system: [[2, 1], [1, 3]] x = [5, 10] → x = [1, 3].
///
/// Verification: 2·1 + 1·3 = 5 ✓, 1·1 + 3·3 = 10 ✓.
#[test]
fn solve_2x2_known() {
    let mut a = vec![2.0, 1.0, 1.0, 3.0];
    let mut b = vec![5.0, 10.0];
    let x = solve_linear_system(&mut a, 2, &mut b).expect("infallible: validated precondition");
    assert!((x[0] - 1.0).abs() < 1e-12, "x[0] = {}", x[0]);
    assert!((x[1] - 3.0).abs() < 1e-12, "x[1] = {}", x[1]);
}

/// Singular 2×2 system: [[1, 1], [1, 1]] x = [1, 1] → None.
#[test]
fn solve_singular_returns_none() {
    let mut a = vec![1.0, 1.0, 1.0, 1.0];
    let mut b = vec![1.0, 1.0];
    assert!(solve_linear_system(&mut a, 2, &mut b).is_none());
}

/// 1×1 system: [4] x = [8] → x = 2.
#[test]
fn solve_1x1() {
    let mut a = vec![4.0];
    let mut b = vec![8.0];
    let x = solve_linear_system(&mut a, 1, &mut b).expect("infallible: validated precondition");
    assert!((x[0] - 2.0).abs() < 1e-12, "x[0] = {}", x[0]);
}
