use super::*;

const TOL: f32 = 1e-4;

// ── compute_hessian_3d ────────────────────────────────────────────────────

/// Quadratic image `I[z, y, x] = x² + y² + z²`.
///
/// Analytical second derivatives: Hxx = Hyy = Hzz = 2, all cross terms = 0.
/// Central differences are exact for degree-2 polynomials.
#[test]
fn test_hessian_quadratic_separable() {
    const N: usize = 5;
    let mut data = vec![0.0f32; N * N * N];
    for iz in 0..N {
        for iy in 0..N {
            for ix in 0..N {
                data[iz * N * N + iy * N + ix] = (ix * ix + iy * iy + iz * iz) as f32;
            }
        }
    }

    let h_out = compute_hessian(&data, [N, N, N], [1.0, 1.0, 1.0]);

    // Interior voxel (iz=2, iy=2, ix=2): central differences apply.
    let voxel = 2 * N * N + 2 * N + 2;
    let [hzz, hzy, hzx, hyy, hyx, hxx] = h_out[voxel];

    assert!((hzz - 2.0).abs() < TOL, "Hzz: expected 2.0, got {hzz}");
    assert!((hyy - 2.0).abs() < TOL, "Hyy: expected 2.0, got {hyy}");
    assert!((hxx - 2.0).abs() < TOL, "Hxx: expected 2.0, got {hxx}");
    assert!(hzy.abs() < TOL, "Hzy: expected 0.0, got {hzy}");
    assert!(hzx.abs() < TOL, "Hzx: expected 0.0, got {hzx}");
    assert!(hyx.abs() < TOL, "Hyx: expected 0.0, got {hyx}");
}

/// Image `I[z, y, x] = x · y`.
///
/// Analytical second derivatives: Hyx = 1, all others = 0.
/// The 4-point bilinear stencil recovers the mixed derivative exactly.
#[test]
fn test_hessian_product_xy() {
    const N: usize = 5;
    let mut data = vec![0.0f32; N * N * N];
    for iz in 0..N {
        for iy in 0..N {
            for ix in 0..N {
                data[iz * N * N + iy * N + ix] = (ix * iy) as f32;
            }
        }
    }

    let h_out = compute_hessian(&data, [N, N, N], [1.0, 1.0, 1.0]);

    // Interior voxel (iz=2, iy=2, ix=2).
    let voxel = 2 * N * N + 2 * N + 2;
    let [hzz, hzy, hzx, hyy, hyx, hxx] = h_out[voxel];

    assert!((hyx - 1.0).abs() < TOL, "Hyx: expected 1.0, got {hyx}");
    assert!(hzz.abs() < TOL, "Hzz: expected 0.0, got {hzz}");
    assert!(hyy.abs() < TOL, "Hyy: expected 0.0, got {hyy}");
    assert!(hxx.abs() < TOL, "Hxx: expected 0.0, got {hxx}");
    assert!(hzy.abs() < TOL, "Hzy: expected 0.0, got {hzy}");
    assert!(hzx.abs() < TOL, "Hzx: expected 0.0, got {hzx}");
}

/// Non-unit spacing: `I[z,y,x] = x²` with spacing `[1.0, 1.0, 2.0]`.
///
/// Hxx = 2.0 / sx² = 2.0 / 4.0 = 0.5 (physical second derivative).
/// All other components = 0.
#[test]
fn test_hessian_nonunit_spacing() {
    const N: usize = 5;
    let mut data = vec![0.0f32; N * N * N];
    for iz in 0..N {
        for iy in 0..N {
            for ix in 0..N {
                data[iz * N * N + iy * N + ix] = (ix * ix) as f32;
            }
        }
    }

    // spacing = [sz=1.0, sy=1.0, sx=2.0]
    let h_out = compute_hessian(&data, [N, N, N], [1.0, 1.0, 2.0]);

    let voxel = 2 * N * N + 2 * N + 2;
    let [_hzz, _hzy, _hzx, _hyy, _hyx, hxx] = h_out[voxel];

    // Central diff gives (ix-1)² - 2*ix² + (ix+1)² = 2 (in voxels).
    // Physical Hxx = 2 / sx² = 2 / 4 = 0.5.
    assert!((hxx - 0.5).abs() < TOL, "Hxx: expected 0.5, got {hxx}");
}

// ── symmetric_3x3_eigenvalues ─────────────────────────────────────────────

/// Diagonal matrix `[1, 0, 0, 2, 0, 3]` → sorted: `[1.0, 2.0, 3.0]`.
#[test]
fn test_eigenvalues_diagonal_positive() {
    let eigs = symmetric_3x3_eigenvalues([1.0f32, 0.0, 0.0, 2.0, 0.0, 3.0]);
    assert!(
        (eigs[0] - 1.0).abs() < TOL,
        "eig[0]: expected 1.0, got {}",
        eigs[0]
    );
    assert!(
        (eigs[1] - 2.0).abs() < TOL,
        "eig[1]: expected 2.0, got {}",
        eigs[1]
    );
    assert!(
        (eigs[2] - 3.0).abs() < TOL,
        "eig[2]: expected 3.0, got {}",
        eigs[2]
    );
}

/// Identity matrix → all eigenvalues 1.0.
#[test]
fn test_eigenvalues_identity() {
    let eigs = symmetric_3x3_eigenvalues([1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0]);
    for (i, &e) in eigs.iter().enumerate() {
        assert!((e - 1.0).abs() < TOL, "eig[{i}]: expected 1.0, got {e}");
    }
}

/// Negative diagonal `[-3, 0, 0, -1, 0, -2]` → sorted by |λ|: `[-1, -2, -3]`.
#[test]
fn test_eigenvalues_negative_diagonal() {
    let eigs = symmetric_3x3_eigenvalues([-3.0f32, 0.0, 0.0, -1.0, 0.0, -2.0]);
    assert!(
        (eigs[0] + 1.0).abs() < TOL,
        "eig[0]: expected -1.0, got {}",
        eigs[0]
    );
    assert!(
        (eigs[1] + 2.0).abs() < TOL,
        "eig[1]: expected -2.0, got {}",
        eigs[1]
    );
    assert!(
        (eigs[2] + 3.0).abs() < TOL,
        "eig[2]: expected -3.0, got {}",
        eigs[2]
    );
}

/// Non-trivial off-diagonal case.
///
/// `M = [[2, 1, 0], [1, 2, 0], [0, 0, 3]]`  →  eigenvalues {1, 3, 3}.
///
/// Input `h = [Hzz=2, Hzy=1, Hzx=0, Hyy=2, Hyx=0, Hxx=3]`.
/// Sorted by |λ|: `[1.0, 3.0, 3.0]`.
///
/// # Derivation
/// The 2×2 block `[[2,1],[1,2]]` has eigenvalues `2±1 = {1, 3}`.
/// The remaining diagonal element 3 is already decoupled.
#[test]
fn test_eigenvalues_offdiagonal_block() {
    let eigs = symmetric_3x3_eigenvalues([2.0f32, 1.0, 0.0, 2.0, 0.0, 3.0]);
    assert!(
        (eigs[0] - 1.0).abs() < TOL,
        "eig[0]: expected 1.0, got {}",
        eigs[0]
    );
    assert!(
        (eigs[1] - 3.0).abs() < TOL,
        "eig[1]: expected 3.0, got {}",
        eigs[1]
    );
    assert!(
        (eigs[2] - 3.0).abs() < TOL,
        "eig[2]: expected 3.0, got {}",
        eigs[2]
    );
}

/// Mixed-sign diagonal: sorted by absolute value, not by sign.
///
/// `h = [-5, 0, 0, 2, 0, -1]` → sorted by |λ|: `[-1, 2, -5]`.
#[test]
fn test_eigenvalues_mixed_sign() {
    let eigs = symmetric_3x3_eigenvalues([-5.0f32, 0.0, 0.0, 2.0, 0.0, -1.0]);
    assert!(
        (eigs[0] + 1.0).abs() < TOL,
        "eig[0]: expected -1.0, got {}",
        eigs[0]
    );
    assert!(
        (eigs[1] - 2.0).abs() < TOL,
        "eig[1]: expected  2.0, got {}",
        eigs[1]
    );
    assert!(
        (eigs[2] + 5.0).abs() < TOL,
        "eig[2]: expected -5.0, got {}",
        eigs[2]
    );
}
