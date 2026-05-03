//! Physical-space cursor position computation (voxel → LPS mm).
//!
//! # Mathematical specification
//!
//! The standard ITK affine transform maps a discrete voxel index
//! `v = [d, r, c]` to physical LPS coordinates `P ∈ ℝ³`:
//!
//! ```text
//! P = origin + D · diag(spacing) · v
//! ```
//!
//! where `D` is the 3×3 direction cosine matrix (row-major, 9 elements),
//! `diag(spacing)` is the diagonal scaling matrix for `[dz, dy, dx]`, and
//! `v` is the 3-component column vector `[d, r, c]` (depth, row, column).
//!
//! Expanding component-wise:
//!
//! ```text
//! P[0] = origin[0] + D[0,0]·dz·d  +  D[0,1]·dy·r  +  D[0,2]·dx·c
//! P[1] = origin[1] + D[1,0]·dz·d  +  D[1,1]·dy·r  +  D[1,2]·dx·c
//! P[2] = origin[2] + D[2,0]·dz·d  +  D[2,1]·dy·r  +  D[2,2]·dx·c
//! ```
//!
//! where `D[i,j] = direction[i*3 + j]` (row-major storage).
//!
//! This is identical to ITK's `itk::ImageBase::TransformContinuousIndexToPhysicalPoint`.
//!
//! # Formal invariants
//!
//! - For identity direction and zero origin:
//!   `P[i] = spacing[i] · v[i]` exactly.
//! - For identity direction, zero origin, and unit spacing:
//!   `P = v` (floating-point cast of integer indices).
//! - For any origin offset `o`:
//!   `voxel_to_lps(v, o, I, s) = o + voxel_to_lps(v, [0,0,0], I, s)`.
//! - The function depends on all four parameters: changing any parameter
//!   while holding the others fixed changes the output (non-cosmetic).

// ── Public API ────────────────────────────────────────────────────────────────

/// Map a discrete voxel index to physical LPS coordinates using the ITK affine.
///
/// # Parameters
/// - `voxel`     — `[depth_idx, row_idx, col_idx]` (0-based, unclamped).
/// - `origin`    — image origin in mm: `[ox, oy, oz]`.
/// - `direction` — 3×3 direction cosine matrix, **row-major** (9 elements).
/// - `spacing`   — voxel spacing `[dz, dy, dx]` in mm/voxel.
///
/// # Returns
/// Physical LPS position `[Px, Py, Pz]` in mm.
///
/// # Examples
/// ```rust
/// use ritk_snap::ui::cursor_info::voxel_to_lps;
///
/// // Identity direction, unit spacing, zero origin — LPS == voxel index.
/// let lps = voxel_to_lps(
///     [2, 3, 4],
///     [0.0; 3],
///     [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
///     [1.0, 1.0, 1.0],
/// );
/// assert!((lps[0] - 2.0_f64).abs() < 1e-12);
/// assert!((lps[1] - 3.0_f64).abs() < 1e-12);
/// assert!((lps[2] - 4.0_f64).abs() < 1e-12);
/// ```
pub fn voxel_to_lps(
    voxel: [usize; 3],
    origin: [f64; 3],
    direction: [f64; 9],
    spacing: [f64; 3],
) -> [f64; 3] {
    // Scaled displacement components: D·diag(spacing)·v, pre-multiplied per column.
    let [dz, dy, dx] = spacing;
    let [d, r, c] = [voxel[0] as f64, voxel[1] as f64, voxel[2] as f64];

    // Each column j of D contributes direction[i*3+j] * spacing[j] * v[j].
    let col0_scale = dz * d; // depth column
    let col1_scale = dy * r; // row column
    let col2_scale = dx * c; // col column

    [
        origin[0]
            + direction[0] * col0_scale
            + direction[1] * col1_scale
            + direction[2] * col2_scale,
        origin[1]
            + direction[3] * col0_scale
            + direction[4] * col1_scale
            + direction[5] * col2_scale,
        origin[2]
            + direction[6] * col0_scale
            + direction[7] * col1_scale
            + direction[8] * col2_scale,
    ]
}

/// Format an LPS position for display in a status bar.
///
/// Output: `"LPS (x.xx, y.yy, z.zz) mm"` with 2 decimal places.
pub fn format_lps(lps: [f64; 3]) -> String {
    format!("LPS ({:.2}, {:.2}, {:.2}) mm", lps[0], lps[1], lps[2])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity direction, 3×3 identity matrix.
    const IDENTITY_DIR: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    /// Tolerance for floating-point equality assertions.
    const EPS: f64 = 1e-10;

    fn assert_lps_eq(got: [f64; 3], expected: [f64; 3]) {
        for i in 0..3 {
            assert!(
                (got[i] - expected[i]).abs() < EPS,
                "LPS[{i}]: got {got:.12?} expected {expected:.12?}"
            );
        }
    }

    /// **Proof**: For identity direction and zero origin with unit spacing,
    /// `P[i] = 1.0 * v[i]`. Derived directly from the affine formula.
    #[test]
    fn identity_direction_unit_spacing_zero_origin() {
        let lps = voxel_to_lps([2, 3, 4], [0.0; 3], IDENTITY_DIR, [1.0, 1.0, 1.0]);
        // Expected: P = I · diag([1,1,1]) · [2,3,4] + [0,0,0] = [2, 3, 4].
        assert_lps_eq(lps, [2.0, 3.0, 4.0]);
    }

    /// **Proof**: At voxel [0,0,0], regardless of direction and spacing, the
    /// affine displacement term is zero so `P = origin`.
    #[test]
    fn zero_voxel_returns_origin() {
        let origin = [10.5, -20.3, 7.1];
        // Use an arbitrary non-identity direction (rotation by 90° around Z).
        let dir90z = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let lps = voxel_to_lps([0, 0, 0], origin, dir90z, [2.5, 1.0, 0.8]);
        assert_lps_eq(lps, origin);
    }

    /// **Proof**: For identity direction and zero origin,
    /// `P[i] = spacing[i] * v[i]`. The spacing `[dz, dy, dx]` maps the
    /// depth, row, and col axes to the three LPS components directly.
    #[test]
    fn identity_direction_non_unit_spacing() {
        let spacing = [2.5, 1.0, 0.75]; // [dz, dy, dx]
        let lps = voxel_to_lps([4, 6, 8], [0.0; 3], IDENTITY_DIR, spacing);
        // Expected: P = [2.5*4, 1.0*6, 0.75*8] = [10.0, 6.0, 6.0].
        assert_lps_eq(lps, [10.0, 6.0, 6.0]);
    }

    /// **Proof**: Origin offset is additive and independent of voxel index and
    /// direction, so `voxel_to_lps(v, o, D, s) = o + voxel_to_lps(v, 0, D, s)`.
    #[test]
    fn origin_is_additive_offset() {
        let origin = [100.0, -50.0, 25.0];
        let spacing = [1.5, 1.5, 1.5];
        let voxel = [3, 5, 7];
        let lps_zero = voxel_to_lps(voxel, [0.0; 3], IDENTITY_DIR, spacing);
        let lps_origin = voxel_to_lps(voxel, origin, IDENTITY_DIR, spacing);
        assert_lps_eq(
            lps_origin,
            [
                origin[0] + lps_zero[0],
                origin[1] + lps_zero[1],
                origin[2] + lps_zero[2],
            ],
        );
    }

    /// **Proof**: A 90° rotation about the Z axis permutes the X and Y
    /// components: `D = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]`.
    /// With unit spacing and zero origin, voxel [0, r, 0] maps to LPS [-r, 0, 0]
    /// because column 1 (row axis) maps to -X in this rotation.
    #[test]
    fn rotation_90_about_z_remaps_axes() {
        // 90° CCW rotation about Z: R_z(90°) = [[0,-1,0],[1,0,0],[0,0,1]].
        let dir90z = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let lps = voxel_to_lps([0, 5, 0], [0.0; 3], dir90z, [1.0, 1.0, 1.0]);
        // column 1 contributes dir[1]*dy*r = -1.0*1.0*5 to P[0],
        //                        dir[4]*dy*r =  0.0*1.0*5 to P[1],
        //                        dir[7]*dy*r =  0.0*1.0*5 to P[2].
        assert_lps_eq(lps, [-5.0, 0.0, 0.0]);
    }

    /// **Proof**: Mixed direction and mixed voxel index exercise all nine
    /// direction-cosine elements. Values are derived by manual matrix-vector
    /// multiplication: P = origin + D·diag(spacing)·v.
    ///
    /// Let D = [[1, 0, 0], [0, 0, -1], [0, 1, 0]] (90° rotation about X),
    /// spacing = [2.0, 3.0, 4.0], origin = [1.0, 2.0, 3.0],
    /// voxel = [1, 2, 3].
    ///
    /// D·diag(s)·v = [[1,0,0],[0,0,-1],[0,1,0]]·[[2,0,0],[0,3,0],[0,0,4]]·[1,2,3]
    ///             = [[1,0,0],[0,0,-1],[0,1,0]]·[2, 6, 12]
    ///             = [2, -12, 6].
    /// P = [1,2,3] + [2,-12,6] = [3,-10,9].
    #[test]
    fn rotation_about_x_mixed_spacing_and_voxel() {
        let dir_rot_x = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        let spacing = [2.0, 3.0, 4.0];
        let origin = [1.0, 2.0, 3.0];
        let voxel = [1, 2, 3];
        let lps = voxel_to_lps(voxel, origin, dir_rot_x, spacing);
        assert_lps_eq(lps, [3.0, -10.0, 9.0]);
    }

    /// **Proof**: `format_lps` must produce the standard two-decimal display
    /// with the "LPS (...) mm" prefix. The exact string is pinned to a known
    /// reference value.
    #[test]
    fn format_lps_output_matches_expected() {
        let lps = [10.1, -20.22, 300.333];
        let s = format_lps(lps);
        assert_eq!(s, "LPS (10.10, -20.22, 300.33) mm");
    }
}
