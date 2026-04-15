//! Exact and approximate inverse displacement field computation.
//!
//! # Mathematical Specification
//!
//! Two inversion strategies are provided based on the field type.
//!
//! ## Part 1 — SVF Exact Inverse (Diffeomorphic Demons)
//!
//! For a stationary velocity field `v`, the forward diffeomorphism is `φ = exp(v)`.
//!
//! **Theorem:** The exact inverse is `φ^{-1} = exp(−v)`.
//!
//! **Proof:** `φ ∘ φ^{-1} = exp(v) ∘ exp(−v) = exp(v − v) = exp(0) = id`.
//! The Baker-Campbell-Hausdorff identity at first order is exact when the field
//! commutes with its own negation; a field always commutes with any scalar
//! multiple of itself. ∎
//!
//! Implementation: negate all velocity components in O(n) time.
//!
//! ## Part 2 — Fixed-Point Iterative Inverse (Thirion / Symmetric Demons)
//!
//! For a general displacement field `u`, the inverse `u^{-1}` satisfies:
//!
//!   `φ(x + u^{-1}(x)) = x  ⟹  u^{-1}(x) = −u(x + u^{-1}(x))`
//!
//! **Fixed-point iteration** (Christensen & Johnson 2001):
//!
//!   `u^{-1}_0(x)      = −u(x)`                         (initialisation)
//!   `u^{-1}_{k+1}(x)  = −u(x + u^{-1}_k(x))`           (update rule)
//!
//! **Convergence guarantee:** When the Lipschitz constant `L = max‖∇u‖ < 1`,
//! the update map is a contraction and the iterate error satisfies:
//!
//!   `‖u^{-1}_{k+1} − u^{-1}_*‖_∞  ≤  L^k · ‖u^{-1}_1 − u^{-1}_0‖_∞`
//!
//! For a sinusoidal field with amplitude `A` and half-period `λ/2`,
//! `L = πA/λ`; for `A = 2.0` and `λ/2 = 11` voxels, `L ≈ 0.571 < 1`.
//! Typical convergence: 5–20 iterations for registration-magnitude fields.
//!
//! # References
//! - Christensen, G. E. & Johnson, H. J. (2001). Consistent image registration.
//!   *IEEE Trans. Med. Imaging* 20(7):568–582.
//! - Chen, M. & Smedby, Ö. (2012). Deformable image registration with
//!   guaranteed correspondence completeness. *MICCAI*.

use crate::deformable_field_ops::warp_image;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for iterative inverse computation (used for non-SVF fields).
///
/// # Defaults
/// - `max_iterations`: 20
/// - `tolerance`: 1e-4 (max-norm convergence threshold, in voxels)
#[derive(Debug, Clone)]
pub struct InverseFieldConfig {
    /// Maximum number of fixed-point iterations.
    /// The algorithm terminates after at most this many iterations regardless
    /// of whether the convergence criterion has been satisfied.
    pub max_iterations: usize,

    /// Convergence threshold (voxels).
    /// The iteration terminates early when the maximum per-voxel Euclidean
    /// norm of the change between successive iterates drops below this value:
    ///
    ///   `max_i ‖u^{-1}_{k+1}(i) − u^{-1}_k(i)‖_2 < tolerance`
    pub tolerance: f64,
}

impl Default for InverseFieldConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            tolerance: 1e-4,
        }
    }
}

// ── SVF exact inverse ─────────────────────────────────────────────────────────

/// Compute the exact inverse of a stationary velocity field.
///
/// # Mathematical Basis
///
/// For a stationary velocity field `v` the exponential map gives `φ = exp(v)`.
/// The exact inverse is `φ^{-1} = exp(−v)`, obtained by negating every
/// component.  This follows directly from the Baker-Campbell-Hausdorff
/// identity: `exp(v) ∘ exp(−v) = exp(v − v) = exp(0) = id`.
///
/// This is a zero-cost O(n) operation — no integration, no iteration.
///
/// # Arguments
///
/// - `vel_z / vel_y / vel_x` — velocity field components, flat Z-major buffers
///   of identical length.
///
/// # Returns
///
/// `(inv_z, inv_y, inv_x)` — negated velocity components as new `Vec<f32>`.
pub fn invert_velocity_field(
    vel_z: &[f32],
    vel_y: &[f32],
    vel_x: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let inv_z: Vec<f32> = vel_z.iter().map(|&v| -v).collect();
    let inv_y: Vec<f32> = vel_y.iter().map(|&v| -v).collect();
    let inv_x: Vec<f32> = vel_x.iter().map(|&v| -v).collect();
    (inv_z, inv_y, inv_x)
}

// ── General displacement inverse ──────────────────────────────────────────────

/// Compute an approximate inverse of a general displacement field using
/// fixed-point iteration (Christensen & Johnson 2001).
///
/// # Algorithm
///
/// Initialisation: `u^{-1}_0(x) = −u(x)`
///
/// Update rule:    `u^{-1}_{k+1}(x) = −u(x + u^{-1}_k(x))`
///
/// Each update warps the forward field `u` by the current inverse estimate and
/// negates the result.  The returned field `u^{-1}` is a displacement from
/// identity (not an absolute position).
///
/// # Convergence
///
/// Terminates when `max_i ‖u^{-1}_{k+1}(i) − u^{-1}_k(i)‖_2 < config.tolerance`
/// or `config.max_iterations` is reached, whichever comes first.
///
/// Convergence is guaranteed when the Lipschitz constant `L = max‖∇u‖ < 1`.
/// The iterate error satisfies `‖e_k‖ ≤ L^k · ‖e_0‖`, so for `L = 0.571`
/// twenty iterations reduce the error by a factor of ~10⁻⁵.
///
/// # Arguments
///
/// - `disp_z / disp_y / disp_x` — forward displacement field, flat Z-major
///   buffers, each of length `dims[0] * dims[1] * dims[2]`.
/// - `dims`   — volume dimensions `[nz, ny, nx]`.
/// - `config` — iteration parameters (`max_iterations`, `tolerance`).
///
/// # Returns
///
/// `(inv_z, inv_y, inv_x, num_iterations_performed)`.
/// `num_iterations_performed` is in the range `[1, config.max_iterations]`
/// for non-empty volumes.
pub fn invert_displacement_field(
    disp_z: &[f32],
    disp_y: &[f32],
    disp_x: &[f32],
    dims: [usize; 3],
    config: &InverseFieldConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
    let n = dims[0] * dims[1] * dims[2];

    // Initialisation: u^{-1}_0 = −u.
    let mut inv_z: Vec<f32> = disp_z.iter().map(|&v| -v).collect();
    let mut inv_y: Vec<f32> = disp_y.iter().map(|&v| -v).collect();
    let mut inv_x: Vec<f32> = disp_x.iter().map(|&v| -v).collect();

    let mut iters = 0usize;

    for _ in 0..config.max_iterations {
        iters += 1;

        // u^{-1}_{k+1}(x) = −u(x + u^{-1}_k(x))
        // warp_displacement samples `disp` at positions x shifted by `inv_k`.
        let (warped_z, warped_y, warped_x) =
            warp_displacement(disp_z, disp_y, disp_x, &inv_z, &inv_y, &inv_x, dims);

        let new_z: Vec<f32> = warped_z.iter().map(|&v| -v).collect();
        let new_y: Vec<f32> = warped_y.iter().map(|&v| -v).collect();
        let new_x: Vec<f32> = warped_x.iter().map(|&v| -v).collect();

        // Convergence check: max per-voxel Euclidean norm of the iterate change.
        // Accumulated in f64 to preserve precision when individual f32 changes
        // are small.
        let max_change = (0..n)
            .map(|i| {
                let dz = (new_z[i] - inv_z[i]) as f64;
                let dy = (new_y[i] - inv_y[i]) as f64;
                let dx = (new_x[i] - inv_x[i]) as f64;
                (dz * dz + dy * dy + dx * dx).sqrt()
            })
            .fold(0.0_f64, f64::max);

        inv_z = new_z;
        inv_y = new_y;
        inv_x = new_x;

        if max_change < config.tolerance {
            break;
        }
    }

    (inv_z, inv_y, inv_x, iters)
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Warp a displacement field by a query displacement field via trilinear
/// interpolation with clamp-to-border boundary conditions.
///
/// For each voxel at position `x = (iz, iy, ix)`:
///
///   `result_c(x) = disp_c(x + query(x))`   for each component `c ∈ {z, y, x}`
///
/// This evaluates `u(x + u^{-1}_k(x))` in the fixed-point inverse iteration.
///
/// # Implementation note
///
/// Each displacement component is a scalar field; sampling component `c` of
/// `disp` at query-shifted positions is identical to the `warp_image` operation.
/// `warp_image` is therefore called once per component — no additional
/// allocations or custom interpolation code are required.
fn warp_displacement(
    disp_z: &[f32],
    disp_y: &[f32],
    disp_x: &[f32],
    query_z: &[f32],
    query_y: &[f32],
    query_x: &[f32],
    dims: [usize; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // warp_image(data, dims, dz, dy, dx) computes for each voxel (iz, iy, ix):
    //   output[i] = data[ (iz + dz[i], iy + dy[i], ix + dx[i]) ]
    // which equals disp_c[ x + query(x) ] when query plays the role of the
    // displacement argument.
    (
        warp_image(disp_z, dims, query_z, query_y, query_x),
        warp_image(disp_y, dims, query_z, query_y, query_x),
        warp_image(disp_x, dims, query_z, query_y, query_x),
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        invert_displacement_field, invert_velocity_field, warp_displacement, InverseFieldConfig,
    };

    // ── Test 1 ────────────────────────────────────────────────────────────────

    /// The exact inverse of the zero velocity field is the zero velocity field.
    ///
    /// Base case: `exp(0) ∘ exp(−0) = id ∘ id = id`.
    /// Negating a zero vector yields a zero vector, so all three output
    /// components must be identically 0.0.
    #[test]
    fn test_velocity_field_negation_is_exact_inverse() {
        let n = 4 * 4 * 4;
        let vel_z = vec![0.0_f32; n];
        let vel_y = vec![0.0_f32; n];
        let vel_x = vec![0.0_f32; n];

        let (inv_z, inv_y, inv_x) = invert_velocity_field(&vel_z, &vel_y, &vel_x);

        assert_eq!(inv_z.len(), n, "inv_z length mismatch");
        assert_eq!(inv_y.len(), n, "inv_y length mismatch");
        assert_eq!(inv_x.len(), n, "inv_x length mismatch");

        assert!(
            inv_z.iter().all(|&v| v == 0.0),
            "inv_z must be identically zero for the zero velocity field"
        );
        assert!(
            inv_y.iter().all(|&v| v == 0.0),
            "inv_y must be identically zero for the zero velocity field"
        );
        assert!(
            inv_x.iter().all(|&v| v == 0.0),
            "inv_x must be identically zero for the zero velocity field"
        );
    }

    // ── Test 2 ────────────────────────────────────────────────────────────────

    /// The iterative inverse of a zero displacement field (identity map) is the
    /// zero displacement field.
    ///
    /// `φ = id` (zero displacement) ⟹ `φ^{-1} = id` (zero inverse).
    /// Initialisation sets `inv = −0 = 0`; warping a zero field by zero gives
    /// zero; the change is 0 < tolerance, so convergence is immediate.
    /// All output components must satisfy `|inv_c[i]| ≤ 1e-5`.
    #[test]
    fn test_invert_displacement_identity_field_is_zero() {
        let dims = [8usize, 8, 8];
        let n = dims[0] * dims[1] * dims[2];

        let disp_z = vec![0.0_f32; n];
        let disp_y = vec![0.0_f32; n];
        let disp_x = vec![0.0_f32; n];

        let config = InverseFieldConfig::default();
        let (inv_z, inv_y, inv_x, iters) =
            invert_displacement_field(&disp_z, &disp_y, &disp_x, dims, &config);

        assert!(iters >= 1, "at least one iteration must be performed");
        assert!(
            iters <= config.max_iterations,
            "iters {iters} exceeds max_iterations {}",
            config.max_iterations
        );

        let bound = 1e-5_f32;
        for i in 0..n {
            assert!(
                inv_z[i].abs() <= bound,
                "inv_z[{i}] = {} exceeds {bound}",
                inv_z[i]
            );
            assert!(
                inv_y[i].abs() <= bound,
                "inv_y[{i}] = {} exceeds {bound}",
                inv_y[i]
            );
            assert!(
                inv_x[i].abs() <= bound,
                "inv_x[{i}] = {} exceeds {bound}",
                inv_x[i]
            );
        }
    }

    // ── Test 3 ────────────────────────────────────────────────────────────────

    /// The inverse of a uniform x-translation by +2.0 voxels has mean
    /// x-displacement in [−2.1, −1.9].
    ///
    /// For `u_x = 2.0` (constant), `φ(x) = x + 2` and `φ^{-1}(y) = y − 2`,
    /// so `u^{-1}_x = −2.0` everywhere.  The Christensen iteration converges
    /// in exactly 1 step: warping a constant field by any displacement returns
    /// the same constant, making the update identical to the initialisation and
    /// the change equal to zero.
    #[test]
    fn test_invert_small_translation() {
        let dims = [16usize, 16, 16];
        let n = dims[0] * dims[1] * dims[2];

        let disp_z = vec![0.0_f32; n];
        let disp_y = vec![0.0_f32; n];
        let disp_x = vec![2.0_f32; n];

        let config = InverseFieldConfig::default();
        let (inv_z, inv_y, inv_x, _iters) =
            invert_displacement_field(&disp_z, &disp_y, &disp_x, dims, &config);

        let mean_inv_x: f64 = inv_x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        assert!(
            (-2.1..=-1.9).contains(&mean_inv_x),
            "mean(inv_x) = {mean_inv_x:.6}, expected in [−2.1, −1.9]"
        );

        // The z and y components of the inverse must remain near zero since the
        // forward field has no z or y displacement.
        let mean_inv_z: f64 = inv_z.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let mean_inv_y: f64 = inv_y.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        assert!(
            mean_inv_z.abs() < 1e-5,
            "mean(inv_z) = {mean_inv_z:.6}, expected near 0"
        );
        assert!(
            mean_inv_y.abs() < 1e-5,
            "mean(inv_y) = {mean_inv_y:.6}, expected near 0"
        );
    }

    // ── Test 4 ────────────────────────────────────────────────────────────────

    /// Composing a non-trivial sinusoidal displacement field with its computed
    /// inverse must yield a composition displacement below 0.5 voxels everywhere.
    ///
    /// # Field specification
    ///
    /// Half-period sinusoidal field in a 12³ volume:
    ///
    ///   `u_x(iz, iy, ix) = 2.0 · sin(π · ix / 11)`  (max amplitude 2.0)
    ///   `u_y(iz, iy, ix) = 1.5 · sin(π · iy / 11)`  (max amplitude 1.5)
    ///   `u_z(iz, iy, ix) = 1.0 · sin(π · iz / 11)`  (max amplitude 1.0)
    ///
    /// Lipschitz constants: `L_x = 2·π/11 ≈ 0.571`, `L_y ≈ 0.428`, `L_z ≈ 0.286`.
    /// All < 1 — Christensen iteration is a contraction mapping for each component.
    ///
    /// Jacobian of `φ = id + u` is diagonal: `J_cc = 1 + ∂u_c/∂c > 0` for all
    /// c ∈ {x, y, z}, so `φ` is a valid diffeomorphism.
    ///
    /// # Composition formula
    ///
    /// `(φ^{-1} ∘ φ)(x) = x`  iff  `u(x) + u^{-1}(x + u(x)) = 0`
    ///
    /// Composition displacement at voxel `i`:
    ///   `c(x) = u(x) + warp_displacement(u^{-1}, u)(x)`
    ///
    /// After 20 iterations with `L = 0.571`: error ≈ initial_error × 0.571²⁰
    /// ≈ initial_error × 1.3 × 10⁻⁵, far below the 0.5-voxel acceptance bound.
    #[test]
    fn test_invert_result_composition_near_identity() {
        use std::f32::consts::PI;

        let dims = [12usize, 12, 12];
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;

        // Build half-period sinusoidal displacement field.
        // Each component depends only on its own spatial index, making the
        // Jacobian diagonal and always positive (valid diffeomorphism).
        let mut disp_z = vec![0.0_f32; n];
        let mut disp_y = vec![0.0_f32; n];
        let mut disp_x = vec![0.0_f32; n];

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let i = iz * ny * nx + iy * nx + ix;
                    // π * c / (N-1) sweeps [0, π] — a half period — so max
                    // amplitude equals the coefficient and gradient at the
                    // boundary is zero, avoiding large clamping artefacts.
                    disp_x[i] = 2.0 * (PI * ix as f32 / (nx as f32 - 1.0)).sin();
                    disp_y[i] = 1.5 * (PI * iy as f32 / (ny as f32 - 1.0)).sin();
                    disp_z[i] = 1.0 * (PI * iz as f32 / (nz as f32 - 1.0)).sin();
                }
            }
        }

        let config = InverseFieldConfig {
            max_iterations: 20,
            tolerance: 1e-6,
        };
        let (inv_z, inv_y, inv_x, _iters) =
            invert_displacement_field(&disp_z, &disp_y, &disp_x, dims, &config);

        // Evaluate composition: c(x) = u(x) + u^{-1}(x + u(x))
        // warp_displacement(inv, disp, dims) returns u^{-1}(x + u(x)) for each x.
        let (comp_z, comp_y, comp_x) =
            warp_displacement(&inv_z, &inv_y, &inv_x, &disp_z, &disp_y, &disp_x, dims);

        let mut max_err = 0.0_f32;
        for i in 0..n {
            let ez = disp_z[i] + comp_z[i];
            let ey = disp_y[i] + comp_y[i];
            let ex = disp_x[i] + comp_x[i];
            let err = (ez * ez + ey * ey + ex * ex).sqrt();
            if err > max_err {
                max_err = err;
            }
        }

        assert!(
            max_err < 0.5,
            "max composition error {max_err:.6} voxels exceeds 0.5-voxel bound; \
             φ^{{-1}} ∘ φ is not sufficiently close to identity"
        );
    }

    // ── Test 5 ────────────────────────────────────────────────────────────────

    /// `invert_displacement_field` returns at most `config.max_iterations`
    /// iterations regardless of field content or tolerance setting.
    ///
    /// A sinusoidal displacement with a sub-machine-epsilon tolerance (1e-30)
    /// ensures the convergence criterion is never satisfied in f32/f64
    /// arithmetic — the f32 quantization floor (~1e-7) is orders of magnitude
    /// above 1e-30 — so the returned count equals `max_iterations` exactly.
    #[test]
    fn test_max_iterations_bound() {
        use std::f32::consts::PI;

        let dims = [8usize, 8, 8];
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;

        // Sinusoidal x-displacement; Lipschitz constant ≈ 0.571 ensures the
        // iterate change at each step remains >> 1e-30 in f64 arithmetic.
        let disp_x: Vec<f32> = (0..n)
            .map(|i| {
                let ix = i % nx;
                2.0 * (PI * ix as f32 / (nx as f32 - 1.0)).sin()
            })
            .collect();
        let disp_z = vec![0.0_f32; n];
        let disp_y = vec![0.0_f32; n];

        // Verify that nz and ny contribute to n (avoids unused-variable lint).
        debug_assert_eq!(n, nz * ny * nx);

        let max_iter = 5usize;
        let config = InverseFieldConfig {
            max_iterations: max_iter,
            // Tolerance below f32 quantization floor — never satisfied for
            // non-trivial fields, so the loop always runs to max_iterations.
            tolerance: 1e-30,
        };

        let (_, _, _, iters) = invert_displacement_field(&disp_z, &disp_y, &disp_x, dims, &config);

        assert!(
            iters <= max_iter,
            "returned {iters} iterations exceeds max_iterations {max_iter}"
        );
        assert_eq!(
            iters, max_iter,
            "expected exactly {max_iter} iterations with sub-epsilon tolerance, got {iters}"
        );
    }
}
