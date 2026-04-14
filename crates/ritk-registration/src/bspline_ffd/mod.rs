//! B-Spline Free-Form Deformation (FFD) registration.
//!
//! # Mathematical Specification
//!
//! Implements the FFD registration framework of Rueckert et al. (1999).
//! The transformation is parameterized by a uniform cubic B-spline control
//! lattice superimposed on the image domain.
//!
//! ## Transformation Model
//!
//! The FFD maps each point **x** to:
//!
//! ```text
//! φ(x) = x + Σᵢ cᵢ · β₃((x − xᵢ) / δ)
//! ```
//!
//! where:
//! - `cᵢ ∈ ℝ³` are the control-point displacement vectors,
//! - `β₃` is the tensor-product cubic B-spline basis function,
//! - `δ` is the (uniform) control-point spacing, and
//! - the sum runs over the 4³ = 64 control points whose support overlaps **x**.
//!
//! The cubic B-spline basis in one dimension for parameter `t ∈ [0, 1)`:
//!
//! ```text
//! β₃₀(t) = (1 − t)³ / 6
//! β₃₁(t) = (3t³ − 6t² + 4) / 6
//! β₃₂(t) = (−3t³ + 3t² + 3t + 1) / 6
//! β₃₃(t) = t³ / 6
//! ```
//!
//! ## Energy Functional
//!
//! Registration minimizes:
//!
//! ```text
//! E(c) = −D(F, M ∘ φ) + λ · R(φ)
//! ```
//!
//! where `D` is a voxel-wise similarity metric (NCC computed globally),
//! `R` is the bending-energy regularizer, and `λ` controls regularization
//! strength.
//!
//! ## Bending Energy
//!
//! ```text
//! R(φ) = (1/|Ω|) Σ_x [ (∂²φ/∂z²)² + (∂²φ/∂y²)² + (∂²φ/∂x²)²
//!                      + 2(∂²φ/∂z∂y)² + 2(∂²φ/∂z∂x)² + 2(∂²φ/∂y∂x)² ]
//! ```
//!
//! Approximated via finite differences on the dense displacement field.
//!
//! ## Optimization
//!
//! Gradient descent on control-point displacements. The gradient of the
//! similarity metric w.r.t. control points is:
//!
//! ```text
//! ∂D/∂cᵢ = Σ_x (∂D/∂φ(x)) · β₃((x − xᵢ) / δ)
//! ```
//!
//! The metric gradient `∂D/∂φ(x)` is computed from the NCC derivative
//! and the spatial gradient of the warped moving image.
//!
//! ## Multi-Resolution Strategy
//!
//! The algorithm proceeds from coarse to fine:
//! 1. Start with `initial_control_spacing`.
//! 2. At each level, optimize to convergence.
//! 3. Refine the control grid by doubling resolution (halving spacing)
//!    and B-spline subdivision of existing control-point displacements.
//!
//! # References
//!
//! - Rueckert, D., Sonoda, L. I., Hayes, C., Hill, D. L. G., Leach, M. O.,
//!   & Hawkes, D. J. (1999). Nonrigid registration using free-form
//!   deformations: application to breast MR images. *IEEE Transactions on
//!   Medical Imaging*, 18(8), 712–721.
//! - Lee, S., Wolberg, G., & Shin, S. Y. (1997). Scattered data interpolation
//!   with multilevel B-splines. *IEEE Transactions on Visualization and
//!   Computer Graphics*, 3(3), 228–244.

use crate::deformable_field_ops::{flat, warp_image};
use crate::error::RegistrationError;

// ── Public Types ──────────────────────────────────────────────────────────────

/// Configuration for B-Spline FFD registration.
#[derive(Debug, Clone)]
pub struct BSplineFFDConfig {
    /// Initial control-point spacing in voxels `[sz, sy, sx]`.
    pub initial_control_spacing: [usize; 3],
    /// Number of multi-resolution levels. Control spacing is halved at each
    /// subsequent level.
    pub num_levels: usize,
    /// Maximum gradient-descent iterations per level.
    pub max_iterations_per_level: usize,
    /// Learning rate (step size) for gradient descent on control displacements.
    pub learning_rate: f64,
    /// Bending-energy regularization weight λ.
    pub regularization_weight: f64,
    /// Convergence threshold: optimization stops when the relative change in
    /// the NCC metric between consecutive iterations falls below this value.
    pub convergence_threshold: f64,
}

impl Default for BSplineFFDConfig {
    fn default() -> Self {
        Self {
            initial_control_spacing: [8, 8, 8],
            num_levels: 3,
            max_iterations_per_level: 100,
            learning_rate: 1.0,
            regularization_weight: 1e-3,
            convergence_threshold: 1e-5,
        }
    }
}

/// Result of B-Spline FFD registration.
#[derive(Debug, Clone)]
pub struct BSplineFFDResult {
    /// Control-point displacements for each spatial component (dz, dy, dx).
    /// Each `Vec<f32>` has length `control_grid_dims[0] * control_grid_dims[1]
    /// * control_grid_dims[2]`.
    pub control_points: (Vec<f32>, Vec<f32>, Vec<f32>),
    /// Control-lattice dimensions `[nz, ny, nx]`.
    pub control_grid_dims: [usize; 3],
    /// Control-point spacing at the finest level `[δz, δy, δx]` in voxels.
    pub control_spacing: [f64; 3],
    /// Moving image warped to the fixed image domain.
    pub warped_moving: Vec<f32>,
    /// Final NCC metric value (higher → better alignment).
    pub final_metric: f64,
    /// Total gradient-descent iterations across all levels.
    pub num_iterations: usize,
}

/// B-Spline FFD registration engine.
///
/// Stateless entry point; all parameters are passed via [`BSplineFFDConfig`].
pub struct BSplineFFDRegistration;

// ── Engine ────────────────────────────────────────────────────────────────────

impl BSplineFFDRegistration {
    /// Register `moving` to `fixed` using multi-resolution B-Spline FFD.
    ///
    /// # Arguments
    /// - `fixed`   — reference image, flat `&[f32]` in Z-major order.
    /// - `moving`  — moving image, same shape as `fixed`.
    /// - `dims`    — image dimensions `[nz, ny, nx]`.
    /// - `spacing` — physical voxel spacing `[sz, sy, sx]`.
    /// - `config`  — algorithm parameters.
    ///
    /// # Errors
    /// Returns [`RegistrationError`] on dimension mismatch or invalid
    /// configuration.
    pub fn register(
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
        config: &BSplineFFDConfig,
    ) -> Result<BSplineFFDResult, RegistrationError> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;

        // ── Input validation ──────────────────────────────────────────────
        if fixed.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "fixed length {} != dims product {}",
                fixed.len(),
                n
            )));
        }
        if moving.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "moving length {} != dims product {}",
                moving.len(),
                n
            )));
        }
        if config.num_levels == 0 {
            return Err(RegistrationError::InvalidConfiguration(
                "num_levels must be >= 1".into(),
            ));
        }
        for d in 0..3 {
            if config.initial_control_spacing[d] == 0 {
                return Err(RegistrationError::InvalidConfiguration(format!(
                    "initial_control_spacing[{}] must be >= 1",
                    d
                )));
            }
        }

        // ── Initialize control grid at coarsest level ─────────────────────
        let mut ctrl_spacing = [
            config.initial_control_spacing[0] as f64,
            config.initial_control_spacing[1] as f64,
            config.initial_control_spacing[2] as f64,
        ];
        let mut ctrl_dims = init_control_grid(dims, &ctrl_spacing);
        let ctrl_n = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];
        let mut cp_z = vec![0.0_f32; ctrl_n];
        let mut cp_y = vec![0.0_f32; ctrl_n];
        let mut cp_x = vec![0.0_f32; ctrl_n];

        let mut total_iters = 0usize;
        let mut final_metric = 0.0_f64;

        // ── Multi-resolution loop ─────────────────────────────────────────
        for level in 0..config.num_levels {
            tracing::info!(
                level,
                ctrl_dims = ?ctrl_dims,
                ctrl_spacing = ?ctrl_spacing,
                "BSpline FFD: starting level"
            );

            let mut prev_metric = f64::NEG_INFINITY;

            for iter in 0..config.max_iterations_per_level {
                // 1. Evaluate dense displacement from current control points.
                let (disp_z, disp_y, disp_x) = evaluate_bspline_displacement(
                    &cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing, dims,
                );

                // 2. Warp moving image.
                let warped = warp_image(moving, dims, &disp_z, &disp_y, &disp_x);

                // 3. Compute NCC metric.
                let ncc = compute_ncc(fixed, &warped);

                // 4. Convergence check.
                let rel_change = if prev_metric.is_finite() && prev_metric.abs() > 1e-12 {
                    ((ncc - prev_metric) / prev_metric.abs()).abs()
                } else {
                    f64::INFINITY
                };

                if rel_change < config.convergence_threshold && iter > 0 {
                    tracing::debug!(iter, ncc, "BSpline FFD: converged");
                    total_iters += iter + 1;
                    final_metric = ncc;
                    break;
                }
                prev_metric = ncc;
                final_metric = ncc;

                // 5. Compute metric gradient w.r.t. control points.
                let (grad_z, grad_y, grad_x) = compute_metric_gradient(
                    fixed, moving, &warped, &disp_z, &disp_y, &disp_x,
                    &ctrl_dims, &ctrl_spacing, dims, spacing,
                );

                // 6. Compute bending-energy gradient w.r.t. control points.
                let (be_gz, be_gy, be_gx) = bending_energy_gradient(
                    &cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing,
                );

                // 7. Gradient descent update.
                let lr = config.learning_rate as f32;
                let lambda = config.regularization_weight as f32;
                let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];
                for i in 0..cn {
                    // Ascend NCC (maximize), descend bending energy (minimize).
                    cp_z[i] += lr * (grad_z[i] - lambda * be_gz[i]);
                    cp_y[i] += lr * (grad_y[i] - lambda * be_gy[i]);
                    cp_x[i] += lr * (grad_x[i] - lambda * be_gx[i]);
                }

                if iter == config.max_iterations_per_level - 1 {
                    total_iters += config.max_iterations_per_level;
                }
            }

            // Refine control grid for next level (except at the last level).
            if level + 1 < config.num_levels {
                let (new_z, new_y, new_x, new_dims, new_spacing) =
                    refine_control_grid(&cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);
                cp_z = new_z;
                cp_y = new_y;
                cp_x = new_x;
                ctrl_dims = new_dims;
                ctrl_spacing = new_spacing;
            }
        }

        // ── Final warp ───────────────────────────────────────────────────
        let (disp_z, disp_y, disp_x) = evaluate_bspline_displacement(
            &cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing, dims,
        );
        let warped_moving = warp_image(moving, dims, &disp_z, &disp_y, &disp_x);

        Ok(BSplineFFDResult {
            control_points: (cp_z, cp_y, cp_x),
            control_grid_dims: ctrl_dims,
            control_spacing: ctrl_spacing,
            warped_moving,
            final_metric,
            num_iterations: total_iters,
        })
    }
}

// ── Cubic B-Spline Basis ──────────────────────────────────────────────────────

/// Evaluate the four cubic B-spline basis values at parameter `t ∈ [0, 1]`.
///
/// Returns `[β₃₀(t), β₃₁(t), β₃₂(t), β₃₃(t)]` where:
///
/// ```text
/// β₃₀(t) = (1 − t)³ / 6
/// β₃₁(t) = (3t³ − 6t² + 4) / 6
/// β₃₂(t) = (−3t³ + 3t² + 3t + 1) / 6
/// β₃₃(t) = t³ / 6
/// ```
///
/// These sum to 1.0 (partition of unity) and are non-negative on `[0, 1]`.
#[inline]
fn cubic_bspline_1d(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    let omt = 1.0 - t;
    let omt3 = omt * omt * omt;

    [
        omt3 / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

/// Evaluate the four cubic B-spline basis *derivatives* at parameter `t ∈ [0, 1]`.
///
/// Returns `d/dt [β₃₀(t), β₃₁(t), β₃₂(t), β₃₃(t)]`:
///
/// ```text
/// β₃₀'(t) = −(1 − t)² / 2
/// β₃₁'(t) = (9t² − 12t) / 6  = (3t² − 4t) / 2
/// β₃₂'(t) = (−9t² + 6t + 3) / 6 = (−3t² + 2t + 1) / 2
/// β₃₃'(t) = t² / 2
/// ```
#[inline]
#[allow(dead_code)]
fn cubic_bspline_1d_deriv(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let omt = 1.0 - t;
    [
        -omt * omt / 2.0,
        (3.0 * t2 - 4.0 * t) / 2.0,
        (-3.0 * t2 + 2.0 * t + 1.0) / 2.0,
        t2 / 2.0,
    ]
}

// ── Control Grid Initialization ───────────────────────────────────────────────

/// Compute control-grid dimensions from image dimensions and control spacing.
///
/// The control lattice extends one extra control point beyond each boundary
/// to ensure full support coverage. Grid dimension along axis `d`:
///
/// ```text
/// n_ctrl[d] = ceil(dims[d] / spacing[d]) + 3
/// ```
///
/// The `+3` accounts for one point before the domain origin and two points
/// after the far boundary, providing the four-point support stencil at every
/// image voxel.
fn init_control_grid(dims: [usize; 3], ctrl_spacing: &[f64; 3]) -> [usize; 3] {
    let mut ctrl_dims = [0usize; 3];
    for d in 0..3 {
        ctrl_dims[d] = (dims[d] as f64 / ctrl_spacing[d]).ceil() as usize + 3;
    }
    ctrl_dims
}

// ── Dense Displacement Evaluation ─────────────────────────────────────────────

/// Evaluate the dense displacement field from B-spline control points.
///
/// For each image voxel `(iz, iy, ix)`, computes the displacement as the
/// tensor-product of 1D cubic B-spline bases evaluated over the 4×4×4
/// neighborhood of control points.
///
/// # Returns
/// `(dz, dy, dx)` — displacement components in voxel units, each of length
/// `dims[0] * dims[1] * dims[2]`.
fn evaluate_bspline_displacement(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
    dims: [usize; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let [cnz, cny, cnx] = *ctrl_dims;

    let mut dz = vec![0.0_f32; n];
    let mut dy = vec![0.0_f32; n];
    let mut dx = vec![0.0_f32; n];

    for iz in 0..nz {
        // Map image coordinate to control-grid parameter space.
        // The control grid origin is at index 1 (one point of padding before
        // the domain), so: u = iz / spacing + 1.
        let uz = iz as f64 / ctrl_spacing[0] + 1.0;
        let kz = uz.floor() as isize - 1; // first control index in stencil
        let tz = uz - (kz + 1) as f64;
        let bz = cubic_bspline_1d(tz);

        for iy in 0..ny {
            let uy = iy as f64 / ctrl_spacing[1] + 1.0;
            let ky = uy.floor() as isize - 1;
            let ty = uy - (ky + 1) as f64;
            let by = cubic_bspline_1d(ty);

            for ix in 0..nx {
                let ux = ix as f64 / ctrl_spacing[2] + 1.0;
                let kx = ux.floor() as isize - 1;
                let tx = ux - (kx + 1) as f64;
                let bx = cubic_bspline_1d(tx);

                let fi = flat(iz, iy, ix, ny, nx);
                let mut sum_z = 0.0_f64;
                let mut sum_y = 0.0_f64;
                let mut sum_x = 0.0_f64;

                for az in 0..4isize {
                    let ciz = kz + az;
                    if ciz < 0 || ciz >= cnz as isize {
                        continue;
                    }
                    let ciz = ciz as usize;
                    let wz = bz[az as usize];

                    for ay in 0..4isize {
                        let ciy = ky + ay;
                        if ciy < 0 || ciy >= cny as isize {
                            continue;
                        }
                        let ciy = ciy as usize;
                        let wzy = wz * by[ay as usize];

                        for ax in 0..4isize {
                            let cix = kx + ax;
                            if cix < 0 || cix >= cnx as isize {
                                continue;
                            }
                            let cix = cix as usize;
                            let w = wzy * bx[ax as usize];

                            let ci = flat(ciz, ciy, cix, cny, cnx);
                            sum_z += w * cp_z[ci] as f64;
                            sum_y += w * cp_y[ci] as f64;
                            sum_x += w * cp_x[ci] as f64;
                        }
                    }
                }

                dz[fi] = sum_z as f32;
                dy[fi] = sum_y as f32;
                dx[fi] = sum_x as f32;
            }
        }
    }

    (dz, dy, dx)
}

// ── NCC Metric ────────────────────────────────────────────────────────────────

/// Compute global normalized cross-correlation between two images.
///
/// ```text
/// NCC = Σ (F − μ_F)(M − μ_M) / sqrt(Σ (F − μ_F)² · Σ (M − μ_M)²)
/// ```
///
/// Returns a value in `[-1, 1]` where 1.0 indicates identical images up to
/// affine intensity scaling.
fn compute_ncc(fixed: &[f32], warped: &[f32]) -> f64 {
    let n = fixed.len() as f64;
    if n < 1.0 {
        return 0.0;
    }

    let mean_f: f64 = fixed.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mean_w: f64 = warped.iter().map(|&v| v as f64).sum::<f64>() / n;

    let mut sum_fw = 0.0_f64;
    let mut sum_ff = 0.0_f64;
    let mut sum_ww = 0.0_f64;

    for i in 0..fixed.len() {
        let f = fixed[i] as f64 - mean_f;
        let w = warped[i] as f64 - mean_w;
        sum_fw += f * w;
        sum_ff += f * f;
        sum_ww += w * w;
    }

    let denom = (sum_ff * sum_ww).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    sum_fw / denom
}

// ── Metric Gradient w.r.t. Control Points ─────────────────────────────────────

/// Compute the gradient of NCC w.r.t. control-point displacements.
///
/// Uses the chain rule:
///
/// ```text
/// ∂NCC/∂cᵢ = Σ_x (∂NCC/∂φ(x)) · (∂φ(x)/∂cᵢ)
/// ```
///
/// where `∂NCC/∂φ(x)` is the voxel-wise NCC gradient propagated through the
/// spatial gradient of the warped moving image, and `∂φ(x)/∂cᵢ = β₃((x−xᵢ)/δ)`.
///
/// # NCC Gradient Derivation
///
/// For the global NCC `ρ = Σ f̃ w̃ / sqrt(Σ f̃² · Σ w̃²)` where `f̃ = F − μ_F`
/// and `w̃ = W − μ_W`:
///
/// ```text
/// ∂ρ/∂W(x) = (1/σ_W) [ (f̃(x) / (n · σ_F)) − ρ · (w̃(x) / (n · σ_W)) ]
/// ```
///
/// and the displacement gradient at voxel x is:
///
/// ```text
/// ∂NCC/∂φ_d(x) = (∂ρ/∂W(x)) · (∂M_warped/∂φ_d)(x) ≈ (∂ρ/∂W(x)) · ∇_d M_warped(x)
/// ```
#[allow(clippy::too_many_arguments)]
fn compute_metric_gradient(
    fixed: &[f32],
    _moving: &[f32],
    warped: &[f32],
    _disp_z: &[f32],
    _disp_y: &[f32],
    _disp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
    dims: [usize; 3],
    spacing: [f64; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let [cnz, cny, cnx] = *ctrl_dims;
    let cn = cnz * cny * cnx;

    // Compute NCC statistics.
    let nf = n as f64;
    let mean_f: f64 = fixed.iter().map(|&v| v as f64).sum::<f64>() / nf;
    let mean_w: f64 = warped.iter().map(|&v| v as f64).sum::<f64>() / nf;

    let mut sum_ff = 0.0_f64;
    let mut sum_ww = 0.0_f64;
    let mut sum_fw = 0.0_f64;
    for i in 0..n {
        let f = fixed[i] as f64 - mean_f;
        let w = warped[i] as f64 - mean_w;
        sum_ff += f * f;
        sum_ww += w * w;
        sum_fw += f * w;
    }
    let sigma_f = (sum_ff / nf).sqrt();
    let sigma_w = (sum_ww / nf).sqrt();
    let rho = if sigma_f * sigma_w > 1e-12 {
        sum_fw / (nf * sigma_f * sigma_w)
    } else {
        0.0
    };

    // Compute spatial gradient of the warped image.
    let (gw_z, gw_y, gw_x) =
        crate::deformable_field_ops::compute_gradient(warped, dims, spacing);

    // Accumulate gradient into control points.
    let mut grad_cz = vec![0.0_f64; cn];
    let mut grad_cy = vec![0.0_f64; cn];
    let mut grad_cx = vec![0.0_f64; cn];

    let inv_n_sigma_f = if sigma_f > 1e-12 { 1.0 / (nf * sigma_f) } else { 0.0 };
    let inv_n_sigma_w = if sigma_w > 1e-12 { 1.0 / (nf * sigma_w) } else { 0.0 };
    let inv_sigma_w = if sigma_w > 1e-12 { 1.0 / sigma_w } else { 0.0 };

    for iz in 0..nz {
        let uz = iz as f64 / ctrl_spacing[0] + 1.0;
        let kz = uz.floor() as isize - 1;
        let tz = uz - (kz + 1) as f64;
        let bz = cubic_bspline_1d(tz);

        for iy in 0..ny {
            let uy = iy as f64 / ctrl_spacing[1] + 1.0;
            let ky = uy.floor() as isize - 1;
            let ty = uy - (ky + 1) as f64;
            let by = cubic_bspline_1d(ty);

            for ix in 0..nx {
                let ux = ix as f64 / ctrl_spacing[2] + 1.0;
                let kx = ux.floor() as isize - 1;
                let tx = ux - (kx + 1) as f64;
                let bx = cubic_bspline_1d(tx);

                let fi = flat(iz, iy, ix, ny, nx);

                let f_tilde = fixed[fi] as f64 - mean_f;
                let w_tilde = warped[fi] as f64 - mean_w;

                // ∂ρ/∂W(x) for the global NCC.
                let drho_dw = inv_sigma_w
                    * (f_tilde * inv_n_sigma_f - rho * w_tilde * inv_n_sigma_w);

                // Voxel-wise displacement gradient = drho_dw * ∇(warped).
                let vg_z = drho_dw * gw_z[fi] as f64;
                let vg_y = drho_dw * gw_y[fi] as f64;
                let vg_x = drho_dw * gw_x[fi] as f64;

                // Splatting: accumulate onto control points weighted by basis.
                for az in 0..4isize {
                    let ciz = kz + az;
                    if ciz < 0 || ciz >= cnz as isize {
                        continue;
                    }
                    let ciz = ciz as usize;
                    let wz = bz[az as usize];

                    for ay in 0..4isize {
                        let ciy = ky + ay;
                        if ciy < 0 || ciy >= cny as isize {
                            continue;
                        }
                        let ciy = ciy as usize;
                        let wzy = wz * by[ay as usize];

                        for ax in 0..4isize {
                            let cix = kx + ax;
                            if cix < 0 || cix >= cnx as isize {
                                continue;
                            }
                            let cix = cix as usize;
                            let w = wzy * bx[ax as usize];

                            let ci = flat(ciz, ciy, cix, cny, cnx);
                            grad_cz[ci] += w * vg_z;
                            grad_cy[ci] += w * vg_y;
                            grad_cx[ci] += w * vg_x;
                        }
                    }
                }
            }
        }
    }

    // Convert to f32.
    let out_z: Vec<f32> = grad_cz.iter().map(|&v| v as f32).collect();
    let out_y: Vec<f32> = grad_cy.iter().map(|&v| v as f32).collect();
    let out_x: Vec<f32> = grad_cx.iter().map(|&v| v as f32).collect();

    (out_z, out_y, out_x)
}

// ── Bending Energy ────────────────────────────────────────────────────────────

/// Compute the bending energy of the displacement field defined by the control
/// points.
///
/// The bending energy is computed directly on the control-point lattice using
/// finite differences of the control-point displacements:
///
/// ```text
/// R = (1/N) Σᵢ [ (Δ²_z cᵢ)² + (Δ²_y cᵢ)² + (Δ²_x cᵢ)²
///              + 2(Δ_zy cᵢ)² + 2(Δ_zx cᵢ)² + 2(Δ_yx cᵢ)² ]
/// ```
///
/// where `Δ²_d` denotes the second-order central difference along axis `d`
/// and `Δ_ab` denotes the cross second difference.
pub fn bending_energy(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
) -> f64 {
    let [cnz, cny, cnx] = *ctrl_dims;
    let mut energy = 0.0_f64;
    let mut count = 0usize;

    let sz2 = (ctrl_spacing[0] * ctrl_spacing[0]) as f32;
    let sy2 = (ctrl_spacing[1] * ctrl_spacing[1]) as f32;
    let sx2 = (ctrl_spacing[2] * ctrl_spacing[2]) as f32;
    let szy = (ctrl_spacing[0] * ctrl_spacing[1]) as f32;
    let szx = (ctrl_spacing[0] * ctrl_spacing[2]) as f32;
    let syx = (ctrl_spacing[1] * ctrl_spacing[2]) as f32;

    for comp in [cp_z, cp_y, cp_x] {
        for iz in 1..cnz.saturating_sub(1) {
            for iy in 1..cny.saturating_sub(1) {
                for ix in 1..cnx.saturating_sub(1) {
                    let c = comp[flat(iz, iy, ix, cny, cnx)];

                    // Pure second derivatives.
                    let dzz = (comp[flat(iz + 1, iy, ix, cny, cnx)]
                        - 2.0 * c
                        + comp[flat(iz - 1, iy, ix, cny, cnx)])
                        / sz2;
                    let dyy = (comp[flat(iz, iy + 1, ix, cny, cnx)]
                        - 2.0 * c
                        + comp[flat(iz, iy - 1, ix, cny, cnx)])
                        / sy2;
                    let dxx = (comp[flat(iz, iy, ix + 1, cny, cnx)]
                        - 2.0 * c
                        + comp[flat(iz, iy, ix - 1, cny, cnx)])
                        / sx2;

                    // Cross second derivatives.
                    let dzy = (comp[flat(iz + 1, iy + 1, ix, cny, cnx)]
                        - comp[flat(iz + 1, iy - 1, ix, cny, cnx)]
                        - comp[flat(iz - 1, iy + 1, ix, cny, cnx)]
                        + comp[flat(iz - 1, iy - 1, ix, cny, cnx)])
                        / (4.0 * szy);
                    let dzx = (comp[flat(iz + 1, iy, ix + 1, cny, cnx)]
                        - comp[flat(iz + 1, iy, ix - 1, cny, cnx)]
                        - comp[flat(iz - 1, iy, ix + 1, cny, cnx)]
                        + comp[flat(iz - 1, iy, ix - 1, cny, cnx)])
                        / (4.0 * szx);
                    let dyx = (comp[flat(iz, iy + 1, ix + 1, cny, cnx)]
                        - comp[flat(iz, iy + 1, ix - 1, cny, cnx)]
                        - comp[flat(iz, iy - 1, ix + 1, cny, cnx)]
                        + comp[flat(iz, iy - 1, ix - 1, cny, cnx)])
                        / (4.0 * syx);

                    energy += (dzz * dzz + dyy * dyy + dxx * dxx
                        + 2.0 * dzy * dzy
                        + 2.0 * dzx * dzx
                        + 2.0 * dyx * dyx) as f64;
                    count += 1;
                }
            }
        }
    }

    if count > 0 {
        energy / count as f64
    } else {
        0.0
    }
}

/// Compute the gradient of the bending energy w.r.t. control-point displacements.
///
/// This is the derivative of `bending_energy` with respect to each control-point
/// component, computed via the chain rule on the finite-difference operators.
/// Each interior control point's gradient accumulates contributions from its
/// own second-difference stencil and from neighboring stencils that include it.
///
/// For efficiency, this implementation applies the discretized biharmonic
/// operator (fourth-order finite difference) to each control point.
fn bending_energy_gradient(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [cnz, cny, cnx] = *ctrl_dims;
    let cn = cnz * cny * cnx;
    let count = count_interior(ctrl_dims);
    let norm = if count > 0 { 2.0 / count as f64 } else { 0.0 };

    let sz2 = (ctrl_spacing[0] * ctrl_spacing[0]) as f64;
    let sy2 = (ctrl_spacing[1] * ctrl_spacing[1]) as f64;
    let sx2 = (ctrl_spacing[2] * ctrl_spacing[2]) as f64;

    let mut out_z = vec![0.0_f32; cn];
    let mut out_y = vec![0.0_f32; cn];
    let mut out_x = vec![0.0_f32; cn];

    for (comp, out) in [(cp_z, &mut out_z), (cp_y, &mut out_y), (cp_x, &mut out_x)] {
        // Apply the biharmonic stencil: gradient of ∫ (∂²c/∂d²)² dd is
        // the fourth-order finite difference operator. For simplicity,
        // compute as the Laplacian of the Laplacian on the interior.
        // First compute the Laplacian on the interior.
        let mut lap = vec![0.0_f64; cn];
        for iz in 1..cnz.saturating_sub(1) {
            for iy in 1..cny.saturating_sub(1) {
                for ix in 1..cnx.saturating_sub(1) {
                    let ci = flat(iz, iy, ix, cny, cnx);
                    let c = comp[ci] as f64;
                    let dzz = (comp[flat(iz + 1, iy, ix, cny, cnx)] as f64
                        - 2.0 * c
                        + comp[flat(iz - 1, iy, ix, cny, cnx)] as f64)
                        / sz2;
                    let dyy = (comp[flat(iz, iy + 1, ix, cny, cnx)] as f64
                        - 2.0 * c
                        + comp[flat(iz, iy - 1, ix, cny, cnx)] as f64)
                        / sy2;
                    let dxx = (comp[flat(iz, iy, ix + 1, cny, cnx)] as f64
                        - 2.0 * c
                        + comp[flat(iz, iy, ix - 1, cny, cnx)] as f64)
                        / sx2;
                    lap[ci] = dzz + dyy + dxx;
                }
            }
        }
        // Gradient = Laplacian of Laplacian (approximation to biharmonic).
        for iz in 2..cnz.saturating_sub(2) {
            for iy in 2..cny.saturating_sub(2) {
                for ix in 2..cnx.saturating_sub(2) {
                    let ci = flat(iz, iy, ix, cny, cnx);
                    let l = lap[ci];
                    let lzz = (lap[flat(iz + 1, iy, ix, cny, cnx)]
                        - 2.0 * l
                        + lap[flat(iz - 1, iy, ix, cny, cnx)])
                        / sz2;
                    let lyy = (lap[flat(iz, iy + 1, ix, cny, cnx)]
                        - 2.0 * l
                        + lap[flat(iz, iy - 1, ix, cny, cnx)])
                        / sy2;
                    let lxx = (lap[flat(iz, iy, ix + 1, cny, cnx)]
                        - 2.0 * l
                        + lap[flat(iz, iy, ix - 1, cny, cnx)])
                        / sx2;
                    out[ci] = (norm * (lzz + lyy + lxx)) as f32;
                }
            }
        }
    }

    (out_z, out_y, out_x)
}

/// Count interior control points (those with at least 1 neighbor in each
/// direction).
fn count_interior(ctrl_dims: &[usize; 3]) -> usize {
    let w = |d: usize| if d >= 3 { d - 2 } else { 0 };
    w(ctrl_dims[0]) * w(ctrl_dims[1]) * w(ctrl_dims[2])
}

// ── Control Grid Refinement ───────────────────────────────────────────────────

/// Double the control-grid resolution via B-spline subdivision.
///
/// Each control-point displacement is subdivided using the cubic B-spline
/// refinement mask so that the represented displacement field is preserved
/// exactly (to within floating-point precision). The control spacing is halved.
///
/// # B-spline Subdivision Rule (1D)
///
/// Given coarse control points `P[i]`, the refined points `Q[j]` are:
///
/// ```text
/// Q[2i]     = (P[i-1] + 6·P[i] + P[i+1]) / 8
/// Q[2i + 1] = (P[i] + P[i+1]) / 2
/// ```
///
/// with boundary extension by clamping.
fn refine_control_grid(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, [usize; 3], [f64; 3]) {
    let [cnz, cny, cnx] = *ctrl_dims;

    let new_spacing = [
        ctrl_spacing[0] / 2.0,
        ctrl_spacing[1] / 2.0,
        ctrl_spacing[2] / 2.0,
    ];

    // New grid dimensions: 2 * old - 1 (exact subdivision), then pad +2 to
    // maintain the boundary extension. But the simpler approach matching
    // init_control_grid is to just recompute from the implicit image dims.
    // However, we don't have image dims here, so we use the subdivision rule:
    // new_dim = 2 * old_dim - 1 (minimum to preserve the field), but we
    // keep at least the same padding by using 2 * (old - 3) + 3 + 2 = 2*old - 1.
    let new_dims = [2 * cnz - 1, 2 * cny - 1, 2 * cnx - 1];
    let [nnz, nny, nnx] = new_dims;
    let nn = nnz * nny * nnx;

    let mut new_z = vec![0.0_f32; nn];
    let mut new_y = vec![0.0_f32; nn];
    let mut new_x = vec![0.0_f32; nn];

    // Apply tensor-product subdivision.
    for comp_pair in [
        (cp_z, &mut new_z),
        (cp_y, &mut new_y),
        (cp_x, &mut new_x),
    ] {
        let (old, new) = comp_pair;
        refine_component_3d(old, new, [cnz, cny, cnx], [nnz, nny, nnx]);
    }

    (new_z, new_y, new_x, new_dims, new_spacing)
}

/// Apply 3D B-spline subdivision to a single displacement component via three
/// sequential 1D passes (separable).
fn refine_component_3d(
    old: &[f32],
    new: &mut [f32],
    old_dims: [usize; 3],
    new_dims: [usize; 3],
) {
    let [oz, oy, ox] = old_dims;
    let [nz, ny, nx] = new_dims;

    // Pass 1: subdivide along X.  old [oz, oy, ox] -> tmp1 [oz, oy, nx]
    let mut tmp1 = vec![0.0_f32; oz * oy * nx];
    for iz in 0..oz {
        for iy in 0..oy {
            for jx in 0..nx {
                let ix = jx / 2;
                let v = if jx % 2 == 0 {
                    // Even: (P[i-1] + 6*P[i] + P[i+1]) / 8
                    let pm = if ix > 0 { old[flat(iz, iy, ix - 1, oy, ox)] } else { old[flat(iz, iy, 0, oy, ox)] };
                    let p0 = old[flat(iz, iy, ix, oy, ox)];
                    let pp = if ix + 1 < ox { old[flat(iz, iy, ix + 1, oy, ox)] } else { old[flat(iz, iy, ox - 1, oy, ox)] };
                    (pm + 6.0 * p0 + pp) / 8.0
                } else {
                    // Odd: (P[i] + P[i+1]) / 2
                    let p0 = old[flat(iz, iy, ix, oy, ox)];
                    let pp = if ix + 1 < ox { old[flat(iz, iy, ix + 1, oy, ox)] } else { old[flat(iz, iy, ox - 1, oy, ox)] };
                    (p0 + pp) / 2.0
                };
                tmp1[iz * oy * nx + iy * nx + jx] = v;
            }
        }
    }

    // Pass 2: subdivide along Y.  tmp1 [oz, oy, nx] -> tmp2 [oz, ny, nx]
    let mut tmp2 = vec![0.0_f32; oz * ny * nx];
    for iz in 0..oz {
        for jy in 0..ny {
            let iy = jy / 2;
            for jx in 0..nx {
                let v = if jy % 2 == 0 {
                    let pm = if iy > 0 { tmp1[iz * oy * nx + (iy - 1) * nx + jx] } else { tmp1[iz * oy * nx + jx] };
                    let p0 = tmp1[iz * oy * nx + iy * nx + jx];
                    let pp = if iy + 1 < oy { tmp1[iz * oy * nx + (iy + 1) * nx + jx] } else { tmp1[iz * oy * nx + (oy - 1) * nx + jx] };
                    (pm + 6.0 * p0 + pp) / 8.0
                } else {
                    let p0 = tmp1[iz * oy * nx + iy * nx + jx];
                    let pp = if iy + 1 < oy { tmp1[iz * oy * nx + (iy + 1) * nx + jx] } else { tmp1[iz * oy * nx + (oy - 1) * nx + jx] };
                    (p0 + pp) / 2.0
                };
                tmp2[iz * ny * nx + jy * nx + jx] = v;
            }
        }
    }

    // Pass 3: subdivide along Z.  tmp2 [oz, ny, nx] -> new [nz, ny, nx]
    for jz in 0..nz {
        let iz = jz / 2;
        for jy in 0..ny {
            for jx in 0..nx {
                let v = if jz % 2 == 0 {
                    let pm = if iz > 0 { tmp2[(iz - 1) * ny * nx + jy * nx + jx] } else { tmp2[jy * nx + jx] };
                    let p0 = tmp2[iz * ny * nx + jy * nx + jx];
                    let pp = if iz + 1 < oz { tmp2[(iz + 1) * ny * nx + jy * nx + jx] } else { tmp2[(oz - 1) * ny * nx + jy * nx + jx] };
                    (pm + 6.0 * p0 + pp) / 8.0
                } else {
                    let p0 = tmp2[iz * ny * nx + jy * nx + jx];
                    let pp = if iz + 1 < oz { tmp2[(iz + 1) * ny * nx + jy * nx + jx] } else { tmp2[(oz - 1) * ny * nx + jy * nx + jx] };
                    (p0 + pp) / 2.0
                };
                new[jz * ny * nx + jy * nx + jx] = v;
            }
        }
    }
}

// ── Warp via B-Spline Field ───────────────────────────────────────────────────

/// Warp an image using the B-spline displacement field.
///
/// Convenience wrapper that evaluates the dense displacement from control
/// points and then applies trilinear-interpolated warping.
///
/// # Returns
/// Warped image as a flat `Vec<f32>` of length `dims[0] * dims[1] * dims[2]`.
pub fn warp_image_bspline(
    moving: &[f32],
    dims: [usize; 3],
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
) -> Vec<f32> {
    let (dz, dy, dx) =
        evaluate_bspline_displacement(cp_z, cp_y, cp_x, ctrl_dims, ctrl_spacing, dims);
    warp_image(moving, dims, &dz, &dy, &dx)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deformable_field_ops::trilinear_interpolate;

    /// Smooth 3D test image: `I[z,y,x] = sin(π z/nz) · cos(π y/ny) · (x + 1)`.
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

    // ── Basis function tests ──────────────────────────────────────────────

    #[test]
    fn bspline_basis_partition_of_unity() {
        // The four cubic B-spline basis values must sum to 1.0 for any t ∈ [0,1].
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
                assert!(
                    val >= -1e-15,
                    "basis {}({}) = {} < 0",
                    j,
                    t,
                    val
                );
            }
        }
    }

    // ── Identity test ─────────────────────────────────────────────────────

    #[test]
    fn zero_control_displacements_produce_identity_warp() {
        let dims = [8, 10, 12];
        let image = make_test_image(dims);
        let ctrl_spacing = [4.0, 4.0, 4.0];
        let ctrl_dims = init_control_grid(dims, &ctrl_spacing);
        let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];

        let cp_z = vec![0.0_f32; cn];
        let cp_y = vec![0.0_f32; cn];
        let cp_x = vec![0.0_f32; cn];

        let warped = warp_image_bspline(&image, dims, &cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);

        for i in 0..image.len() {
            assert!(
                (warped[i] - image[i]).abs() < 1e-5,
                "identity warp mismatch at voxel {}: {} vs {}",
                i,
                warped[i],
                image[i]
            );
        }
    }

    // ── Refinement test ───────────────────────────────────────────────────

    #[test]
    fn refine_doubles_grid_points() {
        let ctrl_dims = [5, 6, 7];
        let ctrl_spacing = [8.0, 8.0, 8.0];
        let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];

        let cp_z = vec![0.0_f32; cn];
        let cp_y = vec![0.0_f32; cn];
        let cp_x = vec![0.0_f32; cn];

        let (_, _, _, new_dims, new_spacing) =
            refine_control_grid(&cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);

        // new_dims = 2 * old - 1
        assert_eq!(new_dims[0], 2 * ctrl_dims[0] - 1);
        assert_eq!(new_dims[1], 2 * ctrl_dims[1] - 1);
        assert_eq!(new_dims[2], 2 * ctrl_dims[2] - 1);

        // Spacing halved.
        for d in 0..3 {
            assert!(
                (new_spacing[d] - ctrl_spacing[d] / 2.0).abs() < 1e-12,
                "spacing mismatch at dim {}: {} vs {}",
                d,
                new_spacing[d],
                ctrl_spacing[d] / 2.0
            );
        }
    }

    // ── Known translation test ────────────────────────────────────────────

    #[test]
    fn constant_displacement_translates_image() {
        let dims = [8, 10, 12];
        let ctrl_spacing = [4.0, 4.0, 4.0];
        let ctrl_dims = init_control_grid(dims, &ctrl_spacing);
        let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];

        // Set all control points to constant displacement of +2 voxels in x.
        let cp_z = vec![0.0_f32; cn];
        let cp_y = vec![0.0_f32; cn];
        let cp_x = vec![2.0_f32; cn];

        // Create a simple ramp in x: I(z,y,x) = x.
        let [nz, ny, nx] = dims;
        let image: Vec<f32> = (0..nz * ny * nx)
            .map(|fi| (fi % nx) as f32)
            .collect();

        let warped = warp_image_bspline(&image, dims, &cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);

        // At interior voxels (away from boundary clamping), warped(z,y,x) ≈
        // moving(z, y, x + 2) = (x + 2). Near the right boundary, clamping
        // limits the sampled coordinate.
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..(nx - 3) {
                    let fi = flat(iz, iy, ix, ny, nx);
                    let expected = (ix + 2) as f32;
                    assert!(
                        (warped[fi] - expected).abs() < 0.5,
                        "translation mismatch at ({},{},{}): got {}, expected {}",
                        iz,
                        iy,
                        ix,
                        warped[fi],
                        expected
                    );
                }
            }
        }
    }

    // ── Bending energy tests ──────────────────────────────────────────────

    #[test]
    fn bending_energy_of_zero_field_is_zero() {
        let ctrl_dims = [6, 6, 6];
        let ctrl_spacing = [4.0, 4.0, 4.0];
        let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];

        let cp_z = vec![0.0_f32; cn];
        let cp_y = vec![0.0_f32; cn];
        let cp_x = vec![0.0_f32; cn];

        let be = bending_energy(&cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);
        assert!(
            be.abs() < 1e-12,
            "bending energy of zero field should be 0, got {}",
            be
        );
    }

    #[test]
    fn bending_energy_of_constant_field_is_zero() {
        // Constant displacement has zero second derivatives.
        let ctrl_dims = [6, 6, 6];
        let ctrl_spacing = [4.0, 4.0, 4.0];
        let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];

        let cp_z = vec![3.0_f32; cn];
        let cp_y = vec![-1.5_f32; cn];
        let cp_x = vec![2.0_f32; cn];

        let be = bending_energy(&cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);
        assert!(
            be.abs() < 1e-10,
            "bending energy of constant field should be ~0, got {}",
            be
        );
    }

    #[test]
    fn bending_energy_positive_for_nonlinear_field() {
        // A quadratic field has nonzero second derivatives → positive bending energy.
        let ctrl_dims = [6, 6, 6];
        let ctrl_spacing = [4.0, 4.0, 4.0];
        let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];
        let [cnz, cny, cnx] = ctrl_dims;

        let mut cp_x = vec![0.0_f32; cn];
        let cp_y = vec![0.0_f32; cn];
        let cp_z = vec![0.0_f32; cn];

        // Set cp_x to a quadratic: ix^2.
        for iz in 0..cnz {
            for iy in 0..cny {
                for ix in 0..cnx {
                    cp_x[flat(iz, iy, ix, cny, cnx)] = (ix as f32) * (ix as f32);
                }
            }
        }

        let be = bending_energy(&cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);
        assert!(
            be > 0.0,
            "bending energy of quadratic field should be > 0, got {}",
            be
        );
    }

    // ── NCC metric tests ──────────────────────────────────────────────────

    #[test]
    fn ncc_identical_images_is_one() {
        let image: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
        let ncc = compute_ncc(&image, &image);
        assert!(
            (ncc - 1.0).abs() < 1e-10,
            "NCC of identical images should be 1.0, got {}",
            ncc
        );
    }

    #[test]
    fn ncc_negated_image_is_minus_one() {
        let image: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let neg: Vec<f32> = image.iter().map(|&v| -v).collect();
        let ncc = compute_ncc(&image, &neg);
        assert!(
            (ncc - (-1.0)).abs() < 1e-10,
            "NCC of negated images should be -1.0, got {}",
            ncc
        );
    }

    // ── Registration convergence test ─────────────────────────────────────

    #[test]
    fn metric_improves_after_iterations() {
        let dims = [8, 10, 12];
        let spacing = [1.0, 1.0, 1.0];
        let fixed = make_test_image(dims);

        // Create a translated version of the fixed image (shift +1 in x).
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let mut moving = vec![0.0_f32; n];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let src_x = (ix as f32 + 1.0).min((nx - 1) as f32);
                    moving[flat(iz, iy, ix, ny, nx)] =
                        trilinear_interpolate(&fixed, dims, iz as f32, iy as f32, src_x);
                }
            }
        }

        // Run a single-level registration with a few iterations.
        let config = BSplineFFDConfig {
            initial_control_spacing: [4, 4, 4],
            num_levels: 1,
            max_iterations_per_level: 10,
            learning_rate: 0.5,
            regularization_weight: 0.0,
            convergence_threshold: 1e-8,
        };

        // Compute initial NCC (before any optimization).
        let initial_ncc = compute_ncc(&fixed, &moving);

        let result =
            BSplineFFDRegistration::register(&fixed, &moving, dims, spacing, &config).unwrap();

        // The final metric should be at least as good as the initial.
        assert!(
            result.final_metric >= initial_ncc - 1e-6,
            "metric should not degrade: initial={}, final={}",
            initial_ncc,
            result.final_metric
        );
    }

    // ── Dimension mismatch error ──────────────────────────────────────────

    #[test]
    fn mismatched_fixed_length_returns_error() {
        let config = BSplineFFDConfig::default();
        let fixed = vec![0.0_f32; 100];
        let moving = vec![0.0_f32; 8];
        let result =
            BSplineFFDRegistration::register(&fixed, &moving, [2, 2, 2], [1.0; 3], &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, RegistrationError::DimensionMismatch(_)),
            "expected DimensionMismatch, got {:?}",
            err
        );
    }

    #[test]
    fn mismatched_moving_length_returns_error() {
        let config = BSplineFFDConfig::default();
        let fixed = vec![0.0_f32; 8];
        let moving = vec![0.0_f32; 100];
        let result =
            BSplineFFDRegistration::register(&fixed, &moving, [2, 2, 2], [1.0; 3], &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, RegistrationError::DimensionMismatch(_)),
            "expected DimensionMismatch, got {:?}",
            err
        );
    }

    // ── Config validation ─────────────────────────────────────────────────

    #[test]
    fn zero_levels_returns_invalid_configuration() {
        let config = BSplineFFDConfig {
            num_levels: 0,
            ..Default::default()
        };
        let img = vec![0.0_f32; 8];
        let result =
            BSplineFFDRegistration::register(&img, &img, [2, 2, 2], [1.0; 3], &config);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RegistrationError::InvalidConfiguration(_)
        ));
    }

    #[test]
    fn zero_spacing_returns_invalid_configuration() {
        let config = BSplineFFDConfig {
            initial_control_spacing: [0, 4, 4],
            ..Default::default()
        };
        let img = vec![0.0_f32; 8];
        let result =
            BSplineFFDRegistration::register(&img, &img, [2, 2, 2], [1.0; 3], &config);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RegistrationError::InvalidConfiguration(_)
        ));
    }

    // ── Control grid init test ────────────────────────────────────────────

    #[test]
    fn init_control_grid_dimensions_correct() {
        // dims = [16, 20, 24], spacing = [8, 8, 8]
        // n_ctrl[d] = ceil(dims[d]/8) + 3
        // z: ceil(16/8)+3 = 2+3 = 5
        // y: ceil(20/8)+3 = 3+3 = 6
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
}
