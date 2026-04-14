//! LDDMM — Large Deformation Diffeomorphic Metric Mapping.
//!
//! # Mathematical Specification
//!
//! LDDMM (Beg et al. 2005) generates geodesic paths in the group of
//! diffeomorphisms by optimising the initial velocity v₀ of a time-dependent
//! velocity field v(t), t ∈ \[0, 1\].
//!
//! ## Energy functional
//!
//! E(v₀) = λ ‖v₀‖²\_V + MSE(I ∘ φ₁, J)
//!
//! where:
//! - ‖·‖\_V is the Sobolev norm induced by Gaussian kernel K\_σ,
//! - φ₁ is the diffeomorphism at t = 1 obtained by integrating v(t),
//! - I is the moving image, J is the fixed image,
//! - λ is the regularisation weight (`regularization_weight`).
//!
//! ## EPDiff shooting (geodesic integration)
//!
//! The velocity field evolves according to the EPDiff equation:
//!
//!   ∂v/∂t = −K\_σ ∗ ad\*\_v(m)
//!
//! where m = K\_σ ∗ v is the momentum and ad\*\_v(m) is the coadjoint
//! operator of the Lie-algebra adjoint:
//!
//!   (ad\*\_v m)\_i = Σ\_j \[v\_j · ∂m\_i/∂x\_j + m\_j · ∂v\_i/∂x\_j\] + m\_i · div(v)
//!
//! Integration proceeds via forward Euler over N\_t steps with dt = 1/N\_t.
//! At each step the displacement field is composed with the incremental map
//! id + v(t)·dt to accumulate the full diffeomorphism φ₁.
//!
//! ## Gradient descent update
//!
//!   ∂E/∂v₀ = 2λ v₀ + K\_σ ∗ \[2(I∘φ₁ − J) · ∇(I∘φ₁)\]
//!
//!   v₀ ← v₀ − lr · ∂E/∂v₀
//!
//! # References
//!
//! - Beg, M. F., Miller, M. I., Trouvé, A. & Younes, L. (2005).
//!   Computing large deformation metric mappings via geodesic flows of
//!   diffeomorphisms. *Int. J. Comput. Vis.* 61(2):139–157.
//! - Vialard, F.-X., Risser, L., Rueckert, D. & Cotter, C. J. (2012).
//!   Diffeomorphic 3D image registration via geodesic shooting using an
//!   efficient adjoint calculation. *Int. J. Comput. Vis.* 97(2):153–174.

use crate::deformable_field_ops::{
    compose_fields, compute_gradient, flat, gaussian_smooth_inplace, warp_image,
};
use crate::error::RegistrationError;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Parameters for LDDMM registration.
#[derive(Debug, Clone)]
pub struct LddmmConfig {
    /// Maximum number of gradient-descent iterations.
    pub max_iterations: usize,
    /// Number of Euler steps for geodesic integration (N\_t).
    pub num_time_steps: usize,
    /// Standard deviation (voxels) of Gaussian kernel K\_σ for the Sobolev norm.
    pub kernel_sigma: f64,
    /// Gradient-descent step size.
    pub learning_rate: f64,
    /// Weight λ on the regularisation term ‖v₀‖²\_V.
    pub regularization_weight: f64,
    /// Stop when |MSE\_{k} − MSE\_{k−1}| / (MSE\_{k−1} + ε) < threshold.
    pub convergence_threshold: f64,
}

impl Default for LddmmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            num_time_steps: 10,
            kernel_sigma: 2.0,
            learning_rate: 0.1,
            regularization_weight: 1.0,
            convergence_threshold: 1e-5,
        }
    }
}

// ── Result ────────────────────────────────────────────────────────────────────

/// Output of LDDMM registration.
#[derive(Debug, Clone)]
pub struct LddmmResult {
    /// Optimised initial velocity (vz, vy, vx) parameterising the geodesic.
    pub initial_velocity: (Vec<f32>, Vec<f32>, Vec<f32>),
    /// Displacement field (dz, dy, dx) at t = 1 in voxel units.
    pub displacement_field: (Vec<f32>, Vec<f32>, Vec<f32>),
    /// Moving image warped by φ₁.
    pub warped_moving: Vec<f32>,
    /// Final MSE after the last forward pass.
    pub final_metric: f64,
    /// Number of gradient-descent iterations executed.
    pub num_iterations: usize,
}

// ── Registration engine ───────────────────────────────────────────────────────

/// LDDMM registration engine.
///
/// Optimises the initial velocity v₀ of a geodesic in diffeomorphism space
/// to align a moving image to a fixed image under the MSE similarity metric
/// with Sobolev-norm regularisation.
#[derive(Debug, Clone)]
pub struct LddmmRegistration {
    /// Algorithm configuration.
    pub config: LddmmConfig,
}

impl LddmmRegistration {
    /// Create a registration instance with the given configuration.
    pub fn new(config: LddmmConfig) -> Self {
        Self { config }
    }

    /// Register `moving` to `fixed` via LDDMM geodesic shooting.
    ///
    /// # Arguments
    /// - `fixed`   — reference image, flat `[f32]` in Z-major order.
    /// - `moving`  — moving image, same length as `fixed`.
    /// - `dims`    — volume dimensions `[nz, ny, nx]`.
    /// - `spacing` — physical voxel spacing `[sz, sy, sx]`.
    ///
    /// # Errors
    /// Returns [`RegistrationError::DimensionMismatch`] when image lengths
    /// differ from `nz * ny * nx`.
    pub fn register(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
    ) -> Result<LddmmResult, RegistrationError> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;

        if fixed.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "fixed length {} != nz*ny*nx = {}",
                fixed.len(),
                n
            )));
        }
        if moving.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "moving length {} != nz*ny*nx = {}",
                moving.len(),
                n
            )));
        }

        let cfg = &self.config;
        let lr = cfg.learning_rate as f32;
        let lam = cfg.regularization_weight as f32;

        // Initial velocity v₀ = 0 (identity geodesic).
        let mut v0z = vec![0.0_f32; n];
        let mut v0y = vec![0.0_f32; n];
        let mut v0x = vec![0.0_f32; n];

        let mut prev_mse = f64::MAX;
        let mut num_iters = 0_usize;

        for iter in 0..cfg.max_iterations {
            // ── Forward pass ──────────────────────────────────────────────
            let (dz, dy, dx) = integrate_geodesic(
                &v0z,
                &v0y,
                &v0x,
                dims,
                spacing,
                cfg.num_time_steps,
                cfg.kernel_sigma,
            );
            let warped = warp_image(moving, dims, &dz, &dy, &dx);

            // MSE = (1/n) Σ (warped − fixed)²
            let mse: f64 = warped
                .iter()
                .zip(fixed.iter())
                .map(|(&w, &f)| {
                    let d = (w - f) as f64;
                    d * d
                })
                .sum::<f64>()
                / n as f64;

            num_iters = iter + 1;

            // ── Convergence check ─────────────────────────────────────────
            if iter > 0 {
                let rel = (prev_mse - mse).abs() / (prev_mse + 1e-12);
                if rel < cfg.convergence_threshold {
                    break;
                }
            }
            prev_mse = mse;

            // ── Gradient w.r.t. v₀ ────────────────────────────────────────
            // Body force: K_σ ∗ [2 (warped − fixed) · ∇(warped)]
            let (gw_z, gw_y, gw_x) = compute_gradient(&warped, dims, spacing);

            let mut bf_z = vec![0.0_f32; n];
            let mut bf_y = vec![0.0_f32; n];
            let mut bf_x = vec![0.0_f32; n];
            for i in 0..n {
                let residual = 2.0 * (warped[i] - fixed[i]);
                bf_z[i] = residual * gw_z[i];
                bf_y[i] = residual * gw_y[i];
                bf_x[i] = residual * gw_x[i];
            }
            gaussian_smooth_inplace(&mut bf_z, dims, cfg.kernel_sigma);
            gaussian_smooth_inplace(&mut bf_y, dims, cfg.kernel_sigma);
            gaussian_smooth_inplace(&mut bf_x, dims, cfg.kernel_sigma);

            // Full gradient: 2λ v₀ + body_force.  Update: v₀ ← v₀ − lr · grad.
            for i in 0..n {
                v0z[i] -= lr * (2.0 * lam * v0z[i] + bf_z[i]);
                v0y[i] -= lr * (2.0 * lam * v0y[i] + bf_y[i]);
                v0x[i] -= lr * (2.0 * lam * v0x[i] + bf_x[i]);
            }
        }

        // ── Final forward pass after last update ──────────────────────────
        let (dz, dy, dx) = integrate_geodesic(
            &v0z,
            &v0y,
            &v0x,
            dims,
            spacing,
            cfg.num_time_steps,
            cfg.kernel_sigma,
        );
        let warped = warp_image(moving, dims, &dz, &dy, &dx);
        let final_mse: f64 = warped
            .iter()
            .zip(fixed.iter())
            .map(|(&w, &f)| {
                let d = (w - f) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        Ok(LddmmResult {
            initial_velocity: (v0z, v0y, v0x),
            displacement_field: (dz, dy, dx),
            warped_moving: warped,
            final_metric: final_mse,
            num_iterations: num_iters,
        })
    }
}

// ── Geodesic integration ──────────────────────────────────────────────────────

/// Integrate the EPDiff equation forward from initial velocity `(v0z, v0y, v0x)`
/// for `num_steps` Euler steps and return the accumulated displacement field
/// at t = 1.
///
/// At each step k ∈ \[0, num\_steps):
/// 1. m = K\_σ ∗ v  (momentum)
/// 2. a = K\_σ ∗ ad\*\_v(m)
/// 3. v ← v − dt · a
/// 4. φ ← (id + v·dt) ∘ φ   (compose incremental step)
fn integrate_geodesic(
    v0z: &[f32],
    v0y: &[f32],
    v0x: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    num_steps: usize,
    kernel_sigma: f64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = dims[0] * dims[1] * dims[2];
    let dt = 1.0 / num_steps as f32;

    let mut vz = v0z.to_vec();
    let mut vy = v0y.to_vec();
    let mut vx = v0x.to_vec();

    let mut dz = vec![0.0_f32; n];
    let mut dy = vec![0.0_f32; n];
    let mut dx = vec![0.0_f32; n];

    for _ in 0..num_steps {
        // 1. Momentum: m = K_σ ∗ v.
        let mut mz = vz.clone();
        let mut my = vy.clone();
        let mut mx = vx.clone();
        gaussian_smooth_inplace(&mut mz, dims, kernel_sigma);
        gaussian_smooth_inplace(&mut my, dims, kernel_sigma);
        gaussian_smooth_inplace(&mut mx, dims, kernel_sigma);

        // 2. EPDiff adjoint ad*_v(m), then smooth.
        let (mut adz, mut ady, mut adx) =
            epdiff_adjoint(&vz, &vy, &vx, &mz, &my, &mx, dims, spacing);
        gaussian_smooth_inplace(&mut adz, dims, kernel_sigma);
        gaussian_smooth_inplace(&mut ady, dims, kernel_sigma);
        gaussian_smooth_inplace(&mut adx, dims, kernel_sigma);

        // 3. Velocity update: v ← v − dt · K_σ ∗ ad*_v(m).
        for i in 0..n {
            vz[i] -= dt * adz[i];
            vy[i] -= dt * ady[i];
            vx[i] -= dt * adx[i];
        }

        // 4. Compose displacement: φ ← (v·dt) ∘ φ.
        let step_z: Vec<f32> = vz.iter().map(|&v| v * dt).collect();
        let step_y: Vec<f32> = vy.iter().map(|&v| v * dt).collect();
        let step_x: Vec<f32> = vx.iter().map(|&v| v * dt).collect();

        let composed = compose_fields(&step_z, &step_y, &step_x, &dz, &dy, &dx, dims);
        dz = composed.0;
        dy = composed.1;
        dx = composed.2;
    }

    (dz, dy, dx)
}

// ── EPDiff adjoint operator ───────────────────────────────────────────────────

/// Compute the EPDiff coadjoint operator ad\*\_v(m).
///
/// For each spatial component i ∈ {z, y, x}:
///
///   (ad\*\_v m)\_i = Σ\_j \[v\_j · ∂m\_i/∂x\_j + m\_j · ∂v\_i/∂x\_j\] + m\_i · div(v)
///
/// Derivatives use central differences at interior voxels and one-sided
/// differences at boundaries, consistent with [`compute_gradient`].
fn epdiff_adjoint(
    vz: &[f32],
    vy: &[f32],
    vx: &[f32],
    mz: &[f32],
    my: &[f32],
    mx: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let sz = spacing[0] as f32;
    let sy = spacing[1] as f32;
    let sx = spacing[2] as f32;

    let mut ad_z = vec![0.0_f32; n];
    let mut ad_y = vec![0.0_f32; n];
    let mut ad_x = vec![0.0_f32; n];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);

                // ∂f/∂z via central / one-sided differences.
                let ddz = |f: &[f32]| -> f32 {
                    if nz == 1 {
                        0.0
                    } else if iz == 0 {
                        (f[flat(1, iy, ix, ny, nx)] - f[fi]) / sz
                    } else if iz == nz - 1 {
                        (f[fi] - f[flat(nz - 2, iy, ix, ny, nx)]) / sz
                    } else {
                        (f[flat(iz + 1, iy, ix, ny, nx)] - f[flat(iz - 1, iy, ix, ny, nx)])
                            / (2.0 * sz)
                    }
                };
                // ∂f/∂y
                let ddy = |f: &[f32]| -> f32 {
                    if ny == 1 {
                        0.0
                    } else if iy == 0 {
                        (f[flat(iz, 1, ix, ny, nx)] - f[fi]) / sy
                    } else if iy == ny - 1 {
                        (f[fi] - f[flat(iz, ny - 2, ix, ny, nx)]) / sy
                    } else {
                        (f[flat(iz, iy + 1, ix, ny, nx)] - f[flat(iz, iy - 1, ix, ny, nx)])
                            / (2.0 * sy)
                    }
                };
                // ∂f/∂x
                let ddx = |f: &[f32]| -> f32 {
                    if nx == 1 {
                        0.0
                    } else if ix == 0 {
                        (f[flat(iz, iy, 1, ny, nx)] - f[fi]) / sx
                    } else if ix == nx - 1 {
                        (f[fi] - f[flat(iz, iy, nx - 2, ny, nx)]) / sx
                    } else {
                        (f[flat(iz, iy, ix + 1, ny, nx)] - f[flat(iz, iy, ix - 1, ny, nx)])
                            / (2.0 * sx)
                    }
                };

                let div_v = ddz(vz) + ddy(vy) + ddx(vx);

                // (ad*_v m)_z = (v·∇)m_z + (m·∇)v_z + m_z · div(v)
                ad_z[fi] = vz[fi] * ddz(mz)
                    + vy[fi] * ddy(mz)
                    + vx[fi] * ddx(mz)
                    + mz[fi] * ddz(vz)
                    + my[fi] * ddy(vz)
                    + mx[fi] * ddx(vz)
                    + mz[fi] * div_v;

                // (ad*_v m)_y = (v·∇)m_y + (m·∇)v_y + m_y · div(v)
                ad_y[fi] = vz[fi] * ddz(my)
                    + vy[fi] * ddy(my)
                    + vx[fi] * ddx(my)
                    + mz[fi] * ddz(vy)
                    + my[fi] * ddy(vy)
                    + mx[fi] * ddx(vy)
                    + my[fi] * div_v;

                // (ad*_v m)_x = (v·∇)m_x + (m·∇)v_x + m_x · div(v)
                ad_x[fi] = vz[fi] * ddz(mx)
                    + vy[fi] * ddy(mx)
                    + vx[fi] * ddx(mx)
                    + mz[fi] * ddz(vx)
                    + my[fi] * ddy(vx)
                    + mx[fi] * ddx(vx)
                    + mx[fi] * div_v;
            }
        }
    }

    (ad_z, ad_y, ad_x)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{epdiff_adjoint, integrate_geodesic, LddmmConfig, LddmmRegistration};
    use crate::error::RegistrationError;

    /// Linear ramp image: intensity = flat\_index / n, range \[0, 1).
    fn make_test_image(dims: [usize; 3]) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        (0..n).map(|i| (i as f32) / (n as f32)).collect()
    }

    /// Isotropic Gaussian blob with peak 1.0.
    fn gaussian_blob(dims: [usize; 3], center: [f32; 3], sigma: f32) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
        (0..n)
            .map(|i| {
                let iz = (i / (ny * nx)) as f32;
                let iy = ((i % (ny * nx)) / nx) as f32;
                let ix = (i % nx) as f32;
                let r2 =
                    (iz - center[0]).powi(2) + (iy - center[1]).powi(2) + (ix - center[2]).powi(2);
                (-r2 * inv_2s2).exp()
            })
            .collect()
    }

    #[test]
    fn identity_registration_low_mse() {
        let dims = [6, 6, 6];
        let img = make_test_image(dims);
        let reg = LddmmRegistration::new(LddmmConfig {
            max_iterations: 5,
            num_time_steps: 2,
            kernel_sigma: 1.0,
            learning_rate: 0.01,
            regularization_weight: 1.0,
            convergence_threshold: 1e-12,
        });
        let result = reg.register(&img, &img, dims, [1.0; 3]).unwrap();

        // Identical images: MSE must be ≈ 0.
        assert!(
            result.final_metric < 1e-10,
            "final_metric = {} exceeds 1e-10",
            result.final_metric
        );
        // Displacement must be ≈ 0.
        let max_disp = result
            .displacement_field
            .0
            .iter()
            .chain(result.displacement_field.1.iter())
            .chain(result.displacement_field.2.iter())
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_disp < 1e-6,
            "max displacement = {} exceeds 1e-6",
            max_disp
        );
    }

    #[test]
    fn metric_improves_over_iterations() {
        let dims = [8, 8, 8];
        let center_f = [3.5_f32, 3.5, 3.5];
        let center_m = [3.5_f32, 3.5, 4.5]; // shifted +1 voxel in x
        let fixed = gaussian_blob(dims, center_f, 2.0);
        let moving = gaussian_blob(dims, center_m, 2.0);

        // Initial MSE before any registration.
        let n = dims[0] * dims[1] * dims[2];
        let initial_mse: f64 = fixed
            .iter()
            .zip(moving.iter())
            .map(|(&f, &m)| {
                let d = (m - f) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        assert!(
            initial_mse > 1e-6,
            "initial_mse {} too small for meaningful test",
            initial_mse
        );

        let reg = LddmmRegistration::new(LddmmConfig {
            max_iterations: 30,
            num_time_steps: 2,
            kernel_sigma: 1.0,
            learning_rate: 0.1,
            regularization_weight: 0.01,
            convergence_threshold: 1e-12,
        });
        let result = reg.register(&fixed, &moving, dims, [1.0; 3]).unwrap();

        assert!(
            result.final_metric < initial_mse,
            "final_metric {} >= initial_mse {}",
            result.final_metric,
            initial_mse
        );
    }

    #[test]
    fn displacement_field_is_finite() {
        let dims = [4, 4, 4];
        let img = make_test_image(dims);
        let reg = LddmmRegistration::new(LddmmConfig {
            max_iterations: 3,
            num_time_steps: 2,
            kernel_sigma: 1.0,
            ..LddmmConfig::default()
        });
        let result = reg.register(&img, &img, dims, [1.0; 3]).unwrap();

        for &v in result
            .displacement_field
            .0
            .iter()
            .chain(result.displacement_field.1.iter())
            .chain(result.displacement_field.2.iter())
        {
            assert!(v.is_finite(), "non-finite displacement value: {}", v);
        }
    }

    #[test]
    fn mismatched_dims_returns_error() {
        let dims = [4, 4, 4];
        let n = 4 * 4 * 4;
        let img = vec![0.0_f32; n];
        let short = vec![0.0_f32; n - 1];
        let reg = LddmmRegistration::new(LddmmConfig::default());

        let err = reg.register(&img, &short, dims, [1.0; 3]);
        assert!(
            matches!(err, Err(RegistrationError::DimensionMismatch(_))),
            "expected DimensionMismatch for short moving, got {:?}",
            err
        );

        let err2 = reg.register(&short, &img, dims, [1.0; 3]);
        assert!(
            matches!(err2, Err(RegistrationError::DimensionMismatch(_))),
            "expected DimensionMismatch for short fixed, got {:?}",
            err2
        );
    }

    #[test]
    fn geodesic_shooting_zero_velocity_produces_identity() {
        let dims = [4, 4, 4];
        let n = 4 * 4 * 4;
        let zeros = vec![0.0_f32; n];
        let (dz, dy, dx) = integrate_geodesic(&zeros, &zeros, &zeros, dims, [1.0; 3], 5, 1.0);

        for i in 0..n {
            assert_eq!(dz[i], 0.0, "dz[{}] = {} != 0", i, dz[i]);
            assert_eq!(dy[i], 0.0, "dy[{}] = {} != 0", i, dy[i]);
            assert_eq!(dx[i], 0.0, "dx[{}] = {} != 0", i, dx[i]);
        }
    }

    #[test]
    fn epdiff_adjoint_zero_momentum_is_zero() {
        let dims = [4, 4, 4];
        let n = 4 * 4 * 4;
        // Non-zero velocity, zero momentum.
        let v: Vec<f32> = (0..n).map(|i| 0.01 * i as f32).collect();
        let zeros = vec![0.0_f32; n];
        let (az, ay, ax) = epdiff_adjoint(&v, &v, &v, &zeros, &zeros, &zeros, dims, [1.0; 3]);

        for i in 0..n {
            assert_eq!(az[i], 0.0, "ad_z[{}] = {} != 0", i, az[i]);
            assert_eq!(ay[i], 0.0, "ad_y[{}] = {} != 0", i, ay[i]);
            assert_eq!(ax[i], 0.0, "ad_x[{}] = {} != 0", i, ax[i]);
        }
    }
}
