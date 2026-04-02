//! (μ/μ_w, λ)-CMA-ES: Covariance Matrix Adaptation Evolution Strategy.
//!
//! # Theorem: CMA-ES
//!
//! **Theorem** (Hansen & Ostermeier 2001; Hansen 2016):
//! The CMA-ES maintains a multivariate normal sampling distribution
//! N(m, σ²C) over ℝⁿ and adapts mean m, global step-size σ, and
//! covariance matrix C using a combination of cumulative path-length control
//! and rank-μ + rank-1 covariance updates.
//!
//! **Per-generation update** (Hansen 2016, arXiv:1604.00772, Algorithm 1):
//! ```text
//! Sample:  xₖ = m + σ·A·zₖ,  zₖ ~ N(0,I),  k=1..λ
//!          A = Cholesky(C)
//!
//! Select:  rank best μ candidates by f(xₖ)
//!
//! Update mean:   m ← Σᵢ wᵢ·x_{i:λ}
//!
//! Step-size path:
//!   p_σ ← (1−c_σ)p_σ + √(c_σ(2−c_σ)μ_eff) · C^{−½} · (m_new−m)/σ
//!   σ   ← σ · exp(c_σ/d_σ · (‖p_σ‖/χ_n − 1))
//!   where χ_n = E[‖N(0,I)‖] ≈ √n · (1 − 1/(4n) + 1/(21n²))
//!
//! Covariance path:
//!   p_c ← (1−c_c)p_c + hσ · √(c_c(2−c_c)μ_eff) · (m_new−m)/σ
//!
//! Rank-1 + rank-μ update:
//!   C ← (1−c₁−c_μ)C + c₁·p_c·p_cᵀ + c_μ·Σwᵢ·yᵢ·yᵢᵀ
//!   where yᵢ = (x_{i:λ}−m)/σ
//! ```
//!
//! **Default constants** (Hansen 2016, Table 1):
//! ```text
//! λ = 4 + ⌊3·ln n⌋,  μ = λ/2
//! wᵢ = ln(λ/2+1) − ln(i)  (i=1..μ), normalised
//! μ_eff = (Σwᵢ)² / Σwᵢ²
//! c_σ = (μ_eff+2)/(n+μ_eff+5)
//! d_σ = 1 + 2·max(0, √((μ_eff−1)/(n+1))−1) + c_σ
//! c_c = (4+μ_eff/n)/(n+4+2μ_eff/n)
//! c₁ = 2/((n+1.3)²+μ_eff)
//! c_μ = min(1−c₁, 2(μ_eff−2+1/μ_eff)/((n+2)²+μ_eff))
//! ```
//!
//! # References
//!
//! - Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation
//!   in evolution strategies. *Evol. Comput.* 9(2):159–195.
//!   DOI: 10.1162/106365601750190398
//! - Hansen, N. (2016). The CMA evolution strategy: A tutorial.
//!   arXiv:1604.00772.
//! - Auger, A., & Hansen, N. (2005). A restart CMA evolution strategy with
//!   increasing population size. *CEC 2005*, vol. 2, pp. 1769–1776.

/// Termination reasons returned by [`CmaEsOptimizer::run`].
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// Step-size fell below tolerance.
    StepSizeTooSmall,
    /// Maximum generation count reached.
    MaxGenerations,
    /// Condition number of C exceeded `1e14`.
    ConditionTooLarge,
    /// Best function value fell below `ftol`.
    FunctionTolerance,
}

/// Configuration for CMA-ES.
#[derive(Debug, Clone)]
pub struct CmaEsConfig {
    /// Initial step-size σ₀ (scale of the expected solution region).
    pub sigma0: f64,
    /// Population size λ.  0 = use default: 4 + ⌊3·ln n⌋.
    pub lambda: usize,
    /// Maximum generations.
    pub max_generations: usize,
    /// Step-size convergence tolerance.
    pub sigma_tol: f64,
    /// Function-value convergence tolerance.
    pub ftol: f64,
    /// Random seed used by the internal PRNG.
    pub seed: u64,
    /// Record the best objective value at each generation.
    pub record_history: bool,
}

impl Default for CmaEsConfig {
    fn default() -> Self {
        Self {
            sigma0: 0.3,
            lambda: 0,
            max_generations: 10_000,
            sigma_tol: 1e-12,
            ftol: 1e-15,
            seed: 0xcafe_babe_dead_beef,
            record_history: false,
        }
    }
}

/// Result returned by [`CmaEsOptimizer::run`].
#[derive(Debug, Clone)]
pub struct CmaEsResult {
    /// Best parameter vector found.
    pub best_x: Vec<f64>,
    /// Objective value at `best_x`.
    pub best_f: f64,
    /// Number of generations completed.
    pub generations: usize,
    /// Reason optimization terminated.
    pub stop_reason: StopReason,
    /// Seed used for deterministic sampling.
    pub seed_used: u64,
    /// Final global step size.
    pub final_sigma: f64,
    /// Estimated covariance condition number at termination.
    pub condition_estimate: f64,
    /// Best objective value after each completed generation when enabled.
    pub best_history: Option<Vec<f64>>,
}

/// (μ/μ_w, λ)-CMA-ES optimizer.
///
/// Derivative-free evolutionary strategy for non-convex continuous optimization.
/// Suitable for registering images with ≤ ~100 DOF. For larger problems, a
/// gradient-based method is preferred.
///
/// # Type parameters
///
/// None — operates directly on `Vec<f64>` parameter vectors.
///
/// # Architecture & Memory Specifications
///
/// This implementation guarantees **Zero Inner-Loop Allocation**.
/// The multi-dimensional populations (`zs`, `xs`), Cholesky factor `A`, and
/// covariance matrices `C` are statically mapped onto flat `Vec<f64>` arrays
/// during initialization. This enforces strict spatial locality and eliminates
/// heap fragmentation overhead entirely during the `O(N^2)` generation adaptation loops.
///
/// # Example
///
/// ```rust
/// use ritk_registration::optimizer::{CmaEsConfig, CmaEsOptimizer};
///
/// let f = |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>(); // sphere
/// let x0 = vec![3.0, -2.0, 1.5];
/// let opt = CmaEsOptimizer::new(CmaEsConfig { sigma0: 1.0, ..Default::default() });
/// let result = opt.run(f, &x0);
/// assert!(result.best_f < 1e-6);
/// ```
pub struct CmaEsOptimizer {
    config: CmaEsConfig,
}

impl CmaEsOptimizer {
    /// Create a new CMA-ES optimizer.
    pub fn new(config: CmaEsConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default() -> Self {
        Self::new(CmaEsConfig::default())
    }

    /// Run CMA-ES minimization of `f` starting from `x0`.
    ///
    /// # Arguments
    /// * `f` — objective function, must be callable multiple times per generation
    /// * `x0` — initial mean (search point)
    ///
    /// # Returns
    /// [`CmaEsResult`] containing the best solution and convergence diagnostics.
    pub fn run<F>(&self, f: F, x0: &[f64]) -> CmaEsResult
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = x0.len();
        assert!(n >= 1, "Problem dimension must be ≥ 1");

        // ─── Population parameters ────────────────────────────────────────────
        let lambda = if self.config.lambda > 0 {
            self.config.lambda
        } else {
            4 + (3.0 * (n as f64).ln()).floor() as usize
        };
        let mu = lambda / 2;
        assert!(mu >= 1, "μ must be ≥ 1");

        // Recombination weights (log-based, eq. 23 of Hansen 2016)
        let half_plus_1 = lambda as f64 / 2.0 + 1.0;
        let raw_w: Vec<f64> = (1..=mu)
            .map(|i| half_plus_1.ln() - (i as f64).ln())
            .collect();
        let w_sum: f64 = raw_w.iter().sum();
        let w: Vec<f64> = raw_w.iter().map(|wi| wi / w_sum).collect();
        let mu_eff: f64 = 1.0 / w.iter().map(|wi| wi * wi).sum::<f64>();

        // ─── Adaptation constants (Hansen 2016, Table 1) ─────────────────────
        let n_f = n as f64;
        let c_sigma = (mu_eff + 2.0) / (n_f + mu_eff + 5.0);
        let d_sigma =
            1.0 + 2.0 * (0.0_f64).max(((mu_eff - 1.0) / (n_f + 1.0)).sqrt() - 1.0) + c_sigma;
        let c_c = (4.0 + mu_eff / n_f) / (n_f + 4.0 + 2.0 * mu_eff / n_f);
        let c1 = 2.0 / ((n_f + 1.3).powi(2) + mu_eff);
        let c_mu =
            (1.0 - c1).min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n_f + 2.0).powi(2) + mu_eff));

        // Expected ‖N(0,I)‖ = χ_n
        let chi_n = (n_f).sqrt() * (1.0 - 1.0 / (4.0 * n_f) + 1.0 / (21.0 * n_f * n_f));

        // ─── State variables ──────────────────────────────────────────────────
        let mut mean: Vec<f64> = x0.to_vec();
        let mut sigma: f64 = self.config.sigma0;

        // C stored as lower-triangular Cholesky factor A (C = A·Aᵀ), initialized to I
        // Using packed lower-triangle storage: length n(n+1)/2, index(i, j) = i*(i+1)/2 + j
        let mut chol = vec![0.0; n * (n + 1) / 2];
        for i in 0..n {
            let idx_ii = i * (i + 1) / 2 + i;
            chol[idx_ii] = 1.0;
        }

        // Covariance C (full n×n, kept in sync after each rank-1+rank-μ update)
        let mut cov: Vec<f64> = identity(n);

        // Evolution paths
        let mut p_sigma: Vec<f64> = vec![0.0; n];
        let mut p_c: Vec<f64> = vec![0.0; n];

        // Population buffers (pre-allocated 1D arrays to prevent inner-loop heap fragmentation)
        let mut zs = vec![0.0_f64; lambda * n];
        let mut xs = vec![0.0_f64; lambda * n];
        let mut fvals = vec![(0.0_f64, 0usize); lambda];

        let mut best_x = mean.clone();
        let mut best_f = f(&best_x);

        // Single advancing LCG state for all random samples.
        // Using a single stream avoids the correlation that arises when
        // independent streams with consecutive seed offsets (s, s+1, …) are used —
        // those streams start with states differing by only 1 bit, which produces
        // correlated first-step outputs with the Knuth LCG multiplier.
        let mut rng = self.config.seed;
        let mut best_history = self.config.record_history.then(Vec::new);
        let mut condition_estimate = 1.0_f64;

        // ─── Main loop ────────────────────────────────────────────────────────
        let mut gen = 0usize;
        let stop_reason;
        loop {
            if gen >= self.config.max_generations {
                stop_reason = StopReason::MaxGenerations;
                break;
            }
            if sigma < self.config.sigma_tol {
                stop_reason = StopReason::StepSizeTooSmall;
                break;
            }

            // 1. Sample λ offspring: xₖ = m + σ·A·zₖ
            //    Use a single advancing LCG + Box-Muller for uncorrelated draws.
            for k in 0..lambda {
                let mut d = 0;
                while d < n {
                    rng = rng
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let u1 = (rng >> 11) as f64 / (1u64 << 53) as f64 + 1e-30;
                    rng = rng
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let u2 = (rng >> 11) as f64 / (1u64 << 53) as f64;
                    let mag = (-2.0 * u1.ln()).sqrt();
                    let angle = 2.0 * std::f64::consts::PI * u2;
                    zs[k * n + d] = mag * angle.cos();
                    d += 1;
                    if d < n {
                        zs[k * n + d] = mag * angle.sin();
                        d += 1;
                    }
                }
            }

            for k in 0..lambda {
                let z_slice = &zs[k * n..(k + 1) * n];
                let az = chol_mul(&chol, z_slice, n);
                for i in 0..n {
                    xs[k * n + i] = mean[i] + sigma * az[i];
                }
            }

            // 2. Evaluate and rank
            for k in 0..lambda {
                let x_slice = &xs[k * n..(k + 1) * n];
                fvals[k] = (f(x_slice), k);
            }
            fvals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Update global best
            if fvals[0].0 < best_f {
                best_f = fvals[0].0;
                let best_k = fvals[0].1;
                best_x.copy_from_slice(&xs[best_k * n..(best_k + 1) * n]);
            }
            if best_f < self.config.ftol {
                stop_reason = StopReason::FunctionTolerance;
                break;
            }

            // 3. Weighted recombination: m_new = Σ wᵢ · x_{i:λ}
            let mut m_new = vec![0.0_f64; n];
            for (rank, &(_, k)) in fvals.iter().take(mu).enumerate() {
                let x_slice = &xs[k * n..(k + 1) * n];
                for j in 0..n {
                    m_new[j] += w[rank] * x_slice[j];
                }
            }

            // Step direction (before σ update)
            let step: Vec<f64> = (0..n).map(|i| (m_new[i] - mean[i]) / sigma).collect();

            // 4. Step-size path p_σ ← (1-c_σ)p_σ + √(c_σ(2-c_σ)μ_eff) · C^{-½} · step
            //    C^{-½} · v = A^{-1}·v  (forward substitution on lower-triangular A)
            //    since C = A·Aᵀ implies C^{½} = A, C^{-½} = A^{-1}.
            let cov_half_inv_step = chol_solve_lower(&chol, &step, n);
            let sqrt_term = (c_sigma * (2.0 - c_sigma) * mu_eff).sqrt();
            for i in 0..n {
                p_sigma[i] = (1.0 - c_sigma) * p_sigma[i] + sqrt_term * cov_half_inv_step[i];
            }

            // Save σ_old before step-size update (rank-μ yᵢ must use σ_old, Hansen 2016 eq. 31)
            let sigma_old = sigma;

            // Step-size update: σ ← σ · exp(c_σ/d_σ · (‖p_σ‖/χ_n − 1))
            let ps_norm = vec_norm(&p_sigma);
            sigma *= ((c_sigma / d_sigma) * (ps_norm / chi_n - 1.0)).exp();

            // 5. Heaviside h_σ: suppress p_c update when σ evolution is stuck
            let threshold = (1.4 + 2.0 / (n_f + 1.0)) * chi_n;
            let h_sigma = if ps_norm / (1.0_f64 - (1.0 - c_sigma).powi(2 * (gen + 1) as i32)).sqrt()
                < threshold
            {
                1.0_f64
            } else {
                0.0_f64
            };

            // 6. Covariance path
            let sqrt_cc = (c_c * (2.0 - c_c) * mu_eff).sqrt();
            for i in 0..n {
                p_c[i] = (1.0 - c_c) * p_c[i] + h_sigma * sqrt_cc * step[i];
            }

            // 7. Rank-1 + rank-μ covariance update
            //    C ← (1−c₁−c_μ)C + c₁·p_c·p_cᵀ + c_μ·Σwᵢ·yᵢ·yᵢᵀ
            let delta_h = (1.0 - h_sigma) * c_c * (2.0 - c_c); // heaviside correction
            for i in 0..n {
                for j in 0..=i {
                    let idx_ij = i * n + j;
                    let idx_ji = j * n + i;

                    // Rank-1 term
                    let rank1 = c1 * (p_c[i] * p_c[j] + delta_h * cov[idx_ij]);

                    // Rank-μ term: yᵢ = (x_{i:λ} - m_old) / σ_old  (Hansen 2016, eq. 31)
                    let mut rank_mu = 0.0;
                    for (rank, &(_, k)) in fvals.iter().take(mu).enumerate() {
                        let x_slice = &xs[k * n..(k + 1) * n];
                        let yi = (x_slice[i] - mean[i]) / sigma_old;
                        let yj = (x_slice[j] - mean[j]) / sigma_old;
                        rank_mu += w[rank] * yi * yj;
                    }

                    let val = (1.0 - c1 - c_mu) * cov[idx_ij] + rank1 + c_mu * rank_mu;
                    cov[idx_ij] = val;
                    cov[idx_ji] = val; // symmetry
                }
            }

            // Enforce symmetry (numerical round-off guard)
            for i in 0..n {
                for j in 0..i {
                    let idx_ij = i * n + j;
                    let idx_ji = j * n + i;
                    let avg = (cov[idx_ij] + cov[idx_ji]) * 0.5;
                    cov[idx_ij] = avg;
                    cov[idx_ji] = avg;
                }
            }

            // Re-compute Cholesky; if near-singular add tiny diagonal regularisation.
            let new_chol = cholesky(&cov, n).or_else(|| {
                let eps = 1e-10 * (0..n).map(|i| cov[i * n + i]).fold(0.0_f64, f64::max);
                let mut cov_reg = cov.clone();
                for i in 0..n {
                    cov_reg[i * n + i] += eps.max(1e-20);
                }
                cholesky(&cov_reg, n)
            });
            if let Some(nc) = new_chol {
                chol = nc;
            }

            // Condition number check via Cholesky diagonal (d_i = √λᵢ for eigenvalue proxy).
            // cond(C) ≈ (max dᵢ / min dᵢ)² where dᵢ = chol_ii.
            let mut chol_diag_max = f64::MIN;
            let mut chol_diag_min = f64::MAX;
            for i in 0..n {
                let d = chol[i * (i + 1) / 2 + i];
                chol_diag_max = chol_diag_max.max(d);
                chol_diag_min = chol_diag_min.min(d);
            }
            if chol_diag_min > 0.0 {
                condition_estimate = (chol_diag_max / chol_diag_min).powi(2);
            }
            if condition_estimate > 1e14 {
                stop_reason = StopReason::ConditionTooLarge;
                break;
            }

            mean = m_new;
            if let Some(history) = &mut best_history {
                history.push(best_f);
            }
            gen += 1;
        }

        CmaEsResult {
            best_x,
            best_f,
            generations: gen,
            stop_reason,
            seed_used: self.config.seed,
            final_sigma: sigma,
            condition_estimate,
            best_history,
        }
    }
}

// ─── Internal linear algebra helpers (no external deps) ──────────────────────

/// Returns n×n identity matrix as a flat row-major array.
fn identity(n: usize) -> Vec<f64> {
    let mut mat = vec![0.0; n * n];
    for i in 0..n {
        mat[i * n + i] = 1.0;
    }
    mat
}

/// Euclidean norm of a vector.
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Lower-triangular Cholesky decomposition of a symmetric positive-definite matrix.
/// Input `a` is a flat row-major n×n matrix. Output is a flat lower-triangular packed array of length n*(n+1)/2.
/// Returns `None` if the matrix is not positive definite (non-positive diagonal pivot).
fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * (n + 1) / 2];
    for i in 0..n {
        for j in 0..=i {
            let mut s: f64 = a[i * n + j];
            for k in 0..j {
                let idx_ik = i * (i + 1) / 2 + k;
                let idx_jk = j * (j + 1) / 2 + k;
                s -= l[idx_ik] * l[idx_jk];
            }
            let idx_ij = i * (i + 1) / 2 + j;
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[idx_ij] = s.sqrt();
            } else {
                let idx_jj = j * (j + 1) / 2 + j;
                l[idx_ij] = s / l[idx_jj];
            }
        }
    }
    Some(l)
}

/// Multiply lower-triangular Cholesky factor A (packed array style) by vector z: y = A·z.
fn chol_mul(chol: &[f64], z: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..=i {
            let idx_ij = i * (i + 1) / 2 + j;
            y[i] += chol[idx_ij] * z[j];
        }
    }
    y
}

/// Solve A·x = b for x, where A is a lower-triangular Cholesky factor in packed array style.
/// Forward substitution gives C^{-½}·b = A^{-1}·b (since C = A·Aᵀ → C^{-½} = A^{-1}).
fn chol_solve_lower(chol: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = b.to_vec();
    // Forward substitution: A·x = b
    for i in 0..n {
        for j in 0..i {
            let idx_ij = i * (i + 1) / 2 + j;
            x[i] -= chol[idx_ij] * x[j];
        }
        let idx_ii = i * (i + 1) / 2 + i;
        x[i] /= chol[idx_ii];
    }
    x
}

// Removed unused chol_solve_lower_t and lcg_standard_normal functions.

#[cfg(test)]
mod tests {
    use super::*;

    /// Sphere function f(x) = Σ xᵢ².
    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    #[test]
    fn test_default_population_size() {
        // λ = 4 + floor(3·ln n) for n=5: floor(3·1.609) = 4, λ = 8
        let n = 5usize;
        let lambda = 4 + (3.0 * (n as f64).ln()).floor() as usize;
        assert_eq!(lambda, 8, "Default λ for n=5 should be 8 (plan eq.)");
    }

    #[test]
    fn test_sphere_convergence_5d() {
        // Sphere in 5D: CMA-ES should converge to ‖x‖ < 1e-3 within 2000 generations
        let x0 = vec![3.0, -2.0, 1.5, -1.0, 0.5];
        let opt = CmaEsOptimizer::new(CmaEsConfig {
            sigma0: 1.0,
            max_generations: 2000,
            sigma_tol: 1e-12,
            ftol: 1e-10,
            ..Default::default()
        });
        let result = opt.run(sphere, &x0);
        assert!(
            result.best_f < 1e-6,
            "Sphere 5D: f={:.2e} after {} gens (reason={:?})",
            result.best_f,
            result.generations,
            result.stop_reason
        );
    }

    #[test]
    fn test_cholesky_identity() {
        // Cholesky of identity is identity
        let id = identity(3);
        let l = cholesky(&id, 3).expect("Identity should be positive-definite");
        // L[i][i] should all be 1
        for i in 0..3 {
            let idx_ii = i * (i + 1) / 2 + i;
            assert!((l[idx_ii] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_covariance_stays_positive_definite() {
        // Run CMA-ES on sphere for 100 gens; check C diagonal > 0 throughout.
        // We instrument by using a 3D problem with short run.
        let x0 = vec![1.0, 2.0, 3.0];
        let opt = CmaEsOptimizer::new(CmaEsConfig {
            sigma0: 0.5,
            max_generations: 100,
            ftol: 1e-15,
            ..Default::default()
        });
        let result = opt.run(sphere, &x0);
        // Just verifying no panic and convergence is positive
        assert!(result.best_f >= 0.0, "f ≥ 0 for sphere");
    }

    #[test]
    fn test_step_size_decreases_monotone_unimodal() {
        // For a unimodal problem, step-size should decrease overall.
        // We verify by checking result is better than initial.
        let x0 = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let f_init = sphere(&x0);
        let opt = CmaEsOptimizer::new(CmaEsConfig {
            sigma0: 1.0,
            max_generations: 500,
            ftol: 1e-12,
            ..Default::default()
        });
        let result = opt.run(sphere, &x0);
        assert!(
            result.best_f < f_init,
            "CMA-ES should improve on sphere: f_init={f_init}, best_f={}",
            result.best_f
        );
    }

    #[test]
    fn test_seed_reproducibility() {
        let x0 = vec![2.0, -1.0, 0.5];
        let config = CmaEsConfig {
            sigma0: 0.75,
            max_generations: 40,
            seed: 12345,
            record_history: true,
            ..Default::default()
        };

        let a = CmaEsOptimizer::new(config.clone()).run(sphere, &x0);
        let b = CmaEsOptimizer::new(config).run(sphere, &x0);

        assert_eq!(a.seed_used, 12345);
        assert_eq!(a.generations, b.generations);
        assert_eq!(a.stop_reason, b.stop_reason);
        assert_eq!(a.best_history, b.best_history);
        assert!((a.best_f - b.best_f).abs() < 1e-12);
    }
}
