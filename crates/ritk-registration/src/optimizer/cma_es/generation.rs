//! Per-generation CMA-ES update step.
//!
//! Owns the mutable state that evolves across generations and the function that
//! advances it by exactly one generation.  Separating this from the outer loop
//! control in `mod.rs` keeps each file focused and below the 500-line limit.

use super::constants::AdaptationConstants;
use super::math::{chol_mul, chol_solve_lower, cholesky, identity, vec_norm};
use super::state::{CmaEsConfig, CmaEsStopReason, HistoryPolicy, PopulationEval};

/// Mutable state that evolves across CMA-ES generations.
///
/// All buffers are pre-allocated once in [`GenerationState::new`] and reused
/// across every generation to satisfy the zero inner-loop allocation invariant.
pub(super) struct GenerationState {
    /// Current distribution mean m.
    pub mean: Vec<f64>,
    /// Global step-size σ.
    pub sigma: f64,
    /// Packed lower-triangular Cholesky factor A where C = A·Aᵀ.
    /// Index formula: `chol[i*(i+1)/2 + j]` for j ≤ i.
    pub chol: Vec<f64>,
    /// Full n×n covariance matrix C (row-major flat storage).
    pub cov: Vec<f64>,
    /// Step-size cumulation path p_σ (length n).
    pub p_sigma: Vec<f64>,
    /// Covariance cumulation path p_c (length n).
    pub p_c: Vec<f64>,
    /// Isotropic samples buffer of length λ·n (row k occupies `[k*n .. (k+1)*n]`).
    pub zs: Vec<f64>,
    /// Transformed offspring buffer of length λ·n.
    pub xs: Vec<f64>,
    /// Per-candidate `(f_value, candidate_index)` sorted ascending by f.
    pub fvals: Vec<(f64, usize)>,
    /// Best parameter vector seen across all generations.
    pub best_x: Vec<f64>,
    /// Lowest observed function value f(best_x).
    pub best_f: f64,
    /// LCG state advancing the Box-Muller sampler.
    pub rng: u64,
    /// Cholesky-diagonal condition estimate (max dᵢ / min dᵢ)².
    pub condition_estimate: f64,
    /// Per-generation best-f trace; `None` when `config.record_history` is false.
    pub best_history: Option<Vec<f64>>,
}

impl GenerationState {
    /// Initialise all buffers from `x0`, performing one objective evaluation
    /// to establish the initial `best_f`.
    pub(super) fn new<F: Fn(&[f64]) -> f64>(
        x0: &[f64],
        f: &F,
        constants: &AdaptationConstants,
        config: &CmaEsConfig,
    ) -> Self {
        let n = x0.len();
        let lambda = constants.lambda;

        // Packed lower-triangular Cholesky factor initialised to I (identity)
        let mut chol = vec![0.0_f64; n * (n + 1) / 2];
        for i in 0..n {
            chol[i * (i + 1) / 2 + i] = 1.0;
        }

        let best_x = x0.to_vec();
        let best_f = f(&best_x);

        Self {
            mean: x0.to_vec(),
            sigma: config.sigma0,
            chol,
            cov: identity(n),
            p_sigma: vec![0.0_f64; n],
            p_c: vec![0.0_f64; n],
            zs: vec![0.0_f64; lambda * n],
            xs: vec![0.0_f64; lambda * n],
            fvals: vec![(0.0_f64, 0_usize); lambda],
            best_x,
            best_f,
            rng: config.seed,
            condition_estimate: 1.0_f64,
            best_history: matches!(config.record_history, HistoryPolicy::Record).then(Vec::new),
        }
    }
}

/// Advance one CMA-ES generation in place.
///
/// Performs one complete per-generation update (Hansen 2016, Algorithm 1):
/// 1. Sample λ offspring xₖ = m + σ·A·zₖ  (LCG Box-Muller + Cholesky multiply)
/// 2. Evaluate and rank candidates
/// 3. Weighted recombination → new mean m
/// 4. Step-size path p_σ update and σ adaptation
/// 5. Heaviside h_σ indicator + covariance path p_c update
/// 6. Rank-1 + rank-μ covariance matrix update
/// 7. Cholesky recompute with diagonal regularisation fallback
/// 8. Condition number check
///
/// Returns `Some(StopReason)` when a stopping criterion is satisfied,
/// `None` to continue.  `state.mean` is committed to the new mean only when
/// `None` is returned; stopping conditions short-circuit before that step.
///
/// # Arguments
/// * `state`     — mutable per-generation state (all buffers live here)
/// * `n`         — problem dimension
/// * `constants` — immutable adaptation constants (Hansen 2016, Table 1)
/// * `config`    — run configuration
/// * `f`         — objective function (`Sync` for parallel evaluation)
/// * `gen`       — zero-based generation index used in the h_σ correction term
pub(super) fn run_one_generation<F>(
    state: &mut GenerationState,
    n: usize,
    constants: &AdaptationConstants,
    config: &CmaEsConfig,
    f: &F,
    gen: usize,
) -> Option<CmaEsStopReason>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let lambda = constants.lambda;
    let mu = constants.mu;
    let mu_eff = constants.mu_eff;
    let w = &constants.w;
    let c_sigma = constants.c_sigma;
    let d_sigma = constants.d_sigma;
    let c_c = constants.c_c;
    let c1 = constants.c1;
    let c_mu = constants.c_mu;
    let chi_n = constants.chi_n;
    let n_f = n as f64;

    // 1. Sample λ offspring: xₖ = m + σ·A·zₖ
    //    Use a single advancing LCG + Box-Muller for uncorrelated draws.
    for k in 0..lambda {
        let mut d = 0;
        while d < n {
            state.rng = state
                .rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u1 = (state.rng >> 11) as f64 / (1u64 << 53) as f64 + 1e-30;
            state.rng = state
                .rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u2 = (state.rng >> 11) as f64 / (1u64 << 53) as f64;
            let mag = (-2.0 * u1.ln()).sqrt();
            let angle = 2.0 * std::f64::consts::PI * u2;
            state.zs[k * n + d] = mag * angle.cos();
            d += 1;
            if d < n {
                state.zs[k * n + d] = mag * angle.sin();
                d += 1;
            }
        }
    }

    for k in 0..lambda {
        // Borrow zs and chol immutably, then write to xs — three disjoint fields.
        let az = chol_mul(&state.chol, &state.zs[k * n..(k + 1) * n], n);
        for ((xi, mi), azi) in state.xs[k * n..(k + 1) * n]
            .iter_mut()
            .zip(state.mean.iter())
            .zip(az.iter())
        {
            *xi = mi + state.sigma * azi;
        }
    }

    // 2. Evaluate and rank
    if config.parallel_population == PopulationEval::Parallel {
        // Parallel evaluation through Moirai — each candidate
        // evaluation is independent (read-only access to the objective
        // closure's captured state).  The objective must be Sync.
        //
        // Bind xs to a local reference so the borrow of state.xs and the
        // mutable borrow of state.fvals are visibly disjoint.
        let xs = &state.xs;
        moirai::enumerate_mut_with::<moirai::Parallel, _, _>(&mut state.fvals, |k, entry| {
            let x_slice = &xs[k * n..(k + 1) * n];
            *entry = (f(x_slice), k);
        });
    } else {
        for k in 0..lambda {
            let x_slice = &state.xs[k * n..(k + 1) * n];
            state.fvals[k] = (f(x_slice), k);
        }
    }
    state
        .fvals
        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Update global best
    if state.fvals[0].0 < state.best_f {
        state.best_f = state.fvals[0].0;
        let best_k = state.fvals[0].1;
        state
            .best_x
            .copy_from_slice(&state.xs[best_k * n..(best_k + 1) * n]);
    }
    if state.best_f < config.ftol {
        return Some(CmaEsStopReason::FunctionTolerance);
    }

    // 3. Weighted recombination: m_new = Σ wᵢ · x_{i:λ}
    let mut m_new = vec![0.0_f64; n];
    for (rank, &(_, k)) in state.fvals.iter().take(mu).enumerate() {
        let x_slice = &state.xs[k * n..(k + 1) * n];
        for j in 0..n {
            m_new[j] += w[rank] * x_slice[j];
        }
    }

    // Step direction (before σ update)
    let step: Vec<f64> = (0..n)
        .map(|i| (m_new[i] - state.mean[i]) / state.sigma)
        .collect();

    // 4. Step-size path p_σ ← (1-c_σ)p_σ + √(c_σ(2-c_σ)μ_eff) · C^{-½} · step
    //    C^{-½} · v = A^{-1}·v  (forward substitution on lower-triangular A)
    //    since C = A·Aᵀ implies C^{½} = A, C^{-½} = A^{-1}.
    let cov_half_inv_step = chol_solve_lower(&state.chol, &step, n);
    let sqrt_term = (c_sigma * (2.0 - c_sigma) * mu_eff).sqrt();
    for (ps, &inv_step) in state.p_sigma.iter_mut().zip(cov_half_inv_step.iter()) {
        *ps = (1.0 - c_sigma) * *ps + sqrt_term * inv_step;
    }

    // Save σ_old before step-size update (rank-μ yᵢ must use σ_old, Hansen 2016 eq. 31)
    let sigma_old = state.sigma;

    // Step-size update: σ ← σ · exp(c_σ/d_σ · (‖p_σ‖/χ_n − 1))
    let ps_norm = vec_norm(&state.p_sigma);
    state.sigma *= ((c_sigma / d_sigma) * (ps_norm / chi_n - 1.0)).exp();

    // 5. Heaviside h_σ: suppress p_c update when σ evolution is stuck
    let threshold = (1.4 + 2.0 / (n_f + 1.0)) * chi_n;
    let h_sigma =
        if ps_norm / (1.0_f64 - (1.0 - c_sigma).powi(2 * (gen + 1) as i32)).sqrt() < threshold {
            1.0_f64
        } else {
            0.0_f64
        };

    // 6. Covariance path
    let sqrt_cc = (c_c * (2.0 - c_c) * mu_eff).sqrt();
    for (pc, &si) in state.p_c.iter_mut().zip(step.iter()) {
        *pc = (1.0 - c_c) * *pc + h_sigma * sqrt_cc * si;
    }

    // 7. Rank-1 + rank-μ covariance update
    //    C ← (1−c₁−c_μ)C + c₁·p_c·p_cᵀ + c_μ·Σwᵢ·yᵢ·yᵢᵀ
    let delta_h = (1.0 - h_sigma) * c_c * (2.0 - c_c); // heaviside correction
    for i in 0..n {
        for j in 0..=i {
            let idx_ij = i * n + j;
            let idx_ji = j * n + i;

            // Rank-1 term
            let rank1 = c1 * (state.p_c[i] * state.p_c[j] + delta_h * state.cov[idx_ij]);

            // Rank-μ term: yᵢ = (x_{i:λ} - m_old) / σ_old  (Hansen 2016, eq. 31)
            let mut rank_mu = 0.0;
            for (rank, &(_, k)) in state.fvals.iter().take(mu).enumerate() {
                let x_slice = &state.xs[k * n..(k + 1) * n];
                let yi = (x_slice[i] - state.mean[i]) / sigma_old;
                let yj = (x_slice[j] - state.mean[j]) / sigma_old;
                rank_mu += w[rank] * yi * yj;
            }

            let val = (1.0 - c1 - c_mu) * state.cov[idx_ij] + rank1 + c_mu * rank_mu;
            state.cov[idx_ij] = val;
            state.cov[idx_ji] = val; // symmetry
        }
    }

    // Enforce symmetry (numerical round-off guard)
    for i in 0..n {
        for j in 0..i {
            let idx_ij = i * n + j;
            let idx_ji = j * n + i;
            let avg = (state.cov[idx_ij] + state.cov[idx_ji]) * 0.5;
            state.cov[idx_ij] = avg;
            state.cov[idx_ji] = avg;
        }
    }

    // Re-compute Cholesky; if near-singular add tiny diagonal regularisation.
    let new_chol = cholesky(&state.cov, n).or_else(|| {
        let eps = 1e-10 * (0..n).map(|i| state.cov[i * n + i]).fold(0.0_f64, f64::max);
        let mut cov_reg = state.cov.clone();
        for i in 0..n {
            cov_reg[i * n + i] += eps.max(1e-20);
        }
        cholesky(&cov_reg, n)
    });
    if let Some(nc) = new_chol {
        state.chol = nc;
    }

    // Condition number check via Cholesky diagonal (d_i = √λᵢ for eigenvalue proxy).
    // cond(C) ≈ (max dᵢ / min dᵢ)² where dᵢ = chol_ii.
    let mut chol_diag_max = f64::MIN;
    let mut chol_diag_min = f64::MAX;
    for i in 0..n {
        let d = state.chol[i * (i + 1) / 2 + i];
        chol_diag_max = chol_diag_max.max(d);
        chol_diag_min = chol_diag_min.min(d);
    }
    if chol_diag_min > 0.0 {
        state.condition_estimate = (chol_diag_max / chol_diag_min).powi(2);
    }
    if state.condition_estimate > 1e14 {
        return Some(CmaEsStopReason::ConditionTooLarge);
    }

    state.mean = m_new;
    if let Some(history) = &mut state.best_history {
        history.push(state.best_f);
    }

    None
}
