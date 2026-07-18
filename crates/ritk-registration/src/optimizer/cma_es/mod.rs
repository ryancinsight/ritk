//! (Î¼/Î¼_w, Î»)-CMA-ES: Covariance Matrix Adaptation Evolution Strategy.
//!
//! # Theorem: CMA-ES
//!
//! **Theorem** (Hansen & Ostermeier 2001; Hansen 2016):
//! The CMA-ES maintains a multivariate normal sampling distribution
//! N(m, ÏƒÂ²C) over â„â¿ and adapts mean m, global step-size Ïƒ, and
//! covariance matrix C using a combination of cumulative path-length control
//! and rank-Î¼ + rank-1 covariance updates.
//!
//! **Per-generation update** (Hansen 2016, arXiv:1604.00772, Algorithm 1):
//! ```text
//! Sample:  xâ‚– = m + ÏƒÂ·AÂ·zâ‚–,  zâ‚– ~ N(0,I),  k=1..Î»
//!          A = Cholesky(C)
//!
//! Select:  rank best Î¼ candidates by f(xâ‚–)
//!
//! Update mean:   m â† Î£áµ¢ wáµ¢Â·x_{i:Î»}
//!
//! Step-size path:
//!   p_Ïƒ â† (1âˆ’c_Ïƒ)p_Ïƒ + âˆš(c_Ïƒ(2âˆ’c_Ïƒ)Î¼_eff) Â· C^{âˆ’Â½} Â· (m_newâˆ’m)/Ïƒ
//!   Ïƒ   â† Ïƒ Â· exp(c_Ïƒ/d_Ïƒ Â· (â€–p_Ïƒâ€–/Ï‡_n âˆ’ 1))
//!   where Ï‡_n = E[â€–N(0,I)â€–] â‰ˆ âˆšn Â· (1 âˆ’ 1/(4n) + 1/(21nÂ²))
//!
//! Covariance path:
//!   p_c â† (1âˆ’c_c)p_c + hÏƒ Â· âˆš(c_c(2âˆ’c_c)Î¼_eff) Â· (m_newâˆ’m)/Ïƒ
//!
//! Rank-1 + rank-Î¼ update:
//!   C â† (1âˆ’câ‚âˆ’c_Î¼)C + câ‚Â·p_cÂ·p_cáµ€ + c_Î¼Â·Î£wáµ¢Â·yáµ¢Â·yáµ¢áµ€
//!   where yáµ¢ = (x_{i:Î»}âˆ’m)/Ïƒ
//! ```
//!
//! **Default constants** (Hansen 2016, Table 1):
//! ```text
//! Î» = 4 + âŒŠ3Â·ln nâŒ‹,  Î¼ = Î»/2
//! wáµ¢ = ln(Î»/2+1) âˆ’ ln(i)  (i=1..Î¼), normalised
//! Î¼_eff = (Î£wáµ¢)Â² / Î£wáµ¢Â²
//! c_Ïƒ = (Î¼_eff+2)/(n+Î¼_eff+5)
//! d_Ïƒ = 1 + 2Â·max(0, âˆš((Î¼_effâˆ’1)/(n+1))âˆ’1) + c_Ïƒ
//! c_c = (4+Î¼_eff/n)/(n+4+2Î¼_eff/n)
//! câ‚ = 2/((n+1.3)Â²+Î¼_eff)
//! c_Î¼ = min(1âˆ’câ‚, 2(Î¼_effâˆ’2+1/Î¼_eff)/((n+2)Â²+Î¼_eff))
//! ```
//!
//! # References
//!
//! - Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation
//!   in evolution strategies. *Evol. Comput.* 9(2):159â€“195.
//!   DOI: 10.1162/106365601750190398
//! - Hansen, N. (2016). The CMA evolution strategy: A tutorial.
//!   arXiv:1604.00772.
//! - Auger, A., & Hansen, N. (2005). A restart CMA evolution strategy with
//!   increasing population size. *CEC 2005*, vol. 2, pp. 1769â€“1776.

pub(crate) mod constants;
mod generation;
pub(crate) mod math;
pub mod state;

use constants::AdaptationConstants;
use generation::{run_one_generation, GenerationState};
pub use state::{CmaEsConfig, CmaEsResult, CmaEsStopReason, HistoryPolicy, PopulationEval};

/// (Î¼/Î¼_w, Î»)-CMA-ES optimizer.
///
/// Derivative-free evolutionary strategy for non-convex continuous optimization.
/// Suitable for registering images with â‰¤ ~100 DOF. For larger problems, a
/// gradient-based method is preferred.
///
/// # Type parameters
///
/// None â€” operates directly on `Vec<f64>` parameter vectors.
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
    config: CmaEsConfig }

impl Default for CmaEsOptimizer {
    fn default() -> Self {
        Self::new(CmaEsConfig::default())
    }
}

impl CmaEsOptimizer {
    /// Create a new CMA-ES optimizer.
    pub fn new(config: CmaEsConfig) -> Self {
        Self { config }
    }

    /// Run CMA-ES minimization of `f` starting from `x0`.
    ///
    /// # Arguments
    /// * `f` â€” objective function, must be callable multiple times per generation
    /// * `x0` â€” initial mean (search point)
    ///
    /// # Returns
    /// [`CmaEsResult`] containing the best solution and convergence diagnostics.
    pub fn run<F>(&self, f: F, x0: &[f64]) -> CmaEsResult
    where
        F: Fn(&[f64]) -> f64 + Sync,
    {
        let n = x0.len();
        assert!(n >= 1, "Problem dimension must be â‰¥ 1");

        let constants = AdaptationConstants::new(n, &self.config);
        let mut state = GenerationState::new(x0, &f, &constants, &self.config);

        // â”€â”€â”€ Main loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        let mut gen = 0usize;
        let stop_reason;
        loop {
            if gen >= self.config.max_generations {
                stop_reason = CmaEsStopReason::MaxGenerations;
                break;
            }
            if state.sigma < self.config.sigma_tol {
                stop_reason = CmaEsStopReason::StepSizeTooSmall;
                break;
            }
            if let Some(reason) =
                run_one_generation(&mut state, n, &constants, &self.config, &f, gen)
            {
                stop_reason = reason;
                break;
            }
            gen += 1;
        }

        CmaEsResult {
            best_x: state.best_x,
            best_f: state.best_f,
            generations: gen,
            stop_reason,
            seed_used: self.config.seed,
            final_sigma: state.sigma,
            condition_estimate: state.condition_estimate,
            best_history: state.best_history }
    }

    /// Run IPOP-CMA-ES: run CMA-ES multiple times with increasing population size,
    /// using **independent random starting points** for restarts to explore different
    /// basins of attraction.
    ///
    /// Unlike the classic IPOP (which restarts from the same x0), this variant draws
    /// each restart's initial mean uniformly from `[-1, 1]^n`.  This is appropriate
    /// when `x0` has been normalised so that the entire feasible region lies within
    /// that box (which is the case for the CMA-ES rigid registration pipeline).
    ///
    /// The first run uses the caller-supplied `x0` (warm start from e.g. CoM init);
    /// subsequent restarts use fresh random starting points so that each restart
    /// independently searches a different region of the landscape.
    ///
    /// # Arguments
    /// * `f` â€” objective function
    /// * `x0` â€” initial mean for the **first** run; restarts ignore this
    /// * `max_restarts` â€” maximum number of additional runs (0 = no restarts, same as `run`)
    ///
    /// # Returns
    /// The best [`CmaEsResult`] found across all runs.
    pub fn run_ipop<F>(&self, f: F, x0: &[f64], max_restarts: usize) -> CmaEsResult
    where
        F: Fn(&[f64]) -> f64 + Sync,
    {
        let n = x0.len();
        let base_lambda = if self.config.lambda > 0 {
            self.config.lambda
        } else {
            4 + (3.0 * (n as f64).ln()).floor() as usize
        };

        let mut best_result = self.run(&f, x0);
        let mut lambda = base_lambda;

        for restart in 0..max_restarts {
            // Double the population for IPOP (classic schedule)
            lambda = lambda.saturating_mul(2);

            // Vary seed per restart
            let restart_seed = self
                .config
                .seed
                .wrapping_add(restart as u64 + 1)
                .wrapping_mul(6_364_136_223_846_793_005);

            // Generate a fresh random starting point in [-1, 1]^n using a
            // deterministic LCG so that results are reproducible.
            // LCG: x_{k+1} = a * x_k + c  (mod 2^64)
            // Parameters from Knuth MMIX.
            let random_x0: Vec<f64> = {
                let mut state = restart_seed;
                (0..n)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        // Map u64 â†’ (-1, 1)
                        (state as i64 as f64) / (i64::MAX as f64)
                    })
                    .collect()
            };

            let restart_config = CmaEsConfig {
                lambda,
                seed: restart_seed,
                ..self.config.clone()
            };

            let restart_result = CmaEsOptimizer::new(restart_config).run(&f, &random_x0);

            if restart_result.best_f < best_result.best_f {
                best_result = restart_result;
            }
        }

        best_result
    }
}

#[cfg(test)]
mod tests;
