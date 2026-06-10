//! Adaptation constants for the (μ/μ_w, λ)-CMA-ES algorithm.
//!
//! All values are derived from Hansen (2016) Table 1 and are pure functions
//! of the problem dimension `n` and the population configuration.

use super::state::CmaEsConfig;

/// Precomputed adaptation constants for a single CMA-ES run.
///
/// All fields are immutable after construction and are derived solely from
/// the problem dimension `n` and the run configuration. Extracting them
/// avoids re-deriving the same values in every access pattern.
#[derive(Debug, Clone)]
pub(super) struct AdaptationConstants {
    /// Population size λ (number of offspring per generation).
    pub lambda: usize,
    /// Parent population size μ = λ/2.
    pub mu: usize,
    /// Effective variance selection mass μ_eff = 1 / Σ wᵢ².
    pub mu_eff: f64,
    /// Normalized recombination weights w₁, …, w_μ summing to 1.
    pub w: Vec<f64>,
    /// Step-size cumulation rate c_σ.
    pub c_sigma: f64,
    /// Damping for step-size adaptation d_σ.
    pub d_sigma: f64,
    /// Rank-1 cumulation rate c_c.
    pub c_c: f64,
    /// Rank-1 update learning rate c₁.
    pub c1: f64,
    /// Rank-μ update learning rate c_μ.
    pub c_mu: f64,
    /// Expected ‖N(0,I)‖ ≈ χ_n.
    pub chi_n: f64,
}

impl AdaptationConstants {
    /// Derive adaptation constants from problem dimension and run config.
    ///
    /// Implements Hansen (2016) Table 1 defaults, overriding λ when
    /// `config.lambda > 0`.
    pub(super) fn new(n: usize, config: &CmaEsConfig) -> Self {
        let n_f = n as f64;
        let lambda = if config.lambda > 0 {
            config.lambda
        } else {
            4 + (3.0 * n_f.ln()).floor() as usize
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

        // Adaptation constants (Hansen 2016, Table 1)
        let c_sigma = (mu_eff + 2.0) / (n_f + mu_eff + 5.0);
        let d_sigma =
            1.0 + 2.0 * (0.0_f64).max(((mu_eff - 1.0) / (n_f + 1.0)).sqrt() - 1.0) + c_sigma;
        let c_c = (4.0 + mu_eff / n_f) / (n_f + 4.0 + 2.0 * mu_eff / n_f);
        let c1 = 2.0 / ((n_f + 1.3).powi(2) + mu_eff);
        let c_mu =
            (1.0 - c1).min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n_f + 2.0).powi(2) + mu_eff));
        let chi_n = n_f.sqrt() * (1.0 - 1.0 / (4.0 * n_f) + 1.0 / (21.0 * n_f * n_f));

        Self {
            lambda,
            mu,
            mu_eff,
            w,
            c_sigma,
            d_sigma,
            c_c,
            c1,
            c_mu,
            chi_n,
        }
    }
}
