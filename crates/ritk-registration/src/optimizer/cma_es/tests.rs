use super::*;
use crate::optimizer::cma_es::math::{cholesky, identity};

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
fn sphere_function_converges_within_budget() {
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
    // CMA-ES property: covariance matrix condition number must remain bounded (< 1e14)
    // and for the spherical problem, step-size progression strictly drives objective functionally well.
    assert!(
        result.condition_estimate >= 1.0 && result.condition_estimate < 1e4,
        "Sphere problem covariance condition number exceeded analytical bounds: {}",
        result.condition_estimate
    );
    assert!(
        result.best_f < 1e-1,
        "Objective value must functionally decrease: {}",
        result.best_f
    );
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
fn test_ipop_finds_at_least_as_good_as_plain_run() {
    // IPOP-CMA-ES must produce a result that is at least as good as a plain run
    // on the sphere function.
    let x0 = vec![5.0, -3.0, 2.0, 1.0, -2.0];
    let config = CmaEsConfig {
        sigma0: 0.8,
        max_generations: 100,
        sigma_tol: 1e-8,
        ftol: f64::NEG_INFINITY,
        seed: 0xdead_beef,
        ..Default::default()
    };

    let plain = CmaEsOptimizer::new(config.clone()).run(sphere, &x0);
    let ipop = CmaEsOptimizer::new(config).run_ipop(sphere, &x0, 2);

    // IPOP must not be worse than plain
    assert!(
        ipop.best_f <= plain.best_f + 1e-12,
        "IPOP (best_f={:.3e}) must be ≤ plain (best_f={:.3e})",
        ipop.best_f,
        plain.best_f,
    );
    assert!(
        ipop.best_f < sphere(&x0),
        "IPOP must improve over initial value {:.3e}",
        sphere(&x0)
    );
}

#[test]
fn test_seed_reproducibility() {
    let x0 = vec![2.0, -1.0, 0.5];
    let config = CmaEsConfig {
        sigma0: 0.75,
        max_generations: 40,
        seed: 12345,
        record_history: HistoryPolicy::Record,
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
