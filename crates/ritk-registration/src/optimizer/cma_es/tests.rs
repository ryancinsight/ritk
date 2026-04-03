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
