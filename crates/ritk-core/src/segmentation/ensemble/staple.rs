//! STAPLE – Simultaneous Truth and Performance Level Estimation.
//!
//! Reference: Warfield, Zou, Wells (2004) IEEE Trans. Med. Imaging 23(7):903–921.
//!
//! # Mathematical Specification
//!
//! Given K binary segmentation masks D = {D_1, ..., D_K} with D_k ∈ {0,1}^N,
//! STAPLE estimates via Expectation–Maximisation (EM):
//! - W ∈ [0,1]^N  probabilistic ground truth
//! - p ∈ (0,1)^K  per-rater sensitivity
//! - q ∈ (0,1)^K  per-rater specificity
//!
//! ## Initialisation
//!
//! ```text
//! p_k = q_k = 0.99,  f = 0.5
//! ```
//!
//! ## E-step (voxel-parallel)
//!
//! For each voxel i:
//! ```text
//! log α_i = log f      + Σ_k [ D_k[i]·log(p_k)   + (1−D_k[i])·log(1−p_k) ]
//! log β_i = log(1−f)   + Σ_k [ D_k[i]·log(1−q_k) + (1−D_k[i])·log(q_k)   ]
//! W_i = 1 / (1 + exp(log β_i − log α_i))
//! ```
//!
//! ## M-step
//!
//! ```text
//! W_sum = Σ_i W_i
//! f     = W_sum / N
//! p_k   = clamp( Σ_i(D_k[i] · W_i)          / W_sum,       ε, 1−ε )
//! q_k   = clamp( Σ_i((1−D_k[i]) · (1−W_i))  / (N−W_sum),   ε, 1−ε )
//! ```
//! where ε = 1e-10 prevents log-domain underflow at parameter boundaries.
//!
//! ## Convergence
//!
//! ```text
//! max_k( |p_k_new − p_k_old| + |q_k_new − q_k_old| ) < tol
//! ```

/// Guard against log(0): clamping boundary for p_k and q_k.
///
/// Chosen as 1e-6 rather than a smaller value so that the clamped f64 value `1 − EPS`
/// is below the rounding midpoint `1 − 2^-25 ≈ 0.9999999702` between 1.0 and the
/// largest sub-1.0 f32 representable value. This guarantees `(1 − EPS) as f32 < 1.0_f32`,
/// preserving the strict-(0,1) invariant in the f32 output without sacrificing log safety
/// (`log(1e-6) ≈ −13.8`, bounded).
const EPS: f64 = 1e-6;

/// Result of the STAPLE algorithm.
#[derive(Debug, Clone)]
pub struct StapleResult {
    /// Probabilistic ground-truth estimate W ∈ \[0,1\]^N.
    pub probabilistic_truth: Vec<f32>,
    /// Per-rater sensitivity p_k ∈ (0,1). Length = K.
    pub sensitivity: Vec<f32>,
    /// Per-rater specificity q_k ∈ (0,1). Length = K.
    pub specificity: Vec<f32>,
    /// Number of EM iterations executed.
    pub iterations: usize,
    /// True if converged before `max_iter`.
    pub converged: bool,
}

/// Numerically stable log-domain sigmoid.
///
/// Computes W = exp(log_a) / (exp(log_a) + exp(log_b)) = 1 / (1 + exp(log_b − log_a)).
/// Finite for all finite inputs; no catastrophic cancellation.
#[inline]
fn sigmoid_log_domain(log_a: f64, log_b: f64) -> f64 {
    let diff = log_b - log_a;
    1.0 / (1.0 + diff.exp())
}

/// Run STAPLE on K binary rater masks (f32, 0.0 = negative, >0.5 = positive).
///
/// # Arguments
/// - `raters`:   slice of K flat masks, each of length N.
/// - `max_iter`: maximum EM iterations.
/// - `tol`:      convergence tolerance on per-rater parameter change.
///
/// # Panics
/// - If `raters` is empty.
/// - If any two masks have different lengths.
///
/// # Reference
/// Warfield, Zou, Wells (2004) IEEE TMI 23(7):903–921.
pub fn staple(raters: &[Vec<f32>], max_iter: usize, tol: f64) -> StapleResult {
    assert!(!raters.is_empty(), "staple: raters must be non-empty");

    let k = raters.len();
    let n = raters[0].len();

    for (idx, r) in raters.iter().enumerate() {
        assert_eq!(
            r.len(),
            n,
            "staple: rater {} has length {} but rater 0 has length {}",
            idx,
            r.len(),
            n,
        );
    }

    // Threshold input masks at 0.5 → bool (avoids repeated f32 comparisons in the hot path).
    let d: Vec<Vec<bool>> = raters
        .iter()
        .map(|r| r.iter().map(|&v| v > 0.5).collect())
        .collect();

    // ── Initialise parameters ────────────────────────────────────────────────
    let mut p = vec![0.99_f64; k];
    let mut q = vec![0.99_f64; k];
    let mut f = 0.5_f64;

    // W: probabilistic ground truth; uniform initialisation.
    let mut w = vec![0.5_f64; n];

    let mut converged = false;
    let mut iterations = 0usize;

    for _iter in 0..max_iter {
        iterations += 1;

        // Pre-compute log-domain parameter vectors (shared read-only across all voxels).
        let log_f = f.ln();
        let log_1mf = (1.0 - f).ln();
        let log_p: Vec<f64> = p.iter().map(|&pk| pk.ln()).collect();
        let log_1mp: Vec<f64> = p.iter().map(|&pk| (1.0 - pk).ln()).collect();
        let log_q: Vec<f64> = q.iter().map(|&qk| qk.ln()).collect();
        let log_1mq: Vec<f64> = q.iter().map(|&qk| (1.0 - qk).ln()).collect();

        // ── E-step: voxels are independent → parallel ────────────────────────
        w = moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |i| {
            let mut log_alpha = log_f;
            let mut log_beta = log_1mf;
            for k_idx in 0..k {
                if d[k_idx][i] {
                    log_alpha += log_p[k_idx];
                    log_beta += log_1mq[k_idx];
                } else {
                    log_alpha += log_1mp[k_idx];
                    log_beta += log_q[k_idx];
                }
            }
            sigmoid_log_domain(log_alpha, log_beta)
        });

        // ── M-step: update prevalence and per-rater parameters ───────────────
        let w_sum: f64 = moirai::reduce_index_with::<moirai::Adaptive, _, _, _>(
            w.len(),
            0.0,
            |i| w[i],
            |a, b| a + b,
        );
        // Epsilon guards prevent division by zero when W collapses to 0 or N.
        let denom_p = w_sum.max(EPS);
        let denom_q = (n as f64 - w_sum).max(EPS);
        f = w_sum / n as f64;

        let mut p_new = vec![0.0_f64; k];
        let mut q_new = vec![0.0_f64; k];

        for k_idx in 0..k {
            // Parallel reduction over N voxels for both numerators simultaneously.
            let (num_p, num_q): (f64, f64) = moirai::reduce_index_with::<moirai::Adaptive, _, _, _>(
                w.len(),
                (0.0_f64, 0.0_f64),
                |i| {
                    let (wi, dki) = (w[i], d[k_idx][i]);
                    if dki {
                        (wi, 0.0_f64)
                    } else {
                        (0.0_f64, 1.0 - wi)
                    }
                },
                |(pa, qa), (pb, qb)| (pa + pb, qa + qb),
            );

            p_new[k_idx] = (num_p / denom_p).clamp(EPS, 1.0 - EPS);
            q_new[k_idx] = (num_q / denom_q).clamp(EPS, 1.0 - EPS);
        }

        // ── Convergence: max over raters of joint parameter change ───────────
        let max_change = p_new
            .iter()
            .zip(p.iter())
            .zip(q_new.iter().zip(q.iter()))
            .map(|((pn, po), (qn, qo))| (pn - po).abs() + (qn - qo).abs())
            .fold(0.0_f64, |acc, x| acc.max(x));

        p = p_new;
        q = q_new;

        if max_change < tol {
            converged = true;
            break;
        }
    }

    StapleResult {
        probabilistic_truth: w.iter().map(|&wi| wi as f32).collect(),
        sensitivity: p.iter().map(|&pk| pk as f32).collect(),
        specificity: q.iter().map(|&qk| qk as f32).collect(),
        iterations,
        converged,
    }
}
