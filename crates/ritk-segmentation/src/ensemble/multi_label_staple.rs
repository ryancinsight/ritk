//! Multi-label STAPLE consensus labeling (`itk::MultiLabelSTAPLEImageFilter`).
//!
//! # Mathematical Specification
//!
//! Generalizes binary [`staple`](super::staple()) to `L` discrete labels. Given `K`
//! label maps `d_k ∈ {0, …, L−1}^N`, it estimates a per-rater confusion matrix
//! `θ_k[j][i] = P(rater k says j | true label is i)` by expectation-maximization
//! and emits the per-voxel maximum-likelihood consensus label.
//!
//! - **Init** (`InitializeConfusionMatrixArrayFromVoting`): the confusion matrix is
//!   seeded from the joint histogram of `(rater label, majority-vote consensus)`,
//!   row-normalized. Majority voting breaks ties to the *undecided* label `L`.
//! - **Prior**: `π_i ∝ Σ_k Σ_n [d_k(n) = i]`, normalized over the `L` real labels.
//! - **E-step**: per voxel, `W_i = π_i · Π_k θ_k[d_k(n)][i]`, normalized to sum 1.
//! - **M-step**: `θ_k[j][i] = Σ_n W_i(n)·[d_k(n)=j]`, column-normalized over `j`.
//! - **Termination**: when the maximum absolute parameter change `< threshold`.
//! - **Output**: `argmax_i W_i`; ties (equal maxima) and all-zero `W` → undecided `L`.
//!
//! Internal arithmetic is `f64`. The output is a discrete label image, so it is
//! float-exact to `sitk.MultiLabelSTAPLE` (the argmax is insensitive to sub-ULP
//! weight jitter except at genuine ties, which both map to undecided).

/// Result of [`multi_label_staple`].
#[derive(Debug, Clone)]
pub struct MultiLabelStapleResult {
    /// Per-voxel consensus label (integer-valued `f32`); undecided voxels carry
    /// [`label_for_undecided`](Self::label_for_undecided).
    pub labels: Vec<f32>,
    /// The label assigned to undecided voxels (default `L`, the max label + 1).
    pub label_for_undecided: f32,
    /// Number of EM iterations performed.
    pub iterations: usize,
}

/// Run multi-label STAPLE on `K` integer-valued label maps (stored as `f32`).
///
/// `max_iter = None` iterates until convergence (`max |Δθ| < termination_threshold`).
/// `label_for_undecided = None` uses `L` (max label + 1).
///
/// # Panics
/// Panics if `raters` is empty or the rater lengths differ (a caller contract
/// violation, not input-dependent).
pub fn multi_label_staple(
    raters: &[Vec<f32>],
    max_iter: Option<usize>,
    termination_threshold: f64,
    label_for_undecided: Option<f32>,
) -> MultiLabelStapleResult {
    assert!(
        !raters.is_empty(),
        "multi_label_staple: raters must be non-empty"
    );
    let k = raters.len();
    let n = raters[0].len();
    for (idx, r) in raters.iter().enumerate() {
        assert_eq!(
            r.len(),
            n,
            "multi_label_staple: rater {idx} length mismatch"
        );
    }

    // Quantize to integer labels in a single voxel-major buffer `d[vox*k + kk]`
    // (one allocation, 4 B/label, and sequential access in the per-voxel rater
    // loops below — the E-step/accumulate hot path iterates k for fixed voxel).
    let mut d = vec![0u32; n * k];
    for (kk, r) in raters.iter().enumerate() {
        for (vox, &v) in r.iter().enumerate() {
            d[vox * k + kk] = v.round().max(0.0) as u32;
        }
    }
    let dl = |vox: usize, kk: usize| d[vox * k + kk] as usize;
    let l = d.iter().copied().max().unwrap_or(0) as usize + 1;
    let undecided = label_for_undecided.unwrap_or(l as f32);

    // Confusion matrices θ_k: (L+1) input-label rows × L output-class columns.
    let row = l; // columns
    let rows = l + 1;
    let mut conf = vec![0.0f64; k * rows * row];
    let cidx = |kk: usize, j: usize, ci: usize| (kk * rows + j) * row + ci;

    // --- Init from majority voting (ties -> undecided = l) ---
    let mut counts = vec![0u32; l];
    for vox in 0..n {
        for c in counts.iter_mut() {
            *c = 0;
        }
        for kk in 0..k {
            counts[dl(vox, kk)] += 1;
        }
        let maxc = *counts.iter().max().expect("infallible: validated precondition");
        let winners = counts.iter().filter(|&&c| c == maxc).count();
        let consensus = if winners == 1 {
            counts.iter().position(|&c| c == maxc).expect("infallible: validated precondition")
        } else {
            l // undecided
        };
        if consensus < l {
            for kk in 0..k {
                conf[cidx(kk, dl(vox, kk), consensus)] += 1.0;
            }
        }
    }
    // Row-normalize.
    for kk in 0..k {
        for j in 0..rows {
            let s: f64 = (0..row).map(|ci| conf[cidx(kk, j, ci)]).sum();
            if s > 0.0 {
                for ci in 0..row {
                    conf[cidx(kk, j, ci)] /= s;
                }
            }
        }
    }

    // --- Prior from label frequencies (over the L real labels) ---
    let mut prior = vec![0.0f64; l];
    for &lab in &d {
        let lab = lab as usize;
        if lab < l {
            prior[lab] += 1.0;
        }
    }
    let total: f64 = prior.iter().sum();
    if total > 0.0 {
        for p in prior.iter_mut() {
            *p /= total;
        }
    }

    // --- EM ---
    let mut updated = vec![0.0f64; k * rows * row];
    let mut w = vec![0.0f64; l];
    let mut iterations = 0usize;
    loop {
        if let Some(m) = max_iter {
            if iterations >= m {
                break;
            }
        }
        updated.iter_mut().for_each(|x| *x = 0.0);
        for vox in 0..n {
            // E-step.
            w.copy_from_slice(&prior);
            for kk in 0..k {
                let j = dl(vox, kk);
                for ci in 0..l {
                    w[ci] *= conf[cidx(kk, j, ci)];
                }
            }
            let sw: f64 = w.iter().sum();
            if sw != 0.0 {
                for x in w.iter_mut() {
                    *x /= sw;
                }
            }
            // Accumulate.
            for kk in 0..k {
                let j = dl(vox, kk);
                for ci in 0..l {
                    updated[cidx(kk, j, ci)] += w[ci];
                }
            }
        }
        // Column-normalize updated confusion matrices.
        for kk in 0..k {
            for ci in 0..l {
                let s: f64 = (0..rows).map(|j| updated[cidx(kk, j, ci)]).sum();
                if s != 0.0 {
                    for j in 0..rows {
                        updated[cidx(kk, j, ci)] /= s;
                    }
                }
            }
        }
        // Max parameter change + apply.
        let mut max_update = 0.0f64;
        for (u, c) in updated.iter().zip(conf.iter_mut()) {
            max_update = max_update.max((u - *c).abs());
            *c = *u;
        }
        iterations += 1;
        if max_update < termination_threshold {
            break;
        }
    }

    // --- Output consensus (repeat E-step, argmax, ties -> undecided) ---
    // `vox`/`ci` index multiple parallel buffers, and the `!(w < win)` tie test
    // mirrors ITK's exact branch (equal-or-incomparable maxima -> undecided), so
    // the index loops and negated partial-order compare are intentional.
    #[allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]
    let labels = {
        let mut labels = vec![undecided; n];
        for vox in 0..n {
            w.copy_from_slice(&prior);
            for kk in 0..k {
                let j = dl(vox, kk);
                for ci in 0..l {
                    w[ci] *= conf[cidx(kk, j, ci)];
                }
            }
            let mut win = undecided;
            let mut win_w = 0.0f64;
            for ci in 0..l {
                if w[ci] > win_w {
                    win_w = w[ci];
                    win = ci as f32;
                } else if !(w[ci] < win_w) {
                    win = undecided;
                }
            }
            labels[vox] = win;
        }
        labels
    };

    MultiLabelStapleResult {
        labels,
        label_for_undecided: undecided,
        iterations,
    }
}

#[cfg(test)]
#[path = "tests_multi_label_staple.rs"]
mod tests_multi_label_staple;
