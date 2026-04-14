//! Multi-atlas label fusion algorithms.
//!
//! Implements two algorithms for combining segmentation results from multiple
//! registered atlas label maps into a single consensus segmentation:
//!
//! 1. **Majority Voting** — baseline: the mode of atlas labels at each voxel.
//! 2. **Joint Label Fusion** (JLF) — weighted voting where per-voxel weights
//!    are derived from local patch intensity similarity between warped atlas
//!    images and the target image.
//!
//! # Joint Label Fusion — Mathematical Specification
//!
//! Given N atlas label maps {L₁, ..., Lₙ} registered to target space,
//! corresponding warped atlas intensity images {A₁, ..., Aₙ}, and target
//! image T:
//!
//! For each voxel x with patch P(x) of radius r (a (2r+1)³ cube clipped to
//! image boundaries):
//!
//! 1. Compute per-atlas patch distance:
//!    dᵢ(x) = ‖P_{Aᵢ}(x) − P_T(x)‖²  =  Σ_{q ∈ P(x)} (Aᵢ(q) − T(q))²
//!
//! 2. Build pairwise similarity matrix M(x) ∈ ℝ^{N×N}:
//!    M_{ij}(x) = dᵢ(x) + dⱼ(x)
//!
//! 3. Regularize: M_{ij}(x) += α · δ_{ij}, where α = β · min_{ij}(M_{ij})
//!    and β is a user-specified parameter (default 0.1).
//!
//! 4. Solve for weights w(x): M w = 1 via Gaussian elimination with partial
//!    pivoting.  Clamp negative weights to 0, then normalize: w = w / Σwᵢ.
//!    If the system is singular or all weights are non-positive, fall back to
//!    uniform weights wᵢ = 1/N.
//!
//! 5. Fused label: L(x) = argmax_l  Σᵢ wᵢ(x) · \[Lᵢ(x) = l\]
//!    Ties are broken by selecting the smallest label value.
//!
//! # References
//!
//! - Wang, H., Suh, J. W., Das, S. R., Pluta, J. B., Craige, C. &
//!   Yushkevich, P. A. (2013). Multi-atlas segmentation with joint label
//!   fusion. *IEEE Trans. Pattern Analysis and Machine Intelligence*
//!   35(3):611–623.

use std::collections::HashMap;

use crate::error::RegistrationError;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for the Joint Label Fusion algorithm.
#[derive(Debug, Clone)]
pub struct LabelFusionConfig {
    /// Patch radius (voxels) for computing local similarity.
    /// A radius of r yields a (2r+1)³ patch clipped at image boundaries.
    pub patch_radius: usize,
    /// Regularization factor β.  The diagonal regularization added to the
    /// pairwise similarity matrix is α = β · min_{ij}(M_{ij}).
    pub beta: f64,
}

impl Default for LabelFusionConfig {
    /// Default: `patch_radius = 2`, `beta = 0.1`.
    fn default() -> Self {
        Self {
            patch_radius: 2,
            beta: 0.1,
        }
    }
}

/// Result of a label fusion operation.
#[derive(Debug, Clone)]
pub struct LabelFusionResult {
    /// Fused label map (flat `Vec<u32>`, shape `[nz, ny, nx]`).
    pub labels: Vec<u32>,
    /// Per-voxel confidence.
    ///
    /// - For majority voting: fraction of atlases that voted for the winning
    ///   label (range \[1/N, 1\]).
    /// - For JLF: sum of weights assigned to the winning label (range
    ///   \[0, 1\]).
    pub confidence: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Majority voting
// ---------------------------------------------------------------------------

/// Majority voting label fusion (baseline).
///
/// For each voxel the output label is the mode of the labels across all
/// atlases.  Ties are broken by selecting the smallest label value.
/// Confidence is the fraction of atlases voting for the winning label.
///
/// # Errors
///
/// - [`RegistrationError::InvalidConfiguration`] if `atlas_labels` is empty.
/// - [`RegistrationError::DimensionMismatch`] if any atlas label map length
///   differs from `dims[0] * dims[1] * dims[2]`.
pub fn majority_vote(
    atlas_labels: &[&[u32]],
    dims: [usize; 3],
) -> Result<LabelFusionResult, RegistrationError> {
    let n_atlases = atlas_labels.len();
    if n_atlases == 0 {
        return Err(RegistrationError::InvalidConfiguration(
            "atlas_labels is empty; at least one atlas is required".into(),
        ));
    }
    let [nz, ny, nx] = dims;
    let n_voxels = nz * ny * nx;
    for (i, lbl) in atlas_labels.iter().enumerate() {
        if lbl.len() != n_voxels {
            return Err(RegistrationError::DimensionMismatch(format!(
                "atlas_labels[{}] length {} != dims product {}",
                i,
                lbl.len(),
                n_voxels
            )));
        }
    }

    let mut labels = vec![0u32; n_voxels];
    let mut confidence = vec![0.0f32; n_voxels];

    let mut counts: HashMap<u32, usize> = HashMap::new();

    for v in 0..n_voxels {
        counts.clear();
        for a in 0..n_atlases {
            *counts.entry(atlas_labels[a][v]).or_insert(0) += 1;
        }

        // Deterministic tie-breaking: highest count wins; on tie, smallest
        // label wins.
        let mut best_label = 0u32;
        let mut best_count = 0usize;
        for (&label, &count) in &counts {
            if count > best_count || (count == best_count && label < best_label) {
                best_count = count;
                best_label = label;
            }
        }
        labels[v] = best_label;
        confidence[v] = best_count as f32 / n_atlases as f32;
    }

    Ok(LabelFusionResult { labels, confidence })
}

// ---------------------------------------------------------------------------
// Joint Label Fusion
// ---------------------------------------------------------------------------

/// Joint Label Fusion (Wang et al. 2013).
///
/// Computes per-voxel atlas weights from local patch similarity between the
/// warped atlas intensity images and the target, then performs weighted voting
/// over the atlas label maps.  See the [module-level documentation](self) for
/// the full mathematical specification.
///
/// # Errors
///
/// - [`RegistrationError::InvalidConfiguration`] if `atlas_images` is empty.
/// - [`RegistrationError::DimensionMismatch`] if `atlas_images` and
///   `atlas_labels` have different lengths, or if any buffer length differs
///   from the product of `dims`.
pub fn joint_label_fusion(
    target: &[f32],
    atlas_images: &[&[f32]],
    atlas_labels: &[&[u32]],
    dims: [usize; 3],
    config: &LabelFusionConfig,
) -> Result<LabelFusionResult, RegistrationError> {
    // ── Validation ────────────────────────────────────────────────────────
    let n_atlases = atlas_images.len();
    if n_atlases == 0 {
        return Err(RegistrationError::InvalidConfiguration(
            "atlas_images is empty; at least one atlas is required".into(),
        ));
    }
    if atlas_labels.len() != n_atlases {
        return Err(RegistrationError::DimensionMismatch(format!(
            "atlas_images count {} != atlas_labels count {}",
            n_atlases,
            atlas_labels.len()
        )));
    }
    let [nz, ny, nx] = dims;
    let n_voxels = nz * ny * nx;
    if target.len() != n_voxels {
        return Err(RegistrationError::DimensionMismatch(format!(
            "target length {} != dims product {}",
            target.len(),
            n_voxels
        )));
    }
    for (i, img) in atlas_images.iter().enumerate() {
        if img.len() != n_voxels {
            return Err(RegistrationError::DimensionMismatch(format!(
                "atlas_images[{}] length {} != dims product {}",
                i,
                img.len(),
                n_voxels
            )));
        }
    }
    for (i, lbl) in atlas_labels.iter().enumerate() {
        if lbl.len() != n_voxels {
            return Err(RegistrationError::DimensionMismatch(format!(
                "atlas_labels[{}] length {} != dims product {}",
                i,
                lbl.len(),
                n_voxels
            )));
        }
    }

    let r = config.patch_radius;
    let mut labels = vec![0u32; n_voxels];
    let mut confidence = vec![0.0f32; n_voxels];

    // Pre-allocate per-voxel working buffers outside the hot loop.
    let mut d = vec![0.0f64; n_atlases];
    let mut m_flat = vec![0.0f64; n_atlases * n_atlases];
    let mut rhs = vec![0.0f64; n_atlases];
    let mut label_weights: HashMap<u32, f64> = HashMap::new();

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let voxel_idx = iz * ny * nx + iy * nx + ix;

                // ── Patch bounds (clipped to image) ──────────────────────
                let z_lo = iz.saturating_sub(r);
                let z_hi = (iz + r).min(nz - 1);
                let y_lo = iy.saturating_sub(r);
                let y_hi = (iy + r).min(ny - 1);
                let x_lo = ix.saturating_sub(r);
                let x_hi = (ix + r).min(nx - 1);

                // ── dᵢ = ‖P_{Aᵢ} − P_T‖² ──────────────────────────────
                for v in d.iter_mut() {
                    *v = 0.0;
                }
                for pz in z_lo..=z_hi {
                    for py in y_lo..=y_hi {
                        for px in x_lo..=x_hi {
                            let pi = pz * ny * nx + py * nx + px;
                            let tv = target[pi] as f64;
                            for (a, d_a) in d.iter_mut().enumerate() {
                                let diff = atlas_images[a][pi] as f64 - tv;
                                *d_a += diff * diff;
                            }
                        }
                    }
                }

                // ── Build M: M_{ij} = dᵢ + dⱼ ──────────────────────────
                let n = n_atlases;
                for i in 0..n {
                    for j in 0..n {
                        m_flat[i * n + j] = d[i] + d[j];
                    }
                }

                // ── Regularize: α = β · min(M), M_{ii} += α ─────────────
                let min_m = m_flat[..n * n].iter().copied().fold(f64::MAX, f64::min);
                let alpha = config.beta * min_m;
                for i in 0..n {
                    m_flat[i * n + i] += alpha;
                }

                // ── Solve M w = 1 ───────────────────────────────────────
                // Copy into row-major Vec<Vec<f64>> for the solver.
                let mut mat: Vec<Vec<f64>> = (0..n)
                    .map(|i| m_flat[i * n..(i + 1) * n].to_vec())
                    .collect();
                for v in rhs.iter_mut().take(n) {
                    *v = 1.0;
                }
                let raw_weights = solve_linear_system(&mut mat, &mut rhs[..n]);

                let mut w: Vec<f64> = match raw_weights {
                    Some(ww) => ww,
                    None => vec![1.0 / n as f64; n],
                };

                // Clamp negatives to 0.
                for v in w.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }

                // Normalize.
                let sum: f64 = w.iter().sum();
                if sum > 1e-30 {
                    for v in w.iter_mut() {
                        *v /= sum;
                    }
                } else {
                    // All weights non-positive → uniform fallback.
                    let u = 1.0 / n as f64;
                    for v in w.iter_mut() {
                        *v = u;
                    }
                }

                // ── Weighted vote ────────────────────────────────────────
                label_weights.clear();
                for a in 0..n {
                    let label = atlas_labels[a][voxel_idx];
                    *label_weights.entry(label).or_insert(0.0) += w[a];
                }

                // argmax with deterministic tie-breaking (smallest label).
                let mut best_label = 0u32;
                let mut best_weight = -1.0f64;
                for (&label, &weight) in &label_weights {
                    if weight > best_weight || (weight == best_weight && label < best_label) {
                        best_weight = weight;
                        best_label = label;
                    }
                }

                labels[voxel_idx] = best_label;
                confidence[voxel_idx] = best_weight as f32;
            }
        }
    }

    Ok(LabelFusionResult { labels, confidence })
}

// ---------------------------------------------------------------------------
// Dense linear system solver (private)
// ---------------------------------------------------------------------------

/// Solve the N×N linear system Ax = b via Gaussian elimination with partial
/// pivoting.
///
/// `a` is modified in place (row-echelon form).  `b` is modified in place
/// (forward-eliminated RHS).  Returns `Some(x)` on success or `None` if the
/// matrix is singular (pivot magnitude < 1e-15).
fn solve_linear_system(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = b.len();
    debug_assert!(a.len() == n);

    // Forward elimination with partial pivoting.
    for col in 0..n {
        // Find pivot row.
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None; // Singular.
        }
        // Swap rows.
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }
        // Eliminate below.
        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for j in col..n {
                // Split borrow: read a[col][j], write a[row][j].
                let a_col_j = a[col][j];
                a[row][j] -= factor * a_col_j;
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }

    Some(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Majority voting – positive tests ──────────────────────────────────

    /// All 3 atlases assign label 7 at every voxel.
    ///
    /// Expected: fused label = 7 everywhere, confidence = 3/3 = 1.0.
    #[test]
    fn majority_vote_unanimous() {
        let dims = [2, 2, 2];
        let n = 8;
        let l1 = vec![7u32; n];
        let l2 = vec![7u32; n];
        let l3 = vec![7u32; n];
        let atlas_labels: Vec<&[u32]> = vec![&l1, &l2, &l3];

        let result = majority_vote(&atlas_labels, dims).unwrap();
        assert_eq!(result.labels.len(), n);
        assert_eq!(result.confidence.len(), n);
        for i in 0..n {
            assert_eq!(result.labels[i], 7, "voxel {}", i);
            assert!((result.confidence[i] - 1.0).abs() < 1e-6, "voxel {}", i);
        }
    }

    /// 3 atlases: two assign label 1, one assigns label 2.
    ///
    /// Expected: fused label = 1 everywhere, confidence = 2/3 ≈ 0.6667.
    #[test]
    fn majority_vote_majority_label() {
        let dims = [2, 2, 2];
        let n = 8;
        let l1 = vec![1u32; n];
        let l2 = vec![1u32; n];
        let l3 = vec![2u32; n];
        let atlas_labels: Vec<&[u32]> = vec![&l1, &l2, &l3];

        let result = majority_vote(&atlas_labels, dims).unwrap();
        let expected_conf = 2.0f32 / 3.0;
        for i in 0..n {
            assert_eq!(result.labels[i], 1, "voxel {}", i);
            assert!(
                (result.confidence[i] - expected_conf).abs() < 1e-6,
                "voxel {} confidence {} != {}",
                i,
                result.confidence[i],
                expected_conf
            );
        }
    }

    /// Tie-breaking: 2 atlases, one votes label 3, one votes label 1.
    /// Tie → smallest label (1) wins.  Confidence = 0.5.
    #[test]
    fn majority_vote_tie_smallest_label_wins() {
        let dims = [1, 1, 2];
        let l1 = vec![3u32; 2];
        let l2 = vec![1u32; 2];
        let atlas_labels: Vec<&[u32]> = vec![&l1, &l2];

        let result = majority_vote(&atlas_labels, dims).unwrap();
        for i in 0..2 {
            assert_eq!(
                result.labels[i], 1,
                "smallest label wins tie at voxel {}",
                i
            );
            assert!((result.confidence[i] - 0.5).abs() < 1e-6);
        }
    }

    // ── Majority voting – boundary tests ──────────────────────────────────

    /// Single atlas: fused labels equal the atlas labels, confidence = 1.0.
    #[test]
    fn majority_vote_single_atlas() {
        let dims = [2, 2, 2];
        let n = 8;
        let mut labels = vec![0u32; n];
        for (i, v) in labels.iter_mut().enumerate() {
            *v = i as u32 + 10;
        }
        let atlas_labels: Vec<&[u32]> = vec![&labels];

        let result = majority_vote(&atlas_labels, dims).unwrap();
        for i in 0..n {
            assert_eq!(result.labels[i], i as u32 + 10, "voxel {}", i);
            assert!((result.confidence[i] - 1.0).abs() < 1e-6);
        }
    }

    // ── Majority voting – negative tests ──────────────────────────────────

    /// Empty atlas list returns `InvalidConfiguration`.
    #[test]
    fn majority_vote_empty_error() {
        let atlas_labels: Vec<&[u32]> = vec![];
        let err = majority_vote(&atlas_labels, [2, 2, 2]).unwrap_err();
        assert!(
            matches!(err, RegistrationError::InvalidConfiguration(_)),
            "expected InvalidConfiguration, got {:?}",
            err
        );
    }

    /// Atlas label map with wrong length returns `DimensionMismatch`.
    #[test]
    fn majority_vote_dimension_mismatch() {
        let l1 = vec![1u32; 8]; // correct for [2,2,2]
        let l2 = vec![1u32; 5]; // wrong
        let atlas_labels: Vec<&[u32]> = vec![&l1, &l2];
        let err = majority_vote(&atlas_labels, [2, 2, 2]).unwrap_err();
        assert!(
            matches!(err, RegistrationError::DimensionMismatch(_)),
            "expected DimensionMismatch, got {:?}",
            err
        );
    }

    // ── JLF – positive tests ─────────────────────────────────────────────

    /// 3 equidistant atlases (all intensity 1.0, target 2.0), all label 5.
    ///
    /// Analytical derivation (patch_radius = 0, single-voxel patches):
    ///   dᵢ = (1.0 − 2.0)² = 1.0  for all i.
    ///   M = [[2, 2, 2], [2, 2, 2], [2, 2, 2]].
    ///   min(M) = 2, α = 0.1 × 2 = 0.2.
    ///   M_reg = [[2.2, 2, 2], [2, 2.2, 2], [2, 2, 2.2]].
    ///   By symmetry w₁ = w₂ = w₃ = w. Row sum: 6.2w = 1 → w = 1/6.2.
    ///   Normalized: wᵢ = 1/3.
    ///   All labels = 5 → fused = 5, confidence = 1.0.
    #[test]
    fn jlf_equidistant_same_labels() {
        let dims = [2, 2, 2];
        let n = 8;
        let a1 = vec![1.0f32; n];
        let a2 = vec![1.0f32; n];
        let a3 = vec![1.0f32; n];
        let target = vec![2.0f32; n];
        let l1 = vec![5u32; n];
        let l2 = vec![5u32; n];
        let l3 = vec![5u32; n];
        let atlas_images: Vec<&[f32]> = vec![&a1, &a2, &a3];
        let atlas_labels: Vec<&[u32]> = vec![&l1, &l2, &l3];

        let config = LabelFusionConfig {
            patch_radius: 0,
            beta: 0.1,
        };
        let result =
            joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

        for i in 0..n {
            assert_eq!(result.labels[i], 5, "voxel {}", i);
            assert!(
                (result.confidence[i] - 1.0).abs() < 1e-4,
                "voxel {} confidence {} != 1.0",
                i,
                result.confidence[i]
            );
        }
    }

    /// 3 equidistant atlases, 2 with label 1, 1 with label 2.
    ///
    /// Same M derivation as above: equal weights wᵢ = 1/3.
    /// Vote for label 1: w₁ + w₂ = 2/3.
    /// Vote for label 2: w₃ = 1/3.
    /// Fused = 1, confidence = 2/3.
    #[test]
    fn jlf_equidistant_majority() {
        let dims = [2, 2, 2];
        let n = 8;
        let a1 = vec![1.0f32; n];
        let a2 = vec![1.0f32; n];
        let a3 = vec![1.0f32; n];
        let target = vec![2.0f32; n];
        let l1 = vec![1u32; n];
        let l2 = vec![1u32; n];
        let l3 = vec![2u32; n];
        let atlas_images: Vec<&[f32]> = vec![&a1, &a2, &a3];
        let atlas_labels: Vec<&[u32]> = vec![&l1, &l2, &l3];

        let config = LabelFusionConfig {
            patch_radius: 0,
            beta: 0.1,
        };
        let result =
            joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

        let expected_conf = 2.0f64 / 3.0;
        for i in 0..n {
            assert_eq!(result.labels[i], 1, "voxel {}", i);
            assert!(
                (result.confidence[i] as f64 - expected_conf).abs() < 1e-4,
                "voxel {} confidence {} != {}",
                i,
                result.confidence[i],
                expected_conf
            );
        }
    }

    /// 2 equidistant atlases with different labels: equal weights → tie →
    /// smallest label wins.
    ///
    /// A1 = 1.0, A2 = 3.0, T = 2.0.  d₁ = d₂ = 1.0.
    /// M = [[2,2],[2,2]], min = 2, α = 0.2.
    /// M_reg = [[2.2,2],[2,2.2]], det = 0.84.
    /// M⁻¹·1 = (1/0.84)[0.2, 0.2] → normalized w = [0.5, 0.5].
    /// L1 = 1, L2 = 2, both weight 0.5 → tie → label 1 wins.
    /// Confidence = 0.5.
    #[test]
    fn jlf_equidistant_tie_smallest_label() {
        let dims = [2, 2, 2];
        let n = 8;
        let a1 = vec![1.0f32; n];
        let a2 = vec![3.0f32; n];
        let target = vec![2.0f32; n];
        let l1 = vec![1u32; n];
        let l2 = vec![2u32; n];
        let atlas_images: Vec<&[f32]> = vec![&a1, &a2];
        let atlas_labels: Vec<&[u32]> = vec![&l1, &l2];

        let config = LabelFusionConfig {
            patch_radius: 0,
            beta: 0.1,
        };
        let result =
            joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

        for i in 0..n {
            assert_eq!(
                result.labels[i], 1,
                "smallest label wins tie at voxel {}",
                i
            );
            assert!(
                (result.confidence[i] - 0.5).abs() < 1e-4,
                "voxel {} confidence {} != 0.5",
                i,
                result.confidence[i]
            );
        }
    }

    /// Atlases identical to target: all dᵢ = 0, min(M) = 0, α = 0.
    /// M is singular (all zeros) → uniform fallback → equal weights.
    /// All labels = 3 → fused = 3, confidence = 1.0.
    #[test]
    fn jlf_zero_distance_singular_fallback() {
        let dims = [2, 2, 2];
        let n = 8;
        let a = vec![5.0f32; n];
        let target = vec![5.0f32; n];
        let l = vec![3u32; n];
        let atlas_images: Vec<&[f32]> = vec![&a, &a, &a];
        let atlas_labels: Vec<&[u32]> = vec![&l, &l, &l];

        let config = LabelFusionConfig {
            patch_radius: 0,
            beta: 0.1,
        };
        let result =
            joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

        for i in 0..n {
            assert_eq!(result.labels[i], 3, "voxel {}", i);
            assert!(
                (result.confidence[i] - 1.0).abs() < 1e-4,
                "voxel {} confidence {} != 1.0",
                i,
                result.confidence[i]
            );
        }
    }

    // ── JLF – boundary tests ─────────────────────────────────────────────

    /// Single atlas: 1×1 system M=[2d+α], w=[1/(2d+α)], normalized to 1.0.
    /// Fused labels equal the atlas labels, confidence = 1.0.
    #[test]
    fn jlf_single_atlas() {
        let dims = [2, 2, 2];
        let n = 8;
        let a = vec![1.0f32; n];
        let target = vec![2.0f32; n];
        let mut l = vec![0u32; n];
        for (i, v) in l.iter_mut().enumerate() {
            *v = i as u32 + 100;
        }
        let atlas_images: Vec<&[f32]> = vec![&a];
        let atlas_labels: Vec<&[u32]> = vec![&l];

        let config = LabelFusionConfig {
            patch_radius: 0,
            beta: 0.1,
        };
        let result =
            joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

        for i in 0..n {
            assert_eq!(result.labels[i], i as u32 + 100, "voxel {}", i);
            assert!(
                (result.confidence[i] - 1.0).abs() < 1e-4,
                "voxel {} confidence {} != 1.0",
                i,
                result.confidence[i]
            );
        }
    }

    /// Patch radius > 0 on a uniform image gives the same result as radius 0
    /// (all voxels in the patch have the same value, so d scales but the
    /// relative weights remain equal for equidistant atlases).
    #[test]
    fn jlf_nonzero_patch_radius_equidistant() {
        let dims = [4, 4, 4];
        let n = 64;
        let a1 = vec![1.0f32; n];
        let a2 = vec![1.0f32; n];
        let target = vec![2.0f32; n];
        let l1 = vec![1u32; n];
        let l2 = vec![1u32; n];
        let atlas_images: Vec<&[f32]> = vec![&a1, &a2];
        let atlas_labels: Vec<&[u32]> = vec![&l1, &l2];

        let config = LabelFusionConfig {
            patch_radius: 1,
            beta: 0.1,
        };
        let result =
            joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

        for i in 0..n {
            assert_eq!(result.labels[i], 1, "voxel {}", i);
            // Equal weights → confidence = 1.0 (all same label).
            assert!(
                (result.confidence[i] - 1.0).abs() < 1e-4,
                "voxel {} confidence {}",
                i,
                result.confidence[i]
            );
        }
    }

    // ── JLF – negative tests ─────────────────────────────────────────────

    /// Empty atlas list returns `InvalidConfiguration`.
    #[test]
    fn jlf_empty_error() {
        let target = vec![1.0f32; 8];
        let atlas_images: Vec<&[f32]> = vec![];
        let atlas_labels: Vec<&[u32]> = vec![];
        let config = LabelFusionConfig::default();
        let err = joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config)
            .unwrap_err();
        assert!(
            matches!(err, RegistrationError::InvalidConfiguration(_)),
            "expected InvalidConfiguration, got {:?}",
            err
        );
    }

    /// Mismatched atlas_images / atlas_labels count → `DimensionMismatch`.
    #[test]
    fn jlf_atlas_count_mismatch() {
        let target = vec![1.0f32; 8];
        let a = vec![1.0f32; 8];
        let l1 = vec![1u32; 8];
        let l2 = vec![1u32; 8];
        let atlas_images: Vec<&[f32]> = vec![&a];
        let atlas_labels: Vec<&[u32]> = vec![&l1, &l2];
        let config = LabelFusionConfig::default();
        let err = joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config)
            .unwrap_err();
        assert!(
            matches!(err, RegistrationError::DimensionMismatch(_)),
            "expected DimensionMismatch, got {:?}",
            err
        );
    }

    /// Target with wrong length → `DimensionMismatch`.
    #[test]
    fn jlf_target_dimension_mismatch() {
        let target = vec![1.0f32; 5]; // wrong for [2,2,2]
        let a = vec![1.0f32; 8];
        let l = vec![1u32; 8];
        let atlas_images: Vec<&[f32]> = vec![&a];
        let atlas_labels: Vec<&[u32]> = vec![&l];
        let config = LabelFusionConfig::default();
        let err = joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config)
            .unwrap_err();
        assert!(
            matches!(err, RegistrationError::DimensionMismatch(_)),
            "expected DimensionMismatch, got {:?}",
            err
        );
    }

    /// Atlas image with wrong length → `DimensionMismatch`.
    #[test]
    fn jlf_atlas_image_dimension_mismatch() {
        let target = vec![1.0f32; 8];
        let a = vec![1.0f32; 5]; // wrong
        let l = vec![1u32; 8];
        let atlas_images: Vec<&[f32]> = vec![&a];
        let atlas_labels: Vec<&[u32]> = vec![&l];
        let config = LabelFusionConfig::default();
        let err = joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config)
            .unwrap_err();
        assert!(
            matches!(err, RegistrationError::DimensionMismatch(_)),
            "expected DimensionMismatch, got {:?}",
            err
        );
    }

    /// Atlas label map with wrong length → `DimensionMismatch`.
    #[test]
    fn jlf_atlas_label_dimension_mismatch() {
        let target = vec![1.0f32; 8];
        let a = vec![1.0f32; 8];
        let l = vec![1u32; 5]; // wrong
        let atlas_images: Vec<&[f32]> = vec![&a];
        let atlas_labels: Vec<&[u32]> = vec![&l];
        let config = LabelFusionConfig::default();
        let err = joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config)
            .unwrap_err();
        assert!(
            matches!(err, RegistrationError::DimensionMismatch(_)),
            "expected DimensionMismatch, got {:?}",
            err
        );
    }

    // ── solve_linear_system unit tests ───────────────────────────────────

    /// 2×2 identity system: Ix = [3, 7] → x = [3, 7].
    #[test]
    fn solve_identity_2x2() {
        let mut a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut b = vec![3.0, 7.0];
        let x = solve_linear_system(&mut a, &mut b).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-12, "x[0] = {}", x[0]);
        assert!((x[1] - 7.0).abs() < 1e-12, "x[1] = {}", x[1]);
    }

    /// 2×2 system: [[2, 1], [1, 3]] x = [5, 10] → x = [5/5, 15/5] = [1, 3].
    ///
    /// Verification: 2·1 + 1·3 = 5 ✓, 1·1 + 3·3 = 10 ✓.
    #[test]
    fn solve_2x2_known() {
        let mut a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let mut b = vec![5.0, 10.0];
        let x = solve_linear_system(&mut a, &mut b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12, "x[0] = {}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-12, "x[1] = {}", x[1]);
    }

    /// Singular 2×2 system: [[1, 1], [1, 1]] x = [1, 1] → None.
    #[test]
    fn solve_singular_returns_none() {
        let mut a = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let mut b = vec![1.0, 1.0];
        assert!(solve_linear_system(&mut a, &mut b).is_none());
    }

    /// 1×1 system: [4] x = [8] → x = 2.
    #[test]
    fn solve_1x1() {
        let mut a = vec![vec![4.0]];
        let mut b = vec![8.0];
        let x = solve_linear_system(&mut a, &mut b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12, "x[0] = {}", x[0]);
    }
}
