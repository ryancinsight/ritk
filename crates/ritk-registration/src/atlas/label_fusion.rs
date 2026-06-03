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
        for label_map in atlas_labels {
            *counts.entry(label_map[v]).or_insert(0) += 1;
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
        for (offset, row_data) in a[(col + 1)..].iter().enumerate() {
            let v = row_data[col].abs();
            if v > max_val {
                max_val = v;
                max_row = col + 1 + offset;
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
            // Split borrow: top = a[..row] (contains a[col]), bottom = a[row..].
            // bottom[0] = a[row]; col < row always holds here.
            let (top, bottom) = a.split_at_mut(row);
            for (a_row_j, &a_col_j) in bottom[0][col..].iter_mut().zip(top[col][col..].iter()) {
                *a_row_j -= factor * a_col_j;
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
#[path = "tests_label_fusion/mod.rs"]
mod tests_label_fusion;
