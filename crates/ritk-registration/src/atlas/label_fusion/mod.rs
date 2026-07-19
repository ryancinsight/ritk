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
//!    dᵢ(x) = —–P_{Aᵢ}(x) − P_T(x)—–² = Σ_{q ∈ P(x)} (Aᵢ(q) − T(q))²
//!
//! 2. Build pairwise similarity matrix M(x) ∈ ℝ^{N×N}:
//!    M_{ij}(x) = dᵢ(x) + dⱼ(x)
//!
//! 3. Regularize: M_{ij}(x) += α · δ_{ij}, where α = β · min_{ij}(M_{ij})
//!    and β is a user-specified parameter (default 0.1).
//!
//! 4. Solve for weights w(x): M w = 1 via Gaussian elimination with partial
//!    pivoting. Clamp negative weights to 0, then normalize: w = w / Σwᵢ.
//!    If the system is singular or all weights are non-positive, fall back to
//!    uniform weights wᵢ = 1/N.
//!
//! 5. Fused label: L(x) = argmax_l Σᵢ wᵢ(x) · \[Lᵢ(x) = l\]
//!    Ties are broken by selecting the smallest label value.
//!
//! # References
//!
//! - Wang, H., Suh, J. W., Das, S. R., Pluta, J. B., Craige, C. &
//!   Yushkevich, P. A. (2013). Multi-atlas segmentation with joint label
//!   fusion. *IEEE Trans. Pattern Analysis and Machine Intelligence*
//!   35(3):611–623.

mod jlf;
mod mv;

pub use jlf::joint_label_fusion;
pub use mv::majority_vote;

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
    /// Regularization factor β. The diagonal regularization added to the
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
// Shared validation helpers
// ---------------------------------------------------------------------------

/// Validate that `atlas_labels` is non-empty and every entry has length
/// `n_voxels`.
///
/// # Errors
///
/// - [`RegistrationError::InvalidConfiguration`] if `atlas_labels` is empty.
/// - [`RegistrationError::DimensionMismatch`] if any entry length differs
///   from `n_voxels`.
pub(crate) fn validate_atlas_labels(
    atlas_labels: &[&[u32]],
    n_voxels: usize,
) -> Result<(), RegistrationError> {
    if atlas_labels.is_empty() {
        return Err(RegistrationError::InvalidConfiguration(
            "atlas_labels is empty; at least one atlas is required".into(),
        ));
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
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../tests_label_fusion/mod.rs"]
mod tests_label_fusion;
