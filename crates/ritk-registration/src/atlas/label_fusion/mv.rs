//! Majority voting label fusion (baseline).

use std::collections::HashMap;

use crate::error::RegistrationError;

use super::{validate_atlas_labels, LabelFusionResult};

/// Majority voting label fusion (baseline).
///
/// For each voxel the output label is the mode of the labels across all
/// atlases. Ties are broken by selecting the smallest label value.
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
    let [nz, ny, nx] = dims;
    let n_voxels = nz * ny * nx;
    validate_atlas_labels(atlas_labels, n_voxels)?;

    let mut labels = vec![0u32; n_voxels];
    let mut confidence = vec![0.0f32; n_voxels];

    // Capacity: bounded by the number of distinct labels across atlases (≤ n_atlases)
    let mut counts: HashMap<u32, usize> = HashMap::with_capacity(n_atlases);

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
