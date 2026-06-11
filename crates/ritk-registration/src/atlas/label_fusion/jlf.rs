//! Joint Label Fusion (Wang et al. 2013).

use std::collections::HashMap;

use crate::error::RegistrationError;

use super::{validate_atlas_labels, LabelFusionConfig, LabelFusionResult};

/// Joint Label Fusion (Wang et al. 2013).
///
/// Computes per-voxel atlas weights from local patch similarity between the
/// warped atlas intensity images and the target, then performs weighted voting
/// over the atlas label maps. See the [module-level documentation](super) for
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
    validate_atlas_labels(atlas_labels, n_voxels)?;

    let r = config.patch_radius;
    let mut labels = vec![0u32; n_voxels];
    let mut confidence = vec![0.0f32; n_voxels];

    // Pre-allocate per-voxel working buffers outside the hot loop.
    let mut d = vec![0.0f64; n_atlases];
    let mut m_flat = vec![0.0f64; n_atlases * n_atlases];
    let mut rhs = vec![0.0f64; n_atlases];
    // Capacity: bounded by the number of distinct labels across atlases (≤ n_atlases)
    let mut label_weights: HashMap<u32, f64> = HashMap::with_capacity(n_atlases);

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
                for v in rhs.iter_mut().take(n) {
                    *v = 1.0;
                }
                let raw_weights = solve_linear_system(&mut m_flat[..n * n], n, &mut rhs[..n]);

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
/// pivoting. `a` is a row-major flat buffer of length `n*n`, modified in
/// place. `b` is the RHS of length `n`, modified in place. Returns `Some(x)`
/// on success or `None` if singular (pivot < 1e-15).
pub(crate) fn solve_linear_system(a: &mut [f64], n: usize, b: &mut [f64]) -> Option<Vec<f64>> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n);

    // Forward elimination with partial pivoting.
    for col in 0..n {
        // Find pivot row.
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None; // Singular.
        }
        // Swap rows col and max_row in the flat slice.
        if max_row != col {
            for k in 0..n {
                a.swap(col * n + k, max_row * n + k);
            }
            b.swap(col, max_row);
        }
        // Eliminate below. Split borrow so that the pivot row (above) and the
        // current row (below) can be accessed simultaneously without aliasing.
        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            let (above, below) = a.split_at_mut(row * n);
            let pivot_row = &above[col * n..col * n + n];
            let current_row = &mut below[..n];
            for k in col..n {
                current_row[k] -= factor * pivot_row[k];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * x[j];
        }
        x[i] = s / a[i * n + i];
    }

    Some(x)
}
