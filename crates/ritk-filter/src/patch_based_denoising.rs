//! Non-local means (patch-based) denoising filter for 3-D volumes.
//!
//! # Algorithm
//!
//! Approximates `itk::PatchBasedDenoisingImageFilter` with
//! `KernelBandwidthEstimation = false` and deterministic grid sampling.
//!
//! 1. Estimate noise via the MAD estimator: σ = median(|I − median(I)|) / 0.6745
//! 2. Set bandwidth h² = σ² (fixed bandwidth path; `kernel_bandwidth_estimation = false`)
//! 3. Sample `number_of_sample_patches` reference positions on a deterministic grid with
//!    stride = max(1, ⌊N / number_of_sample_patches⌋)
//! 4. For each voxel **p**:
//!    - Extract comparison patch P_p with radius `patch_radius` (clamped boundaries)
//!    - For each reference **q**: d_pq = ‖P_p − P_q‖² / patch_volume
//!    - w_pq = exp(−max(d_pq − h², 0) / h²)
//!    - `Output[p] = Σ w_pq · input[q] / Σ w_pq`
//! 5. Repeat for `number_of_iterations` passes.
//!
//! Deterministic grid sampling (instead of ITK's random sampler) ensures
//! reproducible output while achieving the same denoising quality in
//! expectation.
//!
//! # Complexity
//!
//! O(N · n_samples · (2r+1)³) per iteration, where N is the total voxel
//! count, `n_samples = number_of_sample_patches`, and `r = patch_radius`.
//!
//! # References
//!
//! - Buades, A., Coll, B., & Morel, J.-M. (2005). "A Non-Local Algorithm for
//!   Image Denoising." *CVPR*.
//! - Awate, S. P. & Whitaker, R. T. (2006). "Unsupervised, Information-Theoretic,
//!   Adaptive Image Filtering for Image Restoration." *IEEE TPAMI*.

use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

// ── Public API ─────────────────────────────────────────────────────────────────

/// Non-local means (patch-based) denoising for 3-D scalar volumes.
///
/// Approximates `itk::PatchBasedDenoisingImageFilter` with
/// `KernelBandwidthEstimation = false`. Uses deterministic grid sampling for
/// reproducibility.
///
/// # Default parameters
///
/// | Field | Default |
/// |-------|---------|
/// | `number_of_iterations` | 1 |
/// | `number_of_sample_patches` | 200 |
/// | `patch_radius` | 4 |
/// | `kernel_bandwidth_estimation` | false |
#[derive(Debug, Clone)]
pub struct PatchBasedDenoisingImageFilter {
    /// Number of NL-means passes.
    pub number_of_iterations: usize,
    /// Number of reference patch positions sampled per voxel.
    pub number_of_sample_patches: usize,
    /// Half-width of the comparison patch in voxels.
    pub patch_radius: usize,
    /// Ignored when `false` (fixed MAD-based bandwidth). When `true` the same
    /// MAD bandwidth is used; full iterative bandwidth estimation is not
    /// implemented.
    pub kernel_bandwidth_estimation: bool,
}

impl Default for PatchBasedDenoisingImageFilter {
    fn default() -> Self {
        Self {
            number_of_iterations: 1,
            number_of_sample_patches: 200,
            patch_radius: 4,
            kernel_bandwidth_estimation: false,
        }
    }
}

impl PatchBasedDenoisingImageFilter {
    /// Apply patch-based NL-means denoising to a 3-D image.
    ///
    /// Spatial metadata (origin, spacing, direction) is preserved from `image`.
    ///
    /// # Errors
    /// Returns `Err` if the underlying tensor data cannot be read as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (data, dims) = extract_vec(image)?;
        let result = self.run(&data, dims);
        Ok(rebuild(result, dims, image))
    }
}

// ── Core algorithm ─────────────────────────────────────────────────────────────

impl PatchBasedDenoisingImageFilter {
    fn run(&self, data: &[f32], dims: [usize; 3]) -> Vec<f32> {
        let n = dims[0] * dims[1] * dims[2];
        if n == 0 {
            return Vec::new();
        }
        let mut current = data.to_vec();
        for _ in 0..self.number_of_iterations {
            current = nl_means_pass(
                &current,
                dims,
                self.number_of_sample_patches,
                self.patch_radius,
            );
        }
        current
    }
}

/// Single NL-means pass on a Z×Y×X flat `f32` volume.
fn nl_means_pass(
    data: &[f32],
    dims: [usize; 3],
    n_samples: usize,
    patch_radius: usize,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    // MAD noise estimate: σ = median(|I − median(I)|) / 0.6745.
    let sigma = estimate_noise_mad(data);
    // h² = σ²; guard against zero.
    let h2 = (sigma * sigma).max(1e-30_f64);

    // Deterministic grid sampling with stride = max(1, ⌊N/n_samples⌋).
    let stride = ((n as f64 / n_samples as f64).floor() as usize).max(1);
    let samples: Vec<usize> = (0..n).step_by(stride).collect();

    let r = patch_radius as isize;
    let pv = (2 * r + 1).pow(3) as f64; // patch volume

    let mut output = vec![0.0_f32; n];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let p = iz * ny * nx + iy * nx + ix;
                let mut w_sum = 0.0_f64;
                let mut val_sum = 0.0_f64;

                for &q_idx in &samples {
                    let qz = (q_idx / (ny * nx)) as isize;
                    let qy = ((q_idx / nx) % ny) as isize;
                    let qx = (q_idx % nx) as isize;

                    // Mean squared patch distance: d = ‖P_p − P_q‖² / patch_volume.
                    let sq = patch_sq_distance(
                        data,
                        dims,
                        [iz as isize, iy as isize, ix as isize],
                        [qz, qy, qx],
                        r,
                    );
                    let d = sq / pv;

                    // w = exp(−max(d − h², 0) / h²).
                    let excess = d - h2;
                    let w = if excess > 0.0 {
                        (-excess / h2).exp()
                    } else {
                        1.0_f64
                    };

                    w_sum += w;
                    val_sum += w * data[q_idx] as f64;
                }

                output[p] = if w_sum > 1e-20 {
                    (val_sum / w_sum) as f32
                } else {
                    data[p]
                };
            }
        }
    }

    output
}

/// Squared Euclidean patch distance ‖P_p − P_q‖² (sum over all patch offsets).
///
/// Both patches use clamped boundary conditions when any offset falls outside
/// the volume.
#[inline]
fn patch_sq_distance(
    data: &[f32],
    dims: [usize; 3],
    p: [isize; 3],
    q: [isize; 3],
    r: isize,
) -> f64 {
    let [nz, ny, nx] = dims;
    let (nzi, nyi, nxi) = (nz as isize, ny as isize, nx as isize);
    let [pz, py, px] = p;
    let [qz, qy, qx] = q;
    let mut dist = 0.0_f64;

    for dz in -r..=r {
        let pzc = (pz + dz).clamp(0, nzi - 1) as usize;
        let qzc = (qz + dz).clamp(0, nzi - 1) as usize;
        for dy in -r..=r {
            let pyc = (py + dy).clamp(0, nyi - 1) as usize;
            let qyc = (qy + dy).clamp(0, nyi - 1) as usize;
            for dx in -r..=r {
                let pxc = (px + dx).clamp(0, nxi - 1) as usize;
                let qxc = (qx + dx).clamp(0, nxi - 1) as usize;

                let vp = data[pzc * ny * nx + pyc * nx + pxc] as f64;
                let vq = data[qzc * ny * nx + qyc * nx + qxc] as f64;
                dist += (vp - vq) * (vp - vq);
            }
        }
    }

    dist
}

/// MAD noise estimator: σ = median(|I − median(I)|) / 0.6745.
///
/// Returns 1.0 for an empty or constant image (avoids zero bandwidth).
fn estimate_noise_mad(data: &[f32]) -> f64 {
    if data.is_empty() {
        return 1.0;
    }

    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let med = if n & 1 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) as f64 * 0.5
    } else {
        sorted[n / 2] as f64
    };

    let mut devs: Vec<f64> = data.iter().map(|&v| (v as f64 - med).abs()).collect();
    devs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if n & 1 == 0 {
        (devs[n / 2 - 1] + devs[n / 2]) * 0.5
    } else {
        devs[n / 2]
    };

    let sigma = mad / 0.6745;
    // Guard: return 1.0 for constant inputs to keep h² non-zero and avoid
    // degenerate uniform weighting across all samples.
    if sigma < 1e-15 {
        1.0
    } else {
        sigma
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_patch_based_denoising.rs"]
mod tests;
