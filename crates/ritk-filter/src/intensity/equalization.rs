//! Global and adaptive histogram equalization filters.
//!
//! # Mathematical Specification
//!
//! ## Global Histogram Equalization
//!
//! For an image with N pixels and intensity range [v_min, v_max]:
//!
//! 1. **Binning**: `bin(p) = floor((p - v_min) / span * (B - 1))`, clamped to `[0, B-1]`.
//! 2. **Histogram**: `H[b] = |{p : bin(p) = b}|`, with `Σ H[b] = N`.
//! 3. **CDF**: `F[b] = Σ_{i=0}^{b} H[i]` (cumulative histogram count).
//! 4. **Normalised CDF**: `f[b] = F[b] / N`, with `f[B-1] = 1.0`.
//! 5. **Mapping**: `output(p) = v_min + f[bin(p)] * span`.
//!
//! Output invariant: `output ∈ [v_min, v_max]`.
//! When all pixels have the same value (span = 0), output equals input (identity).
//!
//! ## Relationship to CLAHE
//!
//! Global HE is equivalent to CLAHE with a single tile (`n_tiles = 1 × 1`) and
//! no clipping (`clip_limit â— —™ ∞`). CLAHE generalises global HE by applying it
//! locally per tile with clip-limiting to reduce over-enhancement of noise.
//!
//! # References
//!
//! - Gonzalez & Woods (2018). *Digital Image Processing*, 4th ed. Chapter 3.
//! - ITK: `HistogramEqualizationImageFilter`.
//! - ImageJ: Process â— —™ Enhance Contrast (Equalize Histogram).

use anyhow::Result;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Global histogram equalization filter.
///
/// Applies histogram equalization globally to the entire 3-D volume, mapping
/// each voxel intensity to the CDF-normalised value in `[v_min, v_max]`.
///
/// Unlike CLAHE, this is a global operation that does not distinguish between
/// spatial regions. It is computationally simpler but can over-enhance noise in
/// images with concentrated intensity distributions.
///
/// # Parameters
/// - `bins`: number of histogram bins (default 256).
///
/// # Complexity
/// O(N × log(B)) where N is the voxel count and B is the bin count.
pub struct HistogramEqualizationFilter {
    /// Number of histogram bins. Default 256.
    pub bins: usize,
}

impl HistogramEqualizationFilter {
    /// Create a new histogram equalization filter.
    ///
    /// # Arguments
    /// * `bins` ‗ number of histogram bins (minimum 2).
    pub fn new(bins: usize) -> Self {
        Self { bins: bins.max(2) }
    }

    /// Apply global histogram equalization to a 3-D image.
    ///
    /// All voxels are equalized together across the full volume. Spatial
    /// metadata (origin, spacing, direction) is preserved.
    ///
    /// # Errors
    /// Returns `Err` if the tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
        let (vals_vec, dims) = extract_vec_infallible(image);
        let vals = vals_vec;

        let out = histogram_equalize_global(&vals, self.bins);

        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`HistogramEqualizationFilter::apply`].
    ///
    /// Runs the identical global histogram equalization via the shared
    /// `histogram_equalize_global` host core on the image's contiguous host
    /// buffer, so the result is bitwise-identical to the Coeus path. No Coeus
    /// tensor is constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |vals, _dims| {
            histogram_equalize_global(vals, self.bins)
        })
    }
}

// ── Internal ──────────────────────────────────────────────────────────────

/// Apply global histogram equalization to a flat voxel array.
///
/// # Returns
/// New `Vec<f32>` of the same length with each value mapped through the global CDF.
///
/// # Invariant
/// Output values lie in `[v_min, v_max]`. When `span = 0`, output equals input.
pub(crate) fn histogram_equalize_global(vals: &[f32], bins: usize) -> Vec<f32> {
    let n = vals.len();
    if n == 0 {
        return Vec::new();
    }

    let (v_min, v_max) = {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in vals {
            if v.is_finite() {
                mn = mn.min(v);
                mx = mx.max(v);
            }
        }
        if mn.is_infinite() || mn >= mx {
            return vals.to_vec(); // All non-finite or uniform: identity.
        }
        (mn, mx)
    };

    let span = v_max - v_min;
    let mut hist = vec![0u64; bins];

    for &v in vals {
        if v.is_finite() {
            let normalized = ((v - v_min) / span).clamp(0.0, 1.0);
            let bin = ((normalized * (bins as f32 - 1.0)).floor() as usize).min(bins - 1);
            hist[bin] += 1;
        }
    }

    // Build normalised CDF lookup table.
    let mut cdf_lut = Vec::with_capacity(bins);
    let mut cumsum = 0u64;
    for &h in &hist {
        cumsum += h;
        cdf_lut.push(cumsum as f32 / n as f32);
    }

    // Map each value through the CDF.
    vals.iter()
        .map(|&v| {
            if !v.is_finite() {
                return v;
            }
            let normalized = ((v - v_min) / span).clamp(0.0, 1.0);
            let bin = ((normalized * (bins as f32 - 1.0)).floor() as usize).min(bins - 1);
            v_min + cdf_lut[bin].clamp(0.0, 1.0) * span
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_equalization.rs"]
mod tests;
