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
//! no clipping (`clip_limit → ∞`). CLAHE generalises global HE by applying it
//! locally per tile with clip-limiting to reduce over-enhancement of noise.
//!
//! # References
//!
//! - Gonzalez & Woods (2018). *Digital Image Processing*, 4th ed. Chapter 3.
//! - ITK: `HistogramEqualizationImageFilter`.
//! - ImageJ: Process → Enhance Contrast (Equalize Histogram).

use ritk_core::filter::ops::{extract_vec_infallible, rebuild};
use ritk_image::Image;
use anyhow::Result;
use burn::tensor::backend::Backend;

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
    /// * `bins` — number of histogram bins (minimum 2).
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
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec_infallible(image);
        let vals = vals_vec;

        let out = histogram_equalize_global(&vals, self.bins);

        Ok(rebuild(out, dims, image))
    }
}

// ── Internal ──────────────────────────────────────────────────────────────────

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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};

    use burn_ndarray::NdArray;
    type B = NdArray<f32>;
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};

    fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(data, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    // ── histogram_equalize_global ─────────────────────────────────────────────

    #[test]
    fn global_he_empty_input_returns_empty() {
        let out = histogram_equalize_global(&[], 256);
        assert!(out.is_empty());
    }

    #[test]
    fn global_he_uniform_input_is_identity() {
        // All pixels equal 50.0 → span = 0 → identity path → output = input.
        let vals = vec![50.0_f32; 16];
        let out = histogram_equalize_global(&vals, 256);
        for (i, (&inp, &outp)) in vals.iter().zip(out.iter()).enumerate() {
            assert!(
                (inp - outp).abs() < 1e-5,
                "index {i}: input={inp}, output={outp}"
            );
        }
    }

    #[test]
    fn global_he_output_in_input_range() {
        // Analytical: output ∈ [v_min, v_max] for all finite inputs.
        let vals: Vec<f32> = (0..64).map(|i| i as f32 * 3.0 - 50.0).collect();
        let v_min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let v_max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out = histogram_equalize_global(&vals, 256);
        for &o in &out {
            assert!(
                o >= v_min - 1e-4 && o <= v_max + 1e-4,
                "output {o} outside [{v_min}, {v_max}]"
            );
        }
    }

    #[test]
    fn global_he_last_output_is_vmax() {
        // For a strictly increasing ramp, the last pixel (max value) must map to v_max.
        let vals: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let out = histogram_equalize_global(&vals, 32);
        assert!(
            (out[31] - 31.0).abs() < 1.0,
            "last output = {}, expected ~31.0",
            out[31]
        );
    }

    #[test]
    fn global_he_first_output_is_near_vmin() {
        // The first value (minimum) maps to v_min + cdf[0]*span = v_min + (1/N)*span.
        // For N=16, bins=16, cdf[0] = 1/16 → output[0] = 0 + 1/16 * 15 ≈ 0.9375.
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect(); // [0, 1, ..., 15]
        let out = histogram_equalize_global(&vals, 16);
        // First pixel (v=0) maps through bin 0; cdf[0] = 1/16 → output ≈ 0.9375
        assert!(
            out[0] >= 0.0 && out[0] < 3.0,
            "first output = {}, expected small positive value",
            out[0]
        );
    }

    #[test]
    fn global_he_monotone_output_for_sorted_input() {
        // If input is sorted ascending, output must also be non-decreasing
        // (since the CDF mapping is non-decreasing).
        let vals: Vec<f32> = (0..64).map(|i| i as f32 * 1.5).collect();
        let out = histogram_equalize_global(&vals, 64);
        for i in 1..out.len() {
            assert!(
                out[i] >= out[i - 1] - 1e-5,
                "output not monotone at {i}: {:.4} < {:.4}",
                out[i],
                out[i - 1]
            );
        }
    }

    #[test]
    fn global_he_preserves_length() {
        let vals: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let out = histogram_equalize_global(&vals, 256);
        assert_eq!(out.len(), 100);
    }

    // ── HistogramEqualizationFilter::apply ────────────────────────────────────

    #[test]
    fn apply_preserves_shape_and_metadata() {
        let data: Vec<f32> = (0..4 * 16 * 16).map(|i| (i % 256) as f32).collect();
        let img = make_image(data, [4, 16, 16]);
        let origin = *img.origin();
        let spacing = *img.spacing();
        let direction = *img.direction();

        let filter = HistogramEqualizationFilter::new(256);
        let out = filter.apply(&img).expect("HE apply failed");

        assert_eq!(out.shape(), [4, 16, 16]);
        assert_eq!(out.origin(), &origin);
        assert_eq!(out.spacing(), &spacing);
        assert_eq!(out.direction(), &direction);
    }

    #[test]
    fn apply_uniform_volume_is_identity() {
        let data = vec![75.0_f32; 2 * 8 * 8];
        let img = make_image(data.clone(), [2, 8, 8]);
        let filter = HistogramEqualizationFilter::new(256);
        let out = filter.apply(&img).expect("HE apply failed");
        let (out_data, _) = extract_vec_infallible(&out);
        let out_vals: Vec<f32> = out_data.as_slice().to_vec();
        for (i, (&inp, &outp)) in data.iter().zip(out_vals.iter()).enumerate() {
            assert!(
                (inp - outp).abs() < 1e-4,
                "voxel {i}: input={inp}, output={outp}"
            );
        }
    }

    #[test]
    fn apply_output_in_global_range() {
        let data: Vec<f32> = (0..2 * 16 * 16).map(|i| (i % 200) as f32 - 50.0).collect();
        let v_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let v_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let img = make_image(data, [2, 16, 16]);
        let filter = HistogramEqualizationFilter::new(256);
        let out = filter.apply(&img).expect("HE apply failed");
        let (out_data, _) = extract_vec_infallible(&out);
        let out_vals: Vec<f32> = out_data.as_slice().to_vec();
        for &o in &out_vals {
            assert!(
                o >= v_min - 0.5 && o <= v_max + 0.5,
                "output {o} outside global range [{v_min}, {v_max}]"
            );
        }
    }
}
