//! Nyúl-Udupa piecewise-linear histogram standardization.
//!
//! # Mathematical Specification
//!
//! Given K training images {I₁, …, I_K} and a set of target percentile ranks
//! P = {p₁, p₂, …, p_M} (e.g. {1, 10, 20, …, 90, 99}):
//!
//! ## Training phase (`learn_standard`)
//!
//! 1. For each training image Iₖ, compute the intensity landmarks:
//!
//!      Lₖ = [ Q(Iₖ, p₁), Q(Iₖ, p₂), …, Q(Iₖ, p_M) ]
//!
//!    where Q(I, p) is the p-th percentile of the image intensities, computed
//!    via linear interpolation on the sorted value array.
//!
//! 2. Compute the standard (average) landmark vector:
//!
//!      S_j = (1/K) · Σₖ Lₖⱼ,   j = 1, …, M
//!
//! ## Transform phase (`apply`)
//!
//! For a new image I with landmarks L = [ Q(I, p₁), …, Q(I, p_M) ]:
//!
//! 1. For each voxel intensity v, find the interval [Lⱼ, Lⱼ₊₁] containing v.
//! 2. Apply piecewise-linear interpolation to map v → v':
//!
//!      v' = Sⱼ + (v − Lⱼ) · (Sⱼ₊₁ − Sⱼ) / (Lⱼ₊₁ − Lⱼ)
//!
//! 3. Values below L₁ are clamped to S₁; values above L_M are clamped to S_M.
//!
//! ## Percentile computation
//!
//! The p-th percentile (p ∈ [0, 100]) of a sorted array V of length n is
//! computed via linear interpolation:
//!
//!   rank = p / 100 · (n − 1)
//!   lo   = ⌊rank⌋,  hi = ⌈rank⌉
//! Q = V\[lo\] + (rank − lo) · (V\[hi\] − V\[lo\])
//!
//! # References
//!
//! - Nyúl, L. G., Udupa, J. K., & Zhang, X. (2000). New variants of a method
//!   of MRI scale standardization. *IEEE Trans. Med. Imaging*, 19(2), 143–150.

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

// ── Percentile Helper ─────────────────────────────────────────────────────────

/// Compute the p-th percentile of a sorted slice via linear interpolation.
///
/// # Arguments
/// * `sorted` – Non-empty slice sorted in non-decreasing order.
/// * `p`      – Percentile rank in \[0, 100\].
///
/// # Formula
/// ```text
/// rank = p / 100 · (n − 1)
/// Q    = sorted[⌊rank⌋] + (rank − ⌊rank⌋) · (sorted[⌈rank⌉] − sorted[⌊rank⌋])
/// ```
///
/// # Panics
/// Panics if `sorted` is empty.
fn compute_percentile(sorted: &[f32], p: f64) -> f32 {
    assert!(
        !sorted.is_empty(),
        "compute_percentile requires non-empty input"
    );
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = p / 100.0 * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil().min((n - 1) as f64) as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = (rank - lo as f64) as f32;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

// ── Piecewise-Linear Mapping ──────────────────────────────────────────────────

/// Apply piecewise-linear mapping from `source_landmarks` to `target_landmarks`.
///
/// Values below `source_landmarks[0]` clamp to `target_landmarks[0]`.
/// Values above `source_landmarks[last]` clamp to `target_landmarks[last]`.
/// Within each interval \[Lⱼ, Lⱼ₊₁\], linearly interpolates to \[Sⱼ, Sⱼ₊₁\].
#[inline]
fn piecewise_linear_map(value: f32, source_landmarks: &[f32], target_landmarks: &[f32]) -> f32 {
    debug_assert_eq!(source_landmarks.len(), target_landmarks.len());
    let m = source_landmarks.len();

    // Clamp below first landmark.
    if value <= source_landmarks[0] {
        return target_landmarks[0];
    }
    // Clamp above last landmark.
    if value >= source_landmarks[m - 1] {
        return target_landmarks[m - 1];
    }

    // Find the interval [Lⱼ, Lⱼ₊₁] containing value via linear scan.
    // For typical landmark counts (≤ 11 entries), linear scan is faster
    // than binary search due to branch prediction and cache locality.
    for j in 0..m - 1 {
        if value <= source_landmarks[j + 1] {
            let denom = source_landmarks[j + 1] - source_landmarks[j];
            if denom.abs() < f32::EPSILON {
                // Degenerate interval: source landmarks coincide.
                return target_landmarks[j];
            }
            let t = (value - source_landmarks[j]) / denom;
            return target_landmarks[j] + t * (target_landmarks[j + 1] - target_landmarks[j]);
        }
    }

    // Fallback (unreachable for well-formed inputs).
    target_landmarks[m - 1]
}

// ── Nyúl-Udupa Normalizer ─────────────────────────────────────────────────────

/// Nyúl-Udupa piecewise-linear histogram standardization normalizer.
///
/// Two-phase normalizer:
/// 1. **Training** (`learn_standard`): learns average intensity landmarks from
///    a set of training images.
/// 2. **Application** (`apply`): maps a new image's intensity landmarks to the
///    learned standard via piecewise-linear interpolation.
///
/// # Reference
/// Nyúl, L. G., Udupa, J. K., & Zhang, X. (2000). New variants of a method
/// of MRI scale standardization. *IEEE Trans. Med. Imaging*, 19(2), 143–150.
pub struct NyulUdupaNormalizer {
    /// Percentile ranks used as landmarks (values in \[0, 100\]).
    /// Default: \[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99\].
    pub percentiles: Vec<f64>,
    /// Learned standard landmark intensities.
    /// `None` before `learn_standard` has been called.
    pub standard_landmarks: Option<Vec<f32>>,
}

impl NyulUdupaNormalizer {
    /// Create a normalizer with default percentile landmarks
    /// \[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99\].
    pub fn new() -> Self {
        Self {
            percentiles: vec![
                1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 99.0,
            ],
            standard_landmarks: None,
        }
    }

    /// Create a normalizer with custom percentile landmarks.
    ///
    /// # Arguments
    /// * `percentiles` – Percentile ranks in \[0, 100\], must be sorted in
    ///   strictly ascending order with at least 2 entries.
    ///
    /// # Panics
    /// Panics if `percentiles` has fewer than 2 entries or is not strictly ascending.
    pub fn with_percentiles(percentiles: Vec<f64>) -> Self {
        assert!(
            percentiles.len() >= 2,
            "at least 2 percentile landmarks required, got {}",
            percentiles.len()
        );
        for i in 1..percentiles.len() {
            assert!(
                percentiles[i] > percentiles[i - 1],
                "percentiles must be strictly ascending: p[{}]={} <= p[{}]={}",
                i,
                percentiles[i],
                i - 1,
                percentiles[i - 1]
            );
        }
        Self {
            percentiles,
            standard_landmarks: None,
        }
    }

    /// Learn the standard intensity landmarks by averaging per-image landmarks
    /// across all training images.
    ///
    /// # Algorithm
    /// For each image Iₖ:
    /// 1. Extract voxel intensities and sort.
    /// 2. Compute landmarks Lₖ = \[Q(Iₖ, p₁), …, Q(Iₖ, p_M)\].
    ///
    /// Standard landmarks: Sⱼ = (1/K) · Σₖ Lₖⱼ.
    ///
    /// # Panics
    /// Panics if `images` is empty.
    pub fn learn_standard<B: Backend, const D: usize>(&mut self, images: &[&Image<B, D>]) {
        assert!(!images.is_empty(), "at least one training image required");

        let m = self.percentiles.len();
        let k = images.len();
        let mut sum_landmarks = vec![0.0f64; m];

        for image in images {
            let (values_vec, _) = extract_vec_infallible(*image);
            let mut values = values_vec;
            crate::sort_floats(&mut values);

            for (j, &p) in self.percentiles.iter().enumerate() {
                sum_landmarks[j] += compute_percentile(&values, p) as f64;
            }
        }

        self.standard_landmarks = Some(
            sum_landmarks
                .iter()
                .map(|&s| (s / k as f64) as f32)
                .collect(),
        );
    }

    /// Learn standard landmarks from Coeus-native images.
    pub fn learn_standard_native<B, const D: usize>(
        &mut self,
        images: &[&NativeImage<f32, B, D>],
    ) -> anyhow::Result<()>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        anyhow::ensure!(
            !images.is_empty(),
            "at least one training image is required"
        );

        let mut sum_landmarks = vec![0.0_f64; self.percentiles.len()];
        for image in images {
            let mut values = image.data_slice()?.to_vec();
            crate::sort_floats(&mut values);
            for (sum, &percentile) in sum_landmarks.iter_mut().zip(&self.percentiles) {
                *sum += f64::from(compute_percentile(&values, percentile));
            }
        }
        let count = images.len() as f64;
        self.standard_landmarks = Some(
            sum_landmarks
                .into_iter()
                .map(|sum| (sum / count) as f32)
                .collect(),
        );
        Ok(())
    }

    /// Apply the learned piecewise-linear mapping to a new image.
    ///
    /// Computes the input image's own landmarks, then maps each voxel intensity
    /// from the input landmark space to the standard landmark space via
    /// piecewise-linear interpolation.
    ///
    /// # Errors
    /// Returns `Err` if `learn_standard` has not been called (i.e.
    /// `standard_landmarks` is `None`).
    ///
    /// # Spatial metadata
    /// The output image preserves origin, spacing, and direction from the input.
    pub fn apply<B: Backend, const D: usize>(
        &self,
        image: &Image<B, D>,
    ) -> anyhow::Result<Image<B, D>> {
        let standard = self.standard_landmarks.as_ref().ok_or_else(|| {
            anyhow::anyhow!("standard landmarks not learned; call learn_standard before apply")
        })?;

        // ── 1. Extract and sort voxel intensities ─────────────────────────────
        let (mut values, dims) = extract_vec_infallible(image);
        let mut sorted = values.clone();
        crate::sort_floats(&mut sorted);

        // ── 2. Compute input image landmarks ──────────────────────────────────
        let source_landmarks: Vec<f32> = self
            .percentiles
            .iter()
            .map(|&p| compute_percentile(&sorted, p))
            .collect();

        // ── 3. Apply piecewise-linear mapping ─────────────────────────────────
        for value in &mut values {
            *value = piecewise_linear_map(*value, &source_landmarks, standard);
        }

        // ── 4. Reconstruct image ──────────────────────────────────────────────
        Ok(rebuild(values, dims, image))
    }

    /// Apply learned landmarks to a Coeus-native image.
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &NativeImage<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<NativeImage<f32, B, D>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let standard = self.standard_landmarks.as_deref().ok_or_else(|| {
            anyhow::anyhow!("standard landmarks not learned; call learn_standard before apply")
        })?;
        let values = image.data_slice()?;
        let mut sorted = values.to_vec();
        crate::sort_floats(&mut sorted);
        let source_landmarks = self
            .percentiles
            .iter()
            .map(|&percentile| compute_percentile(&sorted, percentile))
            .collect::<Vec<_>>();
        let output = values
            .iter()
            .map(|&value| piecewise_linear_map(value, &source_landmarks, standard))
            .collect();
        NativeImage::from_flat_on(
            output,
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }
}

impl Default for NyulUdupaNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests_nyul_udupa.rs"]
mod tests_nyul_udupa;
