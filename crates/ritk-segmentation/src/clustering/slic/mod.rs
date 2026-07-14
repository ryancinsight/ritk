//! SLIC super-pixel segmentation (Achanta et al. 2012).
//!
//! # Mathematical Specification
//!
//! SLIC (Simple Linear Iterative Clustering) partitions an image into K
//! superpixels (2-D) or supervoxels (3-D) by local k-means clustering in a
//! combined intensity–spatial feature space.
//!
//! ## Feature vector
//!
//! For a D-dimensional image with voxel at spatial position **p** and
//! intensity I(**p**):
//!
//! F(**p**) = \[ I(**p**) / m_c , p₀ / m_s , … , p_{D-1} / m_s \]
//!
//! where:
//! - m_c = max intensity range (normalizes intensity to ~\[0,1\])
//! - m_s = √(N / K), N = total voxels, K = desired superpixels
//! - m (compactness, default 10.0) weights spatial vs. intensity proximity
//!
//! ## Distance metric
//!
//! D²(**p**, **c**) = ((I(**p**) − I_c) / m_c)²
//!                  + m² · Σ_{d=0}^{D-1} (p_d − c_d)² / m_s²
//!
//! where **c** is a cluster center with intensity I_c and spatial position c_d.
//!
//! ## Algorithm
//!
//! 1. **Initialize**: Place K cluster centers on a regular grid with step
//!    S_d = ⌊shape_d / K^(1/D)⌋ per axis. Perturb each center to the
//!    lowest-gradient voxel in a 3^D neighbourhood.
//! 2. **Assign**: For each voxel, find the nearest cluster center within a
//!    2S neighbourhood per axis using a grid-based search-window index.
//!    Amortized complexity: O(N · 2^D) per iteration (not O(N·K)).
//! 3. **Update**: Recompute each center as the mean of all assigned voxels.
//! 4. **Repeat** assignment + update for `max_iterations` or until
//!    convergence (max center shift < tolerance).
//! 5. **Enforce connectivity**: Relabel connected components smaller than
//!    `min_component_size` into the nearest large neighbor by intensity
//!    distance.
//!
//! ## Output
//!
//! A label image where each voxel contains its superpixel label (0..K−1) as
//! f32, preserving spatial metadata.
//!
//! # Complexity
//!
//! Grid initialization: O(N) for gradient computation.
//! Each iteration: O(N · 2^D) assignments + O(N) center updates.
//! Connectivity enforcement: O(N · α(N)) ≈ O(N).
//! Total: O(N · (2^D · I + α(N))) where I = iterations.
//!
//! # References
//!
//! - Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P. & Süsstrunk, S.
//!   (2012). "SLIC Superpixels Compared to State-of-the-Art Superpixel
//!   Methods." *IEEE Trans. Pattern Analysis and Machine Intelligence*,
//!   34(11):2274–2282.

pub mod assign;
pub mod connectivity;
pub mod coords;
pub mod gradient;
pub mod grid;
pub mod itk;
mod itk_filter;
pub use itk_filter::{
    ConnectivityEnforcement, InitializationPerturbation, ItkSlicConfig, ItkSlicFilter,
};

use ritk_core::image::Image;
use ritk_image::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_tensor_ops::extract_vec_infallible;

const MAX_EXACT_LABELS: usize = 1 << f32::MANTISSA_DIGITS;

// ── Public API ────────────────────────────────────────────────────────────────

/// SLIC super-pixel configuration.
#[derive(Debug, Clone)]
pub struct SlicConfig {
    /// Number of desired superpixels (cluster count K). Default: 100.
    n_superpixels: usize,
    /// Compactness parameter m: higher = more regular shapes. Default: 10.0.
    compactness: f32,
    /// Maximum iterations. Default: 10.
    max_iterations: usize,
    /// Convergence tolerance on center shift. Default: 1e-3.
    tolerance: f32,
    /// Minimum component size for connectivity enforcement. Default: 5.
    min_component_size: usize,
}

impl Default for SlicConfig {
    fn default() -> Self {
        Self {
            n_superpixels: 100,
            compactness: 10.0,
            max_iterations: 10,
            tolerance: 1e-3,
            min_component_size: 5,
        }
    }
}

impl SlicConfig {
    /// Create a `SlicConfig` with the given number of superpixels and defaults
    /// for all other fields.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested label count is zero or cannot be
    /// represented exactly by the `f32` output label contract.
    pub fn new(n_superpixels: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            n_superpixels >= 1,
            "SLIC superpixel count must be at least 1, got {n_superpixels}"
        );
        anyhow::ensure!(
            n_superpixels <= MAX_EXACT_LABELS,
            "SLIC superpixel count must not exceed {MAX_EXACT_LABELS}, got {n_superpixels}"
        );
        Ok(Self {
            n_superpixels,
            ..Self::default()
        })
    }

    /// Set the spatial compactness weight.
    pub fn with_compactness(mut self, compactness: f32) -> anyhow::Result<Self> {
        anyhow::ensure!(
            compactness.is_finite()
                && compactness >= 0.0
                && compactness <= f32::MAX.sqrt(),
            "SLIC compactness must be finite, nonnegative, and square-representable, got {compactness}"
        );
        self.compactness = compactness;
        Ok(self)
    }

    /// Set the maximum Lloyd iteration count.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            max_iterations >= 1,
            "SLIC maximum iterations must be at least 1, got {max_iterations}"
        );
        self.max_iterations = max_iterations;
        Ok(self)
    }

    /// Set the convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f32) -> anyhow::Result<Self> {
        anyhow::ensure!(
            tolerance.is_finite() && tolerance >= 0.0,
            "SLIC tolerance must be finite and nonnegative, got {tolerance}"
        );
        self.tolerance = tolerance;
        Ok(self)
    }

    /// Set the minimum connected-component size.
    #[must_use]
    pub fn with_min_component_size(mut self, min_component_size: usize) -> Self {
        self.min_component_size = min_component_size;
        self
    }
}

/// SLIC super-pixel segmentation filter.
///
/// Partitions an image into `config.n_superpixels` superpixels using the
/// SLIC algorithm. Output is a label image where each voxel carries its
/// superpixel index (0..K−1) as f32. Spatial metadata is preserved.
pub struct SlicSuperpixelFilter {
    config: SlicConfig,
}

impl SlicSuperpixelFilter {
    /// Create a new filter with the given configuration.
    pub fn new(config: SlicConfig) -> Self {
        Self { config }
    }

    /// Apply SLIC super-pixel segmentation to `image`.
    ///
    /// For a constant image (zero intensity range), all voxels are assigned
    /// label 0. For n_superpixels=1, all voxels are assigned label 0.
    ///
    /// # Errors
    ///
    /// Returns an error for ranks outside 2-D/3-D, zero extents, shape/sample
    /// mismatch or overflow, inexact coordinate bounds, non-finite samples, or
    /// an unrepresentable intensity range.
    pub fn apply<B: Backend, const D: usize>(
        &self,
        image: &Image<B, D>,
    ) -> anyhow::Result<Image<B, D>> {
        let (vals, shape) = extract_vec_infallible(image);
        validate_standard_input(&vals, &shape, D)?;
        let device = image.data().device();
        let ndim = D;
        let labels = slic_impl(&vals, &shape, ndim, &self.config);
        let tensor = Tensor::<B, D>::from_data(TensorData::new(labels, Shape::new(shape)), &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }

    /// Apply standard SLIC to a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns the validation errors documented by [`Self::apply`], or a
    /// backend storage/output construction error.
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::native::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let values = image.data_slice()?;
        validate_standard_input(values, &image.shape(), D)?;
        crate::native_output::from_values(
            image,
            slic_impl(values, &image.shape(), D, &self.config),
            backend,
        )
    }
}

/// Convenience function: apply SLIC with `n_superpixels` and default parameters.
///
/// # Errors
///
/// Returns an error for an invalid label count or any input error documented
/// by [`SlicSuperpixelFilter::apply`].
pub fn slic_superpixel<B: Backend, const D: usize>(
    image: &Image<B, D>,
    n_superpixels: usize,
) -> anyhow::Result<Image<B, D>> {
    SlicSuperpixelFilter::new(SlicConfig::new(n_superpixels)?).apply(image)
}

// ── Core implementation ────────────────────────────────────────────────────────

/// Core SLIC implementation operating on a flat f32 slice and dynamic shape.
///
/// Returns a `Vec<f32>` of cluster labels (0..K−1).
fn slic_impl(data: &[f32], shape: &[usize], ndim: usize, config: &SlicConfig) -> Vec<f32> {
    let n: usize = shape.iter().product();

    if n == 0 || config.n_superpixels == 0 {
        return vec![0.0_f32; n];
    }

    // Degenerate: K=1 → all label 0.
    let k = config.n_superpixels.min(n);
    if k <= 1 {
        return vec![0.0_f32; n];
    }

    // Intensity range for normalization.
    let (i_min, i_max) = data
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let intensity_range = i_max - i_min;

    // Constant image: all label 0.
    if intensity_range < 1e-10 {
        return vec![0.0_f32; n];
    }

    let m_c = intensity_range.max(1e-10);

    // Spatial normalization factor.
    let m_s = (n as f32 / k as f32).sqrt().max(1.0);

    // Grid step per axis.
    let k_root = (k as f32).powf(1.0 / ndim as f32);
    let steps: Vec<usize> = shape
        .iter()
        .map(|&s| ((s as f32) / k_root).floor().max(1.0) as usize)
        .collect();

    // ── Gradient computation ─────────────────────────────────────────────────
    let gradient = gradient::compute_gradient(data, shape, ndim, m_c);

    // ── Initialize cluster centers on a regular grid ─────────────────────────
    let mut centers = grid::init_centers(data, shape, ndim, &steps, &gradient, k);

    // ── Build grid-to-center mapping ─────────────────────────────────────────
    let grid_sizes: Vec<usize> = steps.iter().map(|&s| s.max(1)).collect();

    // ── Iterative assignment + update ─────────────────────────────────────────
    let mut labels = vec![0u32; n];
    let mut distances = vec![f32::MAX; n];
    let mut grid_map = Vec::new();

    for _iter in 0..config.max_iterations {
        distances.iter_mut().for_each(|d| *d = f32::MAX);

        assign::build_grid_map_into(&centers, &grid_sizes, shape, ndim, &mut grid_map);

        // ── Assignment step ──────────────────────────────────────────────────
        assign::assign_voxels(
            data,
            shape,
            ndim,
            &centers,
            &grid_map,
            &grid_sizes,
            m_c,
            m_s,
            config.compactness,
            &mut distances,
            &mut labels,
        );

        // ── Update step ──────────────────────────────────────────────────────
        let max_shift = assign::update_centers(&mut centers, data, &labels, shape, ndim, k);

        if max_shift < config.tolerance {
            break;
        }
    }

    // ── Final assignment (ensure labels match final centers) ──────────────────
    distances.iter_mut().for_each(|d| *d = f32::MAX);
    assign::build_grid_map_into(&centers, &grid_sizes, shape, ndim, &mut grid_map);
    assign::assign_voxels(
        data,
        shape,
        ndim,
        &centers,
        &grid_map,
        &grid_sizes,
        m_c,
        m_s,
        config.compactness,
        &mut distances,
        &mut labels,
    );

    // ── Connectivity enforcement ──────────────────────────────────────────────
    if config.min_component_size > 0 && ndim >= 2 {
        connectivity::enforce_connectivity(
            &mut labels,
            shape,
            ndim,
            data,
            config.min_component_size,
        );
    }

    labels.iter().map(|&l| l as f32).collect()
}

fn validate_standard_input(data: &[f32], shape: &[usize], ndim: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        (2..=3).contains(&ndim),
        "standard SLIC requires rank 2 or 3, got {ndim}"
    );
    anyhow::ensure!(
        shape.iter().all(|&extent| extent > 0),
        "standard SLIC requires nonzero dimensions, got {shape:?}"
    );
    let expected = shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| anyhow::anyhow!("standard SLIC shape product overflows usize: {shape:?}"))?;
    anyhow::ensure!(
        shape.iter().all(|&extent| extent <= MAX_EXACT_LABELS),
        "standard SLIC dimensions must not exceed {MAX_EXACT_LABELS} for exact f32 coordinates, got {shape:?}"
    );
    anyhow::ensure!(
        expected == data.len(),
        "standard SLIC shape {shape:?} requires {expected} samples, got {}",
        data.len()
    );
    if let Some((index, value)) = data
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("standard SLIC sample at flat index {index} must be finite, got {value}");
    }
    if let Some((&first, rest)) = data.split_first() {
        let (minimum, maximum) = rest
            .iter()
            .copied()
            .fold((first, first), |(low, high), value| {
                (low.min(value), high.max(value))
            });
        anyhow::ensure!(
            (maximum - minimum).is_finite(),
            "standard SLIC sample range must be representable in f32, got [{minimum}, {maximum}]"
        );
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_slic.rs"]
mod tests_slic;
