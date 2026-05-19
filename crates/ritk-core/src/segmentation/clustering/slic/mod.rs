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
//! F(**p**) = [ I(**p**) / m_c , p₀ / m_s , … , p_{D-1} / m_s ]
//!
//! where:
//! - m_c = max intensity range (normalizes intensity to ~[0,1])
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

use crate::filter::ops::extract_vec_infallible;
use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

// ── Public API ────────────────────────────────────────────────────────────────

/// SLIC super-pixel configuration.
#[derive(Debug, Clone)]
pub struct SlicConfig {
    /// Number of desired superpixels (cluster count K). Default: 100.
    pub n_superpixels: usize,
    /// Compactness parameter m: higher = more regular shapes. Default: 10.0.
    pub compactness: f64,
    /// Maximum iterations. Default: 10.
    pub max_iterations: usize,
    /// Convergence tolerance on center shift. Default: 1e-3.
    pub tolerance: f64,
    /// Deterministic seed (reserved for future stochastic extensions). Default: 42.
    pub seed: u64,
    /// Minimum component size for connectivity enforcement. Default: 5.
    pub min_component_size: usize,
}

impl Default for SlicConfig {
    fn default() -> Self {
        Self {
            n_superpixels: 100,
            compactness: 10.0,
            max_iterations: 10,
            tolerance: 1e-3,
            seed: 42,
            min_component_size: 5,
        }
    }
}

impl SlicConfig {
    /// Create a `SlicConfig` with the given number of superpixels and defaults
    /// for all other fields.
    pub fn new(n_superpixels: usize) -> Self {
        Self {
            n_superpixels,
            ..Self::default()
        }
    }
}

/// SLIC super-pixel segmentation filter.
///
/// Partitions an image into `config.n_superpixels` superpixels using the
/// SLIC algorithm. Output is a label image where each voxel carries its
/// superpixel index (0..K−1) as f32. Spatial metadata is preserved.
pub struct SlicSuperpixelFilter {
    pub config: SlicConfig,
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
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let (vals, shape) = extract_vec_infallible(image);
        let device = image.data().device();
        let ndim = D;
        let labels = slic_impl(&vals, &shape, ndim, &self.config);
        let tensor = Tensor::<B, D>::from_data(
            TensorData::new(labels, Shape::new(shape)),
            &device,
        );
        Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        )
    }
}

/// Convenience function: apply SLIC with `n_superpixels` and default parameters.
pub fn slic_superpixel<B: Backend, const D: usize>(
    image: &Image<B, D>,
    n_superpixels: usize,
) -> Image<B, D> {
    SlicSuperpixelFilter::new(SlicConfig::new(n_superpixels)).apply(image)
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

    let intensities: Vec<f64> = data.iter().map(|&v| v as f64).collect();

    // Intensity range for normalization.
    let (i_min, i_max) = intensities
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let intensity_range = i_max - i_min;

    // Constant image: all label 0.
    if intensity_range < 1e-10 {
        return vec![0.0_f32; n];
    }

    let m_c = intensity_range.max(1e-10);

    // Spatial normalization factor.
    let m_s = (n as f64 / k as f64).sqrt().max(1.0);

    // Grid step per axis.
    let k_root = (k as f64).powf(1.0 / ndim as f64);
    let steps: Vec<usize> = shape
        .iter()
        .map(|&s| ((s as f64) / k_root).floor().max(1.0) as usize)
        .collect();

    // ── Gradient computation ─────────────────────────────────────────────────
    let gradient = gradient::compute_gradient(&intensities, shape, ndim);

    // ── Initialize cluster centers on a regular grid ─────────────────────────
    let mut centers = grid::init_centers(&intensities, shape, ndim, &steps, &gradient, k);

    // ── Build grid-to-center mapping ─────────────────────────────────────────
    let grid_sizes: Vec<usize> = steps.iter().map(|&s| s.max(1)).collect();

    // ── Iterative assignment + update ─────────────────────────────────────────
    let mut labels = vec![0u32; n];
    let mut distances = vec![f64::MAX; n];

    for _iter in 0..config.max_iterations {
        distances.iter_mut().for_each(|d| *d = f64::MAX);

        let grid_map = assign::build_grid_map(&centers, &grid_sizes, shape, ndim);

        // ── Assignment step ──────────────────────────────────────────────────
        assign::assign_voxels(
            &intensities,
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
        let max_shift = assign::update_centers(
            &mut centers, &intensities, &labels, shape, ndim, k,
        );

        if max_shift < config.tolerance {
            break;
        }
    }

    // ── Final assignment (ensure labels match final centers) ──────────────────
    distances.iter_mut().for_each(|d| *d = f64::MAX);
    let grid_map = assign::build_grid_map(&centers, &grid_sizes, shape, ndim);
    assign::assign_voxels(
        &intensities,
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
            &mut labels, shape, ndim, &intensities, config.min_component_size,
        );
    }

    labels.iter().map(|&l| l as f32).collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_slic.rs"]
mod tests_slic;
