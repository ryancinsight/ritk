//! Topology-preserving skeletonization (thinning) for binary images.
//!
//! # Mathematical Specification
//!
//! Given a binary image M: ℤá´° → {0, 1}, skeletonization produces a thin
//! subset S ⊆ M that preserves the topology of M (same number of connected
//! components, tunnels, and cavities) while removing as many foreground
//! voxels as possible.
//!
//! ## Definition (Simple Point)
//!
//! A foreground point p is **simple** if its removal does not change the
//! topology of the image. For the (FG=26, BG=6) adjacency pair in 3-D:
//!
//! p is simple ⟺ T₂₆(p) = 1
//!
//! where T₂₆(p) is the number of 26-connected foreground components in
//! N₂₆(p) \ {p} (the 26-neighborhood excluding the center).
//!
//! **Proof sketch (Bertrand & Malandain, 1994):** For (26, 6) adjacency,
//! T₂₆(p) = 1 implies TÌ„₆(p) = 1 (exactly one 6-connected background
//! component 6-adjacent to p in N₁₈(p) \ {p}). Both conditions together
//! are necessary and sufficient for simplicity. Since the first implies the
//! second under (26, 6) adjacency, T₂₆(p) = 1 alone suffices.
//!
//! ## Definition (Endpoint)
//!
//! A foreground point p is an **endpoint** if it has at most one foreground
//! neighbor in its full connectivity neighborhood:
//! - D = 2: |{q ∈ N₈(p) \ {p} : M(q) = 1}| ≤ 1
//! - D = 3: |{q ∈ N₂₆(p) \ {p} : M(q) = 1}| ≤ 1
//!
//! Endpoints are never removed, ensuring the skeleton retains branch tips
//! and produces a **curve skeleton** (medial axis).
//!
//! ## Algorithm — D = 2: Zhang–Suen Thinning (1984)
//!
//! Two sub-iterations per pass on 8-connected foreground / 4-connected background.
//! Neighbors P₂..P₉ are labeled clockwise from north. Let:
//! - B(p) = Σ Pᵢ (count of foreground 8-neighbors)
//! - A(p) = number of 0→1 transitions in the cyclic sequence P₂..P₉,P₂
//!
//! **Sub-iteration 1:** Delete p if 2 ≤ B(p) ≤ 6, A(p) = 1,
//! P₂·P₄·P₆ = 0, and P₄·P₆·P₈ = 0.
//!
//! **Sub-iteration 2:** Delete p if 2 ≤ B(p) ≤ 6, A(p) = 1,
//! P₂·P₄·P₈ = 0, and P₂·P₆·P₈ = 0.
//!
//! Repeat until neither sub-iteration removes any pixel.
//!
//! ## Algorithm — D = 3: 6-Directional Sequential Thinning
//!
//! Each iteration comprises 6 sub-iterations (one per face direction:
//! ±z, ±y, ±x). In each sub-iteration:
//!
//! 1. Identify **border voxels**: foreground voxels whose face-neighbor in
//!    the current direction is background (or out of bounds).
//! 2. For each border voxel p (raster order), re-check:
//!    a. p is still foreground (may have been removed earlier in this pass).
//!    b. p is not an endpoint (|N₂₆(p) ∩ FG| > 1).
//!    c. p is a simple point (T₂₆(p) = 1).
//!    If all hold, delete p immediately (sequential deletion).
//! 3. Repeat until a full pass (all 6 directions) removes zero voxels.
//!
//! **Correctness:** Each individual deletion is of a verified simple point
//! in the current configuration, which preserves topology by definition.
//! Sequential re-checking ensures no two conflicting deletions occur.
//!
//! ## Algorithm — D = 1: Medial Point Extraction
//!
//! For each maximal connected foreground run [a, b], retain only the
//! midpoint ⌊(a + b) / 2⌋. This preserves one point per connected
//! component (topology) and selects the medial position.
//!
//! ## Complexity
//!
//! - D = 1: O(n) single pass.
//! - D = 2: O(n · k) where k is the number of passes (bounded by
//!   max(ny, nx) / 2).
//! - D = 3: O(n · k · 6) where k is the iteration count, plus O(26²)
//!   per simple-point test (constant).
//!
//! # References
//!
//! - Zhang, T.Y. & Suen, C.Y. (1984). "A Fast Parallel Algorithm for
//!   Thinning Digital Patterns." *Communications of the ACM*, 27(3), 236-239.
//! - Bertrand, G. & Malandain, G. (1994). "A New Characterization of
//!   Three-Dimensional Simple Points." *Pattern Recognition Letters*, 15(2),
//!   169-175.
//! - Lee, T.C., Kashyap, R.L. & Chu, C.N. (1994). "Building Skeleton
//!   Models via 3-D Medial Surface/Axis Thinning Algorithms." *CVGIP:
//!   Graphical Models and Image Processing*, 56(6), 462-478.
//! - Palágyi, K. & Kuba, A. (1999). "A Parallel 3D 12-Subiteration
//!   Thinning Algorithm." *Graphical Models and Image Processing*, 61(4),
//!   199-221.

mod thin_1d;
mod thin_2d;
mod thin_3d;
use ritk_core::image::Image;
use ritk_image::tensor::{Backend, Tensor};
use ritk_tensor_ops::extract_vec_infallible;

// ── Public types ─────────────────────────────────────────────────────────────

/// Topology-preserving skeletonization (thinning) filter.
///
/// Reduces a binary mask to its medial axis / curve skeleton while
/// preserving topology (connected components, tunnels, cavities).
/// Endpoints are preserved to retain branch structure.
///
/// Supports D = 1, 2, 3.
pub struct Skeletonization;

impl Skeletonization {
    /// Create a `Skeletonization` filter.
    pub fn new() -> Self {
        Self
    }

    /// Apply skeletonization to a Coeus-native mask.
    ///
    /// # Errors
    ///
    /// Returns an error for dimensions outside 1 through 3, non-finite mask
    /// samples, inaccessible backend storage, or output construction failure.
    pub fn apply_native<B, const D: usize>(
        &self,
        mask: &ritk_image::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        anyhow::ensure!(
            (1..=3).contains(&D),
            "skeletonization supports dimensions 1 through 3, got {D}"
        );
        let values = mask.data_slice()?;
        super::ensure_finite_mask(values)?;
        crate::native_output::from_values(mask, skeleton_nd(values, &mask.shape()), backend)
    }

    /// Apply skeletonization to a binary mask image.
    ///
    /// Returns a binary mask containing the skeleton (values in {0.0, 1.0})
    /// with the same shape and spatial metadata as `mask`.
    ///
    /// # Panics
    /// Panics if D is not 1, 2, or 3.
    pub fn apply<B: Backend, const D: usize>(&self, mask: &Image<f32, B, D>) -> Image<f32, B, D> {
        let shape: [usize; D] = mask.shape();
        let device = B::default();
        let (flat_vals, _shape) = extract_vec_infallible(mask);
        let flat: &[f32] = &flat_vals;
        let output = skeleton_nd(flat, &shape);
        let tensor = Tensor::<f32, B>::from_slice_on(shape, &output, &device);
        Image::new(tensor, *mask.origin(), *mask.spacing(), *mask.direction())
            .expect("invariant: segmentation output tensor preserves the image rank")
    }
}

impl Default for Skeletonization {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend, const D: usize> super::MorphologicalOperation<B, D> for Skeletonization {
    fn apply(&self, mask: &Image<f32, B, D>) -> Image<f32, B, D> {
        self.apply(mask)
    }
}

// ── Dispatcher ───────────────────────────────────────────────────────────────

fn skeleton_nd(flat: &[f32], shape: &[usize]) -> Vec<f32> {
    match shape.len() {
        1 => thin_1d::endpoint_extract(flat, shape[0]),
        2 => thin_2d::zhang_suen(flat, shape[0], shape[1]),
        3 => thin_3d::sequential_thin(flat, shape[0], shape[1], shape[2]),
        d => {
            panic!("Skeletonization: unsupported dimensionality D={d}; only D=1,2,3 are supported")
        }
    }
}

// Re-export for test access.
#[cfg(test)]
pub(crate) use thin_3d::fg_components_26;

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests_skeletonization;
