//! BinShrink image filter: integer sub-sampling with bin averaging.
//!
//! # Mathematical Specification
//!
//! BinShrink reduces image dimensions by integer factors by computing the
//! arithmetic mean of all voxels within each non-overlapping bin:
//!
//! ```text
//! O(o₀, o₁, …, o_{D-1}) = (1/N) · Σ_{b₀=0}^{f₀-1} Σ_{b₁=0}^{f₁-1} … Σ_{b_{D-1}=0}^{f_{D-1}-1}
//!     I(o₀·f₀ + b₀, o₁·f₁ + b₁, …, o_{D-1}·f_{D-1} + b_{D-1})
//!
//! where N = f₀ · f₁ · … · f_{D-1}  (bin volume)
//! ```
//!
//! Output shape per dimension: `out_shape[d] = floor(in_shape[d] / factor[d])`.
//! Remainder voxels at the trailing edge of each dimension are discarded.
//!
//! # Complexity
//!
//! O(V) where V = product of input shape. Each input voxel is visited exactly
//! once across all bin accumulations. The separable multi-pass approach
//! (shrinking one dimension at a time) achieves O(V) with cache-friendly
//! inner loops.
//!
//! # Separable Decomposition
//!
//! The D-dimensional bin average decomposes into a sequence of 1-D averages
//! along each axis. For each axis d with factor f_d:
//!
//! ```text
//! temp'(o_d, rest) = (1/f_d) · Σ_{b=0}^{f_d-1} temp(o_d·f_d + b, rest)
//! ```
//!
//! This is applied iteratively: first shrink axis 0, then axis 1 from the
//! intermediate result, etc. The composition of 1-D means equals the full
//! D-D mean by the linearity of expectation over disjoint partitions.
//!
//! # ITK Parity
//!
//! Equivalent to `itk::BinShrinkImageFilter`. Spacing is multiplied by the
//! shrink factor along each axis; origin and direction are preserved (the
//! physical location of the first voxel is unchanged).

use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_spatial::Spacing;
use ritk_tensor_ops::extract_vec_infallible;

/// BinShrink image filter: integer sub-sampling by bin averaging.
///
/// Reduces image dimensions by integer factors by averaging all voxels
/// within each non-overlapping bin. This provides anti-aliasing compared
/// to naive sub-sampling (which just takes every Nth voxel).
///
/// Output shape\[d\] = floor(input_shape\[d\] / factor\[d\])
/// Spacing\[d\] *= factor\[d\]
pub struct BinShrinkImageFilter {
    /// Shrink factors per dimension. If shorter than D, factors\[0\] is broadcast.
    pub factors: Vec<usize>,
}

impl BinShrinkImageFilter {
    /// Create a new BinShrink filter with the given shrink factors.
    ///
    /// # Arguments
    /// * `factors` - Shrink factor for each dimension (must be >= 1).
    ///   If the vector is shorter than the image dimensionality, `factors\[0\]`
    ///   is broadcast to all remaining dimensions.
    pub fn new(factors: Vec<usize>) -> Self {
        Self { factors }
    }

    /// Apply the BinShrink filter to an image.
    ///
    /// Each input voxel contributes to exactly one output bin. Output voxel
    /// values are the arithmetic mean of the contributing input voxels.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let (data, in_shape) = extract_vec_infallible(image);

        // Resolve per-dimension factors (broadcast factors[0] if shorter than D)
        let factors: [usize; D] = std::array::from_fn(|d| {
            if d < self.factors.len() {
                self.factors[d]
            } else {
                self.factors[0]
            }
        });

        let out_shape: [usize; D] = std::array::from_fn(|d| in_shape[d] / factors[d]);

        if out_shape.iter().product::<usize>() == 0 {
            // Degenerate case: some dimension collapses to zero.
            let device = image.data().device();
            let td = TensorData::new(Vec::<f32>::new(), Shape::new(out_shape));
            let tensor = Tensor::<B, D>::from_data(td, &device);
            let new_spacing = scaled_spacing(image.spacing(), &factors);
            return Image::new(tensor, *image.origin(), new_spacing, *image.direction());
        }

        // Separable multi-pass: shrink one dimension at a time.
        let mut current_data = data;
        let mut current_shape = in_shape;

        for d in 0..D {
            let f = factors[d];
            if f <= 1 {
                continue;
            }
            current_data = shrink_along_dim(&current_data, &current_shape, d, f);
            current_shape[d] /= f;
        }

        let device = image.data().device();
        let td = TensorData::new(current_data, Shape::new(current_shape));
        let tensor = Tensor::<B, D>::from_data(td, &device);
        let new_spacing = scaled_spacing(image.spacing(), &factors);

        Image::new(tensor, *image.origin(), new_spacing, *image.direction())
    }
}

/// Shrink the data along one dimension by averaging consecutive groups of
/// `factor` voxels.
///
/// The data buffer is in column-major (Fortran) order: the leftmost dimension
/// varies fastest in memory. For each output index `o` along dimension `dim`,
/// the output value is the mean of `input[o*factor .. (o+1)*factor]` along
/// that dimension, with all other indices held fixed.
///
/// Uses `rayon::par_iter` over independent "slabs" (1-D slices along `dim`)
/// for parallelism. Each slab is a contiguous or strided run of
/// `shape[dim]` elements, and averaging within a slab is sequential.
fn shrink_along_dim<const D: usize>(
    data: &[f32],
    shape: &[usize; D],
    dim: usize,
    factor: usize,
) -> Vec<f32> {
    let mut out_shape = *shape;
    out_shape[dim] = shape[dim] / factor;

    let out_size: usize = out_shape.iter().product();
    let mut out_data = vec![0.0f32; out_size];

    // Number of "slabs" — independent parallel units.
    // Each slab is a 1-D slice along `dim` with all other indices fixed.
    let n_slabs: usize = (0..D).filter(|&d| d != dim).map(|d| shape[d]).product();

    let out_size_dim = out_shape[dim];

    // Compute column-major strides for input and output shapes.
    let in_strides = col_major_strides(shape);
    let out_strides = col_major_strides(&out_shape);

    // For each slab, we need to:
    // 1. Find the base input/output offsets (at index 0 along `dim`)
    // 2. Walk along `dim`, accumulating and averaging

    // Collect (out_offset, value) pairs in parallel, then scatter.
    // This avoids FnMut borrow issues with rayon.
    let results: Vec<(usize, f32)> =
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n_slabs, |slab_idx| {
            // Decode slab_idx into multi-index with dim fixed at 0.
            // We enumerate non-dim dimensions in order d₀, d₁, … (skipping `dim`).
            let mut in_base = 0usize;
            let mut out_base = 0usize;
            let mut rem = slab_idx;
            for d in 0..D {
                if d == dim {
                    continue;
                }
                let idx = rem % shape[d];
                rem /= shape[d];
                in_base += idx * in_strides[d];
                out_base += idx * out_strides[d];
            }

            // Walk along `dim` for this slab.
            (0..out_size_dim)
                .map(move |o| {
                    let mut sum = 0.0f32;
                    for b in 0..factor {
                        let in_offset = in_base + (o * factor + b) * in_strides[dim];
                        sum += data[in_offset];
                    }
                    (out_base + o * out_strides[dim], sum / factor as f32)
                })
                .collect::<Vec<_>>()
        })
        .into_iter()
        .flatten()
        .collect();

    for (offset, val) in results {
        out_data[offset] = val;
    }

    out_data
}

/// Compute column-major (Fortran-order) strides for a D-dimensional shape.
///
/// # Invariant
/// `strides[d] = product of shape[0..d]` (leftmost dimension varies fastest)
fn col_major_strides<const D: usize>(shape: &[usize; D]) -> [usize; D] {
    let mut strides = [0usize; D];
    strides[0] = 1;
    for d in 1..D {
        strides[d] = strides[d - 1] * shape[d - 1];
    }
    strides
}

/// Scale spacing by the per-dimension shrink factors.
fn scaled_spacing<const D: usize>(spacing: &Spacing<D>, factors: &[usize; D]) -> Spacing<D> {
    let mut new = *spacing;
    for d in 0..D {
        new[d] *= factors[d] as f64;
    }
    new
}

#[cfg(test)]
#[path = "tests_bin_shrink.rs"]
mod tests_bin_shrink;
#[cfg(test)]
#[path = "tests_bin_shrink_edge.rs"]
mod tests_bin_shrink_edge;
