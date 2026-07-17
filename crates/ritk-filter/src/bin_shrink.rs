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

use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_spatial::Spacing;
use ritk_tensor_ops::{extract_vec_infallible, rebuild_with_metadata};

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
            let new_spacing = scaled_spacing(image.spacing(), &factors);
            return rebuild_with_metadata(
                Vec::new(),
                out_shape,
                *image.origin(),
                new_spacing,
                *image.direction(),
                image,
            );
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

        let new_spacing = scaled_spacing(image.spacing(), &factors);

        rebuild_with_metadata(
            current_data,
            current_shape,
            *image.origin(),
            new_spacing,
            *image.direction(),
            image,
        )
    }    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B, const D: usize>(&self, image: &ritk_image::native::Image<f32, B, D>,
        backend: &B) -> anyhow::Result<ritk_image::native::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (data, in_shape) = ritk_tensor_ops::native::extract_image_vec(image)?;

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
            let new_spacing = scaled_spacing(image.spacing(), &factors);
            return crate::native_support::rebuild_with_metadata(Vec::new(), out_shape, *image.origin(), new_spacing, *image.direction(), image, backend);
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

        let new_spacing = scaled_spacing(image.spacing(), &factors);

        crate::native_support::rebuild_with_metadata(current_data, current_shape, *image.origin(), new_spacing, *image.direction(), image, backend)
    
    }

}

/// Shrink the data along one dimension by averaging consecutive groups of
/// `factor` voxels.
///
/// The data buffer is in row-major (C-contiguous) order: the rightmost
/// dimension (X) varies fastest in memory. For each output index `o` along
/// dimension `dim`, the output value is the mean of
/// `input[o*factor .. (o+1)*factor]` along that dimension, with all other
/// indices held fixed.
///
/// Uses Moirai over disjoint output chunks. Each output voxel is written exactly
/// once, and averaging within a bin is sequential over `factor` input samples.
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

    // Compute row-major strides for input and output shapes.
    let in_strides = row_major_strides(shape);
    let out_strides = row_major_strides(&out_shape);
    let geometry = ShrinkGeometry {
        shape,
        out_shape: &out_shape,
        in_strides: &in_strides,
        out_strides: &out_strides,
        dim,
        factor,
    };

    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut out_data,
        1024,
        |chunk_idx, chunk| {
            let base_out = chunk_idx * 1024;
            for (local_idx, out) in chunk.iter_mut().enumerate() {
                let out_flat = base_out + local_idx;
                *out = bin_mean_at_output(data, &geometry, out_flat);
            }
        },
    );

    out_data
}

struct ShrinkGeometry<'a, const D: usize> {
    shape: &'a [usize; D],
    out_shape: &'a [usize; D],
    in_strides: &'a [usize; D],
    out_strides: &'a [usize; D],
    dim: usize,
    factor: usize,
}

#[inline(always)]
fn bin_mean_at_output<const D: usize>(
    data: &[f32],
    geometry: &ShrinkGeometry<'_, D>,
    out_flat: usize,
) -> f32 {
    let mut input_base = 0usize;
    let mut rem = out_flat;
    for d in 0..D {
        let coord = rem / geometry.out_strides[d];
        rem %= geometry.out_strides[d];
        debug_assert!(coord < geometry.out_shape[d]);
        let input_coord = if d == geometry.dim {
            coord * geometry.factor
        } else {
            coord
        };
        debug_assert!(input_coord < geometry.shape[d]);
        input_base += input_coord * geometry.in_strides[d];
    }

    let mut sum = 0.0f32;
    for b in 0..geometry.factor {
        sum += data[input_base + b * geometry.in_strides[geometry.dim]];
    }
    sum / geometry.factor as f32
}

/// Compute column-major (Fortran-order) strides for a D-dimensional shape.
///
/// # Invariant
/// `strides[d] = product of shape[0..d]` (leftmost dimension varies fastest)
/// Row-major (C-contiguous) strides for a `[d₀, …, d_{D-1}]` shape, matching the
/// `[Z, Y, X]` layout of ritk tensors (`vals[z·ny·nx + y·nx + x]`, X innermost).
///
/// The bin-shrink slab walk indexes the flat buffer with these strides; using
/// column-major strides averaged voxels along the wrong axes — invisible on
/// `1×1×N` or constant inputs (where the two stride orders coincide or averaging
/// is layout-invariant), but corrupting any genuine anisotropic 3-D volume.
fn row_major_strides<const D: usize>(shape: &[usize; D]) -> [usize; D] {
    let mut strides = [0usize; D];
    strides[D - 1] = 1;
    for d in (0..D - 1).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
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
