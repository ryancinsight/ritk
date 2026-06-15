use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Shape, Tensor, TensorData};

/// Generate random continuous indices for the given image shape.
///
/// Returns a tensor of shape `[N, D]` where N is num_samples.
/// Samples are drawn uniformly from the continuous index space \[0, shape\[i\]-1\].
///
/// # Arguments
/// * `shape` - The image shape `[D0, D1, ...]`
/// * `num_samples` - The number of random samples to generate
/// * `device` - The device to create the tensor on
///
/// # Returns
/// Tensor of shape `[N, D]` containing random continuous indices
pub fn generate_random_points<B: Backend, const D: usize>(
    shape: [usize; D],
    num_samples: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    // Per-column scaling must use the SAME innermost-first column convention as
    // [`generate_grid`] and the interpolation kernels: column 0 = x = `shape[D-1]`,
    // …, column D-1 = z = `shape[0]`.  Scaling column `j` by `shape[j]-1` (axis
    // order) instead transposes the axes — harmless for cubic images (all extents
    // equal) but catastrophic for anisotropic shapes: e.g. for a [29, 512, 512]
    // thin slab it would draw the z-column over [0, 511] when the z-axis only
    // spans [0, 28], sending almost every sample out of bounds and collapsing the
    // joint histogram (MI ≈ 0).  Reverse the extents so column j ranges over
    // shape[D-1-j].
    let max_vals: Vec<f32> = (0..D).map(|j| (shape[D - 1 - j] as f32) - 1.0).collect();
    let max_vals_tensor =
        Tensor::<B, 1>::from_data(TensorData::new(max_vals, Shape::new([D])), device)
            .reshape([D, 1]);

    // Generate all random numbers at once, scale, and transpose
    Tensor::<B, 2>::random([D, num_samples], Distribution::Uniform(0.0, 1.0), device)
        .mul(max_vals_tensor) // [D, N]
        .transpose() // [N, D]
}

/// Generate a grid of continuous indices for the given image shape.
///
/// Returns a tensor of shape `[N, D]` where N is the total number of voxels
/// and D is the dimensionality.
///
/// The iteration order is row-major with dimension 0 varying fastest,
/// matching the original `generate_grid_2d`/`generate_grid_3d` behavior.
///
/// # Arguments
/// * `shape` - The image shape `[D0, D1, ...]`
/// * `device` - The device to create the tensor on
///
/// # Type Parameters
/// * `B` - The tensor backend
/// * `D` - The spatial dimensionality
///
/// # Returns
/// Tensor of shape `[N, D]` containing continuous indices
pub fn generate_grid<B, const D: usize>(shape: [usize; D], device: &B::Device) -> Tensor<B, 2>
where
    B: Backend,
{
    let total: usize = shape.iter().product();
    let mut grid = Vec::with_capacity(total * D);

    // Column order is INNERMOST-FIRST: column 0 = x = `indices[D-1]`, matching the
    // interpolation kernels (`gather` uses column 0 as the unit-stride/x axis) and
    // `Image::index_to_world_tensor`/`world_to_index_tensor` (which reverse the
    // column↔axis mapping to agree). The ROW order is row-major (innermost varies
    // fastest) so rows match the image's flat data layout.
    let mut indices = [0usize; D];

    for _ in 0..total {
        // Push innermost dimension first: col 0 = x = indices[D-1].
        for d in (0..D).rev() {
            grid.push(indices[d] as f32);
        }

        // Increment innermost first → row-major iteration (rows match flat data).
        for d in (0..D).rev() {
            indices[d] += 1;
            if indices[d] < shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }

    Tensor::<B, 1>::from_data(TensorData::new(grid, Shape::new([total * D])), device)
        .reshape([total, D])
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
#[path = "tests_grid.rs"]
mod tests;
