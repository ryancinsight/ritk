use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;
use rand::Rng;

/// Generate a grid of continuous indices for the given image shape.
///
/// Returns a tensor of shape `[N, D]` where N is the total number of voxels
/// and D is the dimensionality.
///
/// The iteration order is row-major with dimension 0 varying fastest,
/// matching the original `generate_grid_2d`/`generate_grid_3d` behavior.
///
/// Column order is INNERMOST-FIRST: column 0 = x = `indices[D-1]`, matching
/// the interpolation kernels and `Image::index_to_world_tensor`/
/// `world_to_index_tensor`.
///
/// # Arguments
/// * `shape` - The image shape `[D0, D1, ...]`
/// * `backend` - The compute backend to create the tensor on
///
/// # Type Parameters
/// * `T` - The scalar element type (e.g. `f32`)
/// * `B` - The compute backend
/// * `D` - The spatial dimensionality
///
/// # Returns
/// Tensor of shape `[N, D]` containing continuous indices
pub fn generate_grid<T, B, const D: usize>(shape: [usize; D], backend: &B) -> Tensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let total: usize = shape.iter().product();
    let mut grid = Vec::with_capacity(total * D);

    let mut indices = [0usize; D];

    for _ in 0..total {
        for d in (0..D).rev() {
            grid.push(T::from_f64(indices[d] as f64));
        }

        for d in (0..D).rev() {
            indices[d] += 1;
            if indices[d] < shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }

    Tensor::<T, B>::from_slice_on([total, D], &grid, backend)
}

/// Generate uniformly distributed continuous voxel indices.
///
/// Coordinate columns use the same innermost-first convention as
/// [`generate_grid`]. Each coordinate lies in the closed voxel-center range
/// `0..=shape[axis] - 1`.
pub fn generate_random_points<T, B, const D: usize>(
    shape: [usize; D],
    count: usize,
    backend: &B,
) -> Tensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let mut rng = rand::rng();
    let mut points = Vec::with_capacity(count * D);
    for _ in 0..count {
        for column in 0..D {
            let maximum = shape[D - 1 - column].saturating_sub(1) as f64;
            points.push(T::from_f64(rng.random_range(0.0..=maximum)));
        }
    }
    Tensor::from_slice_on([count, D], &points, backend)
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
#[path = "tests_grid.rs"]
mod tests;
