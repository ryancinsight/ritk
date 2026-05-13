use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Shape, Tensor, TensorData};

/// Generate random continuous indices for the given image shape.
///
/// Returns a tensor of shape `[N, D]` where N is num_samples.
/// Samples are drawn uniformly from the continuous index space [0, shape[i]-1].
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
    // Create a scaling tensor for each dimension
    let max_vals: Vec<f32> = shape.iter().map(|&s| (s as f32) - 1.0).collect();
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
    match D {
        3 => {
            let s: &[usize] = shape.as_slice();
            generate_grid_3d(s.try_into().expect("Shape must be 3D"), device)
        }
        2 => {
            let s: &[usize] = shape.as_slice();
            generate_grid_2d(s.try_into().expect("Shape must be 2D"), device)
        }
        _ => panic!("Only 2D and 3D grids are supported"),
    }
}

/// Generate a grid of continuous indices for a 3D image shape.
///
/// Returns a tensor of shape `[N, 3]` where N is the total number of voxels.
///
/// # Arguments
/// * `shape` - The image shape `[D, H, W]`
/// * `device` - The device to create the tensor on
///
/// # Returns
/// Tensor of shape `[N, 3]` containing continuous indices
pub fn generate_grid_3d<B>(shape: [usize; 3], device: &B::Device) -> Tensor<B, 2>
where
    B: Backend,
{
    let d = shape[0];
    let h = shape[1];
    let w = shape[2];
    let total = d * h * w;

    let mut grid = Vec::with_capacity(total * 3);
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                grid.push(x as f32);
                grid.push(y as f32);
                grid.push(z as f32);
            }
        }
    }

    Tensor::<B, 1>::from_data(TensorData::new(grid, Shape::new([total * 3])), device)
        .reshape([total, 3])
}

/// Generate a grid of continuous indices for a 2D image shape.
///
/// Returns a tensor of shape `[N, 2]` where N is the total number of pixels.
///
/// # Arguments
/// * `shape` - The image shape `[H, W]`
/// * `device` - The device to create the tensor on
///
/// # Returns
/// Tensor of shape `[N, 2]` containing continuous indices
pub fn generate_grid_2d<B>(shape: [usize; 2], device: &B::Device) -> Tensor<B, 2>
where
    B: Backend,
{
    let h = shape[0];
    let w = shape[1];
    let total = h * w;

    let mut grid = Vec::with_capacity(total * 2);
    for y in 0..h {
        for x in 0..w {
            grid.push(x as f32);
            grid.push(y as f32);
        }
    }

    Tensor::<B, 1>::from_data(TensorData::new(grid, Shape::new([total * 2])), device)
        .reshape([total, 2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    // ── generate_grid_3d ─────────────────────────────────────────────────────

    /// Shape of a 3D grid is [D*H*W, 3].
    #[test]
    fn grid_3d_shape_is_n_by_3() {
        let device = Default::default();
        let g = generate_grid_3d::<B>([2, 3, 4], &device);
        let [rows, cols] = g.dims();
        assert_eq!(rows, 2 * 3 * 4, "row count must equal D*H*W");
        assert_eq!(cols, 3, "column count must be 3 (x, y, z)");
    }

    /// First voxel (z=0, y=0, x=0) must be [0, 0, 0].
    #[test]
    fn grid_3d_first_voxel_is_origin() {
        let device = Default::default();
        let g = generate_grid_3d::<B>([3, 3, 3], &device);
        let first = g.clone().slice([0..1]).into_data();
        let vals = first.as_slice::<f32>().unwrap();
        // column order in grid.rs: push(x), push(y), push(z)
        assert_eq!(vals[0], 0.0, "x of first voxel");
        assert_eq!(vals[1], 0.0, "y of first voxel");
        assert_eq!(vals[2], 0.0, "z of first voxel");
    }

    /// Last voxel of [D, H, W] must be [W-1, H-1, D-1].
    ///
    /// # Derivation
    /// The innermost loop is over x (0..W), middle y (0..H), outer z (0..D).
    /// Last entry: z=D-1, y=H-1, x=W-1. push order: x, y, z.
    #[test]
    fn grid_3d_last_voxel_matches_shape_minus_one() {
        let device = Default::default();
        let d = 2usize;
        let h = 3usize;
        let w = 4usize;
        let g = generate_grid_3d::<B>([d, h, w], &device);
        let n = d * h * w;
        let last = g.clone().slice([n - 1..n]).into_data();
        let vals = last.as_slice::<f32>().unwrap();
        assert_eq!(vals[0], (w - 1) as f32, "x of last voxel");
        assert_eq!(vals[1], (h - 1) as f32, "y of last voxel");
        assert_eq!(vals[2], (d - 1) as f32, "z of last voxel");
    }

    // ── generate_grid_2d ─────────────────────────────────────────────────────

    /// Shape of a 2D grid is [H*W, 2].
    #[test]
    fn grid_2d_shape_is_n_by_2() {
        let device = Default::default();
        let g = generate_grid_2d::<B>([5, 7], &device);
        let [rows, cols] = g.dims();
        assert_eq!(rows, 5 * 7, "row count must equal H*W");
        assert_eq!(cols, 2, "column count must be 2 (x, y)");
    }

    /// First pixel (y=0, x=0) must be [0, 0].
    #[test]
    fn grid_2d_first_pixel_is_origin() {
        let device = Default::default();
        let g = generate_grid_2d::<B>([4, 6], &device);
        let first = g.clone().slice([0..1]).into_data();
        let vals = first.as_slice::<f32>().unwrap();
        assert_eq!(vals[0], 0.0, "x of first pixel");
        assert_eq!(vals[1], 0.0, "y of first pixel");
    }

    /// generate_grid dispatches correctly for 3D.
    #[test]
    fn generate_grid_3d_dispatch() {
        let device = Default::default();
        let g = generate_grid::<B, 3>([2, 2, 2], &device);
        assert_eq!(g.dims(), [8, 3]);
    }

    /// generate_grid dispatches correctly for 2D.
    #[test]
    fn generate_grid_2d_dispatch() {
        let device = Default::default();
        let g = generate_grid::<B, 2>([3, 4], &device);
        assert_eq!(g.dims(), [12, 2]);
    }

    // ── generate_random_points ───────────────────────────────────────────────

    /// Random points tensor must have shape [N, D].
    #[test]
    fn random_points_shape_is_n_by_d() {
        let device = Default::default();
        let pts = generate_random_points::<B, 3>([10, 20, 30], 50, &device);
        assert_eq!(pts.dims(), [50, 3]);
    }

    /// All random index values must lie in [0, shape[d]-1].
    ///
    /// # Derivation
    /// For shape=[4, 6, 8], the maximum valid index per dimension is [3, 5, 7].
    /// generate_random_points scales Uniform(0,1) by (shape[d]-1).
    #[test]
    fn random_points_within_bounds() {
        let device = Default::default();
        let shape = [4usize, 6, 8];
        let pts = generate_random_points::<B, 3>(shape, 200, &device);
        let data = pts.into_data();
        let vals = data.as_slice::<f32>().unwrap();
        // vals layout: [row0_x, row0_y, row0_z, row1_x, ...]
        for (i, &v) in vals.iter().enumerate() {
            let col = i % 3; // dimension index
            let max_idx = (shape[col] - 1) as f32;
            assert!(
                v >= 0.0 && v <= max_idx + 1e-4, // small epsilon for float rounding
                "random index col={col} out of [0, {max_idx}]: got {v}"
            );
        }
    }
}
