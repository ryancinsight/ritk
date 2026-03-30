use burn::tensor::{Tensor, TensorData, Shape, Distribution};
use burn::tensor::backend::Backend;

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
pub fn generate_grid<B, const D: usize>(
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, 2>
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
pub fn generate_grid_3d<B>(
    shape: [usize; 3],
    device: &B::Device,
) -> Tensor<B, 2>
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
pub fn generate_grid_2d<B>(
    shape: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2>
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

/// Generate random points (indices) within the image grid.
///
/// Returns a tensor of shape `[num_samples, D]` where each row contains
/// random continuous indices within the image bounds.
///
/// # Arguments
/// * `shape` - The image shape `[D0, D1, ...]`
/// * `num_samples` - The number of random samples to generate
/// * `device` - The device to create the tensor on
///
/// # Returns
/// Tensor of shape `[num_samples, D]` containing random continuous indices
pub fn generate_random_points<B, const D: usize>(
    shape: [usize; D],
    num_samples: usize,
    device: &B::Device,
) -> Tensor<B, 2>
where
    B: Backend,
{
    // Generate random values in [0, 1]
    let random = Tensor::<B, 2>::random(
        [num_samples, D],
        Distribution::Uniform(0.0, 1.0),
        device,
    );
    
    // Scale by shape for each dimension
    // We want coordinates in [0, dim_size - 1]
    // random * (dim_size - 1)
    
    let mut scaled = random;
    
    // Create scale tensor [1, D]
    // We use D-1, D-2... order because shape is usually [D, H, W] or [H, W]
    // But index 0 corresponds to first dimension in shape.
    // Wait, grid convention:
    // 3D: z, y, x? or x, y, z?
    // In generate_grid_3d:
    // for z in 0..d, for y in 0..h, for x in 0..w
    // grid.push(x), grid.push(y), grid.push(z)
    // So the output tensor has [x, y, z] order (fastest varying index first).
    // This is Width, Height, Depth.
    // But shape is [D, H, W] (Depth, Height, Width).
    // So shape[0]=D, shape[1]=H, shape[2]=W.
    // And output point is (x, y, z).
    // So point[0] corresponds to shape[2].
    // point[1] corresponds to shape[1].
    // point[2] corresponds to shape[0].
    
    // Let's verify generate_grid_2d:
    // for y in 0..h, for x in 0..w
    // grid.push(x), grid.push(y)
    // So point is (x, y).
    // shape is [H, W].
    // point[0] (x) corresponds to shape[1] (W).
    // point[1] (y) corresponds to shape[0] (H).
    
    // So we need to scale dimension i of the point by the corresponding dimension of the shape.
    // For D=2: point=[x, y]. shape=[H, W].
    // Scale point[0] by (W-1). Scale point[1] by (H-1).
    
    // For D=3: point=[x, y, z]. shape=[D, H, W].
    // Scale point[0] by (W-1). Scale point[1] by (H-1). Scale point[2] by (D-1).
    
    // Generally: point[i] corresponds to shape[D - 1 - i].
    
    let mut scale_factors = Vec::with_capacity(D);
    for i in 0..D {
        let dim_size = shape[D - 1 - i];
        scale_factors.push((dim_size as f32) - 1.0);
    }
    
    let scale_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(scale_factors, Shape::new([D])),
        device,
    ).reshape([1, D]);
    
    scaled = scaled * scale_tensor;
    
    scaled
}
