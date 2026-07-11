//! Differentiable affine image sampling.

use coeus_autograd::{cat, matmul, reshape, scalar_add, scalar_mul, slice, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;
use std::marker::PhantomData;

use crate::ModelError;

/// Applies batched `3 × 4` affine matrices to volumetric images.
#[derive(Debug, Clone, Copy, Default)]
pub struct AffineTransform<B> {
    backend: PhantomData<B>,
}

impl<B> AffineTransform<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Construct a stateless affine transformer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            backend: PhantomData,
        }
    }

    /// Warp an image using flattened `[batch, 12]` affine parameters.
    pub fn forward(
        &self,
        image: &Var<f32, B>,
        theta: &Var<f32, B>,
    ) -> Result<Var<f32, B>, ModelError> {
        let image_shape = image.tensor.shape();
        if image_shape.len() != 5 {
            return Err(ModelError::Shape {
                operation: "AffineTransform::forward image",
                expected: "[batch, channels, depth, height, width]",
                actual: image_shape.to_vec(),
            });
        }
        let [batch, _channels, depth, height, width] = <[usize; 5]>::try_from(image_shape)
            .expect("invariant: image rank was validated as five");
        if theta.tensor.shape() != [batch, 12] {
            return Err(ModelError::Shape {
                operation: "AffineTransform::forward theta",
                expected: "[batch, 12]",
                actual: theta.tensor.shape().to_vec(),
            });
        }

        let theta = reshape(theta, [batch, 3, 4]);
        let grid = normalized_grid::<B>(batch, depth, height, width);
        let warped = matmul(&theta, &grid);
        let warped = reshape(&warped, [batch, 3, depth, height, width]);
        let pixel_grid = denormalize(&warped, batch, depth, height, width);
        Ok(coeus_autograd::trilinear_interpolation(image, &pixel_grid)?)
    }
}

fn normalized_grid<B>(batch: usize, depth: usize, height: usize, width: usize) -> Var<f32, B>
where
    B: Backend + BackendOps<f32>,
{
    let voxels = depth * height * width;
    let mut values = Vec::with_capacity(batch * 4 * voxels);
    for _ in 0..batch {
        for component in 0..4 {
            for z in 0..depth {
                for y in 0..height {
                    for x in 0..width {
                        values.push(match component {
                            0 => normalized_coordinate(z, depth),
                            1 => normalized_coordinate(y, height),
                            2 => normalized_coordinate(x, width),
                            3 => 1.0,
                            _ => unreachable!("invariant: homogeneous grid has four components"),
                        });
                    }
                }
            }
        }
    }
    Var::new(
        Tensor::from_slice_on([batch, 4, voxels], &values, &B::default()),
        false,
    )
}

fn normalized_coordinate(index: usize, length: usize) -> f32 {
    if length == 1 {
        0.0
    } else {
        (2.0 * index as f32 / (length - 1) as f32) - 1.0
    }
}

fn denormalize<B>(
    grid: &Var<f32, B>,
    batch: usize,
    depth: usize,
    height: usize,
    width: usize,
) -> Var<f32, B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let ranges = |channel| {
        [
            (0, batch),
            (channel, channel + 1),
            (0, depth),
            (0, height),
            (0, width),
        ]
    };
    let scale = |component: &Var<f32, B>, length: usize| {
        scalar_mul(
            &scalar_add(component, 1.0),
            (length.saturating_sub(1)) as f32 / 2.0,
        )
    };
    let z = scale(&slice(grid, &ranges(0)), depth);
    let y = scale(&slice(grid, &ranges(1)), height);
    let x = scale(&slice(grid, &ranges(2)), width);
    cat(&[&z, &y, &x], 1)
}
