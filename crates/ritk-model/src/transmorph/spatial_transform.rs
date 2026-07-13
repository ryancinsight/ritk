//! Differentiable displacement-field sampling.

use coeus_autograd::{add, cat, linear_interpolation, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;
use std::marker::PhantomData;

use crate::ModelError;

/// Samples an image at identity-grid coordinates displaced by a flow field.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpatialTransformer<B> {
    backend: PhantomData<B>,
}

impl<B> SpatialTransformer<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Construct a stateless spatial transformer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            backend: PhantomData,
        }
    }

    /// Warp `[batch, channels, depth, height, width]` by a voxel-unit flow.
    pub fn forward(
        &self,
        image: &Var<f32, B>,
        flow: &Var<f32, B>,
    ) -> Result<Var<f32, B>, ModelError> {
        let image_shape = image.tensor.shape();
        if image_shape.len() != 5 {
            return Err(ModelError::Shape {
                operation: "SpatialTransformer::forward image",
                expected: "[batch, channels, depth, height, width]",
                actual: image_shape.to_vec(),
            });
        }
        let [batch, _channels, depth, height, width] = <[usize; 5]>::try_from(image_shape)
            .expect("invariant: image rank was validated as five");
        let expected_flow = [batch, 3, depth, height, width];
        if flow.tensor.shape() != expected_flow {
            return Err(ModelError::Shape {
                operation: "SpatialTransformer::forward flow",
                expected: "[batch, 3, depth, height, width] matching image spatial axes",
                actual: flow.tensor.shape().to_vec(),
            });
        }

        let flow_depth = coeus_autograd::slice(
            flow,
            &[(0, batch), (0, 1), (0, depth), (0, height), (0, width)],
        );
        let flow_height = coeus_autograd::slice(
            flow,
            &[(0, batch), (1, 2), (0, depth), (0, height), (0, width)],
        );
        let flow_width = coeus_autograd::slice(
            flow,
            &[(0, batch), (2, 3), (0, depth), (0, height), (0, width)],
        );

        let backend = B::default();
        let depth_axis = axis_grid::<B>(depth, [1, 1, depth, 1, 1], &backend);
        let height_axis = axis_grid::<B>(height, [1, 1, 1, height, 1], &backend);
        let width_axis = axis_grid::<B>(width, [1, 1, 1, 1, width], &backend);
        let sample_depth = add(&flow_depth, &depth_axis);
        let sample_height = add(&flow_height, &height_axis);
        let sample_width = add(&flow_width, &width_axis);
        let sampling_grid = cat(&[&sample_depth, &sample_height, &sample_width], 1);

        Ok(linear_interpolation::<3, _, _>(
            image,
            &sampling_grid,
            coeus_ops::Replicate,
        )?)
    }
}

fn axis_grid<B>(length: usize, shape: [usize; 5], backend: &B) -> Var<f32, B>
where
    B: Backend + BackendOps<f32>,
{
    let values = (0..length).map(|index| index as f32).collect::<Vec<_>>();
    Var::new(Tensor::from_slice_on(shape, &values, backend), false)
}
