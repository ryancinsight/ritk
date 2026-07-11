use super::core::DisplacementField;
use coeus_autograd::{add, linear_interpolation, reshape, stack, transpose_2d, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::{BackendOps, Dimension, InterpolationError, Replicate, SupportedDimension};

/// Contract failures while applying a trainable displacement transform.
#[derive(Debug, thiserror::Error)]
pub enum DisplacementTransformError {
    /// Physical points are not a rank-2 `[N, D]` matrix.
    #[error("displacement transform requires points shaped [N, {dimension}], got {actual:?}")]
    PointShape {
        /// Spatial dimension.
        dimension: usize,
        /// Supplied shape.
        actual: Vec<usize>,
    },
    /// Coeus interpolation rejected the field or mapped grid.
    #[error(transparent)]
    Interpolation(#[from] InterpolationError),
}

/// Trainable displacement transform with replicated-border interpolation.
#[derive(Clone)]
pub struct DisplacementFieldTransform<B: Backend, const D: usize>
where
    B: BackendOps<f32>,
{
    field: DisplacementField<B, D>,
    boundary: Replicate,
}

impl<B: Backend + BackendOps<f32>, const D: usize> DisplacementFieldTransform<B, D>
where
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Construct a transform over a trainable field.
    #[must_use]
    pub const fn new(field: DisplacementField<B, D>) -> Self {
        Self {
            field,
            boundary: Replicate,
        }
    }

    /// Borrow the trainable field.
    #[must_use]
    pub const fn field(&self) -> &DisplacementField<B, D> {
        &self.field
    }

    /// Transform physical `[N, D]` points through differentiable sampling.
    pub fn transform_points(
        &self,
        points: &Var<f32, B>,
    ) -> Result<Var<f32, B>, DisplacementTransformError> {
        let shape = points.tensor.shape();
        if shape.len() != 2 || shape[1] != D {
            return Err(DisplacementTransformError::PointShape {
                dimension: D,
                actual: shape.to_vec(),
            });
        }
        let displacements = self.field.sample_components(points, self.boundary)?;
        let references = displacements.iter().collect::<Vec<_>>();
        Ok(add(points, &stack(&references, 1)))
    }
}

impl<B: Backend + BackendOps<f32>, const D: usize> DisplacementField<B, D>
where
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    pub(crate) fn sample_components(
        &self,
        points: &Var<f32, B>,
        boundary: Replicate,
    ) -> Result<Vec<Var<f32, B>>, InterpolationError> {
        let indices = self.world_to_index_tensor(points);
        let point_count = points.tensor.shape()[0];
        let mut grid_shape = vec![1, D, point_count];
        grid_shape.resize(D + 2, 1);
        let grid = reshape(&transpose_2d(&indices), grid_shape);
        self.components
            .iter()
            .map(|component| {
                let image_shape = [1, 1]
                    .into_iter()
                    .chain(component.tensor.shape().iter().copied())
                    .collect::<Vec<_>>();
                let image = reshape(component, image_shape);
                linear_interpolation::<D, _, _>(&image, &grid, boundary)
                    .map(|sampled| reshape(&sampled, [point_count]))
            })
            .collect()
    }
}

impl<B: Backend + BackendOps<f32>, const D: usize> Module<f32, B>
    for DisplacementFieldTransform<B, D>
where
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        self.field.components.clone()
    }

    fn named_parameters(&self) -> Vec<coeus_autograd::Parameter<f32, B>> {
        self.field
            .components
            .iter()
            .enumerate()
            .map(|(axis, component)| {
                coeus_autograd::Parameter::new(component.clone(), format!("field.component.{axis}"))
            })
            .collect()
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        self.transform_points(input)
            .expect("invariant: Module::forward receives valid field coordinates")
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        assert_eq!(
            parameters.len(),
            D,
            "invariant: named parameter validation fixes component count"
        );
        self.field.components.clone_from_slice(parameters);
    }
}
