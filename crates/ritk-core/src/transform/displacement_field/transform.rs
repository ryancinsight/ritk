use super::core::DisplacementField;
use crate::interpolation::{Interpolator, LinearInterpolator};
use crate::spatial::{Direction, Point, Spacing};
use crate::transform::{Resampleable, Transform};
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Displacement field transform.
///
/// Transforms points by adding a displacement vector interpolated strictly from a field.
#[derive(Module, Debug)]
pub struct DisplacementFieldTransform<B: Backend, const D: usize> {
    field: DisplacementField<B, D>,
    interpolator: LinearInterpolator,
}

impl<B: Backend, const D: usize> DisplacementFieldTransform<B, D> {
    pub fn new(field: DisplacementField<B, D>, interpolator: LinearInterpolator) -> Self {
        Self {
            field,
            interpolator,
        }
    }

    pub fn field(&self) -> &DisplacementField<B, D> {
        &self.field
    }

    pub fn interpolator(&self) -> &LinearInterpolator {
        &self.interpolator
    }
}

impl<B: Backend, const D: usize> Transform<B, D> for DisplacementFieldTransform<B, D> {
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let indices = self.field.world_to_index_tensor(points.clone());

        let mut displacement_components = Vec::with_capacity(D);
        for i in 0..D {
            let component = &self.field.components[i].val();
            let val = self.interpolator.interpolate(component, indices.clone());
            displacement_components.push(val);
        }

        let displacement = Tensor::stack(displacement_components, 1);
        points + displacement
    }
}

impl<B: Backend, const D: usize> Resampleable<B, D> for DisplacementFieldTransform<B, D> {
    fn resample(
        &self,
        shape: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> Self {
        let new_field = self.field.resample(shape, origin, spacing, direction);
        Self::new(new_field, self.interpolator.clone())
    }
}

pub type DisplacementFieldTransform2D<B> = DisplacementFieldTransform<B, 2>;
pub type DisplacementFieldTransform3D<B> = DisplacementFieldTransform<B, 3>;
