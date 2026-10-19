//! Registration seam implementation for the native trainable displacement field.

use super::traits::Transform;
use coeus_autograd::Var;
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::BackendOps;
use ritk_transform::DisplacementFieldTransform;

impl<B> Transform<f32, B> for DisplacementFieldTransform<B, 3>
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn transform_points(&self, points: &Var<f32, B>) -> Var<f32, B> {
        DisplacementFieldTransform::transform_points(self, points)
            .expect("invariant: registration grids satisfy the field interpolation contract")
    }
}

#[cfg(test)]
mod tests {
    use coeus_autograd::{mean, mul, sub, Var};
    use coeus_core::SequentialBackend;
    use coeus_nn::Module;
    use coeus_optim::{Adam, Optimizer};
    use coeus_tensor::Tensor;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_transform::{DisplacementField, DisplacementFieldTransform};

    #[test]
    fn named_adam_step_reduces_field_point_objective() {
        let backend = SequentialBackend;
        let field = DisplacementField::new(
            (0..3)
                .map(|_| Tensor::zeros_on([2, 2, 2], &backend).expect("valid field component"))
                .collect(),
            Point::origin(),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
        .expect("valid field");
        let mut transform = DisplacementFieldTransform::new(field);
        let points_tensor =
            Tensor::from_slice_on([1, 3], &[0.5, 0.5, 0.5], &backend).expect("valid points tensor");
        let points = Var::new(points_tensor, false).expect("valid points var");
        let target_tensor = Tensor::from_slice_on([1, 3], &[1.5, 0.0, 0.75], &backend)
            .expect("valid target tensor");
        let target = Var::new(target_tensor, false).expect("valid target var");
        let mut optimizer = Adam::new(transform.named_parameters(), 0.1, 0.9, 0.999, 1.0e-8)
            .expect("Adam builds over field parameters");

        let objective = |transform: &DisplacementFieldTransform<SequentialBackend, 3>| {
            let moved = transform
                .transform_points(&points)
                .expect("valid interpolation");
            let residual = sub(&moved, &target).expect("valid residual");
            let squared_residual = mul(&residual, &residual).expect("valid square");
            mean(&squared_residual).expect("valid mean")
        };
        let initial = objective(&transform).tensor.as_slice()[0];
        optimizer.zero_grad().expect("zero_grad succeeds");
        objective(&transform)
            .backward()
            .expect("backward succeeds over the test loss");
        optimizer
            .step()
            .expect("Adam step succeeds for finite registration gradients");
        transform
            .load_named_parameters(&optimizer.params)
            .expect("stable field inventory");
        let final_loss = objective(&transform).tensor.as_slice()[0];
        assert!(
            final_loss < initial,
            "Adam must reduce the analytical field objective: {initial} -> {final_loss}"
        );
    }
}
