use coeus_autograd::{mean, mul, sub, Var};
use coeus_core::SequentialBackend;
use coeus_nn::Module;
use coeus_optim::{Adam, Optimizer};
use coeus_tensor::Tensor;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_registration::metric::autodiff::Transform;
use ritk_transform::{DisplacementField, DisplacementFieldTransform};

fn apply_registration_transform<T: Transform<f32, SequentialBackend>>(
    transform: &T,
    points: &Var<f32, SequentialBackend>,
) -> Var<f32, SequentialBackend> {
    transform.transform_points(points)
}

#[test]
fn named_adam_optimizes_displacement_through_registration_seam() {
    let backend = SequentialBackend;
    let field = DisplacementField::new(
        (0..3)
            .map(|_| Tensor::zeros_on([2, 2, 2], &backend))
            .collect(),
        Point::origin(),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
    .expect("valid field");
    let mut transform = DisplacementFieldTransform::new(field);
    let points = Var::new(
        Tensor::from_slice_on([1, 3], &[0.5, 0.5, 0.5], &backend),
        false,
    );
    let target = Var::new(
        Tensor::from_slice_on([1, 3], &[2.5, 2.5, 2.5], &backend),
        false,
    );
    let mut optimizer = Adam::new(transform.named_parameters(), 0.2, 0.9, 0.999, 1.0e-8);

    let objective = |transform: &DisplacementFieldTransform<SequentialBackend, 3>| {
        let residual = sub(&apply_registration_transform(transform, &points), &target);
        mean(&mul(&residual, &residual))
    };
    let initial = objective(&transform).tensor.as_slice()[0];
    optimizer.zero_grad();
    objective(&transform).backward();
    optimizer.step();
    transform
        .load_named_parameters(&optimizer.params)
        .expect("stable field inventory");
    let final_loss = objective(&transform).tensor.as_slice()[0];
    assert!(
        final_loss < initial,
        "registration objective must decrease: {initial} -> {final_loss}"
    );
    let transformed = apply_registration_transform(&transform, &points);
    for value in transformed.tensor.as_slice() {
        // First-step Adam normalizes the nonzero gradient to a +learning-rate
        // update. All eight center weights sum to one, so 0.5 + 0.2 = 0.7.
        assert!(
            (value - 0.7).abs() <= 32.0 * f32::EPSILON,
            "first Adam point {value}"
        );
    }
}
