//! Integration coverage for the Coeus-native affine registration head.

use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_nn::module::Module;
use coeus_tensor::Tensor;
use ritk_model::affine::{AffineNetworkConfig, AffineTransform};

type TestBackend = SequentialBackend;

fn deterministic(shape: &[usize], scale: f32, bias: f32) -> Vec<f32> {
    let n: usize = shape.iter().product();
    (0..n)
        .map(|i| ((i % 19) as f32) / 19.0 * scale + bias)
        .collect()
}

#[test]
fn affine_network_predicts_twelve_parameters() {
    let shape = [1usize, 2, 32, 32, 32];
    let model = AffineNetworkConfig::default().init::<f32, TestBackend>();
    let input = Var::new(
        Tensor::from_slice_on(shape, &deterministic(&shape, 1.0, -0.5), &SequentialBackend),
        false,
    );
    let output = model.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 12]);
    for &v in output.tensor.as_slice() {
        assert!(v.is_finite(), "affine parameter must be finite: {v}");
    }
}

#[test]
fn affine_transform_applies_predicted_matrix() {
    let shape = [1usize, 1, 32, 32, 32];
    let stn = AffineTransform::new();
    let image = Var::new(
        Tensor::from_slice_on(shape, &deterministic(&shape, 1.0, 0.0), &SequentialBackend),
        false,
    );
    // Identity affine, flattened.
    let theta = Var::new(
        Tensor::from_slice_on(
            [1, 12],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &SequentialBackend,
        ),
        false,
    );
    let output = stn.forward(&image, &theta);
    assert_eq!(output.tensor.shape(), &[1, 1, 32, 32, 32]);

    // Identity warp under align_corners reproduces the input exactly.
    let (input, warped) = (image.tensor.as_slice(), output.tensor.as_slice());
    for (i, (&a, &b)) in input.iter().zip(warped.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-5,
            "identity warp altered voxel {i}: {a} vs {b}"
        );
    }
}

#[test]
fn network_is_differentiable_end_to_end() {
    // The predicted affine parameters are differentiable with respect to the
    // network input through the full conv → norm → relu → pool → linear chain.
    // 36³ keeps every InstanceNorm's spatial extent > 1 (36→18→9→5→3→2); a
    // smaller volume collapses a stage to one voxel, where zero variance makes
    // the norm constant and the gradient vanish (an architectural property).
    let shape = [1usize, 2, 36, 36, 36];
    // Narrow channels keep the 36³ chain tractable in a debug build.
    let model = AffineNetworkConfig {
        channels: vec![2, 3, 4, 5, 6],
    }
    .init::<f32, TestBackend>();
    let input = Var::new(
        Tensor::from_slice_on(shape, &deterministic(&shape, 1.0, -0.5), &SequentialBackend),
        true,
    );
    let theta = model.forward(&input);
    let loss = coeus_autograd::sum(&theta);
    loss.backward();

    let grad = input.grad().expect("gradient reaches the network input");
    assert!(
        grad.as_slice().iter().any(|g| g.abs() > 0.0),
        "end-to-end network gradient must be non-trivial"
    );
}

#[test]
fn network_output_drives_transform_forward() {
    // Composition check: the network's [B,12] output is a valid affine for the
    // transform, and the warp produces a finite, correctly-shaped volume.
    // (An untrained network may steer samples out of bounds, where zero-padding
    // legitimately yields zeros — so this asserts shape and finiteness, while
    // gradient correctness is covered by the per-component FD checks.)
    let feat_shape = [1usize, 2, 36, 36, 36];
    let model = AffineNetworkConfig {
        channels: vec![2, 3, 4, 5, 6],
    }
    .init::<f32, TestBackend>();
    let stn = AffineTransform::new();
    let features = Var::new(
        Tensor::from_slice_on(
            feat_shape,
            &deterministic(&feat_shape, 1.0, -0.5),
            &SequentialBackend,
        ),
        false,
    );
    let theta = model.forward(&features);
    assert_eq!(theta.tensor.shape(), &[1, 12]);

    let moving_shape = [1usize, 1, 16, 16, 16];
    let moving = Var::new(
        Tensor::from_slice_on(
            moving_shape,
            &deterministic(&moving_shape, 1.0, 0.0),
            &SequentialBackend,
        ),
        false,
    );
    let warped = stn.forward(&moving, &theta);
    assert_eq!(warped.tensor.shape(), &[1, 1, 16, 16, 16]);
    for &v in warped.tensor.as_slice() {
        assert!(v.is_finite(), "warped intensity must be finite: {v}");
    }
}
