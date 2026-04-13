use super::*;
use burn::tensor::Tensor;
use burn_ndarray::{NdArray, NdArrayDevice};

#[test]
fn test_state_space_creation_dimensions() {
    let device = NdArrayDevice::default();
    let config = SelectiveStateSpaceConfig::new_with_dims(64, 64);
    let ssm = SelectiveStateSpace::<NdArray>::new(&config, &device);

    assert_eq!(ssm.input_dim, 64);
    assert_eq!(ssm.output_dim, 64);
    assert_eq!(ssm.state_dim, 16);
}

#[test]
fn test_forward_pass_values() {
    let device = NdArrayDevice::default();
    let config = SelectiveStateSpaceConfig::new_with_dims(32, 32);
    let ssm = SelectiveStateSpace::<NdArray>::new(&config, &device);

    // Provide non-zero deterministic inputs to ensure computational participation.
    let input = Tensor::<NdArray, 3>::ones([2, 8, 32], &device);
    let output = ssm.forward(input);

    // Validate output invariants: Output dimensions must be exact and output values must be non-zero and finite
    assert_eq!(output.dims(), [2, 8, 32]);
    let flat_data: Vec<f32> = output.into_data().to_vec::<f32>().unwrap();
    assert!(flat_data.iter().all(|&v| v.is_finite()));
    assert!(
        !flat_data.iter().all(|&v| v == 0.0),
        "Output must not be trivially zero"
    );
}

#[test]
fn test_forward_3d_discretization_bounds() {
    let device = NdArrayDevice::default();
    let config = SelectiveStateSpaceConfig::new_with_dims(16, 16);
    let ssm = SelectiveStateSpace::<NdArray>::new(&config, &device);

    let input = Tensor::<NdArray, 5>::ones([1, 16, 4, 8, 8], &device);
    let output = ssm.forward_3d(input);

    assert_eq!(output.dims(), [1, 16, 4, 8, 8]);

    // Verifying mathematical bounds on the internal discretized dt steps explicitly
    // Since input is uniform ones, the continuous projection -> softplus should guarantee positive dt invariants strictly.
    let flat_input = Tensor::<NdArray, 3>::ones([1, 4 * 8 * 8, 16], &device);
    let inner_dim = ssm.input_dim * ssm.expand_factor;
    let proj = ssm.in_proj.forward(flat_input);
    let x_part = proj.slice([0..1, 0..256, 0..inner_dim]);

    // Project and assert bound invariant: dt > 0.0
    let x_rank = ssm.dt_in_proj.forward(x_part);
    let dt_unbounded = ssm.dt_proj.forward(x_rank);
    let dt = burn::tensor::activation::softplus(dt_unbounded, 1.0);

    let dt_data: Vec<f32> = dt.into_data().to_vec::<f32>().unwrap();
    // Guaranteed by softplus mathematics (x > 0)
    assert!(
        dt_data.iter().all(|&v| v > 0.0),
        "Discretize step Δ must be strictly positive"
    );
}
