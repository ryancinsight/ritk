use super::*;
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

#[test]
fn test_nearest_neighbor_interpolator_planar() {
    let device = Default::default();
    let data = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0], [2.0, 3.0]], &device);
    let interpolator = NearestNeighborInterpolator::new();

    // Test at integer coordinates
    let indices = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0], [1.0, 1.0]], &device);
    let values = interpolator.interpolate(&data, indices);
    let data_slice = values.to_data();
    let data_slice_ref = data_slice.as_slice::<f32>().unwrap();
    assert_eq!(data_slice_ref[0], 0.0);
    assert_eq!(data_slice_ref[1], 3.0);
}

#[test]
fn test_nearest_neighbor_interpolator_rounding() {
    let device = Default::default();
    let data = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0], [2.0, 3.0]], &device);
    let interpolator = NearestNeighborInterpolator::new();

    // Test at half coordinates (should round to nearest)
    let indices = Tensor::<TestBackend, 2>::from_floats([[0.4, 0.4], [0.6, 0.6]], &device);
    let values = interpolator.interpolate(&data, indices);
    let data_slice = values.to_data();
    let data_slice_ref = data_slice.as_slice::<f32>().unwrap();
    // 0.4 rounds to 0, 0.6 rounds to 1
    assert_eq!(data_slice_ref[0], 0.0);
    assert_eq!(data_slice_ref[1], 3.0);
}

#[test]
fn test_nearest_neighbor_interpolator_planar_axes() {
    let device = Default::default();
    // data: [[0, 1],
    //        [2, 3]]
    // Y=0: 0, 1. Y=1: 2, 3.
    let data = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0], [2.0, 3.0]], &device);
    let interpolator = NearestNeighborInterpolator::new();

    // indices: (x, y)
    // (1, 0) -> x=1, y=0. Should correspond to col 1, row 0. Value 1.0.
    let indices = Tensor::<TestBackend, 2>::from_floats([[1.0, 0.0]], &device);
    let values = interpolator.interpolate(&data, indices);
    let val = values.into_data().as_slice::<f32>().unwrap()[0];
    assert_eq!(val, 1.0);
}

#[test]
fn test_nearest_neighbor_interpolator_line() {
    let device = Default::default();
    let data = Tensor::<TestBackend, 1>::from_floats([10.0, 20.0, 30.0], &device);
    let interpolator = NearestNeighborInterpolator::new();

    // x=1.0 -> 20.0
    let indices = Tensor::<TestBackend, 2>::from_floats([[1.0]], &device);
    let val = interpolator
        .interpolate(&data, indices)
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    assert_eq!(val, 20.0);
}

#[test]
fn test_nearest_neighbor_interpolator_4d() {
    let device = Default::default();
    let mut data_vec = vec![0.0; 16];
    data_vec[15] = 100.0; // Last element (1,1,1,1)
    let data = Tensor::<TestBackend, 4>::from_data(
        burn::tensor::TensorData::new(data_vec, burn::tensor::Shape::new([2, 2, 2, 2])),
        &device,
    );
    let interpolator = NearestNeighborInterpolator::new();

    let indices = Tensor::<TestBackend, 2>::from_floats([[1.0, 1.0, 1.0, 1.0]], &device);
    let val = interpolator
        .interpolate(&data, indices)
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    assert_eq!(val, 100.0);
}

#[test]
fn test_nearest_neighbor_zero_pad_volumetric_oob_returns_zero() {
    let device = Default::default();
    let data_vec = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let data = Tensor::<TestBackend, 3>::from_data(
        burn::tensor::TensorData::new(data_vec, burn::tensor::Shape::new([2, 2, 2])),
        &device,
    );
    let interp = NearestNeighborInterpolator::new_zero_pad();

    // Far outside: should be 0.0
    let oob =
        Tensor::<TestBackend, 2>::from_floats([[-5.0, -5.0, -5.0], [10.0, 10.0, 10.0]], &device);
    let result = interp.interpolate(&data, oob);
    let s = result.into_data().as_slice::<f32>().unwrap().to_vec();
    assert!(s[0].abs() < 1e-6, "OOB 3D should give 0, got {}", s[0]);
    assert!(s[1].abs() < 1e-6, "OOB 3D should give 0, got {}", s[1]);
}

#[test]
fn test_nearest_neighbor_zero_pad_volumetric_inbounds_unchanged() {
    let device = Default::default();
    let data_vec = vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let data = Tensor::<TestBackend, 3>::from_data(
        burn::tensor::TensorData::new(data_vec, burn::tensor::Shape::new([2, 2, 2])),
        &device,
    );
    let interp = NearestNeighborInterpolator::new_zero_pad();

    // In-bounds corner at (0,0,0) should return data[0,0,0] = 10.0
    let corner = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0, 0.0]], &device);
    let val = interp
        .interpolate(&data, corner)
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    assert!(
        (val - 10.0).abs() < 1e-5,
        "In-bounds corner should give 10.0, got {}",
        val
    );
}

#[test]
fn test_nearest_neighbor_no_zero_pad_clamps_edge() {
    // Verify backward compat: without zero_pad, OOB clamps to edge.
    let device = Default::default();
    let data_vec = vec![1.0_f32, 2.0, 3.0, 4.0];
    let data = Tensor::<TestBackend, 2>::from_data(
        burn::tensor::TensorData::new(data_vec, burn::tensor::Shape::new([2, 2])),
        &device,
    );
    let interp = NearestNeighborInterpolator::new(); // zero_pad = false
    let oob = Tensor::<TestBackend, 2>::from_floats([[-100.0, -100.0]], &device);
    let val = interp
        .interpolate(&data, oob)
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    // Should clamp to (0,0) -> data[0,0] = 1.0
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Edge clamp should give 1.0, got {}",
        val
    );
}
