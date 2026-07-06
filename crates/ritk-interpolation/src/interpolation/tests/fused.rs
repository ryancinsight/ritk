//! Tests for the fused transform → world-to-index → interpolation kernel.

use crate::interpolation::fused::{is_identity_direction, transform_and_interpolate};
use crate::interpolation::LinearInterpolator;
use ritk_image::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::interpolation::Interpolator;
use ritk_core::transform::Transform;
use ritk_spatial::{Direction, Point, Spacing};

type Backend = NdArray<f32>;
type Point3 = Point<3>;
type Spacing3 = Spacing<3>;
type Direction3 = Direction<3>;

/// A pure-translation transform for testing.
struct TranslationTransform {
    offset: [f32; 3],
}

impl Transform<Backend, 3> for TranslationTransform {
    fn transform_points(&self, points: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
        let device = points.device();
        let offset = Tensor::<Backend, 1>::from_data(
            TensorData::new(self.offset.to_vec(), ritk_image::tensor::Shape::new([3])),
            &device,
        )
        .reshape([1usize, 3]);
        points + offset
    }
}

fn make_identity_image(
    device: &<Backend as ritk_image::tensor::Backend>::Device,
) -> Image<Backend, 3> {
    let data = Tensor::<Backend, 3>::zeros([4, 4, 4], device);
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    let direction = Direction3::identity();
    Image::new(data, origin, spacing, direction)
}

#[test]
fn test_identity_direction_detection() {
    let identity = Direction3::identity();
    assert!(is_identity_direction(&identity));

    let mut non_identity = Direction3::zeros();
    non_identity[(0, 0)] = 0.0;
    non_identity[(0, 1)] = -1.0;
    non_identity[(1, 0)] = 1.0;
    non_identity[(1, 1)] = 0.0;
    non_identity[(2, 2)] = 1.0;
    assert!(!is_identity_direction(&non_identity));
}

#[test]
fn test_fused_identity_image_zero_translation() {
    let device = Default::default();
    let _moving = make_identity_image(&device);

    // Create image data with known values
    let data_vec: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let data = Tensor::<Backend, 3>::from_data(
        TensorData::new(data_vec, ritk_image::tensor::Shape::new([4, 4, 4])),
        &device,
    );
    let moving = Image::new(
        data,
        Point3::new([0.0, 0.0, 0.0]),
        Spacing3::new([1.0, 1.0, 1.0]),
        Direction3::identity(),
    );

    let transform = TranslationTransform {
        offset: [0.0, 0.0, 0.0],
    };
    let interpolator = LinearInterpolator::new();

    // Query at integer grid points → should return exact voxel values
    let points = Tensor::<Backend, 2>::from_floats(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        &device,
    );

    let result = transform_and_interpolate(points, &transform, &moving, &interpolator);
    let values = result.values.into_data().into_vec::<f32>().unwrap();

    assert_eq!(values[0], 0.0, "Voxel (0,0,0) should be 0.0");
    assert_eq!(values[1], 16.0, "Voxel (0,0,1) should be 16.0");
    assert_eq!(values[2], 4.0, "Voxel (0,1,0) should be 4.0");
}

#[test]
fn test_fused_identity_image_with_translation() {
    let device = Default::default();
    let data_vec: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let data = Tensor::<Backend, 3>::from_data(
        TensorData::new(data_vec, ritk_image::tensor::Shape::new([4, 4, 4])),
        &device,
    );
    let moving = Image::new(
        data,
        Point3::new([10.0, 10.0, 10.0]),
        Spacing3::new([1.0, 1.0, 1.0]),
        Direction3::identity(),
    );

    // Transform: fixed_world + [10, 10, 10] → moving_world
    // Then: index = (moving_world - [10,10,10]) / 1.0 = fixed_world
    // So querying fixed_world [0,0,0] should give moving[0,0,0] = 0.0
    let transform = TranslationTransform {
        offset: [10.0, 10.0, 10.0],
    };
    let interpolator = LinearInterpolator::new();

    let points = Tensor::<Backend, 2>::from_floats([[0.0, 0.0, 0.0]], &device);
    let result = transform_and_interpolate(points, &transform, &moving, &interpolator);
    let values = result.values.into_data().into_vec::<f32>().unwrap();

    assert_eq!(
        values[0], 0.0,
        "With translation offset matching origin, index should be 0"
    );
}

#[test]
fn test_fused_non_identity_spacing() {
    let device = Default::default();
    let data_vec: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let data = Tensor::<Backend, 3>::from_data(
        TensorData::new(data_vec, ritk_image::tensor::Shape::new([4, 4, 4])),
        &device,
    );
    let moving = Image::new(
        data,
        Point3::new([0.0, 0.0, 0.0]),
        Spacing3::new([2.0, 2.0, 2.0]),
        Direction3::identity(),
    );

    let transform = TranslationTransform {
        offset: [0.0, 0.0, 0.0],
    };
    let interpolator = LinearInterpolator::new();

    // With spacing=2, a world point at [2,0,0] should map to index [1,0,0]
    let points = Tensor::<Backend, 2>::from_floats([[2.0, 0.0, 0.0]], &device);
    let result = transform_and_interpolate(points, &transform, &moving, &interpolator);
    let values = result.values.into_data().into_vec::<f32>().unwrap();

    assert_eq!(
        values[0], 16.0,
        "World [2,0,0] with spacing 2 → index [0,0,1] = value 16.0"
    );
}

#[test]
fn test_fused_matches_unfused() {
    let device = Default::default();
    let data_vec: Vec<f32> = (0..64).map(|i| i as f32 * 10.0).collect();
    let data = Tensor::<Backend, 3>::from_data(
        TensorData::new(data_vec, ritk_image::tensor::Shape::new([4, 4, 4])),
        &device,
    );
    let origin = Point3::new([5.0, 5.0, 5.0]);
    let spacing = Spacing3::new([2.0, 3.0, 4.0]);
    let direction = Direction3::identity();
    let moving = Image::new(data, origin, spacing, direction);

    let transform = TranslationTransform {
        offset: [3.0, -1.0, 7.0],
    };
    let interpolator = LinearInterpolator::new();

    let fixed_points = Tensor::<Backend, 2>::from_floats(
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [0.0, 0.0, 0.0]],
        &device,
    );

    // Fused path
    let fused_result =
        transform_and_interpolate(fixed_points.clone(), &transform, &moving, &interpolator);

    // Unfused path (3 separate allocations)
    let moving_world = transform.transform_points(fixed_points);
    let indices = moving.world_to_index_tensor(moving_world);
    let unfused_result = interpolator.interpolate(moving.data(), indices);

    let fused_vals = fused_result.values.into_data().into_vec::<f32>().unwrap();
    let unfused_vals = unfused_result.into_data().into_vec::<f32>().unwrap();

    for i in 0..fused_vals.len() {
        let diff = (fused_vals[i] - unfused_vals[i]).abs();
        assert!(
            diff < 1e-4,
            "Fused and unfused results differ at index {i}: fused={}, unfused={}, diff={}",
            fused_vals[i],
            unfused_vals[i],
            diff,
        );
    }
}

#[test]
fn test_fused_identity_direction_anisotropic_matches_unfused() {
    let device = Default::default();
    let data_vec: Vec<f32> = (0..60).map(|i| i as f32 * 1.25).collect();
    let data = Tensor::<Backend, 3>::from_data(
        TensorData::new(data_vec, ritk_image::tensor::Shape::new([3, 4, 5])),
        &device,
    );
    let origin = Point3::new([10.0, -20.0, 30.0]);
    let spacing = Spacing3::new([2.0, 3.0, 5.0]);
    let moving = Image::new(data, origin, spacing, Direction3::identity());

    let transform = TranslationTransform {
        offset: [1.5, -2.0, 7.5],
    };
    let interpolator = LinearInterpolator::new();

    let fixed_points = Tensor::<Backend, 2>::from_floats(
        [
            [9.0, -18.0, 27.5],
            [11.0, -17.0, 32.5],
            [13.0, -14.0, 35.0],
            [16.0, -11.0, 40.0],
        ],
        &device,
    );

    let fused_result =
        transform_and_interpolate(fixed_points.clone(), &transform, &moving, &interpolator);

    let moving_world = transform.transform_points(fixed_points);
    let indices = moving.world_to_index_tensor(moving_world);
    let unfused_result = interpolator.interpolate(moving.data(), indices);

    let fused_vals = fused_result.values.into_data().into_vec::<f32>().unwrap();
    let unfused_vals = unfused_result.into_data().into_vec::<f32>().unwrap();

    for i in 0..fused_vals.len() {
        let diff = (fused_vals[i] - unfused_vals[i]).abs();
        assert!(
            diff < 1e-4,
            "Identity-direction fused and unfused results differ at index {i}: fused={}, unfused={}, diff={}",
            fused_vals[i],
            unfused_vals[i],
            diff,
        );
    }
}

#[test]
fn test_fused_general_direction_matches_unfused() {
    let device = Default::default();
    let data_vec: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let data = Tensor::<Backend, 3>::from_data(
        TensorData::new(data_vec, ritk_image::tensor::Shape::new([4, 4, 4])),
        &device,
    );
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);

    // 90-degree rotation around Z axis
    let mut direction = Direction3::zeros();
    direction[(0, 0)] = 0.0;
    direction[(0, 1)] = -1.0;
    direction[(1, 0)] = 1.0;
    direction[(1, 1)] = 0.0;
    direction[(2, 2)] = 1.0;

    let moving = Image::new(data, origin, spacing, direction);

    let transform = TranslationTransform {
        offset: [0.0, 0.0, 0.0],
    };
    let interpolator = LinearInterpolator::new();

    let fixed_points = Tensor::<Backend, 2>::from_floats(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        &device,
    );

    // Fused path
    let fused_result =
        transform_and_interpolate(fixed_points.clone(), &transform, &moving, &interpolator);

    // Unfused path
    let moving_world = transform.transform_points(fixed_points);
    let indices = moving.world_to_index_tensor(moving_world);
    let unfused_result = interpolator.interpolate(moving.data(), indices);

    let fused_vals = fused_result.values.into_data().into_vec::<f32>().unwrap();
    let unfused_vals = unfused_result.into_data().into_vec::<f32>().unwrap();

    for i in 0..fused_vals.len() {
        let diff = (fused_vals[i] - unfused_vals[i]).abs();
        assert!(
            diff < 1e-4,
            "Fused and unfused results differ at index {i}: fused={}, unfused={}, diff={}",
            fused_vals[i],
            unfused_vals[i],
            diff,
        );
    }
}

#[test]
fn test_fused_oob_mask() {
    let device = Default::default();
    let data_vec: Vec<f32> = (0..64).map(|i| i as f32 * 10.0).collect();
    let data = Tensor::<Backend, 3>::from_data(
        TensorData::new(data_vec, ritk_image::tensor::Shape::new([4, 4, 4])),
        &device,
    );
    let moving = Image::new(
        data,
        Point3::new([0.0, 0.0, 0.0]),
        Spacing3::new([1.0, 1.0, 1.0]),
        Direction3::identity(),
    );

    let transform = TranslationTransform {
        offset: [0.0, 0.0, 0.0],
    };
    let interpolator = LinearInterpolator::new();

    // Mix of in-bounds and out-of-bounds points
    let points = Tensor::<Backend, 2>::from_floats(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ],
        &device,
    );

    let result = transform_and_interpolate(points, &transform, &moving, &interpolator);

    // OOB mask should be Some for 3D images
    let mask = result.oob_mask.expect("OOB mask should be present for 3D");
    let mask_vals = mask.into_data().into_vec::<f32>().unwrap();

    assert_eq!(mask_vals.len(), 5);
    assert!(
        (mask_vals[0] - 1.0).abs() < 1e-6,
        "[0,0,0] should be in-bounds"
    );
    assert!(mask_vals[1].abs() < 1e-6, "[-1,0,0] should be OOB");
    assert!(
        (mask_vals[2] - 1.0).abs() < 1e-6,
        "[2,2,2] should be in-bounds"
    );
    assert!(mask_vals[3].abs() < 1e-6, "[5,0,0] should be OOB");
    assert!(mask_vals[4].abs() < 1e-6, "[0,10,0] should be OOB");
}
