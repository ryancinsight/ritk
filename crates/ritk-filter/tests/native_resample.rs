use coeus_core::SequentialBackend;
use ritk_filter::resample::native::{
    fixed_world_points, resample_image_native, sample_moving_at_world,
};
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::transform::affine::AtlasAffineTransform;

fn image(values: Vec<f32>, shape: [usize; 3]) -> Image<f32, SequentialBackend, 3> {
    Image::from_flat_on(
        values,
        shape,
        Point::new([10.0, 20.0, 30.0]),
        Spacing::new([2.0, 1.5, 0.5]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("test image shape is valid")
}

#[test]
fn native_resample_identity_preserves_values_and_reference_geometry() {
    let input = image((0..24).map(|value| value as f32).collect(), [2, 3, 4]);
    let output = resample_image_native(
        &input,
        &input,
        &AtlasAffineTransform::<SequentialBackend, 3>::identity(None),
    )
    .expect("identity resample is valid");

    assert_eq!(output.shape(), input.shape());
    assert_eq!(*output.origin(), *input.origin());
    assert_eq!(*output.spacing(), *input.spacing());
    assert_eq!(*output.direction(), *input.direction());
    assert_eq!(
        output.data_slice().expect("contiguous output"),
        input.data_slice().expect("contiguous input")
    );
}

#[test]
fn native_resample_uses_axis_major_affine_translation() {
    let reference = image(vec![0.0; 2 * 2 * 4], [2, 2, 4]);
    let moving = image(
        (0..4)
            .flat_map(|row| {
                [
                    row as f32,
                    row as f32 + 0.25,
                    row as f32 + 0.5,
                    row as f32 + 0.75,
                ]
            })
            .collect(),
        [2, 2, 4],
    );
    let transform =
        AtlasAffineTransform::<SequentialBackend, 3>::from_translation(&[0.0, 0.0, 0.5]);
    let output = resample_image_native(&reference, &moving, &transform)
        .expect("translated resample is valid");

    assert_eq!(
        output.data_slice().expect("contiguous output"),
        &[0.25, 0.5, 0.75, 0.0, 1.25, 1.5, 1.75, 0.0, 2.25, 2.5, 2.75, 0.0, 3.25, 3.5, 3.75, 0.0]
    );
}

#[test]
fn native_resample_zero_fills_points_outside_the_half_voxel_buffer() {
    let moving = image(vec![7.0; 8], [2, 2, 2]);
    let samples = sample_moving_at_world(
        &moving,
        &[
            10.0, 20.0, 30.0, // index [0, 0, 0]
            7.0, 20.0, 30.0, // z index -1 outside
            10.0, 20.0, 31.0, // x index 2 outside
        ],
    )
    .expect("sample coordinates are valid");

    assert_eq!(samples, vec![7.0, 0.0, 0.0]);
}

#[test]
fn native_resample_rejects_empty_moving_images() {
    let empty = image(Vec::new(), [0, 2, 2]);
    let error = sample_moving_at_world(&empty, &[10.0, 20.0, 30.0])
        .expect_err("empty moving image must be rejected");

    assert_eq!(
        error.to_string(),
        "cannot sample an empty moving image with shape [0, 2, 2]"
    );
}

#[test]
fn native_resample_fixed_grid_world_points_follow_image_geometry() {
    let fixed = image(vec![0.0; 8], [2, 2, 2]);
    assert_eq!(
        fixed_world_points(&fixed),
        vec![
            10.0, 20.0, 30.0, 10.0, 20.0, 30.5, 10.0, 21.5, 30.0, 10.0, 21.5, 30.5, 12.0, 20.0,
            30.0, 12.0, 20.0, 30.5, 12.0, 21.5, 30.0, 12.0, 21.5, 30.5,
        ]
    );
}
