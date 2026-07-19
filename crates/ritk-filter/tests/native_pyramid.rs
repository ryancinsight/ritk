use coeus_core::SequentialBackend;
use ritk_filter::NativeMultiResolutionPyramid;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

fn image(values: Vec<f32>, shape: [usize; 3]) -> Image<f32, SequentialBackend, 3> {
    Image::from_flat_on(
        values,
        shape,
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("test image shape is valid")
}

#[test]
fn native_pyramid_preserves_identity_level_values_and_metadata() {
    let values: Vec<f32> = (0..24).map(|value| value as f32).collect();
    let input = image(values.clone(), [2, 3, 4]);
    let pyramid = NativeMultiResolutionPyramid::new(
        &input,
        &[[1, 1, 1]],
        &[[0.0, 0.0, 0.0]],
        &SequentialBackend,
    )
    .expect("identity schedule is valid");

    let level = pyramid.get_level(0);
    assert_eq!(pyramid.levels(), 1);
    assert_eq!(level.shape(), [2, 3, 4]);
    assert_eq!(level.data_slice().expect("contiguous result"), values);
    assert_eq!(*level.origin(), *input.origin());
    assert_eq!(*level.spacing(), *input.spacing());
    assert_eq!(*level.direction(), *input.direction());
}

#[test]
fn native_pyramid_samples_integer_strides_and_updates_spacing() {
    let input = image((0..64).map(|value| value as f32).collect(), [4, 4, 4]);
    let pyramid = NativeMultiResolutionPyramid::new(
        &input,
        &[[2, 2, 2]],
        &[[0.0, 0.0, 0.0]],
        &SequentialBackend,
    )
    .expect("stride schedule is valid");

    let level = pyramid.get_level(0);
    assert_eq!(level.shape(), [2, 2, 2]);
    assert_eq!(
        level.data_slice().expect("contiguous result"),
        &[0.0, 2.0, 8.0, 10.0, 32.0, 34.0, 40.0, 42.0]
    );
    assert_eq!(*level.spacing(), Spacing::new([1.0, 2.0, 4.0]));
}

#[test]
fn native_pyramid_default_schedule_is_coarse_to_fine() {
    let input = image(vec![1.0; 8 * 8 * 8], [8, 8, 8]);
    let (shrink, sigmas) = NativeMultiResolutionPyramid::<SequentialBackend>::default_schedule(3);
    let pyramid = NativeMultiResolutionPyramid::new(&input, &shrink, &sigmas, &SequentialBackend)
        .expect("default schedule is valid");

    assert_eq!(shrink, vec![[4, 4, 4], [2, 2, 2], [1, 1, 1]]);
    assert_eq!(
        sigmas,
        vec![[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    );
    assert_eq!(pyramid.levels(), 3);
    assert_eq!(pyramid.get_level(0).shape(), [2, 2, 2]);
    assert_eq!(pyramid.get_level(1).shape(), [4, 4, 4]);
    assert_eq!(pyramid.get_level(2).shape(), [8, 8, 8]);
}

#[test]
fn native_pyramid_rejects_invalid_schedules() {
    let input = image(vec![1.0; 8], [2, 2, 2]);
    let mismatch = NativeMultiResolutionPyramid::new(&input, &[[1, 1, 1]], &[], &SequentialBackend);
    let Err(mismatch) = mismatch else {
        panic!("mismatched schedules must fail");
    };
    assert_eq!(
        mismatch.to_string(),
        "pyramid schedule lengths differ: shrink=1 smoothing=0"
    );

    let zero = NativeMultiResolutionPyramid::new(
        &input,
        &[[1, 0, 1]],
        &[[0.0, 0.0, 0.0]],
        &SequentialBackend,
    );
    let Err(zero) = zero else {
        panic!("zero shrink factor must fail");
    };
    assert_eq!(
        zero.to_string(),
        "pyramid shrink factors must be positive, got [1, 0, 1]"
    );

    let negative = NativeMultiResolutionPyramid::new(
        &input,
        &[[1, 1, 1]],
        &[[-1.0, 0.0, 0.0]],
        &SequentialBackend,
    );
    let Err(negative) = negative else {
        panic!("negative smoothing sigma must fail");
    };
    assert_eq!(
        negative.to_string(),
        "pyramid smoothing sigma must be non-negative, got -1"
    );
}
