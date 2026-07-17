use coeus_core::SequentialBackend;
use ritk_filter::warp_image_native;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};

/// Eight weighted samples and seven additions bound constant-field rounding by
/// `16ε·|value|`, with one extra factor for conservative accumulation error.
const TRILINEAR_CONSTANT_TOLERANCE: f32 = 16.0 * f32::EPSILON * 7.0;

fn image(values: Vec<f32>, shape: [usize; 3]) -> Image<f32, SequentialBackend, 3> {
    Image::from_flat_on(
        values,
        shape,
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([2.0, 1.0, 0.5]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("test image shape is valid")
}

#[test]
fn native_warp_preserves_constant_interior_values() {
    let shape = [6, 6, 6];
    let count: usize = shape.iter().product();
    let moving = image(vec![7.0; count], shape);
    let dz = image(vec![0.2; count], shape);
    let dy = image(vec![-0.4; count], shape);
    let dx = image(
        (0..count)
            .map(|index| 0.15 * (index % 3) as f32 - 0.15)
            .collect(),
        shape,
    );
    let output = warp_image_native(&moving, &dz, &dy, &dx).expect("warp is valid");
    let values = output.data_slice().expect("contiguous output");

    for z in 1..shape[0] - 1 {
        for y in 1..shape[1] - 1 {
            for x in 1..shape[2] - 1 {
                let value = values[(z * shape[1] + y) * shape[2] + x];
                assert!(
                    (value - 7.0).abs() <= TRILINEAR_CONSTANT_TOLERANCE,
                    "constant warp value at [{z}, {y}, {x}] is {value}"
                );
            }
        }
    }
}

#[test]
fn native_warp_zero_fills_out_of_bounds_samples() {
    let shape = [3, 3, 3];
    let moving = image(vec![5.0; 27], shape);
    let dz = image(vec![100.0; 27], shape);
    let dy = image(vec![100.0; 27], shape);
    let dx = image(vec![100.0; 27], shape);

    let output = warp_image_native(&moving, &dz, &dy, &dx).expect("warp is valid");
    assert_eq!(
        output.data_slice().expect("contiguous output"),
        vec![0.0; 27]
    );
}

#[test]
fn native_warp_zero_displacement_is_exact_identity() {
    let shape = [4, 5, 3];
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count).map(|index| (index as f32 * 1.7).sin()).collect();
    let moving = image(values.clone(), shape);
    let zero = image(vec![0.0; count], shape);

    let output = warp_image_native(&moving, &zero, &zero, &zero).expect("warp is valid");
    assert_eq!(output.data_slice().expect("contiguous output"), values);
}

#[test]
fn native_warp_rejects_mismatched_component_shapes() {
    let moving = image(vec![0.0; 8], [2, 2, 2]);
    let dz = image(vec![0.0; 8], [2, 2, 2]);
    let dy = image(vec![0.0; 27], [3, 3, 3]);
    let dx = image(vec![0.0; 8], [2, 2, 2]);

    let error = warp_image_native(&moving, &dz, &dy, &dx)
        .expect_err("mismatched field components must fail");
    assert_eq!(
        error.to_string(),
        "warp: displacement y component shape [3, 3, 3] differs from z component shape [2, 2, 2]"
    );
}

#[test]
fn native_warp_rejects_mismatched_component_geometry() {
    let moving = image(vec![0.0; 8], [2, 2, 2]);
    let dz = image(vec![0.0; 8], [2, 2, 2]);
    let dy = Image::from_flat_on(
        vec![0.0; 8],
        [2, 2, 2],
        Point::new([2.0, 2.0, 3.0]),
        Spacing::new([2.0, 1.0, 0.5]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("test image shape is valid");
    let dx = image(vec![0.0; 8], [2, 2, 2]);

    let error =
        warp_image_native(&moving, &dz, &dy, &dx).expect_err("mismatched field geometry must fail");
    assert_eq!(
        error.to_string(),
        "warp: displacement y component geometry differs from z component geometry"
    );
}
