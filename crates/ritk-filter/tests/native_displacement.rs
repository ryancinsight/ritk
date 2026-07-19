use coeus_core::SequentialBackend;
use ritk_filter::{InverseDisplacementField, NativeDisplacementField};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

fn zero_component() -> Image<f32, SequentialBackend, 3> {
    Image::from_flat_on(
        vec![0.0; 16],
        [1, 4, 4],
        Point::new([1.0, -2.0, 3.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("zero displacement component has a valid shape")
}

#[test]
fn native_inverse_zero_field_preserves_components_and_frame() {
    let x = zero_component();
    let y = zero_component();
    let z = zero_component();

    let NativeDisplacementField {
        x: inverse_x,
        y: inverse_y,
        z: inverse_z,
    } = InverseDisplacementField {
        subsampling_factor: 2,
    }
    .apply_native(&x, &y, &z, &SequentialBackend)
    .expect("zero field inversion succeeds");

    let expected = vec![0.0; 16];
    assert_eq!(
        inverse_x
            .data_slice()
            .expect("inverse x component is contiguous"),
        expected.as_slice()
    );
    assert_eq!(
        inverse_y
            .data_slice()
            .expect("inverse y component is contiguous"),
        expected.as_slice()
    );
    assert_eq!(
        inverse_z
            .data_slice()
            .expect("inverse z component is contiguous"),
        expected.as_slice()
    );
    assert_eq!(inverse_x.origin(), x.origin());
    assert_eq!(inverse_x.spacing(), x.spacing());
    assert_eq!(inverse_x.direction(), x.direction());
    assert_eq!(inverse_y.origin(), y.origin());
    assert_eq!(inverse_y.spacing(), y.spacing());
    assert_eq!(inverse_y.direction(), y.direction());
    assert_eq!(inverse_z.origin(), z.origin());
    assert_eq!(inverse_z.spacing(), z.spacing());
    assert_eq!(inverse_z.direction(), z.direction());
}
