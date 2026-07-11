use coeus_core::MoiraiBackend;
use ritk_core::image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};

type Image3 = Image<f32, MoiraiBackend, 3>;

#[test]
fn rotated_image_maps_physical_point_to_exact_axis_permutation() {
    let image = Image3::from_flat(
        vec![0.0; 10 * 10 * 10],
        [10, 10, 10],
        Point::origin(),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::from_row_major([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    .expect("fixture shape and data length agree");

    let index = image
        .physical_point_to_continuous_index(&Point::new([1.0, 0.0, 0.0]))
        .expect("rotation matrix is invertible");
    assert_eq!(index, Point::new([0.0, -1.0, 0.0]));
}

#[test]
fn singular_direction_is_rejected() {
    let image = Image3::from_flat(
        vec![0.0],
        [1, 1, 1],
        Point::origin(),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::from_row_major([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    .expect("fixture shape and data length agree");

    let error = image
        .physical_point_to_continuous_index(&Point::origin())
        .unwrap_err();
    assert_eq!(error.to_string(), "image direction matrix is singular");
}
