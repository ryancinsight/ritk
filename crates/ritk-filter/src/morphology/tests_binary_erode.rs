//! Value-semantic coverage for canonical native binary erosion.

use coeus_core::MoiraiBackend;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};

use super::erode_binary_3d;
use crate::morphology::{native::binary_erode, ForegroundValue};

type Backend = MoiraiBackend;

fn make_image(values: Vec<f32>, dimensions: [usize; 3]) -> (Image<f32, Backend, 3>, Backend) {
    let backend = Backend::new();
    let image = Image::from_flat_on(
        values,
        dimensions,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &backend,
    )
    .expect("invariant: test data length matches the image dimensions");
    (image, backend)
}

fn flat(image: &Image<f32, Backend, 3>) -> Vec<f32> {
    image
        .data_slice()
        .expect("invariant: from_flat_on constructs contiguous image storage")
        .to_vec()
}

fn apply(
    values: Vec<f32>,
    dimensions: [usize; 3],
    radius: usize,
    foreground: ForegroundValue,
) -> Vec<f32> {
    let (image, backend) = make_image(values, dimensions);
    let output = binary_erode(&image, radius, foreground, &backend)
        .expect("binary erosion of a contiguous native image must succeed");
    flat(&output)
}

fn reference_erode(
    data: &[f32],
    dimensions: [usize; 3],
    radius: usize,
    foreground: ForegroundValue,
) -> Vec<f32> {
    let [depth, height, width] = dimensions;
    let foreground = foreground.0;
    let radius = isize::try_from(radius).expect("test radius fits in isize");
    let mut output = vec![0.0; data.len()];

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let survives = (-radius..=radius).all(|dz| {
                    (-radius..=radius).all(|dy| {
                        (-radius..=radius).all(|dx| {
                            let neighbor = [
                                z.checked_add_signed(dz),
                                y.checked_add_signed(dy),
                                x.checked_add_signed(dx),
                            ];
                            let [Some(nz), Some(ny), Some(nx)] = neighbor else {
                                return false;
                            };
                            nz < depth
                                && ny < height
                                && nx < width
                                && data[nz * height * width + ny * width + nx] == foreground
                        })
                    })
                });
                if survives {
                    output[z * height * width + y * width + x] = foreground;
                }
            }
        }
    }

    output
}

#[test]
fn radius_zero_is_identity_for_binary_input() {
    let values = vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    assert_eq!(
        apply(values.clone(), [2, 2, 2], 0, ForegroundValue::ONE),
        values
    );
}

#[test]
fn border_voxels_erode_to_background() {
    let result = apply(vec![1.0; 27], [3, 3, 3], 1, ForegroundValue::ONE);
    let mut expected = vec![0.0; 27];
    expected[13] = 1.0;
    assert_eq!(result, expected);
}

#[test]
fn background_remains_background() {
    assert_eq!(
        apply(vec![1.0, 0.0, 1.0], [1, 1, 3], 1, ForegroundValue::ONE),
        vec![0.0, 0.0, 0.0]
    );
}

#[test]
fn erosion_strips_one_border_layer() {
    let result = apply(vec![1.0; 45], [3, 3, 5], 1, ForegroundValue::ONE);
    let mut expected = vec![0.0; 45];
    expected[21..=23].fill(1.0);
    assert_eq!(result, expected);
}

#[test]
fn erosion_strips_two_border_layers() {
    let result = apply(vec![1.0; 175], [5, 5, 7], 2, ForegroundValue::ONE);
    let mut expected = vec![0.0; 175];
    expected[86..=88].fill(1.0);
    assert_eq!(result, expected);
}

#[test]
fn custom_foreground_value_is_preserved() {
    let foreground = ForegroundValue::from(255.0);
    let result = apply(vec![255.0; 45], [3, 3, 5], 1, foreground);
    let mut expected = vec![0.0; 45];
    expected[21..=23].fill(255.0);
    assert_eq!(result, expected);
}

#[test]
fn spatial_metadata_is_preserved() {
    let origin = Point::new([3.0, 2.0, 1.0]);
    let spacing = Spacing::new([0.5, 0.5, 1.0]);
    let direction = Direction::identity();
    let backend = Backend::new();
    let image = Image::from_flat_on(
        vec![1.0; 8],
        [2, 2, 2],
        origin,
        spacing,
        direction,
        &backend,
    )
    .expect("invariant: test data length matches the image dimensions");

    let output = binary_erode(&image, 0, ForegroundValue::ONE, &backend)
        .expect("binary erosion of a contiguous native image must succeed");

    assert_eq!(*output.origin(), origin);
    assert_eq!(*output.spacing(), spacing);
    assert_eq!(*output.direction(), direction);
}

#[test]
fn bounded_exhaustive_binary_volumes_match_independent_reference() {
    const DIMENSIONS: [usize; 3] = [2, 2, 3];
    const VOXELS: usize = 12;

    for mask in 0_u16..(1_u16 << VOXELS) {
        let values: Vec<f32> = (0..VOXELS)
            .map(|bit| if mask & (1 << bit) == 0 { 0.0 } else { 1.0 })
            .collect();
        for radius in 0..=2 {
            assert_eq!(
                erode_binary_3d(&values, DIMENSIONS, radius, ForegroundValue::ONE),
                reference_erode(&values, DIMENSIONS, radius, ForegroundValue::ONE),
                "mask={mask:#05x}, radius={radius}"
            );
        }
    }
}
