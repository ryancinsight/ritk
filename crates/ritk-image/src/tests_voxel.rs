//! Runtime-selected voxel representation and image preservation contracts.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use coeus_core::{Scalar, SequentialBackend};
use ritk_spatial::{CoordinateMap, CurvilinearArray, Direction, Point, Spacing};

use crate::{Image, VoxelImage};

fn image<T: Scalar>(values: Vec<T>) -> Image<T, SequentialBackend, 1> {
    let length = values.len();
    Image::from_flat_on(
        values,
        [length],
        Point::new([4.5]),
        Spacing::new([0.75]),
        Direction::identity(),
        &SequentialBackend,
    )
    .unwrap()
}

#[test]
fn runtime_image_preserves_each_supported_stored_type() {
    let cases = [
        VoxelImage::Unsigned8(image(vec![0_u8, u8::MAX])),
        VoxelImage::Signed8(image(vec![i8::MIN, i8::MAX])),
        VoxelImage::Unsigned16(image(vec![0_u16, u16::MAX])),
        VoxelImage::Signed16(image(vec![i16::MIN, i16::MAX])),
        VoxelImage::Unsigned32(image(vec![0_u32, u32::MAX])),
        VoxelImage::Signed32(image(vec![i32::MIN, i32::MAX])),
        VoxelImage::Unsigned64(image(vec![0_u64, u64::MAX])),
        VoxelImage::Signed64(image(vec![i64::MIN, i64::MAX])),
        VoxelImage::Float32(image(vec![0.5_f32, 1.5])),
        VoxelImage::Float64(image(vec![0.5_f64, 1.5])),
    ];

    for image in cases {
        assert_eq!(image.shape(), [2]);
        match image {
            VoxelImage::Unsigned8(image) => assert_eq!(
                image.data_cow_on(&SequentialBackend).as_ref(),
                &[0_u8, u8::MAX]
            ),
            VoxelImage::Signed8(image) => assert_eq!(
                image.data_cow_on(&SequentialBackend).as_ref(),
                &[i8::MIN, i8::MAX]
            ),
            VoxelImage::Unsigned16(image) => assert_eq!(
                image.data_cow_on(&SequentialBackend).as_ref(),
                &[0_u16, u16::MAX]
            ),
            VoxelImage::Signed16(image) => assert_eq!(
                image.data_cow_on(&SequentialBackend).as_ref(),
                &[i16::MIN, i16::MAX]
            ),
            VoxelImage::Unsigned32(image) => assert_eq!(
                image.data_cow_on(&SequentialBackend).as_ref(),
                &[0_u32, u32::MAX]
            ),
            VoxelImage::Signed32(image) => assert_eq!(
                image.data_cow_on(&SequentialBackend).as_ref(),
                &[i32::MIN, i32::MAX]
            ),
            VoxelImage::Unsigned64(image) => assert_eq!(
                image.data_cow_on(&SequentialBackend).as_ref(),
                &[0_u64, u64::MAX]
            ),
            VoxelImage::Signed64(image) => assert_eq!(
                image.data_cow_on(&SequentialBackend).as_ref(),
                &[i64::MIN, i64::MAX]
            ),
            VoxelImage::Float32(image) => assert_eq!(
                image.data_cow_on(&SequentialBackend).as_ref(),
                &[0.5_f32, 1.5]
            ),
            VoxelImage::Float64(image) => assert_eq!(
                image.data_cow_on(&SequentialBackend).as_ref(),
                &[0.5_f64, 1.5]
            ),
        }
    }
}

#[test]
fn wrapping_preserves_integer_values_and_non_cartesian_geometry() {
    let values = [(1_u64 << 53) + 1, u64::MAX];
    let origin = Point::new([12.25, -3.5]);
    let spacing = Spacing::new([0.4, 1.75]);
    let direction = Direction::from_rows([[0.0, -1.0], [1.0, 0.0]]);
    let map = CoordinateMap::CurvilinearArray(
        CurvilinearArray::try_new(0.25, 1.5, 0.125, -0.0625).unwrap(),
    );
    let source = Image::from_flat_on(
        values.to_vec(),
        [2, 1],
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .unwrap()
    .with_coordinate_map(map.clone())
    .unwrap();

    let VoxelImage::Unsigned64(image) = VoxelImage::Unsigned64(source) else {
        panic!("variant construction must retain the selected scalar type");
    };

    assert_eq!(image.data_cow_on(&SequentialBackend).as_ref(), &values);
    assert_eq!(image.shape(), [2, 1]);
    assert_eq!(image.origin(), &origin);
    assert_eq!(image.spacing(), &spacing);
    assert_eq!(image.direction(), &direction);
    assert_eq!(image.coordinate_map(), &map);
}

#[test]
fn wrapping_preserves_float_bits_including_signed_zero_and_nan_payload() {
    let bits = [0x8000_0000_0000_0000, 0x7ff8_0000_0000_0042];
    let values = bits.map(f64::from_bits);
    let source = image(values.to_vec());
    let VoxelImage::Float64(image) = VoxelImage::Float64(source) else {
        panic!("variant construction must retain the selected scalar type");
    };

    assert_eq!(
        image
            .data_cow_on(&SequentialBackend)
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        bits
    );
}

#[test]
fn wrapping_preserves_f32_bits_including_signed_zero_and_nan_payload() {
    let bits = [0x8000_0000, 0x7fc0_0042, 0x3f80_0001];
    let values = bits.map(f32::from_bits);
    let source = image(values.to_vec());
    let VoxelImage::Float32(image) = VoxelImage::Float32(source) else {
        panic!("variant construction must retain the selected scalar type");
    };

    assert_eq!(
        image
            .data_cow_on(&SequentialBackend)
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        bits
    );
}
