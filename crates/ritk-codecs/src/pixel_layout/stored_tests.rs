use std::fmt::Debug;

use crate::sample::Sample;
use crate::ByteOrder;

use super::{decode_stored_pixel_frame, PixelLayout, PixelSignedness, StoredPixelError};

fn layout(allocated: u16, stored: u16, signedness: PixelSignedness) -> PixelLayout {
    PixelLayout {
        rows: 1,
        cols: 3,
        samples_per_pixel: 1,
        bits_allocated: allocated,
        bits_stored: stored,
        pixel_representation: signedness,
        rescale_slope: 2.0,
        rescale_intercept: 5.0,
    }
}

fn encode(values: &[u32], width: u16, order: ByteOrder) -> Vec<u8> {
    let width = usize::from(width / 8);
    let mut bytes = Vec::with_capacity(values.len() * width);
    for value in values {
        match order {
            ByteOrder::LeastSignificantByteFirst => {
                bytes.extend_from_slice(&value.to_le_bytes()[..width]);
            }
            ByteOrder::MostSignificantByteFirst => {
                bytes.extend_from_slice(&value.to_be_bytes()[4 - width..]);
            }
        }
    }
    bytes
}

fn case<T, const N: usize>(layout: PixelLayout, raw: [u32; N], order: ByteOrder, expected: [T; N])
where
    T: Sample + PartialEq + Debug,
{
    let bytes = encode(&raw, layout.bits_allocated, order);
    let decoded = decode_stored_pixel_frame(&bytes, layout, order).expect("valid pixel frame");
    assert_eq!(decoded.sample_type(), T::SAMPLE_TYPE);
    let actual = decoded
        .try_into_samples::<T>()
        .expect("layout selects sample type");
    assert_eq!(actual.as_slice(), &expected);
}

macro_rules! cases {
    ($order:expr; $(($allocated:literal, $stored:literal, $sign:ident, $raw:expr, $expected:expr);)+) => {
        $(case(layout($allocated, $stored, PixelSignedness::$sign), $raw, $order, $expected);)+
    };
}

fn error(bytes: &[u8], layout: PixelLayout) -> StoredPixelError {
    decode_stored_pixel_frame(bytes, layout, ByteOrder::LeastSignificantByteFirst)
        .expect_err("invalid pixel frame")
}

fn rejects_layout(layout: PixelLayout) {
    assert!(matches!(
        error(&[], layout),
        StoredPixelError::InvalidLayout(actual) if actual == layout
    ));
}

#[test]
fn decodes_scalar_widths_and_masks_unused_bits_in_both_byte_orders() {
    for order in [
        ByteOrder::LeastSignificantByteFirst,
        ByteOrder::MostSignificantByteFirst,
    ] {
        cases!(order;
            (8, 8, Signed, [0xff, 0x80, 0x7f], [-1_i8, i8::MIN, i8::MAX]);
            (8, 8, Unsigned, [0, 0x80, 0xff], [0_u8, 128, u8::MAX]);
            (8, 4, Unsigned, [0xf0, 0x0f, 0xa5], [0_u8, 15, 5]);
            (16, 12, Signed, [0xf001, 0xf800, 0x07ff], [1_i16, -2048, 2047]);
            (16, 12, Unsigned, [0xf001, 0xf800, 0x07ff], [1_u16, 2048, 2047]);
            (16, 16, Unsigned, [0xffff, 0x8000, 0x7fff], [u16::MAX, 0x8000, 0x7fff]);
            (24, 24, Signed, [0xff_ffff, 0x80_0000, 0x7f_ffff], [-1_i32, -8_388_608, 8_388_607]);
            (24, 20, Signed, [0xf8_0001, 0xf7_ffff, 0xff_ffff], [-524_287_i32, 524_287, -1]);
            (24, 20, Unsigned, [0xf0_0001, 0xff_ffff, 0x80_0000], [1_u32, 1_048_575, 0]);
            (32, 32, Signed, [0xffff_ffff, 0x8000_0000, 0x7fff_ffff], [-1_i32, i32::MIN, i32::MAX]);
            (32, 12, Unsigned, [0xa000_0fff, 0xffff_f800, 0x0000_07ff], [0x0fff_u32, 0x0800, 0x07ff]);
            (32, 32, Unsigned, [16_777_217, u32::MAX, 1], [16_777_217_u32, u32::MAX, 1]);
        );
    }
}

#[test]
fn stored_samples_exclude_modality_rescale() {
    let mut layout = layout(32, 12, PixelSignedness::Signed);
    layout.rescale_slope = f32::NAN;
    layout.rescale_intercept = f32::INFINITY;
    case(
        layout,
        [0xa000_000f, 0xffff_f800, 0x0000_07ff],
        ByteOrder::LeastSignificantByteFirst,
        [15_i32, -2048, 2047],
    );
}

#[test]
fn rejects_non_scalar_malformed_and_unsupported_frames() {
    let color = PixelLayout {
        samples_per_pixel: 3,
        ..layout(8, 8, PixelSignedness::Unsigned)
    };
    assert!(matches!(
        error(&[], color),
        StoredPixelError::UnsupportedSamplesPerPixel { actual: 3 }
    ));
    assert!(matches!(
        error(&[1, 0], layout(16, 16, PixelSignedness::Unsigned)),
        StoredPixelError::ByteLengthMismatch {
            actual: 2,
            expected: 6
        }
    ));
    let empty = PixelLayout {
        rows: 0,
        ..layout(8, 8, PixelSignedness::Unsigned)
    };
    let overflowing = PixelLayout {
        rows: usize::MAX,
        cols: 2,
        ..layout(8, 8, PixelSignedness::Unsigned)
    };
    for layout in [
        layout(1, 1, PixelSignedness::Unsigned),
        layout(16, 17, PixelSignedness::Unsigned),
        empty,
        overflowing,
    ] {
        rejects_layout(layout);
    }
}
