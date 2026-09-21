//! Baseline JPEG input construction for the codec benchmark.

const DCT_BLOCK_DIMENSION: usize = 8;

/// Builds a baseline DCT stream with zero coefficients at the requested shape.
///
/// # Panics
///
/// Panics if the component count is outside 1..=4 or a dimension is zero.
pub(super) fn baseline_fixture(components: usize, width: u16, height: u16) -> Vec<u8> {
    assert!(
        (1..=4).contains(&components),
        "T.81 section B.2.3 admits 1 to 4 components, got {components}"
    );
    assert!(width > 0 && height > 0, "a frame needs positive dimensions");

    let mut stream = vec![0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x43, 0x00];
    stream.extend(std::iter::repeat_n(0x01, 64));

    stream.extend_from_slice(&[0xFF, 0xC0]);
    let frame_header_len = u16::try_from(8 + 3 * components)
        .expect("invariant: at most four components fit the frame header length");
    let component_count =
        u8::try_from(components).expect("invariant: at most four components fit in a byte");
    stream.extend_from_slice(&frame_header_len.to_be_bytes());
    stream.push(0x08);
    stream.extend_from_slice(&height.to_be_bytes());
    stream.extend_from_slice(&width.to_be_bytes());
    stream.push(component_count);
    for id in 1..=component_count {
        stream.extend_from_slice(&[id, 0x11, 0x00]);
    }

    for class_and_id in [0x00_u8, 0x10] {
        stream.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, class_and_id]);
        stream.push(0x01);
        stream.extend(std::iter::repeat_n(0x00, 15));
        stream.push(0x00);
    }

    stream.extend_from_slice(&[0xFF, 0xDA]);
    let scan_header_len = u16::try_from(6 + 2 * components)
        .expect("invariant: at most four components fit the scan header length");
    stream.extend_from_slice(&scan_header_len.to_be_bytes());
    stream.push(component_count);
    for id in 1..=component_count {
        stream.extend_from_slice(&[id, 0x00]);
    }
    stream.extend_from_slice(&[0x00, 0x3F, 0x00]);

    let blocks_x = usize::from(width).div_ceil(DCT_BLOCK_DIMENSION);
    let blocks_y = usize::from(height).div_ceil(DCT_BLOCK_DIMENSION);
    let bits = 2 * components * blocks_x * blocks_y;
    stream.extend(std::iter::repeat_n(0x00, bits / 8));
    let spare = bits % 8;
    if spare != 0 {
        stream.push(
            u8::try_from(0xFF_u16 >> spare).expect("invariant: shifted padding byte fits in u8"),
        );
    }

    stream.extend_from_slice(&[0xFF, 0xD9]);
    stream
}
