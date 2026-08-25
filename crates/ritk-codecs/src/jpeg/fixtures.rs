//! Minimal conforming baseline JPEG streams, built in code.
//!
//! The committed sample fragments are all single-component lossless (SOF3), so
//! nothing exercised the baseline DCT decoder — which is why its bounds went
//! unchecked until an audit read them. These builders produce the smallest
//! conforming SOF0 streams that reach it, at any size the caller asks for.
//!
//! Every block codes DC category 0 (difference zero) followed immediately by
//! EOB, so all coefficients are zero and each component reconstructs to the
//! level shift, 128. That keeps the entropy stream trivial to generate at any
//! dimension while still driving the real Huffman, dequantisation and IDCT
//! path once per block — which is what makes it usable as a benchmark input as
//! well as a corruption-sweep fixture.
//!
//! Public rather than test-gated: benchmarks are a separate compilation unit
//! from the test harness, so a `#[cfg(test)]` module cannot reach them, and a
//! cargo feature to bridge that would be a build toggle rather than the
//! dependency management features exist for. Downstream crates verifying their
//! own JPEG integration can use these for the same reason.

use crate::jpeg::constants::DCT_BLOCK_DIM;

/// Three-component 8x8 YCbCr baseline stream.
#[must_use]
pub fn baseline_ycbcr_fixture() -> Vec<u8> {
    baseline_fixture(3, 8, 8)
}

/// Single-component 8x8 grayscale baseline stream.
///
/// The grayscale scan path is a separate function reached only when SOF
/// declares one component, so the three-component fixture never covers it.
#[must_use]
pub fn baseline_grayscale_fixture() -> Vec<u8> {
    baseline_fixture(1, 8, 8)
}

/// Build a baseline DCT stream of `components` 1x1-sampled channels at
/// `width` x `height`.
///
/// # Panics
///
/// Panics if `components` is outside 1-4, or either dimension is zero or
/// exceeds `u16::MAX` — none of which describe a stream this builder can emit.
#[must_use]
pub fn baseline_fixture(components: usize, width: u16, height: u16) -> Vec<u8> {
    assert!(
        (1..=4).contains(&components),
        "T.81 §B.2.3 admits 1 to 4 components, got {components}"
    );
    assert!(width > 0 && height > 0, "a frame needs positive dimensions");

    let mut stream = vec![
        0xFF, 0xD8, // SOI
        0xFF, 0xDB, // DQT
        0x00, 0x43, // length 67 = 2 + 1 + 64
        0x00, // Pq = 0 (8-bit), Tq = 0
    ];
    stream.extend(std::iter::repeat_n(0x01, 64)); // flat quantisation

    stream.extend_from_slice(&[0xFF, 0xC0]); // SOF0 (baseline DCT)
                                             // Length: 2 + precision + 2 dimensions + component count + 3 bytes each.
    stream.extend_from_slice(&((8 + 3 * components) as u16).to_be_bytes());
    stream.push(0x08); // precision 8
    stream.extend_from_slice(&height.to_be_bytes());
    stream.extend_from_slice(&width.to_be_bytes());
    stream.push(components as u8);
    for id in 1..=components as u8 {
        stream.extend_from_slice(&[id, 0x11, 0x00]); // 1x1 sampling, quant table 0
    }

    // DC table 0 and AC table 0, each a single one-bit code `0`: DC symbol 0
    // is category 0, AC symbol 0x00 is EOB.
    for class_and_id in [0x00u8, 0x10] {
        stream.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, class_and_id]);
        stream.push(0x01); // BITS[1] = one code of length 1
        stream.extend(std::iter::repeat_n(0x00, 15)); // BITS[2..=16] = 0
        stream.push(0x00); // HUFFVAL[0] = 0
    }

    stream.extend_from_slice(&[0xFF, 0xDA]); // SOS
    stream.extend_from_slice(&((6 + 2 * components) as u16).to_be_bytes());
    stream.push(components as u8);
    for id in 1..=components as u8 {
        stream.extend_from_slice(&[id, 0x00]); // DC table 0, AC table 0
    }
    stream.extend_from_slice(&[
        0x00, // Ss = 0
        0x3F, // Se = 63
        0x00, // Ah = 0, Al = 0
    ]);

    // Two bits per block — DC category 0, then EOB — for every block of every
    // MCU. All-zero bits never produce 0xFF, so no byte stuffing is needed.
    let blocks_x = (width as usize).div_ceil(DCT_BLOCK_DIM);
    let blocks_y = (height as usize).div_ceil(DCT_BLOCK_DIM);
    let bits = 2 * components * blocks_x * blocks_y;
    let full_bytes = bits / 8;
    stream.extend(std::iter::repeat_n(0x00, full_bytes));
    let spare = bits % 8;
    if spare != 0 {
        // Pad the final partial byte with 1-bits per T.81 §F.1.2.3.
        stream.push((0xFFu16 >> spare) as u8);
    }

    stream.extend_from_slice(&[0xFF, 0xD9]); // EOI
    stream
}
