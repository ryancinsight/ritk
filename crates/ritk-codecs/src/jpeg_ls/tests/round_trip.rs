//! Differential encoder↔decoder round-trip tests.
//!
//! The encoder ([`encoder::encode_grayscale_jpeg_ls`]) and the scan decoder
//! are independent code paths over the shared context model; lossless coding
//! (NEAR = 0) requires exact reconstruction: `decoded[i] == original[i]` ∀ i.

use super::*;
use crate::jpeg_ls::encoder::encode_grayscale_jpeg_ls;
use crate::PixelSignedness;

fn layout(rows: usize, cols: usize, bits: u16) -> PixelLayout {
    PixelLayout {
        rows,
        cols,
        samples_per_pixel: 1,
        bits_allocated: bits,
        pixel_representation: PixelSignedness::Unsigned,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    }
}

fn round_trip(samples: &[u16], rows: u32, cols: u32, bpp: u32) {
    let stream = encode_grayscale_jpeg_ls(samples, rows, cols, bpp, 0);
    assert_eq!(&stream[..2], &[0xFF, 0xD8], "stream must start with SOI");
    assert_eq!(
        &stream[stream.len() - 2..],
        &[0xFF, 0xD9],
        "stream must end with EOI"
    );
    let bits = if bpp <= 8 { 8u16 } else { 16 };
    let decoded = decode_jpeg_ls_fragment(&stream, layout(rows as usize, cols as usize, bits))
        .expect("native JPEG-LS round-trip must decode");
    assert_eq!(decoded.len(), samples.len());
    for (i, (&orig, &dec)) in samples.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(
            f32::from(orig),
            dec,
            "sample[{i}] must round-trip exactly (lossless invariant)"
        );
    }
}

#[test]
fn round_trip_uniform_8bit() {
    round_trip(&[128u16; 16], 4, 4, 8);
}

#[test]
fn round_trip_gradient_8bit() {
    let samples: Vec<u16> = (0..64u16).map(|v| v * 4).collect();
    round_trip(&samples, 8, 8, 8);
}

#[test]
fn round_trip_vertical_edges_8bit() {
    // Alternating columns exercise regular mode with strong gradients.
    let samples: Vec<u16> = (0..64u16)
        .map(|i| if (i % 8) < 4 { 10 } else { 240 })
        .collect();
    round_trip(&samples, 8, 8, 8);
}

#[test]
fn round_trip_run_mode_with_interrupts_8bit() {
    // Long flat runs broken by isolated spikes exercise run mode, run
    // interruption samples of both types, and the run-index adaptation.
    let mut samples = vec![77u16; 96];
    samples[13] = 200;
    samples[14] = 200;
    samples[47] = 0;
    samples[95] = 255;
    round_trip(&samples, 8, 12, 8);
}

#[test]
fn round_trip_full_range_12bit() {
    let samples: Vec<u16> = (0..32u16).map(|i| i * 132 + (i % 3)).collect();
    round_trip(&samples, 4, 8, 12);
}

#[test]
fn round_trip_full_range_16bit() {
    let samples: Vec<u16> = vec![
        0, 1, 2, 65535, 65534, 32768, 32767, 4095, 256, 512, 1024, 2048, 100, 200, 400, 800,
    ];
    round_trip(&samples, 4, 4, 16);
}

#[test]
fn round_trip_single_row_and_single_column() {
    round_trip(&[5, 5, 5, 9, 5, 5, 200], 1, 7, 8);
    round_trip(&[5, 5, 5, 9, 5, 5, 200], 7, 1, 8);
}

#[test]
fn round_trip_single_pixel() {
    round_trip(&[42], 1, 1, 8);
}

proptest::proptest! {
    /// Lossless invariant over random images: any sample matrix in the bpp
    /// dynamic range must round-trip exactly through encode → decode.
    #[test]
    fn round_trip_random(
        rows in 1u32..12,
        cols in 1u32..12,
        bpp in proptest::sample::select(vec![8u32, 12, 16]),
        seed in proptest::num::u64::ANY,
    ) {
        let n = (rows * cols) as usize;
        let mask = (1u64 << bpp) - 1;
        let mut state = seed | 1;
        // Mix of random values and runs (LCG decides) to hit both modes.
        let mut samples = Vec::with_capacity(n);
        let mut last = 0u16;
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = state >> 33;
            if r.is_multiple_of(3) {
                samples.push(last); // extend a run
            } else {
                last = (r & mask) as u16;
                samples.push(last);
            }
        }
        let stream = encode_grayscale_jpeg_ls(&samples, rows, cols, bpp, 0);
        let bits = if bpp <= 8 { 8u16 } else { 16 };
        let decoded = decode_jpeg_ls_fragment(
            &stream,
            layout(rows as usize, cols as usize, bits),
        ).expect("random round-trip must decode");
        let expected: Vec<f32> = samples.iter().map(|&v| f32::from(v)).collect();
        proptest::prop_assert_eq!(decoded, expected);
    }

    /// Near-lossless invariant over random images: reconstruction error is
    /// bounded by NEAR per sample (ISO 14495-1 §A.4.4; the bound is exact).
    #[test]
    fn round_trip_random_near_lossless(
        rows in 1u32..10,
        cols in 1u32..10,
        near in 1u32..4,
        seed in proptest::num::u64::ANY,
    ) {
        let n = (rows * cols) as usize;
        let mut state = seed | 1;
        let samples: Vec<u16> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) & 0xFF) as u16
            })
            .collect();
        let stream = encode_grayscale_jpeg_ls(&samples, rows, cols, 8, near);
        let decoded = decode_jpeg_ls_fragment(
            &stream,
            layout(rows as usize, cols as usize, 8),
        ).expect("near-lossless round-trip must decode");
        for (i, (&orig, &dec)) in samples.iter().zip(decoded.iter()).enumerate() {
            let err = (f32::from(orig) - dec).abs();
            proptest::prop_assert!(
                err <= near as f32,
                "sample[{}]: error {} exceeds NEAR={}", i, err, near
            );
        }
    }
}

/// Regression (JLS-16BIT-LOSSLESS): a trailing 0xFF entropy byte directly
/// before EOI was discarded as a marker prefix, corrupting the last samples
/// (fixed by emitting the stuffed 7-bit follow byte at flush).
#[test]
fn round_trip_16bit_regression_seed() {
    // Proptest generator (run-mixing LCG), minimized case rows 3 × cols 8.
    let mut state = 18395098268947010898u64 | 1;
    let mut samples: Vec<u16> = Vec::with_capacity(24);
    let mut last = 0u16;
    for _ in 0..24 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = state >> 33;
        if r.is_multiple_of(3) {
            samples.push(last);
        } else {
            last = (r & 0xFFFF) as u16;
            samples.push(last);
        }
    }
    round_trip(&samples, 3, 8, 16);
}
