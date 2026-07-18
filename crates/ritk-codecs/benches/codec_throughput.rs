//! Codec throughput baselines (criterion).
//!
//! # Methodology (performance_engineering)
//! - Inputs are pinned: deterministic LCG-generated images at fixed sizes.
//! - Reported metric: median wall time per encode/decode of one frame.
//! - Machine class is recorded by criterion in the saved baseline; compare
//!   with `cargo bench -p ritk-codecs -- --baseline <name>`.
//!
//! Workloads:
//! - JPEG-LS lossless 512Ã—512 16-bit (CT-slice-class payload).
//! - JPEG-LS near-lossless (NEAR=2) 512Ã—512 16-bit.
//! - JPEG 2000 lossless 64Ã—64 16-bit (single code-block) and 512Ã—512 16-bit
//!   with 5 DWT levels (multi-code-block, multi-resolution).

use criterion::{criterion_group, criterion_main, Criterion};
use ritk_codecs::jpeg_2000::encoder::{encode_grayscale_j2k, WaveletTransform};
use ritk_codecs::jpeg_ls::encoder::encode_grayscale_jpeg_ls;
use ritk_codecs::{decode_jpeg2000_fragment, decode_jpeg_ls_fragment};
use ritk_codecs::{PixelLayout, PixelSignedness};
use std::hint::black_box;

/// Deterministic 12-bit-range CT-like image: smooth gradient + LCG noise.
fn synthetic_image(rows: usize, cols: usize) -> Vec<u16> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..rows * cols)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((state >> 33) & 0x3F) as usize;
            let base = (i % cols) * 7 + (i / cols) * 3;
            ((base + noise) % 4096) as u16
        })
        .collect()
}

fn layout(rows: usize, cols: usize, bits: u16) -> PixelLayout {
    PixelLayout {
        rows,
        cols,
        samples_per_pixel: 1,
        bits_allocated: bits,
        pixel_representation: PixelSignedness::Unsigned,
        rescale_slope: 1.0,
        rescale_intercept: 0.0 }
}

fn bench_jpeg_ls(c: &mut Criterion) {
    let (rows, cols) = (512usize, 512usize);
    let img = synthetic_image(rows, cols);

    c.bench_function("jpeg_ls_encode_512x512_16bit_lossless", |b| {
        b.iter(|| {
            black_box(encode_grayscale_jpeg_ls(
                black_box(&img),
                rows as u32,
                cols as u32,
                16,
                0,
            ))
        })
    });

    let stream = encode_grayscale_jpeg_ls(&img, rows as u32, cols as u32, 16, 0);
    c.bench_function("jpeg_ls_decode_512x512_16bit_lossless", |b| {
        b.iter(|| {
            black_box(decode_jpeg_ls_fragment(
                black_box(&stream),
                layout(rows, cols, 16),
            ))
        })
    });

    c.bench_function("jpeg_ls_encode_512x512_16bit_near2", |b| {
        b.iter(|| {
            black_box(encode_grayscale_jpeg_ls(
                black_box(&img),
                rows as u32,
                cols as u32,
                16,
                2,
            ))
        })
    });
}

fn bench_jpeg_2000(c: &mut Criterion) {
    let (rows, cols) = (64usize, 64usize);
    let img: Vec<i32> = synthetic_image(rows, cols)
        .iter()
        .map(|&v| v as i32)
        .collect();

    c.bench_function("jpeg2000_encode_64x64_16bit_lossless", |b| {
        b.iter(|| {
            black_box(encode_grayscale_j2k(
                black_box(&img),
                rows as u32,
                cols as u32,
                16,
                PixelSignedness::Unsigned,
                0,
                WaveletTransform::Reversible,
            ))
        })
    });

    let stream = encode_grayscale_j2k(
        &img,
        rows as u32,
        cols as u32,
        16,
        PixelSignedness::Unsigned,
        0,
        WaveletTransform::Reversible,
    );
    c.bench_function("jpeg2000_decode_64x64_16bit_lossless", |b| {
        b.iter(|| {
            black_box(decode_jpeg2000_fragment(
                black_box(&stream),
                layout(rows, cols, 16),
            ))
        })
    });
}

fn bench_jpeg_2000_full(c: &mut Criterion) {
    let (rows, cols) = (512usize, 512usize);
    let img: Vec<i32> = synthetic_image(rows, cols)
        .iter()
        .map(|&v| v as i32)
        .collect();

    c.bench_function("jpeg2000_encode_512x512_16bit_5levels", |b| {
        b.iter(|| {
            black_box(encode_grayscale_j2k(
                black_box(&img),
                rows as u32,
                cols as u32,
                16,
                PixelSignedness::Unsigned,
                5,
                WaveletTransform::Reversible,
            ))
        })
    });

    let stream = encode_grayscale_j2k(
        &img,
        rows as u32,
        cols as u32,
        16,
        PixelSignedness::Unsigned,
        5,
        WaveletTransform::Reversible,
    );
    c.bench_function("jpeg2000_decode_512x512_16bit_5levels", |b| {
        b.iter(|| {
            black_box(decode_jpeg2000_fragment(
                black_box(&stream),
                layout(rows, cols, 16),
            ))
        })
    });
}

criterion_group!(
    benches,
    bench_jpeg_ls,
    bench_jpeg_2000,
    bench_jpeg_2000_full
);
criterion_main!(benches);
