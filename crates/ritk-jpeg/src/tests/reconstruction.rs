use std::f64::consts::{FRAC_1_SQRT_2, PI};

use anyhow::Result;
use coeus_core::SequentialBackend;
use tempfile::tempdir;

use super::{image_from_values, reference_luma_values};
use crate::{read_jpeg, write_jpeg};

const BLOCK_WIDTH: usize = 8;

// ITU-T T.81 Annex K table K.1, scaled by the jpeg-encoder 0.7.1 quality
// rule: scale = 200 - 2 * quality and Q = (K * scale + 50) / 100.
const QUALITY_75_LUMA_QUANTIZATION: [u8; 64] = [
    8, 6, 5, 8, 12, 20, 26, 31, 6, 6, 7, 10, 13, 29, 30, 28, 7, 7, 8, 12, 20, 29, 35, 28, 7, 9, 11,
    15, 26, 44, 40, 31, 9, 11, 19, 28, 34, 55, 52, 39, 12, 18, 28, 32, 41, 52, 57, 46, 25, 32, 39,
    44, 52, 61, 60, 51, 36, 46, 48, 49, 56, 50, 52, 50,
];

const JPEG_ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

fn luma_quantization(path: &std::path::Path) -> [u8; 64] {
    let encoded = std::fs::read(path).expect("JPEG oracle fixture must be readable");
    let marker = encoded
        .windows(2)
        .position(|bytes| bytes == [0xff, 0xdb])
        .expect("quality-75 grayscale JPEG must carry a quantization table");
    let length = usize::from(u16::from_be_bytes([
        encoded[marker + 2],
        encoded[marker + 3],
    ]));
    assert_eq!(length, 67, "fixture must carry one eight-bit table");
    assert_eq!(encoded[marker + 4], 0, "fixture must use luma table zero");

    let mut natural = [0; 64];
    for (&value, &index) in encoded[marker + 5..marker + 69].iter().zip(&JPEG_ZIGZAG) {
        natural[index] = value;
    }
    natural
}

fn basis_scale(frequency: usize) -> f64 {
    if frequency == 0 {
        FRAC_1_SQRT_2
    } else {
        1.0
    }
}

fn basis(sample: usize, frequency: usize) -> f64 {
    let phase_index = u32::try_from((2 * sample + 1) * frequency)
        .expect("invariant: eight-sample DCT phase fits u32");
    (f64::from(phase_index) * PI / 16.0).cos()
}

fn analytical_reconstruction(source: &[f32], width: usize, height: usize) -> Vec<f64> {
    assert_eq!(source.len(), width * height);
    assert_eq!(width % BLOCK_WIDTH, 0);
    assert_eq!(height % BLOCK_WIDTH, 0);

    let mut output = vec![0.0; source.len()];
    for block_y in (0..height).step_by(BLOCK_WIDTH) {
        for block_x in (0..width).step_by(BLOCK_WIDTH) {
            let mut coefficients = [0.0; 64];
            for v in 0..BLOCK_WIDTH {
                for u in 0..BLOCK_WIDTH {
                    let mut sum = 0.0;
                    for y in 0..BLOCK_WIDTH {
                        for x in 0..BLOCK_WIDTH {
                            let sample = f64::from(
                                source[(block_y + y) * width + block_x + x]
                                    .round()
                                    .clamp(0.0, 255.0),
                            ) - 128.0;
                            sum += sample * basis(x, u) * basis(y, v);
                        }
                    }
                    let coefficient = 0.25 * basis_scale(u) * basis_scale(v) * sum;
                    let quantizer = f64::from(QUALITY_75_LUMA_QUANTIZATION[v * 8 + u]);
                    coefficients[v * 8 + u] = (coefficient / quantizer).round() * quantizer;
                }
            }

            for y in 0..BLOCK_WIDTH {
                for x in 0..BLOCK_WIDTH {
                    let mut sum = 0.0;
                    for v in 0..BLOCK_WIDTH {
                        for u in 0..BLOCK_WIDTH {
                            sum += basis_scale(u)
                                * basis_scale(v)
                                * coefficients[v * 8 + u]
                                * basis(x, u)
                                * basis(y, v);
                        }
                    }
                    output[(block_y + y) * width + block_x + x] =
                        (0.25 * sum + 128.0).round().clamp(0.0, 255.0);
                }
            }
        }
    }
    output
}

fn fourth_basis_sign(index: usize) -> f32 {
    if matches!(index % BLOCK_WIDTH, 0 | 3 | 4 | 7) {
        1.0
    } else {
        -1.0
    }
}

#[test]
fn writer_quality_75_gradient_matches_transform_oracle() {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    const SAMPLE_COUNT: usize = WIDTH * HEIGHT;

    let denominator = f32::from(
        u16::try_from(SAMPLE_COUNT - 1).expect("invariant: fixture sample count fits u16"),
    );
    let source: Vec<f32> = (0..SAMPLE_COUNT)
        .map(|index| {
            f32::from(u16::try_from(index).expect("invariant: fixture index fits u16"))
                / denominator
                * 255.0
        })
        .collect();
    let image = image_from_values([1, HEIGHT, WIDTH], source.clone());
    let directory = tempdir().expect("failed to create tempdir");
    let path = directory.path().join("gradient.jpg");

    write_jpeg(&path, &image, &SequentialBackend).expect("write_jpeg failed");

    assert_eq!(luma_quantization(&path), QUALITY_75_LUMA_QUANTIZATION);
    let decoded = reference_luma_values(&path);
    assert_eq!(
        decoded.iter().copied().map(f64::from).collect::<Vec<_>>(),
        analytical_reconstruction(&source, WIDTH, HEIGHT),
        "Annex A DCT, quality-75 quantization, and inverse DCT must predict every sample"
    );
    assert_eq!(
        read_jpeg(&path, &SequentialBackend)
            .expect("RITK must decode the analytical fixture")
            .data_slice()
            .expect("contiguous host data"),
        decoded
    );
}

#[test]
fn writer_quality_75_reconstruction_matches_analytical_blocks() -> Result<()> {
    const HEIGHT: usize = 8;
    const WIDTH: usize = 6 * BLOCK_WIDTH;

    let mut values = Vec::with_capacity(HEIGHT * WIDTH);
    let mut expected = Vec::with_capacity(HEIGHT * WIDTH);
    for y in 0..HEIGHT {
        values.extend(std::iter::repeat_n(-20.0, BLOCK_WIDTH));
        values.extend(std::iter::repeat_n(127.6, BLOCK_WIDTH));
        values.extend(std::iter::repeat_n(300.0, BLOCK_WIDTH));
        expected.extend(std::iter::repeat_n(0.0, BLOCK_WIDTH));
        expected.extend(std::iter::repeat_n(128.0, BLOCK_WIDTH));
        expected.extend(std::iter::repeat_n(255.0, BLOCK_WIDTH));
        let horizontal: [f32; BLOCK_WIDTH] =
            std::array::from_fn(|x| 128.0 + 12.0 * fourth_basis_sign(x));
        let vertical = [128.0 + 9.0 * fourth_basis_sign(y); BLOCK_WIDTH];
        let diagonal: [f32; BLOCK_WIDTH] =
            std::array::from_fn(|x| 128.0 + 17.0 * fourth_basis_sign(x) * fourth_basis_sign(y));
        values.extend(horizontal);
        values.extend(vertical);
        values.extend(diagonal);
        expected.extend(horizontal);
        expected.extend(vertical);
        expected.extend(diagonal);
    }
    let image = image_from_values([1, HEIGHT, WIDTH], values);
    let directory = tempdir()?;
    let path = directory.path().join("analytical-blocks.jpg");

    write_jpeg(&path, &image, &SequentialBackend)?;

    assert_eq!(luma_quantization(&path), QUALITY_75_LUMA_QUANTIZATION);

    // T.81 Annex A gives F(0,0) = 8a for a constant centered level `a`.
    // The fourth basis signs +--++--+ likewise give F(0,4), F(4,0), or
    // F(4,4) = 8a. The covered quantizers divide those coefficients exactly,
    // so the analytical source-to-reconstruction bound is zero.
    assert_eq!(reference_luma_values(&path), expected);
    assert_eq!(
        read_jpeg(&path, &SequentialBackend)?
            .data_slice()
            .expect("contiguous host data"),
        expected
    );

    let mut changed_quantizer = std::fs::read(&path)?;
    let marker = changed_quantizer
        .windows(2)
        .position(|bytes| bytes == [0xff, 0xdb])
        .expect("quality-75 grayscale JPEG must carry a quantization table");
    let horizontal = JPEG_ZIGZAG
        .iter()
        .position(|&index| index == 4)
        .expect("natural coefficient four must have a zig-zag position");
    let table_value = &mut changed_quantizer[marker + 5 + horizontal];
    *table_value = table_value
        .checked_add(1)
        .expect("quality-75 quantizer has room for the mutation");
    let changed_path = directory.path().join("changed-quantizer.jpg");
    std::fs::write(&changed_path, changed_quantizer)?;
    assert_ne!(
        reference_luma_values(&changed_path),
        expected,
        "the analytical oracle must reject a changed horizontal quantizer"
    );
    Ok(())
}
