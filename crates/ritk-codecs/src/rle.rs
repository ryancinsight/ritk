//! DICOM RLE Lossless frame decoding.
//!
//! # Correctness
//! PackBits is lossless and byte-plane reassembly is a permutation from DICOM
//! segment order to little-endian sample order, so decoded integer samples equal
//! the encoded samples exactly before modality LUT application.

use anyhow::{bail, Context, Result};

use crate::packbits_decode;
use crate::{decode_native_pixel_bytes_checked, PixelLayout};

pub fn decode_rle_lossless_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    const HEADER_BYTES: usize = 64;
    let pixels_per_frame = layout.pixels_per_frame()?;
    let bytes_per_sample = layout.bytes_per_sample()?;
    let expected_segments = layout.samples_per_pixel * bytes_per_sample;
    if expected_segments == 0 || expected_segments > 15 {
        bail!(
            "RLE segment count {} is outside DICOM header capacity 1..=15",
            expected_segments
        );
    }
    if fragment.len() < HEADER_BYTES {
        bail!(
            "RLE fragment length {} is smaller than {} byte header",
            fragment.len(),
            HEADER_BYTES
        );
    }

    let n_segments = read_u32_le(fragment, 0, "RLE segment count")? as usize;
    if n_segments != expected_segments {
        bail!(
            "RLE header declares {} segments; expected {}",
            n_segments,
            expected_segments
        );
    }

    let offsets: Vec<usize> = (0..n_segments)
        .map(|k| read_u32_le(fragment, 4 + k * 4, "RLE segment offset").map(|v| v as usize))
        .collect::<Result<Vec<_>>>()?;
    for pair in offsets.windows(2) {
        if pair[0] >= pair[1] {
            bail!(
                "RLE segment offsets are not strictly increasing: {:?}",
                offsets
            );
        }
    }

    let mut segments = Vec::with_capacity(n_segments);
    for (idx, &offset) in offsets.iter().enumerate() {
        if offset >= fragment.len() {
            bail!(
                "RLE segment {} offset {} exceeds fragment length {}",
                idx,
                offset,
                fragment.len()
            );
        }
        let end = if idx + 1 < offsets.len() {
            offsets[idx + 1]
        } else {
            fragment.len()
        };
        let segment = packbits_decode(&fragment[offset..end], pixels_per_frame)
            .with_context(|| format!("RLE PackBits segment {idx} decode failed"))?;
        segments.push(segment);
    }

    // Precompute the segment access order: for each output byte position (sample, le_byte),
    // record which segment index to read. This avoids indexing `segments` with a range
    // loop variable, replacing triple-nested for loops with flat_map iterators.
    let segment_order: Vec<usize> = (0..layout.samples_per_pixel)
        .flat_map(|s| {
            (0..bytes_per_sample).map(move |b| s * bytes_per_sample + (bytes_per_sample - 1 - b))
        })
        .collect();

    let mut raw =
        Vec::with_capacity(pixels_per_frame * layout.samples_per_pixel * bytes_per_sample);
    let segs = &segments;
    raw.extend(
        (0..pixels_per_frame).flat_map(|pi| segment_order.iter().map(move |&si| segs[si][pi])),
    );

    decode_native_pixel_bytes_checked(&raw, layout)
}

/// Encode 16-bit single-channel pixels into one DICOM RLE Lossless fragment.
///
/// The returned bytes are a complete encapsulated fragment payload:
/// 64-byte RLE header + PackBits-compressed segments.
/// Segment ordering follows DICOM PS3.5 Annex G byte-plane order:
/// segment 0 = high byte plane, segment 1 = low byte plane.
pub fn encode_rle_lossless_fragment_u16_grayscale(pixels: &[u16]) -> Vec<u8> {
    let mut high = Vec::with_capacity(pixels.len());
    let mut low = Vec::with_capacity(pixels.len());
    for &p in pixels {
        high.push((p >> 8) as u8);
        low.push((p & 0x00FF) as u8);
    }

    let high_encoded = packbits_encode(&high);
    let low_encoded = packbits_encode(&low);
    const HEADER_BYTES: usize = 64;
    let mut header = [0u32; 16];
    header[0] = 2;
    header[1] = HEADER_BYTES as u32;
    header[2] = (HEADER_BYTES + high_encoded.len()) as u32;

    let mut out = Vec::with_capacity(HEADER_BYTES + high_encoded.len() + low_encoded.len());
    for &w in &header {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out.extend_from_slice(&high_encoded);
    out.extend_from_slice(&low_encoded);
    out
}

fn packbits_encode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 128 + 2);
    let mut i = 0usize;
    while i < data.len() {
        let mut repeat = 1usize;
        while i + repeat < data.len() && data[i + repeat] == data[i] && repeat < 128 {
            repeat += 1;
        }
        if repeat >= 2 {
            out.push((257 - repeat) as u8);
            out.push(data[i]);
            i += repeat;
            continue;
        }
        let lit_start = i;
        let mut lit = 1usize;
        while i + lit < data.len() && lit < 128 {
            if i + lit + 1 < data.len() && data[i + lit] == data[i + lit + 1] {
                break;
            }
            lit += 1;
        }
        out.push((lit - 1) as u8);
        out.extend_from_slice(&data[lit_start..lit_start + lit]);
        i += lit;
    }
    // DICOM RLE segments are even-length padded.
    if out.len() % 2 != 0 {
        out.push(0x00);
    }
    out
}

fn read_u32_le(bytes: &[u8], offset: usize, field: &str) -> Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| anyhow::anyhow!("{field} offset overflows usize"))?;
    let chunk = bytes
        .get(offset..end)
        .ok_or_else(|| anyhow::anyhow!("{field} at offset {offset} exceeds byte buffer"))?;
    Ok(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rle_8bit_grayscale_fragment_decodes_exact_values() {
        let pixels = [42u8, 7, 128, 255];
        let mut fragment = vec![0u8; 64];
        fragment[0..4].copy_from_slice(&1u32.to_le_bytes());
        fragment[4..8].copy_from_slice(&64u32.to_le_bytes());
        fragment.push((pixels.len() - 1) as u8);
        fragment.extend_from_slice(&pixels);

        let decoded = decode_rle_lossless_fragment(
            &fragment,
            PixelLayout {
                rows: 2,
                cols: 2,
                samples_per_pixel: 1,
                bits_allocated: 8,
                pixel_representation: crate::PixelSignedness::Unsigned,
                rescale_slope: 1.0,
                rescale_intercept: 0.0,
            },
        )
        .unwrap();
        assert_eq!(decoded, vec![42.0, 7.0, 128.0, 255.0]);
    }

    #[test]
    fn rle_16bit_grayscale_encode_decode_roundtrip_is_exact() {
        let original: Vec<u16> = vec![0, 1, 255, 256, 1024, 4095, 65535, 42];
        let fragment = encode_rle_lossless_fragment_u16_grayscale(&original);
        let decoded = decode_rle_lossless_fragment(
            &fragment,
            PixelLayout {
                rows: 2,
                cols: 4,
                samples_per_pixel: 1,
                bits_allocated: 16,
                pixel_representation: crate::PixelSignedness::Unsigned,
                rescale_slope: 1.0,
                rescale_intercept: 0.0,
            },
        )
        .expect("RLE decode must succeed");
        let expected: Vec<f32> = original.iter().map(|&v| f32::from(v)).collect();
        assert_eq!(decoded, expected);
    }
}
