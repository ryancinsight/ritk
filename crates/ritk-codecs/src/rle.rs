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

    let mut raw =
        Vec::with_capacity(pixels_per_frame * layout.samples_per_pixel * bytes_per_sample);
    for pixel_idx in 0..pixels_per_frame {
        for sample_idx in 0..layout.samples_per_pixel {
            for le_byte_idx in 0..bytes_per_sample {
                let segment_idx =
                    sample_idx * bytes_per_sample + (bytes_per_sample - 1 - le_byte_idx);
                raw.push(segments[segment_idx][pixel_idx]);
            }
        }
    }

    decode_native_pixel_bytes_checked(&raw, layout)
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
                pixel_representation: 0,
                rescale_slope: 1.0,
                rescale_intercept: 0.0,
            },
        )
        .unwrap();
        assert_eq!(decoded, vec![42.0, 7.0, 128.0, 255.0]);
    }
}
