//! Pixel layout and native sample decoding.
//!
//! # Contract
//! Native byte decode applies `output = sample * slope + intercept`.

use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelLayout {
    pub rows: usize,
    pub cols: usize,
    pub samples_per_pixel: usize,
    pub bits_allocated: u16,
    pub pixel_representation: u16,
    pub rescale_slope: f32,
    pub rescale_intercept: f32,
}

impl PixelLayout {
    pub fn pixels_per_frame(self) -> Result<usize> {
        let pixels = self
            .rows
            .checked_mul(self.cols)
            .ok_or_else(|| anyhow::anyhow!("pixel layout rows*cols overflows usize"))?;
        if pixels == 0 {
            bail!(
                "pixel layout rows={} cols={} yields an empty frame",
                self.rows,
                self.cols
            );
        }
        Ok(pixels)
    }

    pub fn samples_per_frame(self) -> Result<usize> {
        let pixels = self.pixels_per_frame()?;
        if self.samples_per_pixel == 0 {
            bail!("samples_per_pixel=0 is invalid");
        }
        pixels
            .checked_mul(self.samples_per_pixel)
            .ok_or_else(|| anyhow::anyhow!("pixel layout sample count overflows usize"))
    }

    pub fn bytes_per_sample(self) -> Result<usize> {
        if self.bits_allocated % 8 != 0 {
            bail!(
                "bits_allocated={} is not byte-addressable",
                self.bits_allocated
            );
        }
        let bytes = (self.bits_allocated / 8) as usize;
        if !(1..=4).contains(&bytes) {
            bail!(
                "bits_allocated={} gives bytes_per_sample={} outside 1..=4",
                self.bits_allocated,
                bytes
            );
        }
        Ok(bytes)
    }

    pub fn bytes_per_frame(self) -> Result<usize> {
        self.samples_per_frame()?
            .checked_mul(self.bytes_per_sample()?)
            .ok_or_else(|| anyhow::anyhow!("pixel layout byte count overflows usize"))
    }
}

pub fn decode_native_pixel_bytes(bytes: &[u8], layout: PixelLayout) -> Vec<f32> {
    match (layout.bits_allocated, layout.pixel_representation) {
        (8, _) => bytes
            .iter()
            .map(|&b| b as f32 * layout.rescale_slope + layout.rescale_intercept)
            .collect(),
        (16, 1) => bytes
            .chunks_exact(2)
            .map(|c| {
                i16::from_le_bytes([c[0], c[1]]) as f32 * layout.rescale_slope
                    + layout.rescale_intercept
            })
            .collect(),
        _ => bytes
            .chunks_exact(2)
            .map(|c| {
                u16::from_le_bytes([c[0], c[1]]) as f32 * layout.rescale_slope
                    + layout.rescale_intercept
            })
            .collect(),
    }
}

pub fn decode_native_pixel_bytes_checked(bytes: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    let expected = layout.bytes_per_frame()?;
    if bytes.len() != expected {
        bail!(
            "native pixel byte length {} does not match expected frame byte length {}",
            bytes.len(),
            expected
        );
    }
    Ok(decode_native_pixel_bytes(bytes, layout))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_signed_16_decode_applies_linear_modality_lut() {
        let bytes = [-2i16, 0, 10]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let out = decode_native_pixel_bytes(
            &bytes,
            PixelLayout {
                rows: 1,
                cols: 3,
                samples_per_pixel: 1,
                bits_allocated: 16,
                pixel_representation: 1,
                rescale_slope: 2.0,
                rescale_intercept: 5.0,
            },
        );
        assert_eq!(out, vec![1.0, 5.0, 25.0]);
    }

    #[test]
    fn checked_native_decode_rejects_trailing_bytes() {
        let err = decode_native_pixel_bytes_checked(
            &[1, 0, 2],
            PixelLayout {
                rows: 1,
                cols: 1,
                samples_per_pixel: 1,
                bits_allocated: 16,
                pixel_representation: 0,
                rescale_slope: 1.0,
                rescale_intercept: 0.0,
            },
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("byte length"),
            "expected byte-length validation error, got {err:#}"
        );
    }
}
