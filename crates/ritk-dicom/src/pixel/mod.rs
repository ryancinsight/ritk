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

    pub fn validate_pixel_representation(self) -> Result<()> {
        match self.pixel_representation {
            0 | 1 => Ok(()),
            other => bail!("pixel_representation={} is invalid; expected 0 or 1", other),
        }
    }

    pub fn validate_rescale_parameters(self) -> Result<()> {
        if !self.rescale_slope.is_finite() {
            bail!("rescale_slope={} is not finite", self.rescale_slope);
        }
        if !self.rescale_intercept.is_finite() {
            bail!("rescale_intercept={} is not finite", self.rescale_intercept);
        }
        Ok(())
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
        (16, _) => bytes
            .chunks_exact(2)
            .map(|c| {
                u16::from_le_bytes([c[0], c[1]]) as f32 * layout.rescale_slope
                    + layout.rescale_intercept
            })
            .collect(),
        (24, 1) => bytes
            .chunks_exact(3)
            .map(|c| sign_extend_i24(c) as f32 * layout.rescale_slope + layout.rescale_intercept)
            .collect(),
        (24, _) => bytes
            .chunks_exact(3)
            .map(|c| u24_le(c) as f32 * layout.rescale_slope + layout.rescale_intercept)
            .collect(),
        (32, 1) => bytes
            .chunks_exact(4)
            .map(|c| {
                i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 * layout.rescale_slope
                    + layout.rescale_intercept
            })
            .collect(),
        (32, _) => bytes
            .chunks_exact(4)
            .map(|c| {
                u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 * layout.rescale_slope
                    + layout.rescale_intercept
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn u24_le(bytes: &[u8]) -> u32 {
    u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16)
}

fn sign_extend_i24(bytes: &[u8]) -> i32 {
    let raw = u24_le(bytes) as i32;
    if raw & 0x0080_0000 != 0 {
        raw | !0x00FF_FFFF
    } else {
        raw
    }
}

pub fn decode_native_pixel_bytes_checked(bytes: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    layout.validate_pixel_representation()?;
    layout.validate_rescale_parameters()?;
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

    #[test]
    fn checked_native_decode_handles_unsigned_32bit_samples() {
        let bytes = [1u32, 65_535, 1_000_000]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();

        let out = decode_native_pixel_bytes_checked(
            &bytes,
            PixelLayout {
                rows: 1,
                cols: 3,
                samples_per_pixel: 1,
                bits_allocated: 32,
                pixel_representation: 0,
                rescale_slope: 0.5,
                rescale_intercept: -1.0,
            },
        )
        .unwrap();

        assert_eq!(out, vec![-0.5, 32766.5, 499999.0]);
    }

    #[test]
    fn checked_native_decode_handles_signed_24bit_samples() {
        let bytes = [-2i32, 0, 10]
            .iter()
            .flat_map(|v| {
                let le = v.to_le_bytes();
                [le[0], le[1], le[2]]
            })
            .collect::<Vec<_>>();

        let out = decode_native_pixel_bytes_checked(
            &bytes,
            PixelLayout {
                rows: 1,
                cols: 3,
                samples_per_pixel: 1,
                bits_allocated: 24,
                pixel_representation: 1,
                rescale_slope: 2.0,
                rescale_intercept: 5.0,
            },
        )
        .unwrap();

        assert_eq!(out, vec![1.0, 5.0, 25.0]);
    }

    #[test]
    fn checked_native_decode_handles_signed_32bit_samples() {
        let bytes = [-2i32, 0, 10]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();

        let out = decode_native_pixel_bytes_checked(
            &bytes,
            PixelLayout {
                rows: 1,
                cols: 3,
                samples_per_pixel: 1,
                bits_allocated: 32,
                pixel_representation: 1,
                rescale_slope: 2.0,
                rescale_intercept: 5.0,
            },
        )
        .unwrap();

        assert_eq!(out, vec![1.0, 5.0, 25.0]);
    }

    #[test]
    fn checked_native_decode_rejects_invalid_pixel_representation() {
        let err = decode_native_pixel_bytes_checked(
            &[1],
            PixelLayout {
                rows: 1,
                cols: 1,
                samples_per_pixel: 1,
                bits_allocated: 8,
                pixel_representation: 2,
                rescale_slope: 1.0,
                rescale_intercept: 0.0,
            },
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("pixel_representation"),
            "expected pixel representation validation error, got {err:#}"
        );
    }

    #[test]
    fn checked_native_decode_rejects_nonfinite_rescale_slope() {
        let err = decode_native_pixel_bytes_checked(
            &[1],
            PixelLayout {
                rows: 1,
                cols: 1,
                samples_per_pixel: 1,
                bits_allocated: 8,
                pixel_representation: 0,
                rescale_slope: f32::NAN,
                rescale_intercept: 0.0,
            },
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("rescale_slope"),
            "expected rescale_slope validation error, got {err:#}"
        );
    }

    #[test]
    fn checked_native_decode_rejects_nonfinite_rescale_intercept() {
        let err = decode_native_pixel_bytes_checked(
            &[1],
            PixelLayout {
                rows: 1,
                cols: 1,
                samples_per_pixel: 1,
                bits_allocated: 8,
                pixel_representation: 0,
                rescale_slope: 1.0,
                rescale_intercept: f32::INFINITY,
            },
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("rescale_intercept"),
            "expected rescale_intercept validation error, got {err:#}"
        );
    }
}
