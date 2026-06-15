//! Pixel layout and native sample decoding.
//!
//! # Contract
//! Native byte decode applies `output = sample * slope + intercept`.

use anyhow::{bail, Result};

/// Pixel signedness, replacing ad-hoc `u16` / `bool` representations.
///
/// DICOM PixelRepresentation (0028,0103) encodes signedness as 0 = unsigned,
/// 1 = signed two's complement. This enum lifts that convention into the type
/// system so invalid values (2, 3, …) are unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PixelSignedness {
    /// Unsigned pixel representation (PixelRepresentation = 0).
    #[default]
    Unsigned,
    /// Signed (two's complement) pixel representation (PixelRepresentation = 1).
    Signed,
}

impl PixelSignedness {
    /// Returns `true` for [`Signed`](PixelSignedness::Signed).
    pub fn is_signed(self) -> bool {
        matches!(self, Self::Signed)
    }
}

impl From<PixelSignedness> for u16 {
    fn from(value: PixelSignedness) -> Self {
        u16::from(value.is_signed())
    }
}

impl TryFrom<u16> for PixelSignedness {
    type Error = anyhow::Error;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Unsigned),
            1 => Ok(Self::Signed),
            other => bail!("pixel_representation={} is invalid; expected 0 or 1", other),
        }
    }
}

impl std::fmt::Display for PixelSignedness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsigned => write!(f, "Unsigned(0)"),
            Self::Signed => write!(f, "Signed(1)"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelLayout {
    pub rows: usize,
    pub cols: usize,
    pub samples_per_pixel: usize,
    pub bits_allocated: u16,
    pub pixel_representation: PixelSignedness,
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
        if !self.bits_allocated.is_multiple_of(8) {
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

#[deprecated(
    since = "0.14.9",
    note = "use decode_native_pixel_bytes_checked to validate DICOM pixel layout metadata"
)]
pub fn decode_native_pixel_bytes(bytes: &[u8], layout: PixelLayout) -> Vec<f32> {
    decode_native_pixel_bytes_unchecked(bytes, layout)
}

#[inline]
fn apply_rescale(sample: f32, layout: &PixelLayout) -> f32 {
    sample * layout.rescale_slope + layout.rescale_intercept
}

fn decode_native_pixel_bytes_unchecked(bytes: &[u8], layout: PixelLayout) -> Vec<f32> {
    match (layout.bits_allocated, layout.pixel_representation) {
        (8, PixelSignedness::Signed) => bytes
            .iter()
            .map(|&b| apply_rescale((b as i8) as f32, &layout))
            .collect(),
        (8, PixelSignedness::Unsigned) => bytes
            .iter()
            .map(|&b| apply_rescale(b as f32, &layout))
            .collect(),
        (16, PixelSignedness::Signed) => bytes
            .chunks_exact(2)
            .map(|c| apply_rescale(i16::from_le_bytes([c[0], c[1]]) as f32, &layout))
            .collect(),
        (16, PixelSignedness::Unsigned) => bytes
            .chunks_exact(2)
            .map(|c| apply_rescale(u16::from_le_bytes([c[0], c[1]]) as f32, &layout))
            .collect(),
        (24, PixelSignedness::Signed) => bytes
            .chunks_exact(3)
            .map(|c| apply_rescale(sign_extend_i24(c) as f32, &layout))
            .collect(),
        (24, PixelSignedness::Unsigned) => bytes
            .chunks_exact(3)
            .map(|c| apply_rescale(u24_le(c) as f32, &layout))
            .collect(),
        (32, PixelSignedness::Signed) => bytes
            .chunks_exact(4)
            .map(|c| apply_rescale(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32, &layout))
            .collect(),
        (32, PixelSignedness::Unsigned) => bytes
            .chunks_exact(4)
            .map(|c| apply_rescale(u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32, &layout))
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
    layout.validate_rescale_parameters()?;
    let expected = layout.bytes_per_frame()?;
    if bytes.len() != expected {
        bail!(
            "native pixel byte length {} does not match expected frame byte length {}",
            bytes.len(),
            expected
        );
    }
    Ok(decode_native_pixel_bytes_unchecked(bytes, layout))
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
        let out = decode_native_pixel_bytes_checked(
            &bytes,
            PixelLayout {
                rows: 1,
                cols: 3,
                samples_per_pixel: 1,
                bits_allocated: 16,
                pixel_representation: PixelSignedness::Signed,
                rescale_slope: 2.0,
                rescale_intercept: 5.0,
            },
        )
        .unwrap();
        assert_eq!(out, vec![1.0, 5.0, 25.0]);
    }

    #[test]
    fn native_signed_8_decode_applies_linear_modality_lut() {
        let bytes = [-2i8, 0, 10].iter().map(|v| *v as u8).collect::<Vec<_>>();

        let out = decode_native_pixel_bytes_checked(
            &bytes,
            PixelLayout {
                rows: 1,
                cols: 3,
                samples_per_pixel: 1,
                bits_allocated: 8,
                pixel_representation: PixelSignedness::Signed,
                rescale_slope: 2.0,
                rescale_intercept: 5.0,
            },
        )
        .unwrap();

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
                pixel_representation: PixelSignedness::Unsigned,
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
                pixel_representation: PixelSignedness::Unsigned,
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
                pixel_representation: PixelSignedness::Signed,
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
                pixel_representation: PixelSignedness::Signed,
                rescale_slope: 2.0,
                rescale_intercept: 5.0,
            },
        )
        .unwrap();

        assert_eq!(out, vec![1.0, 5.0, 25.0]);
    }

    #[test]
    fn checked_native_decode_rejects_invalid_pixel_representation_from_u16() {
        let err = PixelSignedness::try_from(2u16).unwrap_err();
        assert!(
            err.to_string().contains("pixel_representation"),
            "expected pixel representation conversion error, got {err:#}"
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
                pixel_representation: PixelSignedness::Unsigned,
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
                pixel_representation: PixelSignedness::Unsigned,
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

    /// GAP-R08g regression: signed i16 stored values with RescaleIntercept=-1024
    /// must produce correct Hounsfield units.
    ///
    /// DICOM PS3.3 C.7.6.3.1.4: output = stored_integer x slope + intercept.
    /// For CT with PixelRepresentation=1, BitsAllocated=16, Slope=1,
    /// Intercept=-1024: stored value -1024 -> HU = -1024*1 + (-1024) = -2048.
    /// This was the root cause of GAP-R08g where RITK produced min=-1024
    /// instead of the correct -2048.
    #[test]
    fn ct_signed_i16_rescale_intercept_minus_1024_produces_correct_hu() {
        // Stored values: -1024 (air), 0 (water), 1000 (bone)
        let stored: [i16; 3] = [-1024, 0, 1000];
        let bytes: Vec<u8> = stored.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = decode_native_pixel_bytes_checked(
            &bytes,
            PixelLayout {
                rows: 1,
                cols: 3,
                samples_per_pixel: 1,
                bits_allocated: 16,
                pixel_representation: PixelSignedness::Signed,
                rescale_slope: 1.0,
                rescale_intercept: -1024.0,
            },
        )
        .unwrap();
        // HU = stored * 1 + (-1024)
        assert_eq!(out[0], -2048.0, "air: -1024 * 1 + (-1024) = -2048 HU");
        assert_eq!(out[1], -1024.0, "water: 0 * 1 + (-1024) = -1024 HU");
        assert_eq!(out[2], -24.0, "bone: 1000 * 1 + (-1024) = -24 HU");
    }

    /// Verify that identity rescale (slope=1, intercept=0) passes stored values
    /// through unchanged, which is the correct behavior when rescale has already
    /// been applied upstream (e.g., by dicom-pixeldata in decode_via_dicom_rs).
    #[test]
    fn identity_rescale_preserves_signed_i16_stored_values() {
        let stored: [i16; 4] = [-1024, -512, 0, 3071];
        let bytes: Vec<u8> = stored.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = decode_native_pixel_bytes_checked(
            &bytes,
            PixelLayout {
                rows: 1,
                cols: 4,
                samples_per_pixel: 1,
                bits_allocated: 16,
                pixel_representation: PixelSignedness::Signed,
                rescale_slope: 1.0,
                rescale_intercept: 0.0,
            },
        )
        .unwrap();
        assert_eq!(out, vec![-1024.0, -512.0, 0.0, 3071.0]);
    }

    #[test]
    fn pixel_signedness_try_from_rejects_invalid_u16() {
        assert!(PixelSignedness::try_from(0u16).is_ok());
        assert!(PixelSignedness::try_from(1u16).is_ok());
        assert!(PixelSignedness::try_from(2u16).is_err());
        assert!(PixelSignedness::try_from(255u16).is_err());
    }

    #[test]
    fn pixel_signedness_from_u16_round_trips() {
        assert_eq!(u16::from(PixelSignedness::Unsigned), 0);
        assert_eq!(u16::from(PixelSignedness::Signed), 1);
    }

    #[test]
    fn pixel_signedness_default_is_unsigned() {
        assert_eq!(PixelSignedness::default(), PixelSignedness::Unsigned);
    }
}
