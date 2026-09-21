//! Pixel layout and native sample decoding.
//!
//! # Contract
//! Native byte decode applies `output = sample * slope + intercept`.

use anyhow::{bail, Result};

mod compressed;
pub(crate) use compressed::decode_compressed_samples;

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
    /// Pixel rows in one decoded frame.
    pub rows: usize,
    /// Pixel columns in one decoded frame.
    pub cols: usize,
    /// Samples in each pixel.
    pub samples_per_pixel: usize,
    /// Container width for each sample.
    pub bits_allocated: u16,
    /// Meaningful low-order bits in each sample, including the sign bit.
    pub bits_stored: u16,
    /// Whether meaningful samples use unsigned or two's-complement representation.
    pub pixel_representation: PixelSignedness,
    /// Linear modality transform multiplier.
    pub rescale_slope: f32,
    /// Linear modality transform offset.
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
        if self.bits_stored == 0 || self.bits_stored > self.bits_allocated {
            bail!(
                "bits_stored={} is outside 1..=bits_allocated ({})",
                self.bits_stored,
                self.bits_allocated
            );
        }
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

#[inline]
fn apply_rescale(sample: f32, layout: &PixelLayout) -> f32 {
    sample * layout.rescale_slope + layout.rescale_intercept
}

fn stored_sample(raw: u32, layout: PixelLayout) -> f32 {
    let shift = 32 - u32::from(layout.bits_stored);
    let meaningful = raw.wrapping_shl(shift).wrapping_shr(shift);
    match layout.pixel_representation {
        PixelSignedness::Signed => ((meaningful.wrapping_shl(shift) as i32) >> shift) as f32,
        PixelSignedness::Unsigned => meaningful as f32,
    }
}

fn decode_native_pixel_bytes_unchecked(bytes: &[u8], layout: PixelLayout) -> Vec<f32> {
    match layout.bits_allocated {
        8 => bytes
            .iter()
            .map(|&b| apply_rescale(stored_sample(u32::from(b), layout), &layout))
            .collect(),
        16 => bytes
            .chunks_exact(2)
            .map(|c| {
                let raw = u32::from(u16::from_le_bytes([c[0], c[1]]));
                apply_rescale(stored_sample(raw, layout), &layout)
            })
            .collect(),
        24 => bytes
            .chunks_exact(3)
            .map(|c| apply_rescale(stored_sample(u24_le(c), layout), &layout))
            .collect(),
        32 => bytes
            .chunks_exact(4)
            .map(|c| {
                let raw = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                apply_rescale(stored_sample(raw, layout), &layout)
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn u24_le(bytes: &[u8]) -> u32 {
    u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16)
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
#[path = "pixel_layout/tests.rs"]
mod tests;
