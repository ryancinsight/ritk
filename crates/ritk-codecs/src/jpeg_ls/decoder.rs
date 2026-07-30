//! Header-derived JPEG-LS decoder state and scan dispatch.

use super::bitstream::BitReader;
use super::sample_limits::maximum_near_for_precision;
use super::scan::{decode_scan, Predictor, ScanParams};
use crate::dimensions::checked_pixel_count;
use anyhow::{bail, Context, Result};

/// Interleave mode from the SOS header (JPEG-LS standard §C.1.3).
///
/// Single-component DICOM frames require `None` (0). Multi-component
/// encodings use `LineInterleaved` or `SampleInterleaved` but are not
/// supported by this decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum InterleaveMode {
    None = 0,
    LineInterleaved = 1,
    SampleInterleaved = 2,
}

impl TryFrom<u8> for InterleaveMode {
    type Error = u8;

    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::None),
            1 => Ok(Self::LineInterleaved),
            2 => Ok(Self::SampleInterleaved),
            other => Err(other),
        }
    }
}

/// Per-component decoder metadata populated during SOF55 header parsing.
pub(crate) struct ComponentInfo {}

/// JPEG-LS decoder state populated by header parsing.
pub(crate) struct JpegLsDecoder {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) bits_per_sample: u32,
    pub(crate) components: Vec<ComponentInfo>,
    /// NEAR parameter; 0 = lossless (TS .80), > 0 = near-lossless (TS .81).
    pub(crate) near: u32,
    /// Interleave mode from the SOS header. Single-component scans require `None`.
    pub(crate) interleave_mode: InterleaveMode,
    /// Point transform byte from the SOS header. DICOM lossless frames require zero.
    pub(crate) point_transform: u8,
    /// LSE-specified thresholds; zero values mean ISO defaults.
    pub(crate) t1: i32,
    pub(crate) t2: i32,
    pub(crate) t3: i32,
}

impl JpegLsDecoder {
    /// Create a decoder with default uninitialized header fields.
    pub(crate) fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            bits_per_sample: 8,
            components: Vec::new(),
            near: 0,
            interleave_mode: InterleaveMode::None,
            point_transform: 0,
            t1: 0,
            t2: 0,
            t3: 0,
        }
    }

    /// Decode scan data after the SOS header into DICOM native pixel bytes.
    pub(crate) fn decode_fragment(&self, data: &[u8]) -> Result<Vec<u8>> {
        if self.width == 0 || self.height == 0 {
            bail!(
                "JPEG-LS fragment has invalid dimensions ({}x{})",
                self.width,
                self.height
            );
        }
        if !(2..=16).contains(&self.bits_per_sample) {
            bail!(
                "JPEG-LS precision {} is unsupported; expected 2..=16",
                self.bits_per_sample
            );
        }
        let maximum_near = maximum_near_for_precision(self.bits_per_sample);
        if self.near > maximum_near {
            bail!(
                "JPEG-LS NEAR={} exceeds the precision-dependent limit {}",
                self.near,
                maximum_near
            );
        }
        if self.components.len() != 1 {
            bail!(
                "JPEG-LS multi-component ({}) not supported; use non-interleaved encoding",
                self.components.len()
            );
        }
        if self.interleave_mode != InterleaveMode::None {
            bail!(
                "JPEG-LS interleave mode {:?} not supported for single-component DICOM frames",
                self.interleave_mode
            );
        }
        if self.point_transform != 0 {
            bail!(
                "JPEG-LS point transform {} not supported for DICOM lossless frames",
                self.point_transform
            );
        }

        // Bound the decode against a hostile/corrupt header before allocating
        // the per-pixel working and sample buffers (covers both the `samples`
        // reservation and `decode_scan`'s working buffer).
        let pixel_count =
            checked_pixel_count(self.width, self.height).context("JPEG-LS image dimensions")?;

        let params = ScanParams {
            rows: self.height,
            cols: self.width,
            bpp: self.bits_per_sample,
            near: self.near,
            predictor: Predictor::Adaptive,
            t1: self.t1,
            t2: self.t2,
            t3: self.t3,
        };

        let mut reader = BitReader::new(data);
        let mut samples = Vec::with_capacity(pixel_count);
        decode_scan(&mut reader, &params, &mut samples).context("JPEG-LS scan decode failed")?;

        let bits_per_sample = usize::try_from(self.bits_per_sample)
            .context("JPEG-LS precision does not fit this platform")?;
        let bytes_per_sample = bits_per_sample.div_ceil(8);
        let output_bytes = samples
            .len()
            .checked_mul(bytes_per_sample)
            .context("JPEG-LS decoded byte count overflow")?;
        let mut out = vec![0u8; output_bytes];
        for (&sample, destination) in samples.iter().zip(out.chunks_exact_mut(bytes_per_sample)) {
            if bytes_per_sample == 1 {
                destination[0] =
                    u8::try_from(sample).context("JPEG-LS reconstructed sample exceeds u8")?;
            } else {
                let sample =
                    u16::try_from(sample).context("JPEG-LS reconstructed sample exceeds u16")?;
                destination.copy_from_slice(&sample.to_le_bytes());
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_fragment_rejects_oversized_dimensions_without_oom() {
        // A hostile SOF55 can declare 65535×65535 ≈ 4.29e9 pixels. The guard must
        // reject it with a typed error rather than allocating a ~17 GiB buffer and
        // looping billions of times over an empty scan.
        let mut decoder = JpegLsDecoder::new();
        decoder.width = 65535;
        decoder.height = 65535;
        decoder.components.push(ComponentInfo {});
        let err = decoder
            .decode_fragment(&[])
            .expect_err("oversized JPEG-LS dimensions must error");
        assert!(
            format!("{err:#}").contains("decode limit"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn decode_fragment_rejects_invalid_precision_before_scan() {
        let mut decoder = JpegLsDecoder::new();
        decoder.width = 1;
        decoder.height = 1;
        decoder.bits_per_sample = 17;
        decoder.components.push(ComponentInfo {});
        let error = decoder
            .decode_fragment(&[])
            .expect_err("unsupported precision must error");
        assert!(
            error.to_string().contains("expected 2..=16"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn decode_fragment_rejects_near_outside_precision_range() {
        let mut decoder = JpegLsDecoder::new();
        decoder.width = 1;
        decoder.height = 1;
        decoder.bits_per_sample = 8;
        decoder.near = 128;
        decoder.components.push(ComponentInfo {});
        let error = decoder
            .decode_fragment(&[])
            .expect_err("NEAR above the 8-bit range must error");
        assert!(
            error.to_string().contains("precision-dependent limit 127"),
            "unexpected error: {error:#}"
        );
    }
}
