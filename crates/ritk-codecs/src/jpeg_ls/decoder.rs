//! Header-derived JPEG-LS decoder state and scan dispatch.

use super::bitstream::BitReader;
use super::scan::{decode_scan, Predictor, ScanParams};
use super::ContextState;
use anyhow::{bail, Context, Result};

/// Per-component decoder metadata populated during SOF55 header parsing.
pub(crate) struct ComponentInfo {
    /// Component identifier byte from SOF55.
    #[allow(dead_code)]
    pub(crate) id: u8,
    /// Per-component mapping-table selector byte.
    #[allow(dead_code)]
    pub(crate) mapping_table_selector: u8,
    /// Context states created fresh during decode.
    #[allow(dead_code)]
    pub(crate) context: [ContextState; 365],
}

/// JPEG-LS decoder state populated by header parsing.
pub(crate) struct JpegLsDecoder {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) bits_per_sample: u32,
    pub(crate) components: Vec<ComponentInfo>,
    /// NEAR parameter; DICOM lossless JPEG-LS requires zero.
    pub(crate) near: u32,
    /// Interleave mode from the SOS header. Single-component scans require zero.
    pub(crate) interleave_mode: u8,
    /// Point transform byte from the SOS header. DICOM lossless frames require zero.
    pub(crate) point_transform: u8,
    #[allow(dead_code)]
    pub(crate) restart_interval: u32,
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
            interleave_mode: 0,
            point_transform: 0,
            restart_interval: 0,
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
        if self.near != 0 {
            bail!(
                "JPEG-LS NEAR={} not supported (lossless only, DICOM PS 3.5 §8.2.3)",
                self.near
            );
        }
        if self.components.len() != 1 {
            bail!(
                "JPEG-LS multi-component ({}) not supported; use non-interleaved encoding",
                self.components.len()
            );
        }
        if self.interleave_mode != 0 {
            bail!(
                "JPEG-LS interleave mode {} not supported for single-component DICOM frames",
                self.interleave_mode
            );
        }
        if self.point_transform != 0 {
            bail!(
                "JPEG-LS point transform {} not supported for DICOM lossless frames",
                self.point_transform
            );
        }

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
        let mut samples = Vec::with_capacity(self.height * self.width);
        decode_scan(&mut reader, &params, &mut samples).context("JPEG-LS scan decode failed")?;

        let bytes_per_sample = (self.bits_per_sample as usize).div_ceil(8);
        let mut out = vec![0u8; samples.len() * bytes_per_sample];
        for (i, &sample) in samples.iter().enumerate() {
            if bytes_per_sample == 1 {
                out[i] = sample as u8;
            } else {
                out[i * 2..i * 2 + 2].copy_from_slice(&(sample as u16).to_le_bytes());
            }
        }
        Ok(out)
    }
}
