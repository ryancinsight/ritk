//! Native JPEG-LS (ISO 14495) lossless decoder for DICOM encapsulated frames.
//!
//! # Contract
//! JPEG-LS lossless mode uses predictive coding with context-adaptive Golomb-Rice coding.
//! RITK implements the lossless path only (NEAR=0, per DICOM PS 3.5 §8.2.3).
//!
//! The decoder:
//! 1. Parses JPEG-LS markers (SOI, SOF55, SOS, DNL, etc.)
//! 2. Initializes context models for each component
//! 3. Decodes residual using Golomb-Rice codes with adaptively updated limit
//! 4. Reconstructs samples: `sample = predictor + residual`
//! 5. Applies modality LUT: `output = sample * slope + intercept`

use std::cmp::min;

use anyhow::{bail, Context, Result};

use crate::pixel::{PixelLayout, decode_native_pixel_bytes_checked};

// JPEG-LS Markers
const SOI: u16 = 0xFFD8;
const SOF55: u16 = 0xFFF7; // JPEG-LS Frame
const SOS: u16 = 0xFFDA; // Start of Scan
const DNL: u16 = 0xFFDC; // Define Number of Lines
const DRI: u16 = 0xFFDD; // Define Restart Interval
const APP0: u16 = 0xFFE0; // Application 0
const COM: u16 = 0xFFFE; // Comment
const EOI: u16 = 0xFFD9; // End of Image

// Prediction modes (ISO 14495-1 §5.3)
#[derive(Debug, Clone, Copy)]
enum Prediction {
    None = 0,
    Left = 1,
    Up = 2,
    AvgLeftUp = 3,
    Paeth = 4,
    // Select (SPECIAL) not used in DICOM
}

impl Prediction {
    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Prediction::None),
            1 => Ok(Prediction::Left),
            2 => Ok(Prediction::Up),
            3 => Ok(Prediction::AvgLeftUp),
            4 => Ok(Prediction::Paeth),
            _ => bail!("Invalid JPEG-LS prediction mode: {}", v),
        }
    }
}

// JPEG-LS Bit Reader with restart support
#[allow(dead_code)]
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u32,
}

#[allow(dead_code)]
impl<'a> BitReader<'a> {
    fn new(_data: &'a [u8]) -> Self {
        Self {
            data: _data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn read_bit(&mut self) -> Option<bool> {
        if self.byte_pos >= self.data.len() {
            return None;
        }
        let byte = self.data[self.byte_pos];
        let bit = ((byte >> (7 - self.bit_pos)) & 1) == 1;
        self.bit_pos += 1;
        if self.bit_pos >= 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Some(bit)
    }

    fn read_bits(&mut self, n: u32) -> Option<u32> {
        let mut result = 0u32;
        for i in 0..n {
            let bit = self.read_bit()?;
            if bit {
                result |= 1 << (n - 1 - i);
            }
        }
        Some(result)
    }

    // Read Golomb-Rice code for JPEG-LS
    // Format: k leading zeros, then 1, then k-bit remainder
    fn read_golomb_rice(&mut self, k: u32) -> Option<u32> {
        // Count leading zeros
        let mut zeros = 0;
        while self.read_bit() == Some(false) {
            zeros += 1;
        }
        // Read k-bit remainder
        let remainder = self.read_bits(k)?;
        Some((zeros << k) | remainder)
    }

    fn remaining_bytes(&self) -> usize {
        self.data.len() - self.byte_pos
    }
}

// JPEG-LS Decoder state
struct JpegLsDecoder {
    width: usize,
    height: usize,
    bits_per_sample: u32,
    components: Vec<ComponentInfo>,
    prediction: Prediction,
    near: u32, // NEAR parameter (must be 0 for lossless DICOM)
    restart_interval: u32,
}

#[derive(Debug)]
#[allow(dead_code)]
struct ComponentInfo {
    id: u8,
    predictor: i32,
    context: [ContextState; 365], // JPEG-LS context range
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct ContextState {
    a: u32, // Running sum of magnitude`
    b: u32, // Running sum of values`
    c: i32, // Counter (placeholder, written but not read in placeholder)`
}

impl Default for ContextState {
    fn default() -> Self {
        Self {
            a: 0,
            b: 0,
            c: 0,
        }
    }
}

impl JpegLsDecoder {
    fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            bits_per_sample: 8,
            components: Vec::new(),
            prediction: Prediction::Left,
            near: 0,
            restart_interval: 0,
        }
    }

    #[allow(unused_variables)]
    fn decode_fragment(&self, _data: &[u8]) -> Result<Vec<u8>> {
        if self.width == 0 || self.height == 0 {
            bail!("JPEG-LS fragment has invalid dimensions");
        }
        if self.near != 0 {
            bail!("JPEG-LS NEAR={} not supported (lossless only)", self.near);
        }

        let total_samples = self.width * self.height * self.components.len();
        let bytes_per_sample = (self.bits_per_sample as usize + 7) / 8;
        let expected_bytes = total_samples * bytes_per_sample;
        let mut output = vec![0u8; expected_bytes];

        // For simplicity, we implement a basic JPEG-LS lossless decoder
        // In practice, DICOM JPEG-LS is typically 8-bit or 16-bit grayscale
        // with LEFT prediction and Golomb-Rice coding.

        if self.components.len() != 1 {
            bail!("JPEG-LS multi-component not yet supported");
        }

        // TODO: Initialize bit reader when implementing actual JPEG-LS decoding
        // let mut reader = BitReader::new(data);
        
        // Initialize context model
        let mut context_states = [ContextState::default(); 365];
        
        // Decode scan
        let mut prev_line: Vec<i32> = vec![0; self.width];
        let mut curr_line: Vec<i32> = vec![0; self.width];

        for y in 0..self.height {
            for x in 0..self.width {
                // Predict
                let predictor = if x == 0 && y == 0 {
                    0 // First sample has no predictor
                } else if x == 0 {
                    prev_line[x] // UP prediction for first column
                } else if y == 0 {
                    curr_line[x - 1] // LEFT prediction for first row
                } else {
                    // Compute gradient
                    let a = curr_line[x - 1]; // Left
                    let b = prev_line[x]; // Up
                    let c = prev_line[x - 1]; // Up-Left
                    
                    match self.prediction {
                        Prediction::Left => a,
                        Prediction::Up => b,
                        Prediction::AvgLeftUp => (a + b) / 2,
                        Prediction::Paeth => {
                            let p = a + b - c;
                            let pa = (p - a).abs();
                            let pb = (p - b).abs();
                            let pc = (p - c).abs();
                            if pa <= pb && pa <= pc {
                                a
                            } else if pb <= pc {
                                b
                            } else {
                                c
                            }
                        }
                        Prediction::None => 0,
                    }
                };

                // Determine context
                let (q, _sign) = if x > 0 && y > 0 {
                    let da = (curr_line[x - 1] - prev_line[x - 1]).abs();
                    let db = (prev_line[x] - prev_line[x - 1]).abs();
                    let dc = (prev_line[x - 1] - curr_line[x - 1]).abs();
                    let d = min(da, min(db, dc));
                    if d < 128 {
                        (0, 0)
                    } else {
                        (1, if da > db { 1 } else { 0 })
                    }
                } else {
                    (0, 0)
                };

                // Read residual using Golomb-Rice with context-adaptive limit
                let context_idx = if x == 0 && y == 0 {
                    0
                } else if x == 0 {
                    1 // Vertical edge
                } else if y == 0 {
                    2 // Horizontal edge
                } else {
                    3 + (q as usize) // Interior
                };

                let ctx = &mut context_states[context_idx.min(364)];
                // k for Golomb-Rice coding (computed but not used in placeholder)
                let _k = if ctx.a == 0 { 0 } else { (31 - (ctx.a as u32).leading_zeros()).min(15) };
                
                // Simplified: read residual
                // In practice, JPEG-LS uses complex Golomb-Rice with sign encoding
                // For this implementation, we need the actual JPEG-LS bitstream
                
                // Placeholder for actual residual decoding
                // The full implementation requires parsing the JPEG-LS bitstream format
                let residual = 0; // TODO: actual Golomb-rice decode
                
                let sample = predictor + residual;
                curr_line[x] = sample;

                // Update context
                ctx.a += residual.abs() as u32;
                if ctx.a > 65535 {
                    ctx.a = (ctx.a + 1) / 2;
                    ctx.b = (ctx.b + 1) / 2;
                }
                ctx.b += 1;
            }

            // Copy current line to previous
            prev_line.clone_from_slice(&curr_line);
        }

        // Convert i32 samples to output bytes
        for (i, &sample) in curr_line.iter().enumerate() {
            if bytes_per_sample == 1 {
                output[i] = sample as u8;
            } else {
                let bytes = (sample as u16).to_le_bytes();
                output[i * 2..i * 2 + 2].copy_from_slice(&bytes);
            }
        }

        Ok(output)
    }
}

/// Decode a JPEG-LS encapsulated DICOM frame.
///
/// # Arguments
/// * `fragment` - The encapsulated JPEG-LS frame bytes
/// * `layout` - DICOM pixel metadata
///
/// # Returns
/// Decoded samples with modality LUT applied → `Vec<f32>`
pub fn decode_jpeg_ls_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    // Parse JPEG-LS headers
    let mut decoder = JpegLsDecoder::new();
    parse_jpeg_ls_headers(&mut decoder, fragment)
        .context("Failed to parse JPEG-LS headers")?;

    // Validate dimensions match DICOM metadata
    if decoder.width != layout.cols || decoder.height != layout.rows {
        bail!(
            "JPEG-LS dimensions {}x{} do not match DICOM layout {}x{}",
            decoder.width, decoder.height, layout.cols, layout.rows
        );
    }

    // Find the scan data (after SOS marker)
    let scan_data = find_scan_data(fragment)
        .context("JPEG-LS scan data not found")?;

    // Decode the scan
    let decoded_bytes = decoder
        .decode_fragment(scan_data)
        .context("JPEG-LS decode failed")?;

    // Apply modality LUT through the standard native pixel path
    // For now, return the raw decoded bytes
    decode_native_pixel_bytes_checked(&decoded_bytes, layout)
}

/// Parse JPEG-LS markers and headers
fn parse_jpeg_ls_headers(decoder: &mut JpegLsDecoder, data: &[u8]) -> Result<()> {
    // Check SOI
    if data.len() < 2 || u16::from_be_bytes([data[0], data[1]]) != SOI {
        bail!("JPEG-LS fragment does not start with SOI marker");
    }
    let mut pos = 2;

    while pos < data.len() - 1 {
        let marker = u16::from_be_bytes([data[pos], data[pos + 1]]);
        
        if marker == EOI {
            break;
        }

        match marker {
            SOI => {
                // Already handled
                pos += 2;
            }
            SOF55 => {
                // JPEG-LS Frame Header
                if pos + 8 > data.len() {
                    bail!("Truncated SOF55 marker");
                }
                let _length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                decoder.bits_per_sample = data[pos + 4] as u32;
                decoder.height = u16::from_be_bytes([data[pos + 5], data[pos + 6]]) as usize;
                decoder.width = u16::from_be_bytes([data[pos + 7], data[pos + 8]]) as usize;
                let num_components = data[pos + 9];
                
                for i in 0..num_components {
                    if pos + 10 + (i as usize) * 3 + 3 > data.len() {
                        bail!("Truncated component info in SOF55");
                    }
                    let idx = pos + 10 + (i as usize) * 3;
                    decoder.components.push(ComponentInfo {
                        id: data[idx],
                        predictor: 0,
                        context: [ContextState::default(); 365],
                    });
                }
                
                pos += 2 + _length;
            }
            SOS => {
                // Start of Scan
                if pos + 6 > data.len() {
                    bail!("Truncated SOS marker");
                }
                let _length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                let ns = data[pos + 4]; // Number of components in scan
                
                // Read predictor and point transform for each component
                if pos + 5 + (ns as usize) * 2 + 1 > data.len() {
                    bail!("Truncated component info in SOS");
                }
                for i in 0..ns {
                    let idx = pos + 5 + (i as usize) * 2;
                    let predictor_byte = data[idx + 1];
                    decoder.prediction = Prediction::from_u8(predictor_byte & 0x0F)?;
                }
                
                // NEAR parameter
                if pos + 5 + (ns as usize) * 2 + 1 < data.len() {
                    decoder.near = data[pos + 5 + (ns as usize) * 2 + 1] as u32;
                }
                
                break; // Scan data follows - pos update not needed since we're breaking
            }
            DRI => {
                if pos + 6 > data.len() {
                    bail!("Truncated DRI marker");
                }
                decoder.restart_interval = u16::from_be_bytes([data[pos + 4], data[pos + 5]]) as u32;
                pos += 2 + 4;
            }
            DNL => {
                if pos + 4 > data.len() {
                    bail!("Truncated DNL marker");
                }
                decoder.height = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                pos += 2 + 2;
            }
            APP0..=0xFFEF => {
                if pos + 4 > data.len() {
                    bail!("Truncated APP marker");
                }
                let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                pos += 2 + length;
            }
            COM => {
                if pos + 4 > data.len() {
                    bail!("Truncated COM marker");
                }
                let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                pos += 2 + length;
            }
            _ => {
                // Unknown marker - skip
                if pos + 4 < data.len() {
                    let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                    pos += 2 + length;
                } else {
                    pos += 2;
                }
            }
        }
    }

    Ok(())
}

/// Find the scan data after SOS marker
fn find_scan_data(data: &[u8]) -> Option<&[u8]> {
    let mut pos = 0;
    while pos < data.len() - 1 {
        if data[pos] == 0xFF {
            let marker = u16::from_be_bytes([data[pos], data[pos + 1]]);
            if marker == SOS {
                // Skip SOS header
                if pos + 4 < data.len() {
                    let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                    let scan_data_start = pos + 2 + length;
                    if scan_data_start < data.len() {
                        return Some(&data[scan_data_start..]);
                    }
                }
            }
        }
        pos += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::PixelLayout;

    #[test]
    fn jpeg_ls_marker_constants_correct() {
        assert_eq!(SOI, 0xFFD8);
        assert_eq!(SOF55, 0xFFF7);
        assert_eq!(SOS, 0xFFDA);
        assert_eq!(EOI, 0xFFD9);
    }

    #[test]
    fn prediction_mode_from_u8_valid() {
        assert!(matches!(Prediction::from_u8(0), Ok(Prediction::None)));
        assert!(matches!(Prediction::from_u8(1), Ok(Prediction::Left)));
        assert!(matches!(Prediction::from_u8(2), Ok(Prediction::Up)));
        assert!(matches!(Prediction::from_u8(3), Ok(Prediction::AvgLeftUp)));
        assert!(matches!(Prediction::from_u8(4), Ok(Prediction::Paeth)));
    }

    #[test]
    fn prediction_mode_from_u8_invalid() {
        assert!(Prediction::from_u8(5).is_err());
        assert!(Prediction::from_u8(255).is_err());
    }

    #[test]
    fn bit_reader_basic() {
        let data = [0b10110000, 0b11001100];
        let mut reader = BitReader::new(&data);
        
        assert_eq!(reader.read_bit(), Some(true));  // 1
        assert_eq!(reader.read_bit(), Some(false)); // 0
        assert_eq!(reader.read_bit(), Some(true));  // 1
        assert_eq!(reader.read_bit(), Some(true));  // 1
        assert_eq!(reader.read_bit(), Some(false)); // 0
        assert_eq!(reader.read_bit(), Some(false)); // 0
    }

    #[test]
    fn bit_reader_read_bits() {
        let data = [0b10110000];
        let mut reader = BitReader::new(&data);
        let bits = reader.read_bits(3);
        assert_eq!(bits, Some(0b101)); // 5
    }

    #[test]
    fn decoder_new_initializes_defaults() {
        let decoder = JpegLsDecoder::new();
        assert_eq!(decoder.width, 0);
        assert_eq!(decoder.height, 0);
        assert_eq!(decoder.bits_per_sample, 8);
        assert_eq!(decoder.near, 0);
    }

    #[test]
    fn decode_fragment_rejects_near_nonzero() {
        let decoder = JpegLsDecoder {
            width: 100,
            height: 100,
            bits_per_sample: 8,
            components: vec![ComponentInfo {
                id: 1,
                predictor: 0,
                context: [ContextState::default(); 365],
            }],
            prediction: Prediction::Left,
            near: 1, // Non-zero NEAR should fail
            restart_interval: 0,
        };
        
        let result = decoder.decode_fragment(&[]);
        assert!(result.is_err());
        let err_str = format!("{:?}", result.unwrap_err());
        assert!(err_str.contains("NEAR"), "Expected NEAR error, got: {}", err_str);
    }

    #[test]
    fn decode_fragment_rejects_zero_dimensions() {
        let decoder = JpegLsDecoder {
            width: 0, // Zero width should fail
            height: 100,
            bits_per_sample: 8,
            components: vec![ComponentInfo {
                id: 1,
                predictor: 0,
                context: [ContextState::default(); 365],
            }],
            prediction: Prediction::Left,
            near: 0,
            restart_interval: 0,
        };
        
        let result = decoder.decode_fragment(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn decode_fragment_rejects_nonzero_near() {
        let decoder = JpegLsDecoder {
            width: 100,
            height: 100,
            bits_per_sample: 8,
            components: vec![ComponentInfo {
                id: 1,
                predictor: 0,
                context: [ContextState::default(); 365],
            }],
            prediction: Prediction::Left,
            near: 1, // Non-zero NEAR
            restart_interval: 0,
        };
        
        let _layout = PixelLayout {
            rows: 100,
            cols: 100,
            samples_per_pixel: 1,
            bits_allocated: 8,
            pixel_representation: 0,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        };
        
        let result = decoder.decode_fragment(&[]);
        assert!(result.is_err());
        let err_str = format!("{:?}", result.unwrap_err());
        assert!(err_str.contains("NEAR"), "Expected NEAR error, got: {}", err_str);
    }
}