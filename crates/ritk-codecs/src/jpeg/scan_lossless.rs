//! JPEG lossless (SOF3) scan decode.
//!
//! # Specification
//! ITU-T T.81 §H: Lossless sequential Huffman-coded JPEG.
//! Each sample is decoded as: Rx = clamp(predictor(Ra,Rb,Rc) + diff, 0, MAXVAL)
//! where diff is Huffman-coded using the DC Huffman table.
//!
//! Predictors (T.81 §H.1.2):
//!   Ss=1: Ra       Ss=2: Rb       Ss=3: Rc
//!   Ss=4: Ra+Rb-Rc Ss=5: Ra+((Rb-Rc)>>1)
//!   Ss=6: Rb+((Ra-Rc)>>1) Ss=7: (Ra+Rb)>>1
//!
//! Initial conditions (first row and first column of each row):
//!   - (0,0): predictor = 2^(P−Pt−1)  where P=precision, Pt=point_transform
//!   - row=0, x>0: use predictor Ss=1 (Ra=left) — equivalent to predicting left
//!   - row>0, x=0: use Rb (above pixel) as predictor

use anyhow::{bail, Context, Result};

use super::backend::{JpegDecoded, JpegPixelFormat};
use super::huffman::{receive_and_extend, BitReader};
use super::marker::{JpegFrameData, SOF3};

/// Compute lossless predictor from causal neighbors and Ss selection.
#[inline]
fn predict(ra: i32, rb: i32, rc: i32, ss: u8) -> i32 {
    match ss {
        1 => ra,
        2 => rb,
        3 => rc,
        4 => ra + rb - rc,
        5 => ra + ((rb - rc) >> 1),
        6 => rb + ((ra - rc) >> 1),
        7 => (ra + rb) >> 1,
        _ => 0, // Ss=0 means no prediction (differential coding)
    }
}

/// Decode a JPEG Lossless (SOF3) scan.
///
/// Returns `JpegDecoded` with:
/// - `pixel_format = L8`  for precision ≤ 8
/// - `pixel_format = L16` for precision 9..=16 (2 bytes per pixel, native endian)
/// - Multi-component images are not supported; single-component only.
pub(crate) fn decode_lossless_scan(
    frame: &JpegFrameData,
    entropy_data: &[u8],
) -> Result<JpegDecoded> {
    if frame.sof.sof_marker != SOF3 {
        bail!(
            "decode_lossless_scan called with SOF marker 0x{:04X}",
            frame.sof.sof_marker
        );
    }

    let precision = frame.sof.precision;
    if !(2..=16).contains(&precision) {
        bail!("JPEG Lossless: unsupported precision {precision}");
    }
    if frame.sos.se != 0 || frame.sos.ah != 0 {
        bail!(
            "JPEG Lossless scan parameters unsupported: Se={} Ah={}",
            frame.sos.se,
            frame.sos.ah
        );
    }
    let maxval: i32 = (1 << precision) - 1;
    let point_transform = frame.sos.al as i32;
    if point_transform as u8 >= precision {
        bail!(
            "JPEG Lossless point transform {} must be less than precision {}",
            point_transform,
            precision
        );
    }
    let initial_pred: i32 = 1 << (precision - point_transform as u8 - 1);

    let width = frame.sof.width as usize;
    let height = frame.sof.height as usize;
    let ncomp = frame.sos.components.len();
    if ncomp != 1 {
        bail!("JPEG Lossless: multi-component ({ncomp}) scans not yet supported");
    }

    let scan_comp = &frame.sos.components[0];
    let dc_table = frame.dc_huff[scan_comp.dc_table_id as usize]
        .as_ref()
        .with_context(|| {
            format!(
                "JPEG Lossless: DC Huffman table {} not present",
                scan_comp.dc_table_id
            )
        })?;

    let predictor_sel = frame.sos.ss;
    let n_pixels = width * height;
    let mut samples = vec![0i32; n_pixels];
    let mut reader = BitReader::new(entropy_data);

    for y in 0..height {
        for x in 0..width {
            // Determine causal neighbors.
            let ra = if x > 0 { samples[y * width + x - 1] } else { 0 };
            let rb = if y > 0 {
                samples[(y - 1) * width + x]
            } else {
                0
            };
            let rc = if y > 0 && x > 0 {
                samples[(y - 1) * width + x - 1]
            } else {
                0
            };

            let px = if y == 0 && x == 0 {
                initial_pred
            } else if y == 0 {
                // Top row: use Ra (left neighbor) regardless of Ss
                ra
            } else if x == 0 {
                // First column of non-top row: use Rb (above)
                rb
            } else {
                predict(ra, rb, rc, predictor_sel)
            };

            // Decode Huffman category and difference.
            let category = dc_table.decode(&mut reader)?;
            let diff = receive_and_extend(&mut reader, category)?;
            let reconstructed = (px + diff) & maxval;
            samples[y * width + x] = reconstructed;
        }
    }

    // Pack samples into bytes.
    if precision <= 8 {
        let pixels: Vec<u8> = samples.into_iter().map(|s| s as u8).collect();
        Ok(JpegDecoded {
            width,
            height,
            pixel_format: JpegPixelFormat::L8,
            pixels,
        })
    } else {
        // 9–16 bit: pack as native-endian u16.
        let mut pixels = Vec::with_capacity(n_pixels * 2);
        for s in samples {
            pixels.extend_from_slice(&(s as u16).to_ne_bytes());
        }
        Ok(JpegDecoded {
            width,
            height,
            pixel_format: JpegPixelFormat::L16,
            pixels,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::jpeg::marker::parse_jpeg;

    /// Hand-crafted lossless 8-bit 1×1 JPEG fixture.
    /// Huffman: BITS=[1,0,...], HUFFVAL=[0] (one code of length 1 → symbol 0).
    /// Predictor Ss=1 (Ra). Initial predictor = 2^7 = 128. diff=0 → pixel=128.
    /// Entropy byte 0x7F: first bit is 0 → Huffman code 0 → category 0 → diff=0.
    pub(crate) fn lossless_8bit_fixture() -> Vec<u8> {
        vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC3, // SOF3
            0x00, 0x0B, // length 11
            0x08, // precision 8
            0x00, 0x01, // height 1
            0x00, 0x01, // width 1
            0x01, // 1 component
            0x01, 0x11, 0x00, // id=1, h=1,v=1, quant=0
            0xFF, 0xC4, // DHT
            0x00, 0x14, // length 20
            0x00, // DC table 0
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, // HUFFVAL[0]=0 (category 0)
            0xFF, 0xDA, // SOS
            0x00, 0x08, // length 8
            0x01, // 1 component
            0x01, 0x00, // id=1, DC table=0
            0x01, // Ss=1 (predictor Ra)
            0x00, // Se=0
            0x00, // Ah=0, Al=0
            0x7F, // entropy: bit 0 → Huffman→category=0 → diff=0 → pixel=128
            0xFF, 0xD9, // EOI
        ]
    }

    /// Hand-crafted lossless 16-bit 1×1 JPEG fixture.
    /// Huffman: BITS=[1,0,...], HUFFVAL=[15] (one code of length 1 → symbol 15).
    /// Initial predictor = 2^15 = 32768. diff=-28108 → pixel = 32768-28108 = 4660 = 0x1234.
    /// Entropy [0x12,0x33]: first bit 0 → category 15, next 15 bits = 0x1233=4659,
    /// receive_and_extend(15): 4659 < 2^14 → negative → 4659-32767 = -28108.
    pub(crate) fn lossless_16bit_fixture() -> Vec<u8> {
        vec![
            0xFF, 0xD8, 0xFF, 0xC3, 0x00, 0x0B, 0x10, // precision 16
            0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x0F, // HUFFVAL[0]=15
            0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x12,
            0x33, // entropy data
            0xFF, 0xD9,
        ]
    }

    #[test]
    fn lossless_8bit_decodes_pixel_128() {
        let data = lossless_8bit_fixture();
        let frame = parse_jpeg(&data).unwrap();
        let entropy = &data[frame.scan_data_start..];
        let decoded = decode_lossless_scan(&frame, entropy).unwrap();
        assert_eq!(decoded.width, 1);
        assert_eq!(decoded.height, 1);
        assert_eq!(decoded.pixel_format, JpegPixelFormat::L8);
        assert_eq!(decoded.pixels, vec![128u8]);
    }

    #[test]
    fn lossless_16bit_decodes_pixel_0x1234() {
        let data = lossless_16bit_fixture();
        let frame = parse_jpeg(&data).unwrap();
        let entropy = &data[frame.scan_data_start..];
        let decoded = decode_lossless_scan(&frame, entropy).unwrap();
        assert_eq!(decoded.pixel_format, JpegPixelFormat::L16);
        assert_eq!(decoded.pixels, 0x1234u16.to_ne_bytes().to_vec());
    }
}
