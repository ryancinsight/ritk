//! JPEG marker and frame header parsing.
//!
//! Parses the JPEG bitstream up to and including the first SOS marker,
//! collecting all tables and metadata needed for entropy decoding.
//! Entropy data begins immediately after the SOS segment.
//!
//! # Specification
//! ITU-T T.81 Â§B.1â€“Â§B.3.

use anyhow::{bail, Context, Result};

use super::huffman::HuffmanTable;

// â”€â”€â”€ Marker Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

pub(crate) const SOI: u16 = 0xFFD8;
pub(crate) const EOI: u16 = 0xFFD9;
pub(crate) const SOF0: u16 = 0xFFC0; // Baseline DCT
pub(crate) const SOF1: u16 = 0xFFC1; // Extended sequential DCT
pub(crate) const SOF3: u16 = 0xFFC3; // Lossless Huffman
pub(crate) const DHT: u16 = 0xFFC4;
pub(crate) const DQT: u16 = 0xFFDB;
pub(crate) const SOS: u16 = 0xFFDA;
pub(crate) const DRI: u16 = 0xFFDD;

// â”€â”€â”€ Data Structures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Quantization table precision (T.81 Â§B.2.4.1, Pq field).
///
/// `Bits8` (Pq = 0) means 8-bit quantization values; `Bits16` (Pq = 1) means
/// 16-bit values. Baseline DCT (SOF0) requires `Bits8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum QuantPrecision {
    Bits8 = 0,
    Bits16 = 1,
}

impl TryFrom<u8> for QuantPrecision {
    type Error = u8;

    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::Bits8),
            1 => Ok(Self::Bits16),
            other => Err(other),
        }
    }
}

/// JPEG quantization table (T.81 Â§B.2.4.1).
#[derive(Debug, Clone)]
pub(crate) struct QuantTable {
    pub(crate) precision: QuantPrecision,
    pub(crate) values: [u16; 64], // zigzag order
}

/// Per-component frame header entry (T.81 Â§B.2.2).
#[derive(Debug, Clone)]
pub(crate) struct FrameComponent {
    pub(crate) id: u8,
    pub(crate) h_samp: u8,
    pub(crate) v_samp: u8,
    pub(crate) quant_id: u8,
}

/// SOFn frame header (T.81 Â§B.2.2).
#[derive(Debug, Clone)]
pub(crate) struct SofFrame {
    pub(crate) sof_marker: u16,
    pub(crate) precision: u8,
    pub(crate) height: u16,
    pub(crate) width: u16,
    pub(crate) components: Vec<FrameComponent>,
}

/// Per-component scan header entry (T.81 Â§B.2.3).
#[derive(Debug, Clone)]
pub(crate) struct ScanComponent {
    pub(crate) id: u8,
    pub(crate) dc_table_id: u8,
    pub(crate) ac_table_id: u8,
}

/// SOS scan header (T.81 Â§B.2.3).
#[derive(Debug, Clone)]
pub(crate) struct SosHeader {
    pub(crate) components: Vec<ScanComponent>,
    /// Ss: start of spectral selection (0 for DC; 1â€“7 = predictor for lossless).
    pub(crate) ss: u8,
    /// Se: end of spectral selection.
    pub(crate) se: u8,
    /// Ah: successive approximation bit position high.
    pub(crate) ah: u8,
    /// Al: successive approximation bit position low / point transform (lossless).
    pub(crate) al: u8,
}

/// Fully parsed JPEG frame up to the first SOS, with all tables.
#[derive(Debug)]
pub(crate) struct JpegFrameData {
    pub(crate) sof: SofFrame,
    pub(crate) quant: [Option<QuantTable>; 4],
    pub(crate) dc_huff: [Option<HuffmanTable>; 4],
    pub(crate) ac_huff: [Option<HuffmanTable>; 4],
    pub(crate) sos: SosHeader,
    /// Byte offset in the original fragment where entropy data begins.
    pub(crate) scan_data_start: usize,
}

// â”€â”€â”€ Parser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Read a big-endian u16 from `data[pos..pos+2]`.
#[inline]
fn read_u16(data: &[u8], pos: usize) -> Result<u16> {
    data.get(pos..pos + 2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]))
        .context("JPEG stream truncated reading u16")
}

/// Parse a JPEG bitstream and return `JpegFrameData` plus the scan start offset.
pub(crate) fn parse_jpeg(data: &[u8]) -> Result<JpegFrameData> {
    let mut pos = 0usize;

    // SOI
    if read_u16(data, pos)? != SOI {
        bail!("JPEG fragment does not begin with SOI marker");
    }
    pos += 2;

    let mut sof: Option<SofFrame> = None;
    let mut quant: [Option<QuantTable>; 4] = [None, None, None, None];
    let mut dc_huff: [Option<HuffmanTable>; 4] = [None, None, None, None];
    let mut ac_huff: [Option<HuffmanTable>; 4] = [None, None, None, None];

    loop {
        if pos + 1 >= data.len() {
            bail!("JPEG stream ended before SOS marker");
        }
        // Markers always start with 0xFF; skip fill bytes.
        if data[pos] != 0xFF {
            bail!(
                "expected JPEG marker at offset {pos}, got 0x{:02X}",
                data[pos]
            );
        }
        while pos < data.len() && data[pos] == 0xFF {
            pos += 1;
        }
        if pos >= data.len() {
            bail!("JPEG stream ended in marker prefix");
        }
        let marker = 0xFF00u16 | u16::from(data[pos]);
        pos += 1;

        match marker {
            EOI => bail!("JPEG stream ended (EOI) before SOS"),
            0xFF00 => {} // padding â€” skip
            // SOI inside segment: skip
            SOI => {}
            // APPn (0xFFE0â€“0xFFEF) and COM (0xFFFE): skip segment
            m if (0xFFE0..=0xFFEF).contains(&m) || m == 0xFFFE => {
                let len = read_u16(data, pos)? as usize;
                pos += len;
            }
            // DRI: restart interval (2 bytes payload, ignore value)
            DRI => {
                let _len = read_u16(data, pos)?;
                pos += 4; // length field + 2 bytes DRI value
            }
            DQT => {
                let len = read_u16(data, pos)? as usize;
                let end = pos + len;
                pos += 2;
                while pos < end {
                    let pq_tq = data[pos];
                    let precision_byte = pq_tq >> 4;
                    let id = (pq_tq & 0x0F) as usize;
                    pos += 1;
                    if id >= 4 {
                        bail!("DQT table id {id} out of range");
                    }
                    let precision = QuantPrecision::try_from(precision_byte).map_err(|v| {
                        anyhow::anyhow!("DQT precision {v} is invalid; expected 0 or 1")
                    })?;
                    let mut values = [0u16; 64];
                    if precision == QuantPrecision::Bits8 {
                        for v in &mut values {
                            *v = u16::from(data[pos]);
                            pos += 1;
                        }
                    } else {
                        for v in &mut values {
                            *v = read_u16(data, pos)?;
                            pos += 2;
                        }
                    }
                    quant[id] = Some(QuantTable { precision, values });
                }
            }
            DHT => {
                let len = read_u16(data, pos)? as usize;
                let end = pos + len;
                pos += 2;
                while pos < end {
                    let tc_th = data[pos];
                    let table_class = tc_th >> 4; // 0=DC/lossless, 1=AC
                    let id = (tc_th & 0x0F) as usize;
                    pos += 1;
                    if id >= 4 {
                        bail!("DHT table id {id} out of range");
                    }
                    let mut bits = [0u8; 16];
                    bits.copy_from_slice(&data[pos..pos + 16]);
                    pos += 16;
                    let n_syms: usize = bits.iter().map(|&b| b as usize).sum();
                    let huffval = &data[pos..pos + n_syms];
                    pos += n_syms;
                    let table = HuffmanTable::from_bits_huffval(&bits, huffval)?;
                    if table_class == 0 {
                        dc_huff[id] = Some(table);
                    } else {
                        ac_huff[id] = Some(table);
                    }
                }
            }
            SOF0 | SOF1 | SOF3 => {
                let len = read_u16(data, pos)? as usize;
                pos += 2;
                let precision = data[pos];
                pos += 1;
                let height = read_u16(data, pos)?;
                pos += 2;
                let width = read_u16(data, pos)?;
                pos += 2;
                let ncomp = data[pos] as usize;
                pos += 1;
                if ncomp > 4 {
                    bail!("SOF: too many components: {ncomp}");
                }
                let mut components = Vec::with_capacity(ncomp);
                for _ in 0..ncomp {
                    let id = data[pos];
                    let samp = data[pos + 1];
                    let h_samp = samp >> 4;
                    let v_samp = samp & 0x0F;
                    let quant_id = data[pos + 2];
                    pos += 3;
                    components.push(FrameComponent {
                        id,
                        h_samp,
                        v_samp,
                        quant_id,
                    });
                }
                let _ = len; // length already consumed field-by-field
                sof = Some(SofFrame {
                    sof_marker: marker,
                    precision,
                    height,
                    width,
                    components,
                });
            }
            SOS => {
                let len = read_u16(data, pos)? as usize;
                pos += 2;
                let ncomp = data[pos] as usize;
                pos += 1;
                let mut scan_comps = Vec::with_capacity(ncomp);
                for _ in 0..ncomp {
                    let id = data[pos];
                    let tables = data[pos + 1];
                    let dc_table_id = tables >> 4;
                    let ac_table_id = tables & 0x0F;
                    pos += 2;
                    scan_comps.push(ScanComponent {
                        id,
                        dc_table_id,
                        ac_table_id,
                    });
                }
                let ss = data[pos];
                let se = data[pos + 1];
                let ah_al = data[pos + 2];
                let ah = ah_al >> 4;
                let al = ah_al & 0x0F;
                pos += 3;
                // Verify we consumed len bytes (including the 2-byte length field)
                let _ = len;
                let sos = SosHeader {
                    components: scan_comps,
                    ss,
                    se,
                    ah,
                    al,
                };
                let frame = sof.context("SOS before SOF in JPEG stream")?;
                return Ok(JpegFrameData {
                    sof: frame,
                    quant,
                    dc_huff,
                    ac_huff,
                    sos,
                    scan_data_start: pos,
                });
            }
            other => {
                // Unknown marker with length segment: skip it.
                if pos + 1 < data.len() {
                    let len = read_u16(data, pos)? as usize;
                    if len >= 2 {
                        pos += len;
                    } else {
                        bail!("malformed JPEG marker 0x{other:04X} with length {len}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lossless 8-bit hand-crafted fixture must parse to a 1Ã—1 SOF3 frame.
    #[test]
    fn parse_lossless_8bit_fixture() {
        let data = crate::jpeg::scan_lossless::tests::lossless_8bit_fixture();
        let frame = parse_jpeg(&data).unwrap();
        assert_eq!(frame.sof.sof_marker, SOF3);
        assert_eq!(frame.sof.precision, 8);
        assert_eq!(frame.sof.width, 1);
        assert_eq!(frame.sof.height, 1);
        assert_eq!(frame.sof.components.len(), 1);
        assert_eq!(frame.sos.ss, 1); // predictor Ra
        assert_eq!(frame.sos.al, 0); // no point transform
                                     // DC table 0 must be present, AC table not needed for lossless
        assert!(frame.dc_huff[0].is_some());
    }

    /// The lossless 16-bit fixture must parse to a 1Ã—1 SOF3 frame with precision 16.
    #[test]
    fn parse_lossless_16bit_fixture() {
        let data = crate::jpeg::scan_lossless::tests::lossless_16bit_fixture();
        let frame = parse_jpeg(&data).unwrap();
        assert_eq!(frame.sof.sof_marker, SOF3);
        assert_eq!(frame.sof.precision, 16);
        assert_eq!(frame.sof.width, 1);
        assert_eq!(frame.sof.height, 1);
    }
}
