//! JPEG-LS marker parser.

use super::{ComponentInfo, InterleaveMode, JpegLsDecoder, DNL, DRI, EOI, LSE, SOF55, SOI, SOS};
use anyhow::{bail, Result};

/// Parse all JPEG-LS markers before scan data and populate `decoder`.
pub(crate) fn parse_jpeg_ls_headers(decoder: &mut JpegLsDecoder, data: &[u8]) -> Result<()> {
    if data.len() < 2 || u16::from_be_bytes([data[0], data[1]]) != SOI {
        bail!("JPEG-LS fragment does not start with SOI marker (0xFFD8)");
    }
    let mut pos = 2usize;

    while pos + 1 < data.len() {
        let marker = u16::from_be_bytes([data[pos], data[pos + 1]]);

        if marker == EOI || marker == SOS {
            break;
        }

        match marker {
            SOI => pos += 2,
            SOF55 => pos = parse_sof55(decoder, data, pos)?,
            DNL => pos = parse_dnl(decoder, data, pos)?,
            DRI => pos = parse_dri(decoder, data, pos)?,
            LSE => pos = parse_lse(decoder, data, pos)?,
            _ => pos = skip_variable_marker(data, pos),
        }
    }

    parse_sos(decoder, data, pos)?;
    Ok(())
}

/// Find the scan data bytes immediately following the SOS header.
pub(crate) fn find_scan_data(data: &[u8]) -> Option<&[u8]> {
    let mut pos = 0usize;
    while pos + 1 < data.len() {
        if data[pos] == 0xFF {
            let marker = u16::from_be_bytes([data[pos], data[pos + 1]]);
            if marker == SOS && pos + 3 < data.len() {
                let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                let scan_start = pos + 2 + length;
                return (scan_start < data.len()).then_some(&data[scan_start..]);
            }
        }
        pos += 1;
    }
    None
}

fn parse_sof55(decoder: &mut JpegLsDecoder, data: &[u8], pos: usize) -> Result<usize> {
    if pos + 9 > data.len() {
        bail!("Truncated SOF55 marker at offset {}", pos);
    }
    let length = marker_length(data, pos);
    decoder.bits_per_sample = data[pos + 4] as u32;
    decoder.height = u16::from_be_bytes([data[pos + 5], data[pos + 6]]) as usize;
    decoder.width = u16::from_be_bytes([data[pos + 7], data[pos + 8]]) as usize;
    let num_comp = if pos + 9 < data.len() {
        data[pos + 9]
    } else {
        1
    };
    decoder.components.clear();
    for i in 0..(num_comp as usize) {
        let idx = pos + 10 + i * 3;
        if idx + 2 < data.len() {
            decoder.components.push(ComponentInfo {});
        }
    }
    Ok(pos + 2 + length)
}

fn parse_dnl(decoder: &mut JpegLsDecoder, data: &[u8], pos: usize) -> Result<usize> {
    if pos + 6 > data.len() {
        bail!("Truncated DNL marker at offset {}", pos);
    }
    let length = marker_length(data, pos);
    if pos + 4 + length <= data.len() && length >= 2 {
        decoder.height = u16::from_be_bytes([data[pos + 4], data[pos + 5]]) as usize;
    }
    Ok(pos + 2 + length)
}

fn parse_dri(_decoder: &mut JpegLsDecoder, data: &[u8], pos: usize) -> Result<usize> {
    if pos + 6 > data.len() {
        bail!("Truncated DRI marker at offset {}", pos);
    }
    // DRI restart interval is parsed but no longer stored (not used for decode).
    Ok(pos + 6)
}

fn parse_lse(decoder: &mut JpegLsDecoder, data: &[u8], pos: usize) -> Result<usize> {
    if pos + 4 > data.len() {
        bail!("Truncated LSE marker at offset {}", pos);
    }
    let length = marker_length(data, pos);
    if pos + 2 + length <= data.len() && length >= 13 && data[pos + 4] == 1 {
        decoder.t1 = u16::from_be_bytes([data[pos + 7], data[pos + 8]]) as i32;
        decoder.t2 = u16::from_be_bytes([data[pos + 9], data[pos + 10]]) as i32;
        decoder.t3 = u16::from_be_bytes([data[pos + 11], data[pos + 12]]) as i32;
    }
    Ok(pos + 2 + length)
}

fn parse_sos(decoder: &mut JpegLsDecoder, data: &[u8], pos: usize) -> Result<()> {
    if pos + 1 >= data.len() {
        return Ok(());
    }
    let marker = u16::from_be_bytes([data[pos], data[pos + 1]]);
    if marker != SOS || pos + 4 >= data.len() {
        return Ok(());
    }

    let ns = if pos + 4 < data.len() {
        data[pos + 4] as usize
    } else {
        1
    };
    let comp_end = pos + 5 + ns * 2;
    if comp_end + 3 <= data.len() {
        decoder.near = data[comp_end] as u32;
        decoder.interleave_mode =
            InterleaveMode::try_from(data[comp_end + 1]).unwrap_or(InterleaveMode::None);
        decoder.point_transform = data[comp_end + 2];
    }
    Ok(())
}

fn skip_variable_marker(data: &[u8], pos: usize) -> usize {
    if pos + 4 <= data.len() {
        pos + 2 + marker_length(data, pos)
    } else {
        pos + 2
    }
}

fn marker_length(data: &[u8], pos: usize) -> usize {
    u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize
}
