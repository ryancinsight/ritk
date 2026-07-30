//! JPEG-LS marker parser.

use super::{ComponentInfo, InterleaveMode, JpegLsDecoder, DNL, DRI, EOI, LSE, SOF55, SOI, SOS};
use anyhow::{anyhow, bail, Result};

/// Parse all JPEG-LS markers and return the entropy-coded scan bytes.
pub(crate) fn parse_jpeg_ls_headers<'a>(
    decoder: &mut JpegLsDecoder,
    data: &'a [u8],
) -> Result<&'a [u8]> {
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
            _ => pos = skip_variable_marker(data, pos)?,
        }
    }

    let scan_start = parse_sos(decoder, data, pos)?;
    if scan_start >= data.len() {
        bail!("JPEG-LS scan data is missing after SOS marker at offset {pos}");
    }
    Ok(&data[scan_start..])
}

fn parse_sof55(decoder: &mut JpegLsDecoder, data: &[u8], pos: usize) -> Result<usize> {
    let (length, end) = marker_segment_bounds(data, pos, "SOF55")?;
    if length < 8 {
        bail!("JPEG-LS SOF55 length {length} is shorter than 8 at offset {pos}");
    }
    decoder.bits_per_sample = u32::from(data[pos + 4]);
    decoder.height = usize::from(u16::from_be_bytes([data[pos + 5], data[pos + 6]]));
    decoder.width = usize::from(u16::from_be_bytes([data[pos + 7], data[pos + 8]]));
    let num_comp = usize::from(data[pos + 9]);
    let expected_length = num_comp
        .checked_mul(3)
        .and_then(|component_bytes| component_bytes.checked_add(8))
        .ok_or_else(|| anyhow!("JPEG-LS SOF55 component count overflows at offset {pos}"))?;
    if length != expected_length {
        bail!(
            "JPEG-LS SOF55 length {length} does not match {num_comp} components; expected {expected_length}"
        );
    }
    decoder.components.clear();
    for _ in 0..num_comp {
        decoder.components.push(ComponentInfo {});
    }
    Ok(end)
}

fn parse_dnl(decoder: &mut JpegLsDecoder, data: &[u8], pos: usize) -> Result<usize> {
    let (length, end) = marker_segment_bounds(data, pos, "DNL")?;
    if length != 4 {
        bail!("JPEG-LS DNL length {length} is not 4 at offset {pos}");
    }
    decoder.height = usize::from(u16::from_be_bytes([data[pos + 4], data[pos + 5]]));
    Ok(end)
}

fn parse_dri(_decoder: &mut JpegLsDecoder, data: &[u8], pos: usize) -> Result<usize> {
    let (length, end) = marker_segment_bounds(data, pos, "DRI")?;
    if length != 4 {
        bail!("JPEG-LS DRI length {length} is not 4 at offset {pos}");
    }
    let restart_interval = u16::from_be_bytes([data[pos + 4], data[pos + 5]]);
    if restart_interval != 0 {
        bail!("JPEG-LS restart interval {restart_interval} is unsupported at offset {pos}");
    }
    Ok(end)
}

fn parse_lse(decoder: &mut JpegLsDecoder, data: &[u8], pos: usize) -> Result<usize> {
    let (length, end) = marker_segment_bounds(data, pos, "LSE")?;
    if length < 3 {
        bail!("JPEG-LS LSE length {length} is shorter than 3 at offset {pos}");
    }
    let parameter_id = data[pos + 4];
    if parameter_id != 1 {
        bail!("JPEG-LS LSE parameter ID {parameter_id} is unsupported at offset {pos}");
    }
    if length != 13 {
        bail!("JPEG-LS preset LSE length {length} is not 13 at offset {pos}");
    }
    decoder.t1 = i32::from(u16::from_be_bytes([data[pos + 7], data[pos + 8]]));
    decoder.t2 = i32::from(u16::from_be_bytes([data[pos + 9], data[pos + 10]]));
    decoder.t3 = i32::from(u16::from_be_bytes([data[pos + 11], data[pos + 12]]));
    Ok(end)
}

fn parse_sos(decoder: &mut JpegLsDecoder, data: &[u8], pos: usize) -> Result<usize> {
    let marker_end = pos
        .checked_add(2)
        .ok_or_else(|| anyhow!("JPEG-LS SOS marker offset overflows"))?;
    if marker_end > data.len() {
        bail!("JPEG-LS SOS marker is missing at offset {pos}");
    }
    let marker = u16::from_be_bytes([data[pos], data[pos + 1]]);
    if marker != SOS {
        bail!("JPEG-LS SOS marker is missing at offset {pos}");
    }

    let (length, end) = marker_segment_bounds(data, pos, "SOS")?;
    if length < 6 {
        bail!("JPEG-LS SOS length {length} is shorter than 6 at offset {pos}");
    }
    let component_count = usize::from(data[pos + 4]);
    let expected_length = component_count
        .checked_mul(2)
        .and_then(|component_bytes| component_bytes.checked_add(6))
        .ok_or_else(|| anyhow!("JPEG-LS SOS component count overflows at offset {pos}"))?;
    if length != expected_length {
        bail!(
            "JPEG-LS SOS length {length} does not match {component_count} components; expected {expected_length}"
        );
    }
    for component in 0..component_count {
        let mapping_table_selector = data[pos + 6 + component * 2];
        if mapping_table_selector != 0 {
            bail!(
                "JPEG-LS mapping table selector {mapping_table_selector} for scan component {component} is unsupported"
            );
        }
    }
    let component_end = pos + 5 + component_count * 2;
    decoder.near = u32::from(data[component_end]);
    let interleave = data[component_end + 1];
    decoder.interleave_mode = InterleaveMode::try_from(interleave)
        .map_err(|value| anyhow!("invalid JPEG-LS interleave mode {value}; expected 0, 1, or 2"))?;
    decoder.point_transform = data[component_end + 2];
    Ok(end)
}

fn skip_variable_marker(data: &[u8], pos: usize) -> Result<usize> {
    marker_segment_bounds(data, pos, "unknown marker").map(|(_, end)| end)
}

fn marker_segment_bounds(data: &[u8], pos: usize, marker_name: &str) -> Result<(usize, usize)> {
    let length_end = pos
        .checked_add(4)
        .ok_or_else(|| anyhow!("JPEG-LS {marker_name} offset overflows"))?;
    if length_end > data.len() {
        bail!("truncated JPEG-LS {marker_name} length at offset {pos}");
    }
    let length = usize::from(u16::from_be_bytes([data[pos + 2], data[pos + 3]]));
    if length < 2 {
        bail!("JPEG-LS {marker_name} length {length} is shorter than 2 at offset {pos}");
    }
    let end = pos
        .checked_add(2)
        .and_then(|offset| offset.checked_add(length))
        .ok_or_else(|| anyhow!("JPEG-LS {marker_name} length overflows at offset {pos}"))?;
    if end > data.len() {
        bail!(
            "truncated JPEG-LS {marker_name} segment at offset {pos}: declared end {end}, input length {}",
            data.len()
        );
    }
    Ok((length, end))
}
