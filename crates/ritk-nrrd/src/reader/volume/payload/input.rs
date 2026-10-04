use std::collections::HashMap;
use std::io::{self, BufRead, Read, Seek, SeekFrom};

use ritk_codecs::SampleType;

use super::ascii::read_ascii_payload;
use super::NrrdEncoding;
use crate::reader::stored::NrrdStoredReadError;

pub(super) fn parse_line_skip(
    headers: &HashMap<String, String>,
) -> Result<i32, NrrdStoredReadError> {
    let Some(value) = headers.get("line skip").map(String::as_str) else {
        return Ok(0);
    };
    let parsed = value
        .parse::<i32>()
        .map_err(|source| NrrdStoredReadError::InvalidSkipField {
            field: "line skip",
            value: value.to_owned(),
            source,
        })?;
    if parsed < 0 {
        return Err(NrrdStoredReadError::NegativeLineSkip { value: parsed });
    }
    Ok(parsed)
}

pub(super) fn parse_byte_skip(
    headers: &HashMap<String, String>,
) -> Result<i32, NrrdStoredReadError> {
    let Some(value) = headers.get("byte skip").map(String::as_str) else {
        return Ok(0);
    };
    let parsed = value
        .parse::<i32>()
        .map_err(|source| NrrdStoredReadError::InvalidSkipField {
            field: "byte skip",
            value: value.to_owned(),
            source,
        })?;
    if parsed < -1 {
        return Err(NrrdStoredReadError::InvalidByteSkip {
            value: parsed,
            reason: "only nonnegative values and -1 are defined",
        });
    }
    Ok(parsed)
}

pub(super) fn read_nrrd_payload<R: BufRead + Seek>(
    reader: &mut R,
    encoding: NrrdEncoding,
    expected_bytes: usize,
    expected_samples: usize,
    sample_type: SampleType,
    element_type: &str,
    line_skip: i32,
    byte_skip: i32,
    data_start: u64,
) -> Result<Vec<u8>, NrrdStoredReadError> {
    if byte_skip == -1 && encoding != NrrdEncoding::Raw {
        return Err(NrrdStoredReadError::InvalidByteSkip {
            value: byte_skip,
            reason: "-1 is only defined for raw encoding",
        });
    }

    if byte_skip == -1 {
        seek_from_payload_end(reader, expected_bytes, data_start)?;
    } else {
        skip_lines(reader, line_skip)?;
    }

    match encoding {
        NrrdEncoding::Raw => {
            if byte_skip > 0 {
                skip_bytes(reader, byte_skip)?;
            }
            verify_raw_payload_length(reader, expected_bytes)?;
            read_exact_payload(reader, expected_bytes)
        }
        NrrdEncoding::Ascii => {
            if byte_skip > 0 {
                skip_bytes(reader, byte_skip)?;
            }
            read_ascii_payload(
                reader,
                expected_samples,
                expected_bytes,
                sample_type,
                element_type,
            )
        }
        NrrdEncoding::Gzip => {
            let mut decoder = flate2::read::MultiGzDecoder::new(reader);
            if byte_skip > 0 {
                skip_bytes(&mut decoder, byte_skip)?;
            }
            let bytes = read_exact_payload(&mut decoder, expected_bytes)?;
            io::copy(&mut decoder.take(1), &mut io::sink())
                .map_err(|source| NrrdStoredReadError::PayloadIo { source })?;
            Ok(bytes)
        }
    }
}

fn skip_lines<R: BufRead>(reader: &mut R, line_skip: i32) -> Result<(), NrrdStoredReadError> {
    let requested = u64::try_from(line_skip)
        .map_err(|_| NrrdStoredReadError::NegativeLineSkip { value: line_skip })?;
    let mut actual = 0_u64;
    let mut partial_line = false;
    while actual < requested {
        let buffer = reader
            .fill_buf()
            .map_err(|source| NrrdStoredReadError::PayloadIo { source })?;
        if buffer.is_empty() {
            if partial_line {
                actual += 1;
                partial_line = false;
                continue;
            }
            return Err(NrrdStoredReadError::InsufficientPayloadSkip {
                field: "line skip",
                requested,
                actual,
            });
        }
        if let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
            reader.consume(newline + 1);
            actual += 1;
            partial_line = false;
        } else {
            let consumed = buffer.len();
            reader.consume(consumed);
            partial_line = true;
        }
    }
    Ok(())
}

fn skip_bytes<R: Read>(reader: &mut R, byte_skip: i32) -> Result<(), NrrdStoredReadError> {
    let requested = u64::try_from(byte_skip).map_err(|_| NrrdStoredReadError::InvalidByteSkip {
        value: byte_skip,
        reason: "negative byte skips must be handled as raw end-relative offsets",
    })?;
    let mut actual = 0_u64;
    let mut buffer = [0_u8; 8192];
    while actual < requested {
        let remaining = requested - actual;
        let request = usize::try_from(remaining)
            .expect("invariant: an i32 byte skip fits usize")
            .min(buffer.len());
        let count = reader
            .read(&mut buffer[..request])
            .map_err(|source| NrrdStoredReadError::PayloadIo { source })?;
        if count == 0 {
            return Err(NrrdStoredReadError::InsufficientPayloadSkip {
                field: "byte skip",
                requested,
                actual,
            });
        }
        actual += u64::try_from(count).map_err(|_| {
            NrrdStoredReadError::PayloadLengthNotRepresentable {
                expected_bytes: count,
            }
        })?;
    }
    Ok(())
}

fn seek_from_payload_end<R: Seek>(
    reader: &mut R,
    expected_bytes: usize,
    data_start: u64,
) -> Result<(), NrrdStoredReadError> {
    let expected = u64::try_from(expected_bytes)
        .map_err(|_| NrrdStoredReadError::PayloadLengthNotRepresentable { expected_bytes })?;
    let end = reader
        .seek(SeekFrom::End(0))
        .map_err(|source| NrrdStoredReadError::PayloadIo { source })?;
    let start = end
        .checked_sub(expected)
        .ok_or(NrrdStoredReadError::TruncatedPayload {
            expected_bytes,
            actual_bytes: usize::try_from(end.saturating_sub(data_start))
                .expect("invariant: truncated payload is shorter than its usize length"),
        })?;
    if start < data_start {
        let available = end.saturating_sub(data_start);
        return Err(NrrdStoredReadError::TruncatedPayload {
            expected_bytes,
            actual_bytes: usize::try_from(available)
                .expect("invariant: available payload is shorter than its usize length"),
        });
    }
    reader
        .seek(SeekFrom::Start(start))
        .map_err(|source| NrrdStoredReadError::PayloadIo { source })?;
    Ok(())
}

fn verify_raw_payload_length<R: Seek>(
    reader: &mut R,
    expected_bytes: usize,
) -> Result<(), NrrdStoredReadError> {
    let expected = u64::try_from(expected_bytes)
        .map_err(|_| NrrdStoredReadError::PayloadLengthNotRepresentable { expected_bytes })?;
    let start = reader
        .stream_position()
        .map_err(|source| NrrdStoredReadError::PayloadIo { source })?;
    let end = reader
        .seek(SeekFrom::End(0))
        .map_err(|source| NrrdStoredReadError::PayloadIo { source })?;
    reader
        .seek(SeekFrom::Start(start))
        .map_err(|source| NrrdStoredReadError::PayloadIo { source })?;
    let available = end.saturating_sub(start);
    if available < expected {
        return Err(NrrdStoredReadError::TruncatedPayload {
            expected_bytes,
            actual_bytes: usize::try_from(available)
                .expect("invariant: truncated payload is shorter than its usize length"),
        });
    }
    Ok(())
}

fn read_exact_payload<R: Read>(
    reader: &mut R,
    expected_bytes: usize,
) -> Result<Vec<u8>, NrrdStoredReadError> {
    const CHUNK_BYTES: usize = 65_536;
    let mut output = Vec::new();
    let mut chunk = Vec::new();
    chunk
        .try_reserve_exact(CHUNK_BYTES)
        .map_err(|source| NrrdStoredReadError::Allocation {
            operation: "payload read buffer",
            source,
        })?;
    chunk.resize(CHUNK_BYTES, 0);
    while output.len() < expected_bytes {
        let remaining = expected_bytes - output.len();
        let request = remaining.min(CHUNK_BYTES);
        output
            .try_reserve(request)
            .map_err(|source| NrrdStoredReadError::Allocation {
                operation: "payload chunk",
                source,
            })?;
        let count = reader
            .read(&mut chunk[..request])
            .map_err(|source| NrrdStoredReadError::PayloadIo { source })?;
        if count == 0 {
            return Err(NrrdStoredReadError::TruncatedPayload {
                expected_bytes,
                actual_bytes: output.len(),
            });
        }
        output.extend_from_slice(&chunk[..count]);
    }
    Ok(output)
}
