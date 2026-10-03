use std::io::BufRead;

use ritk_codecs::SampleType;

use crate::reader::stored::NrrdStoredReadError;

const MAX_ASCII_TOKEN_BYTES: usize = 128;

pub(super) fn read_ascii_payload<R: BufRead>(
    reader: &mut R,
    expected_samples: usize,
    expected_bytes: usize,
    sample_type: SampleType,
    element_type: &str,
) -> Result<Vec<u8>, NrrdStoredReadError> {
    let calculated_bytes = expected_samples
        .checked_mul(sample_type.byte_width())
        .ok_or(NrrdStoredReadError::PayloadByteCountOverflow {
            voxel_count: expected_samples,
            sample_width: sample_type.byte_width(),
        })?;
    if calculated_bytes != expected_bytes {
        return Err(NrrdStoredReadError::PayloadSampleCountMismatch {
            expected_bytes,
            actual_bytes: calculated_bytes,
        });
    }
    let mut output = Vec::new();
    let mut token = [0_u8; MAX_ASCII_TOKEN_BYTES];
    for sample_index in 0..expected_samples {
        let Some((token_length, too_long)) = read_ascii_token(reader, &mut token)? else {
            return Err(NrrdStoredReadError::TruncatedAsciiPayload {
                expected_samples,
                actual_samples: sample_index,
            });
        };
        if too_long {
            return Err(NrrdStoredReadError::AsciiTokenTooLong {
                sample_index,
                maximum_bytes: MAX_ASCII_TOKEN_BYTES,
            });
        }
        output
            .try_reserve(sample_type.byte_width())
            .map_err(|source| NrrdStoredReadError::Allocation {
                operation: "ASCII sample payload",
                source,
            })?;
        append_ascii_sample(
            &token[..token_length],
            sample_index,
            sample_type,
            element_type,
            &mut output,
        )?;
    }
    Ok(output)
}

fn read_ascii_token<R: BufRead>(
    reader: &mut R,
    token: &mut [u8; MAX_ASCII_TOKEN_BYTES],
) -> Result<Option<(usize, bool)>, NrrdStoredReadError> {
    let mut length = 0;
    let mut started = false;
    let mut too_long = false;
    loop {
        let buffer = reader
            .fill_buf()
            .map_err(|source| NrrdStoredReadError::PayloadIo { source })?;
        if buffer.is_empty() {
            return Ok(started.then_some((length, too_long)));
        }
        let mut consumed = 0;
        for byte in buffer.iter().copied() {
            consumed += 1;
            if is_c_whitespace(byte) {
                if started {
                    reader.consume(consumed);
                    return Ok(Some((length, too_long)));
                }
                continue;
            }
            started = true;
            if length < token.len() {
                token[length] = byte;
                length += 1;
            } else {
                too_long = true;
            }
        }
        reader.consume(consumed);
    }
}

const fn is_c_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn append_ascii_sample(
    token: &[u8],
    sample_index: usize,
    sample_type: SampleType,
    element_type: &str,
    output: &mut Vec<u8>,
) -> Result<(), NrrdStoredReadError> {
    let value = std::str::from_utf8(token)
        .map_err(|_| invalid_ascii_sample(token, sample_index, element_type))?;
    let invalid = || invalid_ascii_sample(token, sample_index, element_type);
    match sample_type {
        SampleType::U8 => output.push(value.parse::<u8>().map_err(|_| invalid())?),
        SampleType::I8 => {
            output.extend_from_slice(&value.parse::<i8>().map_err(|_| invalid())?.to_le_bytes())
        }
        SampleType::U16 => {
            output.extend_from_slice(&value.parse::<u16>().map_err(|_| invalid())?.to_le_bytes())
        }
        SampleType::I16 => {
            output.extend_from_slice(&value.parse::<i16>().map_err(|_| invalid())?.to_le_bytes())
        }
        SampleType::U32 => {
            output.extend_from_slice(&value.parse::<u32>().map_err(|_| invalid())?.to_le_bytes())
        }
        SampleType::I32 => {
            output.extend_from_slice(&value.parse::<i32>().map_err(|_| invalid())?.to_le_bytes())
        }
        SampleType::U64 => {
            output.extend_from_slice(&value.parse::<u64>().map_err(|_| invalid())?.to_le_bytes())
        }
        SampleType::I64 => {
            output.extend_from_slice(&value.parse::<i64>().map_err(|_| invalid())?.to_le_bytes())
        }
        SampleType::F32 => {
            output.extend_from_slice(&parse_float::<f32>(value).ok_or_else(invalid)?.to_le_bytes())
        }
        SampleType::F64 => {
            output.extend_from_slice(&parse_float::<f64>(value).ok_or_else(invalid)?.to_le_bytes())
        }
        _ => {
            return Err(NrrdStoredReadError::UnsupportedElementType {
                element_type: element_type.to_owned(),
            })
        }
    }
    Ok(())
}

trait AsciiFloat: Sized {
    fn parse_decimal(value: &str) -> Option<Self>;
    fn nan() -> Self;
    fn positive_infinity() -> Self;
    fn negative_infinity() -> Self;
}

impl AsciiFloat for f32 {
    fn parse_decimal(value: &str) -> Option<Self> {
        value.parse().ok()
    }

    fn nan() -> Self {
        Self::NAN
    }

    fn positive_infinity() -> Self {
        Self::INFINITY
    }

    fn negative_infinity() -> Self {
        Self::NEG_INFINITY
    }
}

impl AsciiFloat for f64 {
    fn parse_decimal(value: &str) -> Option<Self> {
        value.parse().ok()
    }

    fn nan() -> Self {
        Self::NAN
    }

    fn positive_infinity() -> Self {
        Self::INFINITY
    }

    fn negative_infinity() -> Self {
        Self::NEG_INFINITY
    }
}

fn parse_float<T: AsciiFloat>(value: &str) -> Option<T> {
    if contains_ascii_case_insensitive(value, b"nan") {
        Some(T::nan())
    } else if contains_ascii_case_insensitive(value, b"-inf") {
        Some(T::negative_infinity())
    } else if contains_ascii_case_insensitive(value, b"inf") {
        Some(T::positive_infinity())
    } else {
        T::parse_decimal(value)
    }
}

fn contains_ascii_case_insensitive(value: &str, needle: &[u8]) -> bool {
    value
        .as_bytes()
        .windows(needle.len())
        .any(|candidate| candidate.eq_ignore_ascii_case(needle))
}

fn invalid_ascii_sample(
    token: &[u8],
    sample_index: usize,
    element_type: &str,
) -> NrrdStoredReadError {
    NrrdStoredReadError::InvalidAsciiSample {
        sample_index,
        value: String::from_utf8_lossy(token).into_owned(),
        sample_type: element_type.to_owned(),
    }
}
