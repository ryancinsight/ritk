use consus_core::types::datatype::ByteOrder as ConsusByteOrder;
use std::io::{self, Read, Write};

use crate::ByteOrder;

use super::buffer::{SampleBuffer, SampleType, StoredSamples};
use super::element::Sample;
use super::error::SampleError;

// F64 is a widest stored sample in SampleType::ALL.
const MAX_SAMPLE_WIDTH: usize = SampleType::F64.byte_width();
// Keep stream staging bounded to the standard library's current 8 KiB
// BufReader capacity, independent of the image's sample count.
const READ_BUFFER_BYTES: usize = 8 * 1024;

pub(super) fn decode(
    sample_type: SampleType,
    bytes: &[u8],
    byte_order: ByteOrder,
) -> Result<SampleBuffer, SampleError> {
    let width = sample_type.byte_width();
    let trailing_bytes = bytes.len() % width;
    if trailing_bytes != 0 {
        return Err(SampleError::PartialSample {
            sample_type,
            byte_length: bytes.len(),
            trailing_bytes,
        });
    }
    let mut reader = io::Cursor::new(bytes);
    read(sample_type, &mut reader, bytes.len() / width, byte_order)
}

pub(super) fn read<R: Read>(
    sample_type: SampleType,
    reader: &mut R,
    sample_count: usize,
    byte_order: ByteOrder,
) -> Result<SampleBuffer, SampleError> {
    let samples = match sample_type {
        SampleType::U8 => {
            StoredSamples::U8(read_samples::<u8, _>(reader, sample_count, byte_order)?)
        }
        SampleType::I8 => {
            StoredSamples::I8(read_samples::<i8, _>(reader, sample_count, byte_order)?)
        }
        SampleType::U16 => {
            StoredSamples::U16(read_samples::<u16, _>(reader, sample_count, byte_order)?)
        }
        SampleType::I16 => {
            StoredSamples::I16(read_samples::<i16, _>(reader, sample_count, byte_order)?)
        }
        SampleType::U32 => {
            StoredSamples::U32(read_samples::<u32, _>(reader, sample_count, byte_order)?)
        }
        SampleType::I32 => {
            StoredSamples::I32(read_samples::<i32, _>(reader, sample_count, byte_order)?)
        }
        SampleType::U64 => {
            StoredSamples::U64(read_samples::<u64, _>(reader, sample_count, byte_order)?)
        }
        SampleType::I64 => {
            StoredSamples::I64(read_samples::<i64, _>(reader, sample_count, byte_order)?)
        }
        SampleType::F32 => {
            StoredSamples::F32(read_samples::<f32, _>(reader, sample_count, byte_order)?)
        }
        SampleType::F64 => {
            StoredSamples::F64(read_samples::<f64, _>(reader, sample_count, byte_order)?)
        }
    };
    Ok(SampleBuffer { samples })
}

pub(super) fn encode(
    samples: &StoredSamples,
    byte_order: ByteOrder,
) -> Result<Vec<u8>, SampleError> {
    match samples {
        StoredSamples::U8(values) => encode_samples(values, byte_order),
        StoredSamples::I8(values) => encode_samples(values, byte_order),
        StoredSamples::U16(values) => encode_samples(values, byte_order),
        StoredSamples::I16(values) => encode_samples(values, byte_order),
        StoredSamples::U32(values) => encode_samples(values, byte_order),
        StoredSamples::I32(values) => encode_samples(values, byte_order),
        StoredSamples::U64(values) => encode_samples(values, byte_order),
        StoredSamples::I64(values) => encode_samples(values, byte_order),
        StoredSamples::F32(values) => encode_samples(values, byte_order),
        StoredSamples::F64(values) => encode_samples(values, byte_order),
    }
}

pub(super) fn write<W: Write>(
    samples: &StoredSamples,
    byte_order: ByteOrder,
    writer: &mut W,
) -> Result<(), SampleError> {
    match samples {
        StoredSamples::U8(values) => write_samples(values, byte_order, writer),
        StoredSamples::I8(values) => write_samples(values, byte_order, writer),
        StoredSamples::U16(values) => write_samples(values, byte_order, writer),
        StoredSamples::I16(values) => write_samples(values, byte_order, writer),
        StoredSamples::U32(values) => write_samples(values, byte_order, writer),
        StoredSamples::I32(values) => write_samples(values, byte_order, writer),
        StoredSamples::U64(values) => write_samples(values, byte_order, writer),
        StoredSamples::I64(values) => write_samples(values, byte_order, writer),
        StoredSamples::F32(values) => write_samples(values, byte_order, writer),
        StoredSamples::F64(values) => write_samples(values, byte_order, writer),
    }
}

fn read_samples<T: Sample, R: Read>(
    reader: &mut R,
    sample_count: usize,
    byte_order: ByteOrder,
) -> Result<Vec<T>, SampleError> {
    let width = T::BYTE_WIDTH;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(sample_count)
        .map_err(SampleError::Allocation)?;

    let byte_order = consus_byte_order(byte_order);
    let mut encoded = [0_u8; READ_BUFFER_BYTES];
    let mut completed_samples = 0;
    while completed_samples < sample_count {
        let block_samples = (sample_count - completed_samples).min(READ_BUFFER_BYTES / width);
        let block_bytes = block_samples * width;
        let mut bytes_read = 0;
        while bytes_read < block_bytes {
            let destination = encoded.get_mut(bytes_read..block_bytes).ok_or(
                SampleError::ScalarCodecRejected {
                    sample_type: T::SAMPLE_TYPE,
                },
            )?;
            match reader.read(destination) {
                Ok(0) => {
                    return Err(truncated_input(
                        T::SAMPLE_TYPE,
                        sample_count,
                        completed_samples + bytes_read / width,
                    ));
                }
                Ok(count) => bytes_read += count,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                    return Err(truncated_input(
                        T::SAMPLE_TYPE,
                        sample_count,
                        completed_samples + bytes_read / width,
                    ));
                }
                Err(error) => return Err(SampleError::Io(error)),
            }
        }
        let block = encoded
            .get(..block_bytes)
            .ok_or(SampleError::ScalarCodecRejected {
                sample_type: T::SAMPLE_TYPE,
            })?;
        for raw in block.chunks_exact(width) {
            let sample =
                T::from_bytes(raw, byte_order).ok_or(SampleError::ScalarCodecRejected {
                    sample_type: T::SAMPLE_TYPE,
                })?;
            samples.push(sample);
        }
        completed_samples += block_samples;
    }
    Ok(samples)
}

fn truncated_input(
    sample_type: SampleType,
    sample_count: usize,
    completed_samples: usize,
) -> SampleError {
    SampleError::TruncatedInput {
        sample_type,
        sample_count,
        completed_samples,
    }
}

fn encode_samples<T: Sample>(samples: &[T], byte_order: ByteOrder) -> Result<Vec<u8>, SampleError> {
    let sample_width = T::BYTE_WIDTH;
    let byte_length =
        samples
            .len()
            .checked_mul(sample_width)
            .ok_or(SampleError::EncodedLengthOverflow {
                sample_count: samples.len(),
                sample_width,
            })?;

    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(byte_length)
        .map_err(SampleError::Allocation)?;

    let byte_order = consus_byte_order(byte_order);
    let mut scratch = [0_u8; MAX_SAMPLE_WIDTH];
    for sample in samples {
        bytes.extend_from_slice(encode_sample(sample, byte_order, &mut scratch)?);
    }
    Ok(bytes)
}

fn write_samples<T: Sample, W: Write>(
    samples: &[T],
    byte_order: ByteOrder,
    writer: &mut W,
) -> Result<(), SampleError> {
    let byte_order = consus_byte_order(byte_order);
    let mut scratch = [0_u8; MAX_SAMPLE_WIDTH];
    for sample in samples {
        writer.write_all(encode_sample(sample, byte_order, &mut scratch)?)?;
    }
    Ok(())
}

fn encode_sample<'a, T: Sample>(
    sample: &T,
    byte_order: ConsusByteOrder,
    scratch: &'a mut [u8; MAX_SAMPLE_WIDTH],
) -> Result<&'a [u8], SampleError> {
    let encoded = scratch
        .get_mut(..T::BYTE_WIDTH)
        .ok_or(SampleError::ScalarCodecRejected {
            sample_type: T::SAMPLE_TYPE,
        })?;
    sample
        .to_bytes(encoded, byte_order)
        .ok_or(SampleError::ScalarCodecRejected {
            sample_type: T::SAMPLE_TYPE,
        })?;
    Ok(encoded)
}

const fn consus_byte_order(byte_order: ByteOrder) -> ConsusByteOrder {
    match byte_order {
        ByteOrder::MostSignificantByteFirst => ConsusByteOrder::BigEndian,
        ByteOrder::LeastSignificantByteFirst => ConsusByteOrder::LittleEndian,
    }
}
