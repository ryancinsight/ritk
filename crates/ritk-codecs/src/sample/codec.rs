use consus_core::types::datatype::ByteOrder as ConsusByteOrder;
use std::io::Write;

use crate::ByteOrder;

use super::buffer::{SampleBuffer, SampleType, StoredSamples};
use super::element::Sample;
use super::error::SampleError;

// F64 is a widest stored sample in SampleType::ALL.
const MAX_SAMPLE_WIDTH: usize = SampleType::F64.byte_width();

pub(super) fn decode(
    sample_type: SampleType,
    bytes: &[u8],
    byte_order: ByteOrder,
) -> Result<SampleBuffer, SampleError> {
    let samples = match sample_type {
        SampleType::U8 => StoredSamples::U8(decode_samples::<u8>(bytes, byte_order)?),
        SampleType::I8 => StoredSamples::I8(decode_samples::<i8>(bytes, byte_order)?),
        SampleType::U16 => StoredSamples::U16(decode_samples::<u16>(bytes, byte_order)?),
        SampleType::I16 => StoredSamples::I16(decode_samples::<i16>(bytes, byte_order)?),
        SampleType::U32 => StoredSamples::U32(decode_samples::<u32>(bytes, byte_order)?),
        SampleType::I32 => StoredSamples::I32(decode_samples::<i32>(bytes, byte_order)?),
        SampleType::U64 => StoredSamples::U64(decode_samples::<u64>(bytes, byte_order)?),
        SampleType::I64 => StoredSamples::I64(decode_samples::<i64>(bytes, byte_order)?),
        SampleType::F32 => StoredSamples::F32(decode_samples::<f32>(bytes, byte_order)?),
        SampleType::F64 => StoredSamples::F64(decode_samples::<f64>(bytes, byte_order)?),
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

fn decode_samples<T: Sample>(bytes: &[u8], byte_order: ByteOrder) -> Result<Vec<T>, SampleError> {
    let width = T::BYTE_WIDTH;
    let trailing_bytes = bytes.len() % width;
    if trailing_bytes != 0 {
        return Err(SampleError::PartialSample {
            sample_type: T::SAMPLE_TYPE,
            byte_length: bytes.len(),
            trailing_bytes,
        });
    }

    let sample_count = bytes.len() / width;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(sample_count)
        .map_err(SampleError::Allocation)?;

    let byte_order = consus_byte_order(byte_order);
    for chunk in bytes.chunks_exact(width) {
        let sample = T::from_bytes(chunk, byte_order).ok_or(SampleError::ScalarCodecRejected {
            sample_type: T::SAMPLE_TYPE,
        })?;
        samples.push(sample);
    }
    Ok(samples)
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
