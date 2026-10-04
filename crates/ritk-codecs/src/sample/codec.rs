//! Bulk conversion between packed bytes and typed samples.

use consus_core::{read_integer, write_integer, ByteOrder, EndianScalar};

use super::{Sample, SampleError};

/// Decode `bytes`, packed `T` samples stored in `order`, into a vector of `T`.
///
/// The byte order is passed to Consus's fixed-width scalar reader, which
/// converts each exact-width chunk directly into `T`. The output vector
/// reserves the exact sample count once before conversion begins.
///
/// # Errors
///
/// Returns [`SampleError::PartialSample`] when `bytes.len()` is not a whole
/// number of `T` samples, or [`SampleError::Allocation`] when the output
/// vector cannot reserve space. A trailing partial sample is never dropped.
pub fn decode_samples<T: Sample>(bytes: &[u8], order: ByteOrder) -> Result<Vec<T>, SampleError> {
    let width = T::BYTE_WIDTH;
    if !bytes.len().is_multiple_of(width) {
        return Err(SampleError::PartialSample {
            sample_type: T::TYPE,
            byte_len: bytes.len(),
        });
    }
    let sample_count = bytes.len() / width;
    let mut values = Vec::new();
    values
        .try_reserve_exact(sample_count)
        .map_err(|source| SampleError::Allocation {
            sample_type: T::TYPE,
            sample_count,
            requested_bytes: bytes.len(),
            source,
        })?;
    for chunk in bytes.chunks_exact(width) {
        values.push(
            read_integer::<T>(chunk, order)
                .expect("invariant: chunks_exact yields one complete sample"),
        );
    }
    Ok(values)
}

/// Encode typed samples into packed bytes in `order`.
///
/// The result contains exactly `values.len() * T::BYTE_WIDTH` bytes, with no
/// metadata, padding, or implicit sample conversion.
///
/// # Errors
///
/// Returns [`SampleError::LengthOverflow`] when the packed byte length cannot
/// fit in `usize`, or [`SampleError::Allocation`] when the output vector
/// cannot reserve that many bytes.
pub fn encode_samples<T: Sample>(
    values: &[T],
    order: ByteOrder,
) -> Result<Vec<u8>, SampleError> {
    let sample_count = values.len();
    let requested_bytes = sample_count
        .checked_mul(T::BYTE_WIDTH)
        .ok_or(SampleError::LengthOverflow {
            sample_type: T::TYPE,
            sample_count,
        })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(requested_bytes)
        .map_err(|source| SampleError::Allocation {
            sample_type: T::TYPE,
            sample_count,
            requested_bytes,
            source,
        })?;
    let mut sample_bytes = [0_u8; std::mem::size_of::<u64>()];
    for &value in values {
        let sample_bytes = &mut sample_bytes[..T::BYTE_WIDTH];
        write_integer(sample_bytes, value, order)
            .expect("invariant: sample_bytes has exactly T::BYTE_WIDTH bytes");
        bytes.extend_from_slice(sample_bytes);
    }
    Ok(bytes)
}
