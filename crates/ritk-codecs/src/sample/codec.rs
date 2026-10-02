//! Bulk conversion between packed bytes and typed samples.

use consus_core::ByteOrder;

use super::{Sample, SampleError};

/// Byte order of the compilation target.
pub const NATIVE_BYTE_ORDER: ByteOrder = if cfg!(target_endian = "big") {
    ByteOrder::BigEndian
} else {
    ByteOrder::LittleEndian
};

/// Decode `bytes`, packed `T` samples stored in `order`, into a vector of `T`.
///
/// The bytes are copied into the typed vector in one bulk copy; when `order`
/// differs from [`NATIVE_BYTE_ORDER`] each sample's bytes are then reversed in
/// place. The byte order is resolved once per buffer, never per sample. The
/// vector is created zeroed, which for a primitive `T` is a zeroed allocation
/// the copy overwrites, not a fill pass.
///
/// # Errors
///
/// Returns [`SampleError::PartialSample`] when `bytes.len()` is not a whole
/// number of `T` samples; a trailing partial sample is never dropped.
pub fn decode_samples<T: Sample>(bytes: &[u8], order: ByteOrder) -> Result<Vec<T>, SampleError> {
    let width = T::TYPE.byte_width();
    if !bytes.len().is_multiple_of(width) {
        return Err(SampleError::PartialSample {
            sample_type: T::TYPE,
            byte_len: bytes.len(),
        });
    }
    let mut values = vec![T::zero(); bytes.len() / width];
    bytemuck::cast_slice_mut::<T, u8>(&mut values).copy_from_slice(bytes);
    if order != NATIVE_BYTE_ORDER {
        swap_sample_bytes(&mut values);
    }
    Ok(values)
}

/// Reverse the byte order of every sample in `values` in place.
fn swap_sample_bytes<T: Sample>(values: &mut [T]) {
    let width = T::TYPE.byte_width();
    if width > 1 {
        bytemuck::cast_slice_mut::<T, u8>(values)
            .chunks_exact_mut(width)
            .for_each(<[u8]>::reverse);
    }
}
