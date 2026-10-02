//! Streaming encode of typed samples into packed bytes.

use std::io::{self, Write};

use consus_core::{extend_encoded, ByteOrder};

use super::Sample;

/// Samples encoded per `write_all` call.
///
/// Bounds the scratch buffer at 64 KiB for the widest (8-byte) type, whatever
/// the volume size, so a writer never holds a second full copy of the volume;
/// at that size the per-call overhead of `write_all` is small against the
/// copy it carries.
const SAMPLES_PER_WRITE: usize = 8192;

/// Write `values` to `writer` as packed samples in `order`.
///
/// Encoding goes through consus-core's bulk encoder, which resolves the byte
/// order once per block.
///
/// # Errors
///
/// Returns an allocation error when the bounded scratch buffer cannot be
/// reserved, or the writer's error.
pub fn write_samples<T: Sample, W: Write + ?Sized>(
    values: &[T],
    order: ByteOrder,
    writer: &mut W,
) -> io::Result<()> {
    let capacity = SAMPLES_PER_WRITE.min(values.len()) * T::TYPE.byte_width();
    let mut block = Vec::new();
    block
        .try_reserve_exact(capacity)
        .map_err(|error| io::Error::new(io::ErrorKind::OutOfMemory, error))?;
    for chunk in values.chunks(SAMPLES_PER_WRITE) {
        block.clear();
        extend_encoded(&mut block, chunk.iter().copied(), order);
        writer.write_all(&block)?;
    }
    Ok(())
}
