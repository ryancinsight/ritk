//! Samples read from a byte stream in bounded steps.

use std::io::{self, Read};

use consus_core::ByteOrder;

use super::{Sample, SampleBuffer, SampleType};

impl SampleBuffer {
    /// Read `count` samples of `sample_type` stored in `order` from `reader`.
    ///
    /// The streaming counterpart of [`decode`](Self::decode) for payloads
    /// behind a decompressor or file handle: `consus_core::read_extend`
    /// reads in fixed steps and grows the buffer only by samples already
    /// read, so a `count` taken from an untrusted header never reserves more
    /// than the stream supplies.
    ///
    /// # Errors
    ///
    /// Returns `UnexpectedEof` naming the first missing sample when the
    /// stream ends early, `OutOfMemory` when the buffer cannot grow, or the
    /// reader's error.
    pub fn read_from<R: Read + ?Sized>(
        reader: &mut R,
        sample_type: SampleType,
        order: ByteOrder,
        count: usize,
    ) -> io::Result<Self> {
        Ok(match sample_type {
            SampleType::U8 => Self::U8(read_values(reader, order, count)?),
            SampleType::I8 => Self::I8(read_values(reader, order, count)?),
            SampleType::U16 => Self::U16(read_values(reader, order, count)?),
            SampleType::I16 => Self::I16(read_values(reader, order, count)?),
            SampleType::U32 => Self::U32(read_values(reader, order, count)?),
            SampleType::I32 => Self::I32(read_values(reader, order, count)?),
            SampleType::U64 => Self::U64(read_values(reader, order, count)?),
            SampleType::I64 => Self::I64(read_values(reader, order, count)?),
            SampleType::F32 => Self::F32(read_values(reader, order, count)?),
            SampleType::F64 => Self::F64(read_values(reader, order, count)?),
        })
    }
}

/// `count` values of `T`, or the stream's failure with the index of the first
/// missing sample when the stream ends early.
fn read_values<T: Sample, R: Read + ?Sized>(
    reader: &mut R,
    order: ByteOrder,
    count: usize,
) -> io::Result<Vec<T>> {
    let mut values = Vec::new();
    match consus_core::read_extend(reader, order, count, &mut values) {
        Ok(()) => Ok(values),
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "stream ended at {} sample {} of {count}",
                T::TYPE,
                values.len()
            ),
        )),
        Err(error) => Err(error),
    }
}
