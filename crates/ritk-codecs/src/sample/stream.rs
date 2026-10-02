//! Samples read from a byte stream in bounded steps.

use std::io::{self, Read, Seek, SeekFrom};

use consus_core::ByteOrder;

use super::{Sample, SampleBuffer, SampleType};

/// Count a payload through `limit`, returning `None` as soon as it exceeds it.
///
/// The reader is probed once beyond the limit, so `Some(limit)` means the
/// stream reached EOF at exactly that length. The fixed-size copy buffer keeps
/// memory independent of the payload size; reading to EOF also makes a
/// decompressor verify its trailer before the caller allocates typed samples.
///
/// # Errors
///
/// Returns the reader's error, including decompression or integrity failures,
/// when the limit or measured byte count cannot fit the platform integer.
///
/// # Examples
///
/// ```
/// use ritk_codecs::sample::count_payload_bytes;
///
/// let mut source = &b"voxel"[..];
/// assert_eq!(count_payload_bytes(&mut source, 5)?, Some(5));
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn count_payload_bytes<R: Read + ?Sized>(
    reader: &mut R,
    limit: usize,
) -> io::Result<Option<usize>> {
    let limit_usize = limit;
    let limit = u64::try_from(limit_usize).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("payload limit does not fit u64: {error}"),
        )
    })?;
    let copied = {
        let mut bounded = (&mut *reader).take(limit);
        io::copy(&mut bounded, &mut io::sink())?
    };

    let copied = usize::try_from(copied).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("payload byte count does not fit usize: {error}"),
        )
    })?;
    if copied < limit_usize {
        return Ok(Some(copied));
    }

    let mut excess = [0_u8; 1];
    if reader.read(&mut excess)? == 0 {
        Ok(Some(copied))
    } else {
        Ok(None)
    }
}

/// Check that the bytes from the current position to EOF have `expected` size.
///
/// The original position is restored on success and a length mismatch.
///
/// # Errors
///
/// Returns `UnexpectedEof` for a short payload, `InvalidData` for an excess
/// payload, `InvalidInput` when `expected` does not fit `u64`, or the seek
/// error.
///
/// # Examples
///
/// ```
/// use ritk_codecs::sample::validate_remaining_payload;
/// use std::io::{Cursor, Seek, SeekFrom};
///
/// let mut source = Cursor::new(b"headerdata");
/// source.seek(SeekFrom::Start(6))?;
/// validate_remaining_payload(&mut source, 4)?;
/// assert_eq!(source.stream_position()?, 6);
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn validate_remaining_payload<R: Seek + ?Sized>(
    reader: &mut R,
    expected: usize,
) -> io::Result<()> {
    let expected = u64::try_from(expected).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("expected payload size does not fit u64: {error}"),
        )
    })?;
    let start = reader.stream_position()?;
    let end = reader.seek(SeekFrom::End(0))?;
    let restored = reader.seek(SeekFrom::Start(start))?;
    if restored != start {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("payload position restored to {restored}, expected {start}"),
        ));
    }
    let actual = end.checked_sub(start).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "stream end precedes the current payload position",
        )
    })?;
    if actual == expected {
        Ok(())
    } else if actual < expected {
        Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("payload has {actual} bytes; expected {expected}"),
        ))
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("payload has {actual} bytes; expected {expected}"),
        ))
    }
}

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
