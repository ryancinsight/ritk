use std::fmt;
use std::io::{Read, Write};

use crate::ByteOrder;

use super::codec;
use super::element::Sample;
use super::error::{SampleError, SampleExtractionError};

/// The fixed-width numeric representation stored by an image format.
///
/// # Examples
///
/// ```
/// use ritk_codecs::SampleType;
///
/// assert_eq!(SampleType::F64.byte_width(), 8);
/// assert!(SampleType::ALL.contains(&SampleType::U16));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SampleType {
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// IEEE 754 32-bit floating-point value.
    F32,
    /// IEEE 754 64-bit floating-point value.
    F64,
}

impl SampleType {
    /// Every fixed-width sample representation supported by the codec.
    pub const ALL: [Self; 10] = [
        Self::U8,
        Self::I8,
        Self::U16,
        Self::I16,
        Self::U32,
        Self::I32,
        Self::U64,
        Self::I64,
        Self::F32,
        Self::F64,
    ];

    /// Returns the byte width of this stored representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_codecs::SampleType;
    ///
    /// assert_eq!(SampleType::U32.byte_width(), 4);
    /// ```
    #[must_use]
    pub const fn byte_width(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

/// Owned samples decoded without changing their fixed-width representation.
pub struct SampleBuffer {
    pub(super) samples: StoredSamples,
}

pub(super) enum StoredSamples {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl SampleBuffer {
    /// Decodes a complete byte buffer into the declared stored sample type.
    ///
    /// Both byte orders are supported. A partial final sample and allocation
    /// failure are reported without returning partially decoded data.
    ///
    /// # Errors
    ///
    /// Returns [`SampleError::PartialSample`] when the byte length is not a
    /// multiple of [`SampleType::byte_width`], or an allocation or scalar
    /// codec error.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_codecs::{ByteOrder, SampleBuffer, SampleType};
    ///
    /// let samples = SampleBuffer::decode(
    ///     SampleType::U32,
    ///     &16_777_217_u32.to_le_bytes(),
    ///     ByteOrder::LeastSignificantByteFirst,
    /// )?;
    /// assert_eq!(samples.try_into_samples::<u32>()?, [16_777_217]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decode(
        sample_type: SampleType,
        bytes: &[u8],
        byte_order: ByteOrder,
    ) -> Result<Self, SampleError> {
        codec::decode(sample_type, bytes, byte_order)
    }

    /// Reads an exact number of stored samples from a stream.
    ///
    /// The buffer retains the typed sample allocation and stages encoded bytes
    /// through a fixed 8 KiB block; it does not copy the full encoded payload.
    /// Bytes after `sample_count` samples remain unread.
    ///
    /// # Errors
    ///
    /// Returns [`SampleError::TruncatedInput`] when the stream ends before all
    /// samples arrive, [`SampleError::Io`] for another stream error, or an
    /// allocation or scalar codec error.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Cursor;
    /// use ritk_codecs::{ByteOrder, SampleBuffer, SampleType};
    ///
    /// let mut input = Cursor::new(16_777_217_u32.to_le_bytes());
    /// let samples = SampleBuffer::read_from(
    ///     SampleType::U32,
    ///     &mut input,
    ///     1,
    ///     ByteOrder::LeastSignificantByteFirst,
    /// )?;
    /// assert_eq!(samples.try_into_samples::<u32>()?, [16_777_217]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn read_from<R: Read>(
        sample_type: SampleType,
        reader: &mut R,
        sample_count: usize,
        byte_order: ByteOrder,
    ) -> Result<Self, SampleError> {
        codec::read(sample_type, reader, sample_count, byte_order)
    }

    /// Takes ownership of values while recording their stored sample type.
    ///
    /// This operation does not allocate or convert values.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_codecs::{SampleBuffer, SampleType};
    ///
    /// let samples = SampleBuffer::from_samples(vec![0_u16, 16_777]);
    /// assert_eq!(samples.sample_type(), SampleType::U16);
    /// assert_eq!(samples.len(), 2);
    /// ```
    #[must_use]
    pub fn from_samples<T: Sample>(samples: Vec<T>) -> Self {
        T::into_sample_buffer(samples)
    }

    /// Returns the fixed-width type stored in this buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_codecs::{SampleBuffer, SampleType};
    ///
    /// let samples = SampleBuffer::from_samples(vec![7_u16]);
    /// assert_eq!(samples.sample_type(), SampleType::U16);
    /// ```
    #[must_use]
    pub const fn sample_type(&self) -> SampleType {
        match &self.samples {
            StoredSamples::U8(_) => SampleType::U8,
            StoredSamples::I8(_) => SampleType::I8,
            StoredSamples::U16(_) => SampleType::U16,
            StoredSamples::I16(_) => SampleType::I16,
            StoredSamples::U32(_) => SampleType::U32,
            StoredSamples::I32(_) => SampleType::I32,
            StoredSamples::U64(_) => SampleType::U64,
            StoredSamples::I64(_) => SampleType::I64,
            StoredSamples::F32(_) => SampleType::F32,
            StoredSamples::F64(_) => SampleType::F64,
        }
    }

    /// Returns the number of stored samples.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_codecs::SampleBuffer;
    ///
    /// let samples = SampleBuffer::from_samples(vec![7_u16, 9]);
    /// assert_eq!(samples.len(), 2);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.samples {
            StoredSamples::U8(values) => values.len(),
            StoredSamples::I8(values) => values.len(),
            StoredSamples::U16(values) => values.len(),
            StoredSamples::I16(values) => values.len(),
            StoredSamples::U32(values) => values.len(),
            StoredSamples::I32(values) => values.len(),
            StoredSamples::U64(values) => values.len(),
            StoredSamples::I64(values) => values.len(),
            StoredSamples::F32(values) => values.len(),
            StoredSamples::F64(values) => values.len(),
        }
    }

    /// Reports whether this buffer contains no samples.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_codecs::SampleBuffer;
    ///
    /// let samples = SampleBuffer::from_samples(Vec::<u16>::new());
    /// assert!(samples.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Encodes all samples in the requested byte order.
    ///
    /// Floating-point values are written from their bit representation, so
    /// signed zero and NaN payloads are retained.
    ///
    /// # Errors
    ///
    /// Returns an allocation or encoded-length overflow error.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_codecs::{ByteOrder, SampleBuffer};
    ///
    /// let samples = SampleBuffer::from_samples(vec![0x1234_u16]);
    /// assert_eq!(
    ///     samples.encode(ByteOrder::MostSignificantByteFirst)?,
    ///     [0x12, 0x34]
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn encode(&self, byte_order: ByteOrder) -> Result<Vec<u8>, SampleError> {
        codec::encode(&self.samples, byte_order)
    }

    /// Writes stored samples without creating a separate encoded byte buffer.
    ///
    /// The caller selects output buffering. Use a buffered writer for files to
    /// combine the per-sample writes into larger I/O operations.
    ///
    /// # Errors
    ///
    /// Returns [`SampleError::ScalarCodecRejected`] if the scalar codec rejects
    /// an exact-width sample, or [`SampleError::Io`] if the stream fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_codecs::{ByteOrder, SampleBuffer};
    ///
    /// let samples = SampleBuffer::from_samples(vec![0x1234_u16]);
    /// let mut bytes = Vec::new();
    /// samples.write_to(&mut bytes, ByteOrder::LeastSignificantByteFirst)?;
    /// assert_eq!(bytes, [0x34, 0x12]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write_to<W: Write>(
        &self,
        writer: &mut W,
        byte_order: ByteOrder,
    ) -> Result<(), SampleError> {
        codec::write(&self.samples, byte_order, writer)
    }

    /// Extracts samples only when the requested type matches exactly.
    ///
    /// A mismatch returns [`SampleExtractionError`] containing this buffer,
    /// so a failed type selection does not discard or convert any samples.
    ///
    /// # Errors
    ///
    /// Returns the unchanged buffer when `T` differs from its stored type.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_codecs::{SampleBuffer, SampleType};
    ///
    /// let samples = SampleBuffer::from_samples(vec![1_u32, 2]);
    /// assert_eq!(samples.sample_type(), SampleType::U32);
    /// assert_eq!(samples.try_into_samples::<u32>()?, [1, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn try_into_samples<T: Sample>(self) -> Result<Vec<T>, SampleExtractionError> {
        T::try_from_sample_buffer(self)
    }
}

impl fmt::Debug for SampleBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SampleBuffer")
            .field("sample_type", &self.sample_type())
            .field("len", &self.len())
            .finish()
    }
}
