use thiserror::Error;

use crate::sample::{Sample, SampleBuffer, SampleError, SampleType};
use crate::ByteOrder;

use super::{PixelLayout, PixelSignedness};

/// Failure to decode stored scalar values from one DICOM pixel frame.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StoredPixelError {
    /// The pixel layout is invalid or exceeds the supported sample widths.
    #[error("invalid pixel layout: {0:?}")]
    InvalidLayout(PixelLayout),
    /// The frame byte length differs from the dimensions and allocated width.
    #[error("pixel frame has {actual} bytes; expected {expected}")]
    ByteLengthMismatch {
        /// Number of bytes supplied for the frame.
        actual: usize,
        /// Number of bytes required by the layout.
        expected: usize,
    },
    /// The scalar decoder received color or multi-component pixels.
    #[error("stored scalar decoding requires SamplesPerPixel=1, got {actual}")]
    UnsupportedSamplesPerPixel {
        /// Number of samples declared for each pixel.
        actual: usize,
    },
    /// The low-level sample decoder rejected the payload or could not allocate storage.
    #[error(transparent)]
    SampleDecode(#[from] SampleError),
}

/// Decode one scalar DICOM pixel frame into its fixed-width stored values.
///
/// The frame is decoded in the byte order of its transfer syntax. The returned
/// [`SampleBuffer`] retains integer precision and signedness and excludes the
/// modality transform in `PixelLayout`. For BitsAllocated=24, samples occupy
/// the next wider fixed-width representation because [`SampleType`] has no
/// 24-bit integer variant.
///
/// This decoder handles the low-order BitsStored arrangement (`HighBit =
/// BitsStored - 1`) from DICOM PS3.5 Section 8.1.1, which leaves unused
/// allocated bits unspecified; it masks them before signed interpretation.
/// See the [DICOM pixel data encoding rules](https://dicom.nema.org/medical/DICOM/current/output/chtml/part05/chapter_8.html).
/// Pass exactly one frame, without trailing Value Field padding.
///
/// # Errors
///
/// Returns [`StoredPixelError`] when the layout is invalid, its byte length
/// differs from one frame, it declares multiple samples per pixel, decoding
/// fails, or sample storage cannot be allocated.
///
/// # Examples
///
/// ```
/// use ritk_codecs::{
///     decode_stored_pixel_frame, ByteOrder, PixelLayout, PixelSignedness,
/// };
///
/// let layout = PixelLayout {
///     rows: 1,
///     cols: 2,
///     samples_per_pixel: 1,
///     bits_allocated: 16,
///     bits_stored: 12,
///     pixel_representation: PixelSignedness::Signed,
///     rescale_slope: 2.0,
///     rescale_intercept: 5.0,
/// };
/// let frame = [0xff, 0x0f, 0x00, 0x08];
/// let samples = decode_stored_pixel_frame(
///     &frame,
///     layout,
///     ByteOrder::LeastSignificantByteFirst,
/// )?;
/// assert_eq!(samples.try_into_samples::<i16>()?, [-1, -2048]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn decode_stored_pixel_frame(
    bytes: &[u8],
    layout: PixelLayout,
    byte_order: ByteOrder,
) -> Result<SampleBuffer, StoredPixelError> {
    if layout.samples_per_pixel != 1 {
        return Err(StoredPixelError::UnsupportedSamplesPerPixel {
            actual: layout.samples_per_pixel,
        });
    }
    let expected_bytes = layout
        .bytes_per_frame()
        .map_err(|_| StoredPixelError::InvalidLayout(layout))?;
    if bytes.len() != expected_bytes {
        return Err(StoredPixelError::ByteLengthMismatch {
            actual: bytes.len(),
            expected: expected_bytes,
        });
    }

    match (layout.bits_allocated, layout.pixel_representation) {
        (8, PixelSignedness::Unsigned) => {
            decode_typed::<u8>(bytes, layout, SampleType::U8, byte_order)
        }
        (8, PixelSignedness::Signed) => {
            decode_typed::<i8>(bytes, layout, SampleType::I8, byte_order)
        }
        (16, PixelSignedness::Unsigned) => {
            decode_typed::<u16>(bytes, layout, SampleType::U16, byte_order)
        }
        (16, PixelSignedness::Signed) => {
            decode_typed::<i16>(bytes, layout, SampleType::I16, byte_order)
        }
        (24, PixelSignedness::Unsigned) => {
            decode_typed::<u32>(bytes, layout, SampleType::U32, byte_order)
        }
        (24, PixelSignedness::Signed) => {
            decode_typed::<i32>(bytes, layout, SampleType::I32, byte_order)
        }
        (32, PixelSignedness::Unsigned) => {
            decode_typed::<u32>(bytes, layout, SampleType::U32, byte_order)
        }
        (32, PixelSignedness::Signed) => {
            decode_typed::<i32>(bytes, layout, SampleType::I32, byte_order)
        }
        _ => Err(StoredPixelError::InvalidLayout(layout)),
    }
}

fn decode_typed<T>(
    bytes: &[u8],
    layout: PixelLayout,
    sample_type: SampleType,
    byte_order: ByteOrder,
) -> Result<SampleBuffer, StoredPixelError>
where
    T: Sample + Into<i64> + TryFrom<i64>,
{
    let mut samples = if layout.bits_allocated == 24 {
        decode_24_bit_samples::<T>(bytes, layout, byte_order)?
    } else {
        match SampleBuffer::decode(sample_type, bytes, byte_order)?.try_into_samples::<T>() {
            Ok(samples) => samples,
            Err(_) => unreachable!("invariant: sample buffer retains its requested type"),
        }
    };

    if layout.bits_stored == layout.bits_allocated
        && (layout.bits_allocated != 24 || layout.pixel_representation == PixelSignedness::Unsigned)
    {
        return Ok(SampleBuffer::from_samples(samples));
    }

    normalize_stored_bits::<T>(
        &mut samples,
        layout.bits_stored,
        layout.pixel_representation,
    );
    Ok(SampleBuffer::from_samples(samples))
}

fn decode_24_bit_samples<T>(
    bytes: &[u8],
    layout: PixelLayout,
    byte_order: ByteOrder,
) -> Result<Vec<T>, StoredPixelError>
where
    T: Sample + TryFrom<i64>,
{
    let sample_count = layout
        .pixels_per_frame()
        .map_err(|_| StoredPixelError::InvalidLayout(layout))?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(sample_count)
        .map_err(|source| StoredPixelError::SampleDecode(SampleError::Allocation(source)))?;

    for bytes in bytes.chunks_exact(3) {
        let raw = i64::from(super::u24(bytes, byte_order));
        samples.push(T::try_from(raw).unwrap_or_else(|_| {
            unreachable!("invariant: 24-bit value fits its selected 32-bit sample")
        }));
    }
    Ok(samples)
}

fn normalize_stored_bits<T>(samples: &mut [T], bits_stored: u16, signedness: PixelSignedness)
where
    T: Sample + Into<i64> + TryFrom<i64>,
{
    let modulus = 1_i64 << u32::from(bits_stored);
    let sign_bit = modulus / 2;

    for sample in samples {
        let low_bits = (*sample).into().rem_euclid(modulus);
        let value = match signedness {
            PixelSignedness::Unsigned => low_bits,
            PixelSignedness::Signed if low_bits >= sign_bit => low_bits - modulus,
            PixelSignedness::Signed => low_bits,
        };
        *sample = T::try_from(value).unwrap_or_else(|_| {
            unreachable!("invariant: normalized BitsStored fits its allocated sample type")
        });
    }
}
