//! Typed failures from DICOM image writers.

/// An invalid image or metadata value prevented a DICOM image write.
///
/// Pixel-data attributes follow DICOM PS3.5 section 8.1.1: the allocated width,
/// stored width, high bit, signedness, and serialized sample representation must
/// describe the same pixel bytes. Writer APIs retain their `anyhow::Result`
/// signatures; callers can recover this cause with `downcast_ref`.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum DicomWriteError {
    /// A frame count, row count, or column count is zero.
    #[error("DICOM image dimensions must be non-zero (depth={depth} rows={rows} cols={columns})")]
    InvalidDimensions {
        /// Number of frames or slices.
        depth: usize,
        /// Number of rows in a frame.
        rows: usize,
        /// Number of columns in a frame.
        columns: usize,
    },
    /// The product of dimensions exceeded the addressable sample count.
    #[error("DICOM pixel count overflows the addressable sample range")]
    PixelCountOverflow,
    /// The buffer length differs from the dimensions' sample count.
    #[error("DICOM pixel buffer has {actual} samples; dimensions require {expected}")]
    PixelCountMismatch {
        /// Required sample count.
        expected: usize,
        /// Actual sample count.
        actual: usize,
    },
    /// Rows cannot be represented by the DICOM US Rows attribute.
    #[error("DICOM Rows value {rows} exceeds the unsigned 16-bit range")]
    RowsOutOfRange {
        /// Actual row count.
        rows: usize,
    },
    /// Columns cannot be represented by the DICOM US Columns attribute.
    #[error("DICOM Columns value {columns} exceeds the unsigned 16-bit range")]
    ColumnsOutOfRange {
        /// Actual column count.
        columns: usize,
    },
    /// The input pixel buffer contains NaN or infinity.
    #[error("DICOM pixel at linear index {index} is not finite")]
    NonFinitePixel {
        /// Linear index of the invalid sample.
        index: usize,
    },
    /// A finite pixel range cannot be represented by the writer's scale.
    #[error("DICOM pixel range cannot be represented by a finite rescale")]
    PixelRangeOutOfRange,
    /// An encoded sample did not fit its selected unsigned sample type.
    #[error("normalized DICOM pixel at linear index {index} is outside its sample range")]
    EncodedPixelOutOfRange {
        /// Linear index of the sample that could not be encoded.
        index: usize,
    },
    /// Allocation for encoded pixel data or its per-frame plans failed.
    #[error("cannot reserve memory for DICOM pixel encoding")]
    PixelAllocationFailed,
    /// Declared pixel width contradicts the chosen encoding.
    #[error("DICOM pixel width {declared} contradicts encoded width {encoded}")]
    PixelDescriptionMismatch {
        /// Caller-supplied width.
        declared: u16,
        /// Width used by the encoded samples.
        encoded: u16,
    },
    /// Per-frame metadata does not cover every frame.
    #[error("DICOM per-frame metadata has {actual} entries; n_frames is {expected}")]
    FrameMetadataCountMismatch {
        /// Required frame count.
        expected: usize,
        /// Actual per-frame entry count.
        actual: usize,
    },
    /// A BINARY segmentation sample is neither zero nor one.
    #[error("DICOM BINARY sample at linear index {index} is outside 0..=1")]
    InvalidBinaryPixel {
        /// Linear sample index.
        index: usize,
    },
    /// Only some source BitsAllocated, BitsStored, and HighBit values were set.
    #[error("source DICOM pixel description must provide all three bit attributes or none")]
    IncompleteSourcePixelDescription,
    /// Source pixel bit attributes do not satisfy the DICOM pixel-module rules.
    #[error(
        "invalid source pixel description: BitsAllocated={bits_allocated}, BitsStored={bits_stored}, HighBit={high_bit}"
    )]
    InvalidSourcePixelDescription {
        /// Source BitsAllocated value.
        bits_allocated: u16,
        /// Source BitsStored value.
        bits_stored: u16,
        /// Source HighBit value.
        high_bit: u16,
    },
    /// The scalar writer cannot preserve the source photometric interpretation.
    #[error("scalar DICOM output supports MONOCHROME1 and MONOCHROME2 only")]
    UnsupportedPhotometricInterpretation,
    /// Spatial metadata is non-finite, has non-positive spacing, or invalid direction cosines.
    #[error(
        "DICOM spatial metadata requires finite values, positive spacing, and orthonormal axes"
    )]
    InvalidSpatialMetadata,
    /// A decimal-string attribute cannot represent a non-finite value.
    #[error("DICOM Decimal String requires a finite value")]
    NonFiniteDecimalStringValue,
    /// A finite value has no legal decimal-string representation within 16 bytes.
    #[error("DICOM Decimal String value cannot fit the 16-byte component limit")]
    DecimalStringValueOutOfRange,
}
