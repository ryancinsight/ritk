#![forbid(unsafe_code)]
#![deny(missing_docs)]

/// Error returned when reading or writing a `.trk` file.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TrkError {
    /// Input exhausted before the expected number of bytes was read.
    #[error("unexpected end of file at byte offset {offset}")]
    UnexpectedEof {
        /// Byte position where the read stopped.
        offset: usize,
    },

    /// The header magic bytes do not match `TRACK`.
    #[error("invalid .trk magic bytes: expected b\"TRACK\", got {got:?}")]
    InvalidMagic {
        /// First 5 bytes that were read.
        got: [u8; 5],
    },

    /// Header size field does not equal 1000.
    #[error("invalid header size {value}; expected 1000")]
    InvalidHeaderSize {
        /// The `hdr_size` value read from the file.
        value: i32,
    },

    /// A streamline declares a negative or unreasonably large point count.
    #[error("invalid point count {count} in streamline {index}")]
    InvalidPointCount {
        /// Streamline index (0-based).
        index: usize,
        /// The `n_points` value read from the file.
        count: i32,
    },

    /// The header declares a negative or unreasonably large streamline count.
    #[error("invalid streamline count {count}")]
    InvalidStreamlineCount {
        /// The `n_count` value read from the header.
        count: i32,
    },

    /// The header declares more per-point scalars than the format can name.
    ///
    /// `n_scalars` indexes the fixed `scalar_name[10][20]` table, so a value
    /// outside `0..=10` describes names the format has no room to store. Left
    /// unchecked it also sets the per-point stride, which turned a 1 KiB header
    /// into a multi-gigabyte reservation.
    #[error("invalid per-point scalar count {count}; the TrackVis scalar-name table holds at most 10")]
    InvalidScalarCount {
        /// The `n_scalars` value read from the header.
        count: i16,
    },

    /// The header declares more per-streamline properties than the format can name.
    ///
    /// Mirrors [`TrkError::InvalidScalarCount`] for the `property_name[10][20]`
    /// table and the per-streamline property stride.
    #[error("invalid per-streamline property count {count}; the TrackVis property-name table holds at most 10")]
    InvalidPropertyCount {
        /// The `n_properties` value read from the header.
        count: i16,
    },

    /// A coordinate component is NaN or infinite.
    #[error("non-finite coordinate in streamline {index}, point {point_index}")]
    NonFiniteCoordinate {
        /// Streamline index.
        index: usize,
        /// Point index within the streamline.
        point_index: usize,
    },

    /// Gaia rejected the point sequence.
    #[error("invalid streamline {index}: {source}")]
    InvalidPolyline {
        /// Streamline index.
        index: usize,
        /// Error from Gaia.
        #[source]
        source: gaia::PolylineError,
    },
}
