//! Typed failures for block-matching input and derived geometry.

use thiserror::Error;

/// Failure raised while validating image dimensions or buffer ownership.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BlockMatchingError {
    /// A three-dimensional voxel count cannot be represented by `usize`.
    #[error("{label} dimensions {dims:?} overflow")]
    VoxelCountOverflow {
        /// Operation whose dimensions overflowed.
        label: &'static str,
        /// Dimensions supplied to the operation.
        dims: [usize; 3],
    },
    /// A three-dimensional buffer exceeds the platform allocation byte limit.
    #[error("{label} buffer with dimensions {dims:?} exceeds the allocation byte limit")]
    ByteCountOverflow {
        /// Operation whose buffer exceeds the allocation limit.
        label: &'static str,
        /// Dimensions supplied to the operation.
        dims: [usize; 3],
        /// Size of one buffer element in bytes.
        element_size: usize,
    },
    /// A window extent `2 * radius + 1` cannot be represented by `usize`.
    #[error("{label} extent overflows on axis {axis}")]
    WindowExtentOverflow {
        /// Operation whose window overflowed.
        label: &'static str,
        /// Axis whose extent overflowed.
        axis: usize,
        /// Radius supplied on the overflowing axis.
        radius: usize,
    },
    /// A min/max pyramid cannot double its axial extent.
    #[error("min/max pyramid axial extent overflows")]
    AxialPairExtentOverflow {
        /// Coarse dimensions before storing the min/max pair.
        base_dims: [usize; 3],
    },
    /// An FFT convolution extent cannot be represented by `usize`.
    #[error("FFT convolution extent overflows on axis {axis}: {roi} + {block} - 1")]
    FftConvolutionExtentOverflow {
        /// Axis whose convolution extent overflowed.
        axis: usize,
        /// Moving ROI extent on the overflowing axis.
        roi: usize,
        /// Fixed block extent on the overflowing axis.
        block: usize,
    },
    /// An FFT moving ROI reach cannot be represented by `usize`.
    #[error("FFT block/search reach overflows on axis {axis}: {block_radius} + {search_radius}")]
    FftReachExtentOverflow {
        /// Axis whose reach overflowed.
        axis: usize,
        /// Block radius on the overflowing axis.
        block_radius: usize,
        /// Search radius on the overflowing axis.
        search_radius: usize,
    },
    /// An FFT padded extent cannot be rounded to the next power of two.
    #[error("FFT padded extent overflows on axis {axis}: {extent}")]
    FftPaddingExtentOverflow {
        /// Axis whose padded extent overflowed.
        axis: usize,
        /// Convolution extent that cannot be padded.
        extent: usize,
    },
    /// The fixed and moving buffers do not contain one sample per voxel.
    #[error(
        "fixed ({fixed}) and moving ({moving}) buffers must both hold {expected} voxels for dims {dims:?}"
    )]
    BufferLengthMismatch {
        /// Number of fixed samples supplied.
        fixed: usize,
        /// Number of moving samples supplied.
        moving: usize,
        /// Required voxel count.
        expected: usize,
        /// Image dimensions used to derive `expected`.
        dims: [usize; 3],
    },
}
