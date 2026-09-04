//! Typed failures for MIND-SSC construction and evaluation.

use thiserror::Error;

use crate::classical::RigidPhysicalAffineError;

/// Failure while configuring, preparing, or evaluating MIND-SSC.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MindSscError {
    /// A patch radius is zero, which would not describe a 3-D patch.
    #[error("MIND-SSC patch radius on axis {axis} must be positive, got {value}")]
    InvalidPatchRadius { axis: usize, value: usize },
    /// A six-neighbour offset is zero.
    #[error("MIND-SSC neighbour dilation on axis {axis} must be positive, got {value}")]
    InvalidNeighbourDilation { axis: usize, value: usize },
    /// A radius cannot be represented safely by the integer support kernel.
    #[error("MIND-SSC support on axis {axis} overflows: patch radius {patch_radius}, neighbour dilation {neighbour_dilation}")]
    SupportOverflow {
        axis: usize,
        patch_radius: usize,
        neighbour_dilation: usize,
    },
    /// A deterministic sample budget is empty.
    #[error("MIND-SSC maximum sample count must be positive")]
    EmptySampleBudget,
    /// A caller-provided sample set is empty.
    #[error("MIND-SSC caller-provided sample indices must not be empty")]
    EmptySampleIndices,
    /// Caller-provided sample indices contain a duplicate.
    #[error("MIND-SSC caller-provided sample index {index} is duplicated")]
    DuplicateSampleIndex { index: usize },
    /// An image is not an ordinary Cartesian raster.
    #[error("MIND-SSC supports Cartesian images only; {image} uses {coordinate_map}")]
    NonCartesianImage {
        image: &'static str,
        coordinate_map: String,
    },
    /// An image is too small for complete descriptor support.
    #[error(
        "MIND-SSC fixed image shape {shape:?} cannot contain complete support with halo {halo:?}"
    )]
    ImageTooSmall { shape: [usize; 3], halo: [usize; 3] },
    /// The moving image has an empty axis.
    #[error("MIND-SSC moving image shape {shape:?} contains an empty axis")]
    EmptyMovingImage { shape: [usize; 3] },
    /// A fixed-domain mask has the wrong number of entries.
    #[error("MIND-SSC mask length {actual} does not match fixed voxel count {expected}")]
    MaskLength { expected: usize, actual: usize },
    /// A fixed-domain weight array has the wrong number of entries.
    #[error("MIND-SSC weight length {actual} does not match fixed voxel count {expected}")]
    WeightLength { expected: usize, actual: usize },
    /// No complete-support center survived fixed-domain selection.
    #[error("MIND-SSC fixed domain contains no selected complete-support centers")]
    EmptyFixedDomain,
    /// A caller-provided linear index is outside the fixed volume.
    #[error("MIND-SSC sample index {index} is outside fixed voxel count {voxel_count}")]
    SampleIndexOutOfBounds { index: usize, voxel_count: usize },
    /// A caller-provided center cannot support every SSC patch.
    #[error("MIND-SSC sample index {index} is outside the complete-support interior")]
    SampleOutsideCompleteSupport { index: usize },
    /// A caller-provided center is excluded by the fixed mask.
    #[error("MIND-SSC sample index {index} is excluded by the fixed mask")]
    SampleExcludedByMask { index: usize },
    /// A selected fixed weight is invalid.
    #[error("MIND-SSC weight at fixed index {index} must be finite and non-negative, got {value}")]
    InvalidWeight { index: usize, value: f32 },
    /// Selected weights have no mass.
    #[error("MIND-SSC selected fixed-domain weights sum to zero")]
    ZeroSelectedWeight,
    /// Finite selected weights overflowed while their sum was accumulated.
    #[error("MIND-SSC selected fixed-domain weight sum overflowed to {value}")]
    NonFiniteSelectedWeightSum { value: f32 },
    /// A finite selected-weight sum cannot form the 60-bit loss denominator.
    #[error("MIND-SSC denominator overflowed for selected weight sum {weight_sum}")]
    NonFiniteDenominator { weight_sum: f32 },
    /// Host sampling requires contiguous image storage.
    #[error("MIND-SSC {image} image must have contiguous row-major storage: {reason}")]
    NonContiguousImage { image: &'static str, reason: String },
    /// Physical image metadata is non-finite or non-positive.
    #[error("MIND-SSC {image} image has invalid {field} metadata at element {index}: {value}")]
    InvalidGeometry {
        image: &'static str,
        field: &'static str,
        index: usize,
        value: f64,
    },
    /// A direction-cosine matrix cannot be inverted.
    #[error("MIND-SSC {image} direction-cosine matrix is singular")]
    SingularDirection { image: &'static str },
    /// An image sample used by a selected descriptor is non-finite.
    #[error("MIND-SSC {image} sample at linear index {index} is non-finite: {value}")]
    NonFiniteImageSample {
        image: &'static str,
        index: usize,
        value: f32,
    },
    /// The affine contains a non-finite coefficient.
    #[error("MIND-SSC affine {field} element {index} is non-finite: {value}")]
    NonFiniteTransform {
        field: &'static str,
        index: usize,
        value: f32,
    },
    /// The classical physical affine cannot be represented natively.
    #[error("MIND-SSC physical affine conversion failed: {0}")]
    NativeAffine(#[from] RigidPhysicalAffineError),
    /// Transform arithmetic produced a non-finite moving coordinate.
    #[error("MIND-SSC transformed moving index on axis {axis} is non-finite: {value}")]
    NonFiniteMovingCoordinate { axis: usize, value: f32 },
    /// Descriptor arithmetic overflowed for a selected center.
    #[error("MIND-SSC patch-distance arithmetic produced a non-finite value")]
    NonFinitePatchDistance,
    /// Shape or index arithmetic overflowed.
    #[error("MIND-SSC index arithmetic overflowed")]
    IndexOverflow,
}
