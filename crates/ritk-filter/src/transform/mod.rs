//! Geometric image transform filters.
//!
//! # Filters
//!
//! - [`FlipImageFilter`] — reverses voxel ordering along any combination of axes
//!   (ITK `FlipImageFilter`, ImageJ Image > Transform)
//! - [`ShrinkImageFilter`] — integer downsampling by subsampling (ITK `Shrink`)
//! - [`TileMeanShrinkFilter`] — integer downsampling by tile-averaging (display)
//! - [`ConstantPadImageFilter`], [`MirrorPadImageFilter`], [`WrapPadImageFilter`] — padding

pub mod cyclic_shift;
pub mod expand;
pub mod flip;
pub mod pad;
pub mod paste;
pub mod permute_axes;
pub mod roi;
pub mod shrink;

pub use cyclic_shift::CyclicShiftImageFilter;
pub use expand::ExpandImageFilter;
pub use flip::{FlipImageFilter, FlipPolicy};
pub use pad::{
    ConstantPadImageFilter, MirrorPadImageFilter, Padding, WrapPadImageFilter,
    ZeroFluxNeumannPadImageFilter,
};
pub use paste::PasteImageFilter;
pub use permute_axes::PermuteAxesImageFilter;
pub use roi::RegionOfInterestImageFilter;
pub use shrink::{ShrinkImageFilter, TileMeanShrinkFilter};
