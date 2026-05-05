//! Geometric image transform filters.
//!
//! # Filters
//!
//! - [`FlipImageFilter`] — reverses voxel ordering along any combination of axes
//!   (ITK `FlipImageFilter`, ImageJ Image > Transform)
//! - [`ShrinkImageFilter`] — integer downsampling by tile-averaging
//! - [`ConstantPadImageFilter`], [`MirrorPadImageFilter`], [`WrapPadImageFilter`] — padding

pub mod flip;
pub mod permute_axes;
pub mod paste;
pub mod roi;
pub mod shrink;
pub mod pad;

pub use flip::FlipImageFilter;
pub use permute_axes::PermuteAxesImageFilter;
pub use paste::PasteImageFilter;
pub use roi::RegionOfInterestImageFilter;
pub use shrink::ShrinkImageFilter;
pub use pad::{ConstantPadImageFilter, MirrorPadImageFilter, WrapPadImageFilter};
