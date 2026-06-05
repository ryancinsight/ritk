//! Structuring element types and operations.
//!
//! This module provides the [`StructuringElement`] value type and Zero-Sized
//! Type shape markers ([`Cube`], [`Cross`], [`Ball`]) for mathematical
//! morphology. All shapes implement the sealed [`SeShape`] trait, enabling
//! compile-time monomorphization and zero-cost static dispatch.
//!
//! # Submodules
//!
//! - [`offset`]: `Offset3D` — a `#[repr(transparent)]` newtype over `[i32; 3]`
//!   for individual voxel offsets.
//! - [`shape_markers`]: `Cube` / `Cross` / `Ball` ZSTs and the sealed
//!   `SeShape` trait, plus `const fn` cardinality evaluators.
//! - [`structuring_element`]: the `StructuringElement` value type that
//!   stores a list of `Offset3D` plus the half-width that produced them.
//!
//! # See also
//!
//! - [`crate::filter::rank`] — rank/percentile filters that consume
//!   [`StructuringElement`] for their footprint.

pub mod offset;
pub mod shape_markers;
pub mod structuring_element;

pub use offset::Offset3D;
pub use shape_markers::{
    ball_cardinality_upper, cross_cardinality, cube_cardinality, sealed, Ball, Cross, Cube, SeShape,
};
pub use structuring_element::StructuringElement;
