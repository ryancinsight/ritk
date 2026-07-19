//! Structuring element types for mathematical morphology.
//!
//! This crate provides the [`StructuringElement`] value type and Zero-Sized
//! Type shape markers ([`Cube`], [`Cross`], [`Ball`]) for mathematical
//! morphology. All shapes implement the sealed [`SeShape`] trait, enabling
//! compile-time monomorphization and zero-cost static dispatch.
//!
//! # Key design properties
//!
//! - **Zero-Sized Types**: [`Cube`], [`Cross`], [`Ball`] are all ZSTs
//!   (`size_of == 0`). Shape selection is resolved at compile time.
//! - **Sealed trait**: [`SeShape`] cannot be implemented outside this crate.
//! - **Const cardinality**: [`cube_cardinality`], [`cross_cardinality`], and
//!   [`ball_cardinality_upper`] are `const fn` evaluators for compile-time
//!   pre-sizing.
//! - **Zero-copy**: [`StructuringElement::offsets`] returns a borrowed
//!   `&[Offset3D]` slice — no allocation at call sites.
//! - **`#[repr(transparent)]`**: [`Offset3D`] is ABI-compatible with `[i32; 3]`.
//!
//! # Submodules
//!
//! - [`offset`]: `Offset3D` — a `#[repr(transparent)]` newtype over `[i32; 3]`.
//! - [`shape_markers`]: `Cube` / `Cross` / `Ball` ZSTs and the sealed
//!   `SeShape` trait, plus `const fn` cardinality evaluators.
//! - [`structuring_element`]: the `StructuringElement` value type that
//!   stores a list of `Offset3D` plus the half-width that produced them.

pub mod offset;
pub mod shape_markers;
pub mod structuring_element;

pub use offset::Offset3D;
pub use shape_markers::{
    ball_cardinality_upper, cross_cardinality, cube_cardinality, sealed, Ball, Cross, Cube, SeShape,
};
pub use structuring_element::StructuringElement;
