//! Interpolation module.
//!
//! The `Interpolator` trait lives here. All concrete interpolator
//! implementations are in the `ritk-interpolation` crate.
//!
//! This module is intentionally minimal to avoid circular dependencies:
//! `ritk-interpolation` depends on `ritk-core`, and several `ritk-core`
//! modules (e.g. `transform`) only need the trait, not the concrete types.

pub mod trait_;

pub use trait_::Interpolator;
