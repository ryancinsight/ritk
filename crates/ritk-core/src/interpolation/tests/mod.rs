//! Consolidated interpolation test module (Sprint 353).
//!
//! Each kernel's tests live in its own sub-module file under `tests/`,
//! included from here so `cargo test -p ritk-core --lib interpolation::tests`
//! runs them all as a single suite.

mod linear;
mod sinc;
