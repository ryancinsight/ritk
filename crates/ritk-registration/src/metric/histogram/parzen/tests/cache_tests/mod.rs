//! Cache tests for Parzen joint histogram cache dispatch.
//!
//! Split from the monolithic cache_tests.rs for structural compliance.

#![allow(clippy::needless_range_loop)]

mod fingerprint;
mod integration;
mod lazy;
mod parallel;
mod property;
