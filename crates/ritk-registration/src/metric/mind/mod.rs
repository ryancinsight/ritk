//! Modality-independent neighbourhood descriptor with self-similarity context.
//!
//! This module implements Heinrich et al.'s 12-component six-neighbour SSC
//! descriptor and 60-bit unary packing for bounded rigid CT/MR registration.
//! Fixed descriptors are prepared only at a pose-invariant selected domain;
//! moving descriptors are evaluated on demand in the fixed physical frame.

mod config;
mod descriptor;
mod error;
mod geometry;
mod native;
mod sampling;

pub use config::{MindSscConfig, MindSscSampling};
pub use error::MindSscError;
pub use native::{mind_ssc_value, MindSscFixedPrep, MindSscMemoryUsage};

#[cfg(test)]
mod tests;
