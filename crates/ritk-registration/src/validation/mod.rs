//! Structural and numerical algorithm evaluations validating registration spaces safely.

pub mod config;
pub mod numerical;
pub mod shape;

pub use config::ValidationConfig;
pub use numerical::*;
pub use shape::*;
