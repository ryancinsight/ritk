//! CLI error types.
//!
//! `CliError` is the leaf error type for argument validation and feature-availability
//! checks.  All fallible command handlers return `anyhow::Result<()>`; `CliError`
//! is constructed explicitly where a specific variant adds clarity.

use thiserror::Error;

/// Top-level error variants for the RITK CLI.
///
/// Defined as the public API contract for the CLI error hierarchy.
/// Command handlers return `anyhow::Result<()>` for ergonomics; `CliError`
/// is the leaf type for argument-validation and feature-availability checks.
#[allow(dead_code)]
#[derive(Error, Debug)]
pub enum CliError {
    /// Wraps a standard IO error (file not found, permission denied, …).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Wraps an `anyhow` error originating from image read/write operations.
    #[error("Image read error: {0}")]
    Read(String),

    /// Signals that a required CLI argument is absent, malformed, or out of range.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Signals that a requested filter is not yet compiled into this build.
    #[error("Filter not available: {0}")]
    FilterNotAvailable(String),
}

#[allow(dead_code)]
impl CliError {
    /// Construct a `Read` variant from any displayable error.
    pub fn read<E: std::fmt::Display>(e: E) -> Self {
        Self::Read(e.to_string())
    }

    /// Construct an `InvalidArgument` variant.
    pub fn invalid_argument<S: Into<String>>(msg: S) -> Self {
        Self::InvalidArgument(msg.into())
    }

    /// Construct a `FilterNotAvailable` variant.
    pub fn filter_not_available<S: Into<String>>(name: S) -> Self {
        Self::FilterNotAvailable(name.into())
    }
}
