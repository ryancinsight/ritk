use thiserror::Error;

/// A failure while parsing or constructing a NIfTI header.
#[derive(Debug, Error)]
#[error(transparent)]
pub struct NiftiHeaderError(#[from] anyhow::Error);
