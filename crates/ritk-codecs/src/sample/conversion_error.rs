//! Failures while converting a stored sample buffer.

use std::fmt;

use super::{SampleBuffer, SampleType};

/// Why a requested sample conversion did not produce a vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SampleConversionFailure {
    /// At least one value might change under the requested type pair.
    Inexact,
    /// The destination vector could not reserve the required capacity.
    Allocation,
}

/// A failed sample conversion that retains the original sample buffer.
pub struct SampleConversionError {
    buffer: SampleBuffer,
    requested: SampleType,
    failure: Failure,
}

#[derive(Debug)]
enum Failure {
    Inexact,
    Allocation(std::collections::TryReserveError),
}

impl SampleConversionError {
    pub(super) fn inexact(buffer: SampleBuffer, requested: SampleType) -> Self {
        Self {
            buffer,
            requested,
            failure: Failure::Inexact,
        }
    }

    pub(super) fn allocation(
        buffer: SampleBuffer,
        requested: SampleType,
        source: std::collections::TryReserveError,
    ) -> Self {
        Self {
            buffer,
            requested,
            failure: Failure::Allocation(source),
        }
    }

    /// The type the samples are stored in.
    #[must_use]
    pub fn stored(&self) -> SampleType {
        self.buffer.sample_type()
    }

    /// The type the samples were requested in.
    #[must_use]
    pub fn requested(&self) -> SampleType {
        self.requested
    }

    /// The reason the conversion failed.
    #[must_use]
    pub fn failure(&self) -> SampleConversionFailure {
        match &self.failure {
            Failure::Inexact => SampleConversionFailure::Inexact,
            Failure::Allocation(_) => SampleConversionFailure::Allocation,
        }
    }

    /// The original buffer, unchanged.
    #[must_use]
    pub fn into_buffer(self) -> SampleBuffer {
        self.buffer
    }
}

impl fmt::Display for SampleConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.failure {
            Failure::Inexact => write!(
                f,
                "{} samples do not all have exact {} values; read them as a type {} widens to, or convert explicitly",
                self.stored(),
                self.requested,
                self.stored()
            ),
            Failure::Allocation(source) => write!(
                f,
                "cannot allocate {} {} samples requested as {}: {source}",
                self.buffer.len(),
                self.stored(),
                self.requested
            ),
        }
    }
}

/// Names the types and count without formatting the sample values.
impl fmt::Debug for SampleConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SampleConversionError")
            .field("stored", &self.stored())
            .field("requested", &self.requested)
            .field("sample_count", &self.buffer.len())
            .field("failure", &self.failure())
            .finish()
    }
}

impl std::error::Error for SampleConversionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.failure {
            Failure::Inexact => None,
            Failure::Allocation(source) => Some(source),
        }
    }
}
