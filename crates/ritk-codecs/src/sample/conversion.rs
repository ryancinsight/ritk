//! Policies for converting stored samples to a requested sample type.

use super::{Sample, SampleBuffer, SampleConversionError, SampleType};

/// How stored samples become the requested sample type.
///
/// A reader takes the policy as a zero-sized value, so each policy is
/// monomorphized into the reader with no runtime dispatch, and the caller that
/// owns the precision decision chooses it.
pub trait Conversion: Copy {
    /// Convert `samples` to `T`.
    ///
    /// # Errors
    ///
    /// Returns [`SampleConversionError`], which owns the untouched samples,
    /// when the policy refuses the conversion or destination allocation fails.
    fn convert<T: Sample>(self, samples: SampleBuffer) -> Result<Vec<T>, SampleConversionError>;

    /// Describe whether this policy accepts the stored type and may change values.
    fn report<T: Sample>(self, samples: &SampleBuffer) -> ConversionReport;
}

/// Whether this type pair admits a sample value that conversion can change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValueChange {
    /// The stored type widens to the requested type, so every value survives.
    None,
    /// The type pair admits rounding, truncation, wrapping, saturation, or NaN mapping.
    Possible,
}

/// Whether the conversion policy applies or refuses the requested type pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConversionDisposition {
    /// The policy permits the requested conversion.
    Applied,
    /// The exact policy refuses a type pair that can change values.
    Refused,
}

/// Type-level value-preservation report for a sample conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConversionReport {
    stored: SampleType,
    requested: SampleType,
    sample_count: usize,
    value_change: ValueChange,
    disposition: ConversionDisposition,
}

impl ConversionReport {
    fn new(
        stored: SampleType,
        requested: SampleType,
        sample_count: usize,
        disposition: ConversionDisposition,
    ) -> Self {
        let value_change = if stored.widens_to(requested) {
            ValueChange::None
        } else {
            ValueChange::Possible
        };
        Self {
            stored,
            requested,
            sample_count,
            value_change,
            disposition,
        }
    }

    /// The type named by the input format.
    #[must_use]
    pub fn stored(&self) -> SampleType {
        self.stored
    }

    /// The type requested by the caller.
    #[must_use]
    pub fn requested(&self) -> SampleType {
        self.requested
    }

    /// Number of samples covered by this report.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Whether the source and target types admit a value change.
    #[must_use]
    pub fn value_change(&self) -> ValueChange {
        self.value_change
    }

    /// Policy outcome for this type pair.
    #[must_use]
    pub fn disposition(&self) -> ConversionDisposition {
        self.disposition
    }
}

/// Refuse every conversion that could change a value: accept the stored type
/// or a type it widens to ([`SampleBuffer::into_vec`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Exact;

/// Convert by Rust's primitive numeric conversion rules
/// ([`SampleBuffer::cast_into_vec`]), logging a warning when the stored type
/// does not widen to the requested one, since some values may then round,
/// truncate, wrap, or saturate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cast;

impl Conversion for Exact {
    fn convert<T: Sample>(self, samples: SampleBuffer) -> Result<Vec<T>, SampleConversionError> {
        samples.into_vec()
    }

    fn report<T: Sample>(self, samples: &SampleBuffer) -> ConversionReport {
        let stored = samples.sample_type();
        let disposition = if stored.widens_to(T::TYPE) {
            ConversionDisposition::Applied
        } else {
            ConversionDisposition::Refused
        };
        ConversionReport::new(stored, T::TYPE, samples.len(), disposition)
    }
}

impl Conversion for Cast {
    fn convert<T: Sample>(self, samples: SampleBuffer) -> Result<Vec<T>, SampleConversionError> {
        let stored = samples.sample_type();
        if !stored.widens_to(T::TYPE) {
            tracing::warn!(
                %stored,
                requested = %T::TYPE,
                samples = samples.len(),
                "samples cast to a type that cannot hold every stored value"
            );
        }
        samples.cast_into_vec()
    }

    fn report<T: Sample>(self, samples: &SampleBuffer) -> ConversionReport {
        ConversionReport::new(
            samples.sample_type(),
            T::TYPE,
            samples.len(),
            ConversionDisposition::Applied,
        )
    }
}
