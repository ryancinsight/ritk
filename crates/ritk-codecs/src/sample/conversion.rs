//! Explicit conversions between stored sample types.

use eunomia::CastFrom;

use super::{Sample, SampleType};

/// The representation changes found by an explicitly lossy conversion.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConversionReport {
    /// Type in which the input buffer was stored.
    pub source_type: SampleType,
    /// Type requested by the caller.
    pub target_type: SampleType,
    /// Number of samples whose numeric value or floating-point representation
    /// changed during conversion.
    pub changed_samples: usize,
    /// First index whose source representation changed, if one changed.
    pub first_changed_sample: Option<usize>,
}

/// Converted samples together with the representation-change report.
#[must_use = "inspect the conversion report before using converted samples"]
#[derive(Debug, Clone, PartialEq)]
pub struct ConvertedSamples<T> {
    samples: Vec<T>,
    report: ConversionReport,
}

impl<T> ConvertedSamples<T> {
    /// The conversion report.
    #[must_use]
    pub const fn report(&self) -> ConversionReport {
        self.report
    }

    /// Consume the conversion and return its samples and report.
    #[must_use]
    pub fn into_parts(self) -> (Vec<T>, ConversionReport) {
        (self.samples, self.report)
    }
}

/// An exact sample conversion failed or its output could not be allocated.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum SampleConversionError {
    /// Conversion changed one or more source values or representations.
    #[error(
        "conversion from {source_type} to {target_type} changes sample {first_changed_sample} ({changed_samples} samples total)"
    )]
    Changed {
        /// Type in which the input buffer was stored.
        source_type: SampleType,
        /// Type requested by the caller.
        target_type: SampleType,
        /// First sample index whose source representation changed.
        first_changed_sample: usize,
        /// Number of samples whose source representation changed.
        changed_samples: usize,
    },
    /// Reserving memory for converted samples failed.
    #[error("cannot reserve memory for {sample_count} samples converting {source_type} to {target_type}")]
    Allocation {
        /// Type in which the input buffer was stored.
        source_type: SampleType,
        /// Type requested by the caller.
        target_type: SampleType,
        /// Number of output samples requested.
        sample_count: usize,
        /// The allocator's capacity or allocation error.
        #[source]
        source: std::collections::TryReserveError,
    },
}

pub(crate) fn convert<S, T>(samples: Vec<S>) -> Result<ConvertedSamples<T>, SampleConversionError>
where
    S: Sample + CastFrom<T>,
    T: Sample + CastFrom<S>,
{
    let source_type = S::TYPE;
    let sample_count = samples.len();
    let mut converted = Vec::new();
    converted
        .try_reserve_exact(sample_count)
        .map_err(|source| SampleConversionError::Allocation {
            source_type,
            target_type: T::TYPE,
            sample_count,
            source,
        })?;
    let mut changed_samples = 0_usize;
    let mut first_changed_sample = None;

    for (index, source) in samples.into_iter().enumerate() {
        let target = <T as CastFrom<S>>::cast_from(source);
        let changed = if S::TYPE.is_float() && T::TYPE.is_float() {
            let round_trip = <S as CastFrom<T>>::cast_from(target);
            !source.same_representation(round_trip)
        } else if S::TYPE.is_float() {
            let round_trip = <S as CastFrom<T>>::cast_from(target);
            source.exact_integer_value() != target.exact_integer_value()
                || !source.same_representation(round_trip)
        } else {
            source.exact_integer_value() != target.exact_integer_value()
        };
        if changed {
            changed_samples = changed_samples
                .checked_add(1)
                .expect("invariant: a buffer cannot contain more than usize::MAX samples");
            first_changed_sample.get_or_insert(index);
        }
        converted.push(target);
    }

    Ok(ConvertedSamples {
        samples: converted,
        report: ConversionReport {
            source_type,
            target_type: T::TYPE,
            changed_samples,
            first_changed_sample,
        },
    })
}

pub(crate) fn exact_error(report: ConversionReport) -> Option<SampleConversionError> {
    Some(SampleConversionError::Changed {
        source_type: report.source_type,
        target_type: report.target_type,
        first_changed_sample: report.first_changed_sample?,
        changed_samples: report.changed_samples,
    })
}
