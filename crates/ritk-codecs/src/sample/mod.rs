//! Typed voxel samples for volume formats.
//!
//! A volume file stores its samples as one fixed-width numeric type named in
//! its header. This module is the sample vocabulary the RITK format readers
//! converge on (ADR 0053), so a reader keeps the stored type instead of
//! converting every file to one working type:
//!
//! - [`SampleType`] — the runtime descriptor a header parses into.
//! - [`Sample`] — the compile-time counterpart, implemented for the ten stored
//!   primitives, with [`Sample::TYPE`] linking the two.
//! - [`SampleBuffer`] — an owned vector in the stored type, selected at run
//!   time; [`SampleBuffer::decode`] fills it from packed bytes through
//!   consus-core's bulk decoder, and [`SampleBuffer::read_from`] from a byte
//!   stream in bounded steps. [`SampleBuffer::into_vec`] hands the samples
//!   out exactly — the stored vector itself, or a lossless widening — and
//!   [`SampleBuffer::cast_into_vec`] is the explicit lossy conversion.
//! - [`Conversion`] — the policy a reader applies, chosen by its caller:
//!   [`Exact`] refuses any conversion that could change a value, [`Cast`]
//!   casts and warns when it might. [`Conversion::report`] returns a
//!   value-semantic [`ConversionReport`] describing the stored/requested types,
//!   sample count, possible value change, and policy outcome.
//! - [`Rescale`] — the linear map from stored to physical values some headers
//!   carry beside the samples, applied only in floating-point arithmetic.
//! - [`write_samples`] — the encoding counterpart of
//!   [`SampleBuffer::decode`], streaming in bounded blocks.
//!
//! Integer samples convert from exact signed or unsigned integer carriers;
//! integer-to-float conversion rounds once at the requested precision. A
//! 64-bit integer therefore stays exact until the caller explicitly requests
//! a floating-point type. Whether a potentially lossy conversion is
//! acceptable is the caller's policy; [`SampleType::widens_to`] describes
//! pairs whose complete numeric range is representable by the target.

mod buffer;
mod conversion;
mod encode;
mod error;
mod kind;
mod rescale;
mod stream;

pub use buffer::SampleBuffer;
pub use conversion::{
    Cast, Conversion, ConversionDisposition, ConversionReport, Exact, Sample,
    SampleConversionError, SampleConversionFailure, ValueChange,
};
pub use encode::write_samples;
pub use error::SampleError;
pub use kind::SampleType;
pub use rescale::Rescale;
pub use stream::{count_payload_bytes, validate_remaining_payload};

#[cfg(test)]
mod tests;
