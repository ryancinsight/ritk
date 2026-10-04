//! Typed voxel samples for RITK format adapters.
//!
//! A volume file stores its samples as one fixed-width numeric type named in
//! its header. This module is the single place where packed bytes become typed
//! samples, so readers keep the stored type instead of converting every file
//! to one working type:
//!
//! - [`SampleType`] — the runtime descriptor a header parses into.
//! - [`Sample`] — the compile-time counterpart, implemented for the ten stored
//!   primitives, with [`Sample::TYPE`] linking the two.
//! - [`SampleBuffer`] — an owned vector in the stored type, selected at run
//!   time; [`SampleBuffer::try_into_vec`] moves it out unchanged, while
//!   conversion requires an exact or explicitly lossy method.
//! - [`decode_samples`] — bulk decode of packed bytes in a given byte order.
//!
//! Conversion between sample types never passes through a third type: a
//! 64-bit integer read as `i64` stays exact, where a detour through `f32` or
//! `f64` would round it.

mod buffer;
mod codec;
mod conversion;
mod element;
mod error;
mod kind;

pub use buffer::SampleBuffer;
pub use codec::{decode_samples, encode_samples};
pub use conversion::{ConversionReport, ConvertedSamples, SampleConversionError};
pub use element::Sample;
pub use error::SampleError;
pub use kind::SampleType;

#[cfg(test)]
mod tests;
