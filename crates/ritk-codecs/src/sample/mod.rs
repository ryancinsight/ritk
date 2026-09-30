//! Typed voxel samples shared by every volume format.
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
//!   time; [`SampleBuffer::into_vec`] moves it out unchanged when the caller
//!   asks for the stored type and converts each sample directly otherwise.
//! - [`decode_samples`] — bulk decode of packed bytes in a given byte order.
//!
//! Conversion between sample types never passes through a third type: a
//! 64-bit integer read as `i64` stays exact, where a detour through `f32` or
//! `f64` would round it.

mod buffer;
mod codec;
mod element;
mod error;
mod kind;

pub use buffer::SampleBuffer;
pub use codec::{decode_samples, NATIVE_BYTE_ORDER};
pub use element::{FromSample, Sample};
pub use error::SampleError;
pub use kind::SampleType;

#[cfg(test)]
mod tests;
