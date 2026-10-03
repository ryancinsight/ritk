//! Fixed-width stored-sample buffers shared by RITK format adapters.
//!
//! [`SampleBuffer`] decodes and encodes the stored representation without
//! converting values to the scalar type used by image algorithms. Format
//! adapters retain their geometry and calibration metadata beside this
//! buffer and request a numeric conversion explicitly when needed.
//!
//! ```
//! use ritk_codecs::{ByteOrder, SampleBuffer, SampleType};
//!
//! let source = [0x01, 0x00, 0x01, 0x01];
//! let samples = SampleBuffer::decode(
//!     SampleType::U16,
//!     &source,
//!     ByteOrder::LeastSignificantByteFirst,
//! )?;
//! assert_eq!(samples.sample_type(), SampleType::U16);
//! assert_eq!(samples.try_into_samples::<u16>()?, [1, 257]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod buffer;
mod codec;
mod element;
mod error;

pub use buffer::{SampleBuffer, SampleType};
pub use element::Sample;
pub use error::{SampleError, SampleExtractionError, SampleWriteError};

#[cfg(test)]
mod tests;
