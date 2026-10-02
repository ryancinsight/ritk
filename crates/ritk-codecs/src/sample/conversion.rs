//! Numeric conversion operations and their caller-selected policies.

mod element;
mod error;
mod policy;
mod primitive;

pub use element::Sample;
pub use error::{SampleConversionError, SampleConversionFailure};
pub use policy::{Cast, Conversion, ConversionDisposition, ConversionReport, Exact, ValueChange};
