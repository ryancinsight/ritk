//! Mutable DICOM datasets, file metadata, and the Part 10 writer.
//!
//! Each data set contains one entry per tag in numerical order. A parsed
//! binary value retains a slice of the backing input. Mutating, cloning, or
//! writing a dataset therefore need not duplicate Pixel Data.

mod data;
mod element;
mod file;
mod primitive;
mod sequence;
pub(crate) mod wire;

pub use crate::DicomTag;
pub use data::Dataset;
pub use element::{DataElement, VR};
pub use file::{DicomFile, FileMeta, FileMetaBuilder, IMPLEMENTATION_CLASS_UID};
pub use primitive::PrimitiveValue;
pub use sequence::{Length, PixelFragments, Sequence, Value};
pub use smallvec::{smallvec, SmallVec};
