#![doc = include_str!("../README.md")]
//!
//! # Module map
//!
//! | Module | Responsibility |
//! |--------|----------------|
//! | `model` | The document: [`GiftiImage`], [`DataArray`], [`ArrayData`], the label table |
//! | `read` | XML event parsing into the model, checked against the DTD |
//! | `decode` | `Data` element payloads: ASCII, base64, zlib, byte order |
//! | `io` | [`GiftiImage::read`] and [`GiftiImage::write`] |
//! | `surface` | The coordinate and triangle arrays as a mesh |

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod decode;
mod error;
mod io;
mod model;
mod read;
mod surface;

pub use error::GiftiError;
pub use model::{
    ArrayData, CoordinateTransform, DataArray, DataEncoding, GiftiImage, GiftiLabel, IndexingOrder,
    Intent, MetaData,
};
pub use surface::GiftiSurface;

#[cfg(test)]
mod tests;
