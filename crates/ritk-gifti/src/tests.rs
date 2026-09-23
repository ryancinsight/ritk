//! Tests over hand-written documents laid out from the GIFTI 1.0 specification.

mod decoding;
mod documents;
mod hostile;
mod round_trip;

use crate::{GiftiError, GiftiImage};

/// A one-array document with `attributes` on the `DataArray` and `data` as
/// its payload.
fn one_array(attributes: &str, data: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<GIFTI Version="1.0" NumberOfDataArrays="1">
  <DataArray {attributes}>
    <Data>{data}</Data>
  </DataArray>
</GIFTI>
"#
    )
}

fn read(document: &str) -> Result<GiftiImage, GiftiError> {
    GiftiImage::read(document.as_bytes())
}

fn error_of(document: &str) -> GiftiError {
    read(document).expect_err("invalid document must be rejected")
}
