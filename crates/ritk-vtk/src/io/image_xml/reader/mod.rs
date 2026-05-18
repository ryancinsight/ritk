//! VTK XML ImageData (.vti) reader (ASCII inline and binary-appended formats).

mod binary;
mod parse;
mod xml_helpers;

pub use binary::{read_vti_binary_appended, read_vti_binary_appended_bytes};
#[cfg(test)]
pub(crate) use parse::parse_vti;
pub use parse::read_vti_image_data;

#[cfg(test)]
mod tests;
