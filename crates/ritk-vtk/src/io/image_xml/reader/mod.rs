//! VTK XML ImageData (.vti) reader (ASCII inline and binary-appended formats).

mod xml_helpers;
mod parse;
mod binary;

pub use parse::read_vti_image_data;
#[cfg(test)]
pub(crate) use parse::parse_vti;
pub use binary::{read_vti_binary_appended, read_vti_binary_appended_bytes};

#[cfg(test)]
mod tests;
