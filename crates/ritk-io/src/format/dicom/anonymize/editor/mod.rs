//! Validated metadata pseudonymization for interactive editors.
//!
//! This is not a certificate of anonymous data: pixels, recognizable anatomy,
//! unlisted standard attributes, and user-entered values may identify a person.
//! Export writes a new directory and never modifies an input file.
mod export;
mod fields;
#[cfg(test)]
mod tests;

pub use export::{export_batch, ExportReport};
pub use fields::EditorFields;
