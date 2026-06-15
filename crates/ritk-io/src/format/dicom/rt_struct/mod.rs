//! DICOM RT Structure Set Storage (SOP Class 1.2.840.10008.5.1.4.1.1.481.3) reader and writer.
//!
//! # Specification
//!
//! SOP Class UID: `1.2.840.10008.5.1.4.1.1.481.3` (RT Structure Set Storage)
//!
//! ## Invariants
//! 1. The file must carry SOP Class UID `1.2.840.10008.5.1.4.1.1.481.3`.
//! 2. ROIs are de-duplicated by ROINumber; the result is sorted ascending.
//! 3. Contour data length is always a multiple of 3 (X, Y, Z per point).
//! 4. `display_color` is `Some([r,g,b])` when present and parseable as three u8 values.
//! 5. `rt_roi_to_polydata` maps geometric type to the correct VTK cell bucket;
//!    unknown types fall back to `lines`.
//! 6. `write_rt_struct` followed by `read_rt_struct` preserves all fields invariantly.

mod converter;
mod reader;
mod types;
mod utils;
mod writer;

pub use converter::{label_map_to_rt_struct, rt_roi_to_polydata};
pub use reader::read_rt_struct;
pub use types::{ContourGeometricType, RtContour, RtRoiInfo, RtRoiInterpretedType, RtStructureSet};
pub use writer::write_rt_struct;

#[cfg(test)]
mod tests;
