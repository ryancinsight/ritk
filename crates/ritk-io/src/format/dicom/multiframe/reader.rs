//! Multi-frame DICOM reader.
//!
//! Split at the point pixel data becomes necessary: [`header`] answers what an
//! object is from its tags alone, [`volume`] decodes it. The file they came
//! from reached 520 lines when the volume path landed.

mod header;
mod volume;

pub(crate) use header::parse_ds_backslash;
pub use header::read_multiframe_info;
pub(super) use header::read_multiframe_info_from_object;
pub(super) use volume::load_multiframe_flat_from_object;
pub use volume::{
    load_dicom_multiframe, load_dicom_multiframe_flat, load_dicom_multiframe_native,
    MultiFrameVolume,
};
