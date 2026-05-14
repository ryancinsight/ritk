//! Multi-frame DICOM image reader and writer.
//!
//! # Reader specification
//!
//! A multi-frame DICOM file stores N frames in one file:
//! - (0028,0008) NumberOfFrames: N (absent ⇒ 1)
//! - (0028,0010) Rows, (0028,0011) Columns
//! - (7FE0,0010) PixelData: `N × Rows × Cols × (BitsAllocated/8)` bytes
//!
//! ## Reader invariants
//! - `n_frames >= 1`
//! - Output tensor shape: `[n_frames, rows, cols]`
//! - RescaleSlope (absent ⇒ 1.0) and RescaleIntercept (absent ⇒ 0.0) applied.
//! - 8-bit and 16-bit BitsAllocated are both supported.
//! - ImagePositionPatient (0020,0032) sets the image origin when present.
//! - ImageOrientationPatient (0020,0037) sets the direction matrix when present;
//!   the normal vector is computed as the cross product of the row and column cosines.
//!
//! # Writer specification (`write_dicom_multiframe`)
//!
//! Writes a 3-D `Image<B, 3>` with shape `[n_frames, rows, cols]` as a single
//! DICOM Part 10 file. The writer enforces the following constraints:
//!
//! ## Encoding constraints
//! - **SOP Class**: Multi-Frame Grayscale Word Secondary Capture Image Storage
//!   (`1.2.840.10008.5.1.4.1.1.7.3`). The output is not an Enhanced Multi-Frame
//!   CT, MR, or PET object. Viewers that enforce strict modality-to-SOP-class
//!   binding may reject the file.
//! - **Transfer Syntax**: Explicit VR Little Endian (`1.2.840.10008.1.2.1`).
//!   Compressed transfer syntaxes (JPEG, JPEG-LS, JPEG 2000) are not supported.
//! - **Pixel depth**: always 16-bit unsigned (BitsAllocated=16, BitsStored=16,
//!   HighBit=15, PixelRepresentation=0).
//!
//! ## Rescale constraints
//! - A **single global linear rescale** maps the entire f32 volume to the u16 range
//!   [0, 65535]: `rescale_slope = (max - min) / 65535; rescale_intercept = min`.
//! - When max == min (flat image), slope = 1.0 and intercept = min_val.
//! - **All frames share one slope/intercept pair.** Per-frame rescaling is not
//!   supported. Images whose frames have widely varying intensity ranges will
//!   lose intra-frame contrast fidelity relative to inter-frame range.
//!
//! ## Spatial metadata
//! - `write_dicom_multiframe` emits no spatial metadata (IPP/IOP/PixelSpacing/
//!   SliceThickness). Use [`write_dicom_multiframe_with_options`] with a
//!   [`MultiFrameSpatialMetadata`] value to include spatial tags.
//!
//! ## Interoperability limits
//! - The file is readable by `load_dicom_multiframe` (round-trip invariant:
//!   |recovered − original| ≤ rescale_slope + 1.0).
//! - DICOM conformance: the file satisfies the Multi-Frame Grayscale Word SC IOD
//!   but does NOT carry a conformance statement or General Series / Frame Of
//!   Reference modules required for Enhanced Multi-Frame objects.

mod per_frame;
mod reader;
mod types;
mod writer;

pub use reader::{load_dicom_multiframe, read_multiframe_info};
pub use types::{MultiFrameInfo, MultiFrameSpatialMetadata, MultiFrameWriterConfig};
pub use writer::{
    write_dicom_multiframe, write_dicom_multiframe_with_config, write_dicom_multiframe_with_options,
};

#[cfg(test)]
mod tests;
