//! RT Dose Storage (SOP Class 1.2.840.10008.5.1.4.1.1.481.2) reader/writer.
//!
//! # Specification
//!
//! An RT Dose file contains a 3-D dose grid:
//! - (3004,000E) DoseGridScaling: multiply raw pixel values to get dose in Gy.
//! - (3004,0002) DoseSummationType: PLAN, BEAM, FRACTION, CONTROL_PT, etc.
//! - (3004,0004) DoseType: PHYSICAL, EFFECTIVE, or ERROR.
//! - (3004,000C) GridFrameOffsetVector: z-positions of each dose plane (DS, multi-value).
//! - (0028,0010) Rows, (0028,0011) Columns, (0028,0008) NumberOfFrames.
//! - (7FE0,0010) PixelData: Uint32 LE voxel values (BitsAllocated = 32).
//! - Dose(x) = PixelValue(x) * DoseGridScaling.
//!
//! ## Dose computation invariant
//!
//! For voxel index `k = frame * rows * cols + row * cols + col`:
//!   `raw_u32 = u32::from_le_bytes(pixel_bytes[k*4 .. k*4+4])`
//!   `dose_gy[k] = raw_u32 as f64 * dose_grid_scaling`

mod reader;
mod types;
mod utils;
mod writer;

pub use reader::read_rt_dose;
pub use types::{RtDoseGrid, RtDoseSummationType, RtDoseType, RT_DOSE_SOP_CLASS_UID};
pub use writer::write_rt_dose;

#[cfg(test)]
mod tests;
