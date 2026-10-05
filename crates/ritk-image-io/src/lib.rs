#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

//! Typed values shared by RITK image-format readers, writers, and conversions.
//!
//! Stored samples stay in their file representation. Physical metadata,
//! coordinate mapping, and calibration travel beside those samples so an
//! adapter cannot silently replace one with a compute scalar.

mod calibration;
mod read_budget;
mod series;
mod volume;

pub use calibration::{
    CalibrationError, CalibrationShapeError, IntensityCalibration, LinearCalibration,
    LutOutputBits, ModalityLookupTable,
};
pub use read_budget::{ImageReadBudget, ImageReadBudgetError, ImageReadResource};
pub use series::{SeriesAxis, StoredSeries, StoredSeriesError};
pub use volume::{validate_coordinate_map, validate_physical_geometry, StoredVolume, VolumeError};

#[cfg(test)]
mod tests;
