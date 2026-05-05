//! Intensity transformation filters.
//!
//! ITK/SimpleITK-style pixel-wise intensity transform filters.
//! Includes a CT bed separation filter for body masking and foreground extraction,
//! contrast-limited adaptive histogram equalization (CLAHE), global
//! histogram equalization (ImageJ/ITK parity), and unsharp masking
//! (ITK `UnsharpMaskingImageFilter` / ImageJ "Unsharp Mask" parity).

pub mod bed_separation;
pub mod binary_threshold;
pub mod clahe;
pub mod equalization;
pub mod rescale;
pub mod sigmoid;
pub mod threshold;
pub mod unsharp_mask;
pub mod windowing;

pub use bed_separation::{BedSeparationConfig, BedSeparationFilter};
pub use binary_threshold::BinaryThresholdImageFilter;
pub use clahe::ClaheFilter;
pub use equalization::HistogramEqualizationFilter;
pub use rescale::RescaleIntensityFilter;
pub use sigmoid::SigmoidImageFilter;
pub use threshold::{ThresholdImageFilter, ThresholdMode};
pub use unsharp_mask::UnsharpMaskFilter;
pub use windowing::IntensityWindowingFilter;
