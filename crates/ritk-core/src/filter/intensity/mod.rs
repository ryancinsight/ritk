//! Intensity transformation filters.
//!
//! ITK/SimpleITK-style pixel-wise intensity transform filters.
//! Includes a CT bed separation filter for body masking and foreground extraction,
//! contrast-limited adaptive histogram equalization (CLAHE), global
//! histogram equalization (ImageJ/ITK parity), unsharp masking
//! (ITK `UnsharpMaskingImageFilter` / ImageJ "Unsharp Mask" parity), and
//! pixelwise arithmetic transforms (ITK `AbsImageFilter`, `InvertIntensityImageFilter`,
//! `NormalizeImageFilter`, `SquareImageFilter`, `SqrtImageFilter`,
//! `LogImageFilter`, `ExpImageFilter`).

pub mod arithmetic;
pub mod bed_separation;
pub mod binary_ops;
pub mod binary_threshold;
pub mod clahe;
pub mod clamp;
pub mod equalization;
pub mod mask;
pub mod rescale;
pub mod sigmoid;
pub mod threshold;
pub mod unsharp_mask;
pub mod windowing;
pub mod suv;

pub use arithmetic::{
    AbsImageFilter, ExpImageFilter, InvertIntensityFilter, LogImageFilter, NormalizeImageFilter,
    SqrtImageFilter, SquareImageFilter,
};
pub use bed_separation::{BedSeparationConfig, BedSeparationFilter};
pub use binary_ops::{
    AddImageFilter, DivideImageFilter, ImageMaxFilter, ImageMinFilter, MultiplyImageFilter,
    SubtractImageFilter,
};
pub use binary_threshold::BinaryThresholdImageFilter;
pub use clahe::ClaheFilter;
pub use clamp::ClampImageFilter;
pub use equalization::HistogramEqualizationFilter;
pub use mask::{MaskImageFilter, MaskNegatedImageFilter};
pub use rescale::RescaleIntensityFilter;
pub use sigmoid::SigmoidImageFilter;
pub use threshold::{ThresholdImageFilter, ThresholdMode};
pub use unsharp_mask::UnsharpMaskFilter;
pub use windowing::IntensityWindowingFilter;

pub mod shift_scale;
pub mod trig;
pub mod zero_crossing;
pub use shift_scale::ShiftScaleImageFilter;
pub use trig::{
    AcosImageFilter, AsinImageFilter, AtanImageFilter, BoundedReciprocalImageFilter,
    CosImageFilter, SinImageFilter, TanImageFilter,
};
pub use zero_crossing::ZeroCrossingImageFilter;
pub use suv::SuvBodyWeightImageFilter;
