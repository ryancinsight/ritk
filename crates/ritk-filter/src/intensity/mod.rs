//! Intensity transformation filters.
//!
//! ITK/SimpleITK-style pixel-wise intensity transform filters.
//! Includes a CT bed separation filter for body masking and foreground extraction,
//! contrast-limited adaptive histogram equalization (CLAHE), global
//! histogram equalization (ImageJ/ITK parity), unsharp masking
//! (ITK `UnsharpMaskingImageFilter` / ImageJ "Unsharp Mask" parity), and
//! pixelwise arithmetic transforms (ITK `AbsImageFilter`, `InvertIntensityImageFilter`,
//! `NormalizeImageFilter`, `SquareImageFilter`, `SqrtImageFilter`,
//! `LogImageFilter`, `Log10ImageFilter`, `ExpImageFilter`,
//! `ExpNegativeImageFilter`).

pub mod arithmetic;
pub mod bed_separation;
pub mod binary_ops;
pub mod binary_threshold;
pub mod blend;
pub mod clahe;
pub mod clamp;
pub mod equalization;
pub mod mask;
pub mod rescale;
pub mod sigmoid;
pub mod suv;
pub mod threshold;
pub mod unsharp_mask;
pub mod windowing;

pub use arithmetic::{
    AbsImageFilter, ExpImageFilter, ExpNegativeImageFilter, InvertIntensityFilter,
    Log10ImageFilter, LogImageFilter, NormalizeImageFilter, SqrtImageFilter, SquareImageFilter,
};
pub use bed_separation::{BedSeparationConfig, BedSeparationFilter, ComponentPolicy};
pub use binary_ops::{
    AddImageFilter, AddOp, BinaryOp, BinaryOpFilter, DivideImageFilter, DivideOp, ImageMaxFilter,
    ImageMinFilter, MaxOp, MinOp, MultiplyImageFilter, MultiplyOp, SubtractImageFilter, SubtractOp,
};
pub use binary_threshold::BinaryThresholdImageFilter;
pub use blend::BlendImageFilter;
pub use clahe::{ClaheFilter, ClaheScratch};
pub use clamp::ClampImageFilter;
pub use equalization::HistogramEqualizationFilter;
pub use mask::{MaskImageFilter, MaskNegatedImageFilter};
pub use rescale::RescaleIntensityFilter;
pub use sigmoid::SigmoidImageFilter;
pub use threshold::{ThresholdImageFilter, ThresholdMode};
pub use unsharp_mask::{ClampPolicy, UnsharpMaskFilter};
pub use windowing::IntensityWindowingFilter;

pub mod shift_scale;
pub mod trig;
pub mod zero_crossing;
pub use shift_scale::ShiftScaleImageFilter;
pub use suv::SuvBodyWeightImageFilter;
pub use trig::{
    AcosImageFilter, AsinImageFilter, AtanImageFilter, BoundedReciprocalImageFilter,
    CosImageFilter, SinImageFilter, TanImageFilter,
};
pub use zero_crossing::ZeroCrossingImageFilter;
