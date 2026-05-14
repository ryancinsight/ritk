//! Pixelwise arithmetic intensity transform filters.
//!
//! Each filter is a pure pixelwise map `f : f32 → f32` applied independently to
//! every voxel in a 3-D image. Spatial metadata (origin, spacing, direction) is
//! preserved identically in every output image.
//!
//! # ITK / ImageJ / SimpleITK parity
//!
//! | Filter                       | ITK class                          | ImageJ (Process > Math) |
//! |------------------------------|------------------------------------|-------------------------|
//! | `AbsImageFilter`             | `AbsImageFilter`                   | Abs                     |
//! | `InvertIntensityFilter`      | `InvertIntensityImageFilter`       | (Image > Adjust > Invert) |
//! | `NormalizeImageFilter`       | `NormalizeImageFilter`             | —                       |
//! | `SquareImageFilter`          | `SquareImageFilter`                | Square                  |
//! | `SqrtImageFilter`            | `SqrtImageFilter`                  | Square Root             |
//! | `LogImageFilter`             | `LogImageFilter`                   | Log                     |
//! | `ExpImageFilter`             | `ExpImageFilter`                   | Exp                     |



pub mod abs;
pub mod invert;
pub mod normalize;
pub mod square;
pub mod sqrt;
pub mod log;
pub mod exp;

pub use abs::AbsImageFilter;
pub use invert::InvertIntensityFilter;
pub use normalize::NormalizeImageFilter;
pub use square::SquareImageFilter;
pub use sqrt::SqrtImageFilter;
pub use log::LogImageFilter;
pub use exp::ExpImageFilter;
