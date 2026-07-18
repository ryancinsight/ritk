//! Pixelwise arithmetic intensity transform filters.
//!
//! Each filter is a pure pixelwise map `f : f32 â†’ f32` applied independently to
//! every voxel in a D-dimensional image. Spatial metadata (origin, spacing,
//! direction) is preserved identically in every output image.
//!
//! The uniform-scaffold filters (`Abs`, `Sqrt`, `Exp`, `Log`, `Square`,
//! `Log10`, `ExpNegative`) share a single generic implementation in [`unary`];
//! type aliases maintain the original names.  [`invert`] stays separate because
//! it carries state (`maximum: Option<f32>`).
//!
//! # ITK / ImageJ / SimpleITK parity
//!
//! | Filter                       | ITK class                          | ImageJ (Process > Math) |
//! |------------------------------|------------------------------------|-------------------------|
//! | `AbsImageFilter`             | `AbsImageFilter`                   | Abs                     |
//! | `InvertIntensityFilter`      | `InvertIntensityImageFilter`       | (Image > Adjust > Invert) |
//! | `NormalizeImageFilter`       | `NormalizeImageFilter`             | â€”                       |
//! | `SquareImageFilter`          | `SquareImageFilter`                | Square                  |
//! | `SqrtImageFilter`            | `SqrtImageFilter`                  | Square Root             |
//! | `LogImageFilter`             | `LogImageFilter`                   | Log                     |
//! | `Log10ImageFilter`          | `Log10ImageFilter`                 | Log (base 10)           |
//! | `ExpImageFilter`             | `ExpImageFilter`                   | Exp                     |
//! | `ExpNegativeImageFilter`    | `ExpNegativeImageFilter`           | â€”                       |

// â”€â”€ Generic unary infrastructure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pub mod unary;
pub use unary::{
    Abs, AbsImageFilter, Acos, AcosImageFilter, Asin, AsinImageFilter, Atan, AtanImageFilter,
    BoundedReciprocal, BoundedReciprocalImageFilter, Cos, CosImageFilter, Exp, ExpImageFilter,
    ExpNegative, ExpNegativeImageFilter, Log, Log10, Log10ImageFilter, LogImageFilter, Not,
    NotImageFilter, Round, RoundImageFilter, Sin, SinImageFilter, Sqrt, SqrtImageFilter, Square,
    SquareImageFilter, Tan, TanImageFilter, UnaryImageFilter, UnaryMinus, UnaryMinusImageFilter,
    UnaryPixelOp,
};

// â”€â”€ Test-hosting modules (one per filter; contain only the #[cfg(test)] block) â”€
pub mod abs;
pub mod exp;
pub mod log;
pub mod sqrt;
pub mod square;

// â”€â”€ Filters with unique state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pub mod binary_not;
pub mod bitwise_not;
pub mod invert;
pub mod modulus;
pub mod normalize;

pub use binary_not::BinaryNotImageFilter;
pub use bitwise_not::BitwiseNotImageFilter;
pub use invert::InvertIntensityFilter;
pub use modulus::ModulusImageFilter;
pub use normalize::{NormalizeImageFilter, NormalizeToConstantImageFilter};
