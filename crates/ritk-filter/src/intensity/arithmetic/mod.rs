//! Pixelwise arithmetic intensity transform filters.
//!
//! Each filter is a pure pixelwise map `f : f32 → f32` applied independently to
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
//! | `NormalizeImageFilter`       | `NormalizeImageFilter`             | —                       |
//! | `SquareImageFilter`          | `SquareImageFilter`                | Square                  |
//! | `SqrtImageFilter`            | `SqrtImageFilter`                  | Square Root             |
//! | `LogImageFilter`             | `LogImageFilter`                   | Log                     |
//! | `Log10ImageFilter`          | `Log10ImageFilter`                 | Log (base 10)           |
//! | `ExpImageFilter`             | `ExpImageFilter`                   | Exp                     |
//! | `ExpNegativeImageFilter`    | `ExpNegativeImageFilter`           | —                       |

// ── Generic unary infrastructure ─────────────────────────────────────────────
pub mod unary;
pub use unary::{
    Abs, AbsImageFilter, Exp, ExpImageFilter, ExpNegative, ExpNegativeImageFilter, Log, Log10,
    Log10ImageFilter, LogImageFilter, Not, NotImageFilter, Round, RoundImageFilter, Sqrt,
    SqrtImageFilter, Square, SquareImageFilter, UnaryImageFilter, UnaryMinus,
    UnaryMinusImageFilter, UnaryPixelOp,
};

// ── Test-hosting modules (one per filter; contain only the #[cfg(test)] block) ─
pub mod abs;
pub mod exp;
pub mod log;
pub mod sqrt;
pub mod square;

// ── Filters with unique state ─────────────────────────────────────────────────
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
