//! Pixelwise two-image arithmetic filters.
//!
//! Each filter combines two co-registered images with matching shapes via a
//! pointwise binary operation applied independently to every voxel.
//!
//! # Mathematical Specification
//!
//! Let `A, B : â„¤Â³ â†’ â„` be two images with identical shape `[nz, ny, nx]`:
//!
//! - `AddImageFilter`: `out(x) = A(x) + B(x)`
//! - `SubtractImageFilter`: `out(x) = A(x) âˆ’ B(x)`
//! - `MultiplyImageFilter`: `out(x) = A(x) Ã— B(x)`
//! - `DivideImageFilter`: `out(x) = A(x) / B(x)` (division by zero yields 0)
//! - `ImageMinFilter`: `out(x) = min(A(x), B(x))`
//! - `ImageMaxFilter`: `out(x) = max(A(x), B(x))`
//! - `SquaredDifferenceImageFilter`: `out(x) = (A(x) âˆ’ B(x))Â²`
//! - `AbsoluteValueDifferenceImageFilter`: `out(x) = |A(x) âˆ’ B(x)|`
//! - `Atan2ImageFilter`: `out(x) = atan2(A(x), B(x))`
//! - `PowImageFilter`: `out(x) = A(x)^{B(x)}`
//!
//! Spatial metadata (origin, spacing, direction) is taken from the **first** input image.
//! Both images must have identical shapes; a shape mismatch returns `Err`.
//!
//! # Architecture
//!
//! All filters share a single generic [`BinaryOpFilter<Op>`] implementation
//! parameterized by a ZST operation type implementing [`BinaryOp`]. This
//! eliminates duplicated `apply` bodies while producing monomorphized,
//! zero-cost specializations per operation.
//!
//! # ITK / SimpleITK / ImageJ Parity
//!
//! | Filter | ITK class | ImageJ (Process > Image Calculator) |
//! |------------------------|------------------------|--------------------------------------|
//! | `AddImageFilter` | `AddImageFilter` | Add |
//! | `SubtractImageFilter` | `SubtractImageFilter` | Subtract |
//! | `MultiplyImageFilter` | `MultiplyImageFilter` | Multiply |
//! | `DivideImageFilter` | `DivideImageFilter` | Divide |
//! | `ImageMinFilter` | `MinimumImageFilter` | Min |
//! | `ImageMaxFilter` | `MaximumImageFilter` | Max |
//! | `SquaredDifferenceImageFilter` | `SquaredDifferenceImageFilter` | â€” |
//! | `AbsoluteValueDifferenceImageFilter` | `AbsoluteValueDifferenceImageFilter` | Difference |
//! | `Atan2ImageFilter` | `Atan2ImageFilter` | â€” |
//! | `PowImageFilter` | `PowImageFilter` | â€” |

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec as extract, rebuild};

// â”€â”€ Trait â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Trait for pointwise binary operations on voxel pairs.
///
/// Each implementor is a zero-sized type (ZST) that encodes the operation
/// in the type system. The compiler monomorphizes `BinaryOpFilter<Op>::apply`
/// into a specialized, branch-free loop identical to a hand-written version.
pub trait BinaryOp: Default {
    /// Apply the binary operation to a single voxel pair.
    fn apply(a: f32, b: f32) -> f32;
}

// â”€â”€ ZST operation types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Addition: `a + b`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AddOp;

/// Subtraction: `a âˆ’ b`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SubtractOp;

/// Multiplication: `a Ã— b`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MultiplyOp;

/// Division: `a / b` (returns 0 where `b = 0`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DivideOp;

/// Real division: `a / b`, returning `f32::MAX` where `b = 0` (ITK
/// `Functor::Div` convention, distinct from [`DivideOp`]'s `0` guard).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DivideRealOp;

/// Floored division: `âŒŠa / bâŒ‹`, returning `f32::MAX` where `b = 0` (ITK
/// `DivideFloorImageFilter`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DivideFloorOp;

/// Elementwise minimum: `min(a, b)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MinOp;

/// Elementwise maximum: `max(a, b)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaxOp;

/// Squared difference: `(a âˆ’ b)Â²`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SquaredDifferenceOp;

/// Absolute value of the difference: `|a âˆ’ b|`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AbsoluteValueDifferenceOp;

/// Four-quadrant arctangent: `atan2(a, b)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Atan2Op;

/// Power: `a^b`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PowOp;

/// Binary magnitude: `âˆš(aÂ² + bÂ²)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BinaryMagnitudeOp;

/// Equality test: `1.0` where `a == b`, else `0.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EqualOp;

/// Inequality test: `1.0` where `a != b`, else `0.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NotEqualOp;

/// Logical AND on binary masks: `1.0` where both `> 0.5`, else `0.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AndOp;

/// Logical OR on binary masks: `1.0` where either `> 0.5`, else `0.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OrOp;

/// Logical XOR on binary masks: `1.0` where exactly one `> 0.5`, else `0.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct XorOp;

/// Greater-than test: `1.0` where `a > b`, else `0.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GreaterOp;

/// Greater-or-equal test: `1.0` where `a >= b`, else `0.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GreaterEqualOp;

/// Less-than test: `1.0` where `a < b`, else `0.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LessOp;

/// Less-or-equal test: `1.0` where `a <= b`, else `0.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LessEqualOp;

impl BinaryOp for AddOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        a + b
    }
}

impl BinaryOp for SubtractOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        a - b
    }
}

impl BinaryOp for MultiplyOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        a * b
    }
}

impl BinaryOp for DivideOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        if b == 0.0 {
            0.0
        } else {
            a / b
        }
    }
}

impl BinaryOp for DivideRealOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        if b == 0.0 {
            f32::MAX
        } else {
            a / b
        }
    }
}

impl BinaryOp for DivideFloorOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        if b == 0.0 {
            f32::MAX
        } else {
            (a / b).floor()
        }
    }
}

impl BinaryOp for MinOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        a.min(b)
    }
}

impl BinaryOp for MaxOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        a.max(b)
    }
}

impl BinaryOp for SquaredDifferenceOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        let d = a - b;
        d * d
    }
}

impl BinaryOp for AbsoluteValueDifferenceOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        (a - b).abs()
    }
}

impl BinaryOp for Atan2Op {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        a.atan2(b)
    }
}

impl BinaryOp for PowOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        a.powf(b)
    }
}

impl BinaryOp for BinaryMagnitudeOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        (a * a + b * b).sqrt()
    }
}

impl BinaryOp for EqualOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        (a == b) as u8 as f32
    }
}

impl BinaryOp for NotEqualOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        (a != b) as u8 as f32
    }
}

impl BinaryOp for AndOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        (a > 0.5 && b > 0.5) as u8 as f32
    }
}

impl BinaryOp for OrOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        (a > 0.5 || b > 0.5) as u8 as f32
    }
}

impl BinaryOp for XorOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        ((a > 0.5) != (b > 0.5)) as u8 as f32
    }
}

impl BinaryOp for GreaterOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        (a > b) as u8 as f32
    }
}

impl BinaryOp for GreaterEqualOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        (a >= b) as u8 as f32
    }
}

impl BinaryOp for LessOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        (a < b) as u8 as f32
    }
}

impl BinaryOp for LessEqualOp {
    #[inline]
    fn apply(a: f32, b: f32) -> f32 {
        (a <= b) as u8 as f32
    }
}

// â”€â”€ Generic filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Generic pixelwise binary image filter parameterized by operation type.
///
/// `Op` is a ZST implementing [`BinaryOp`]. The filter monomorphizes to a
/// specialized loop with zero runtime overhead compared to a hand-written
/// per-operation implementation.
///
/// # Invariants
///
/// - Both input images must have identical shapes.
/// - Spatial metadata (origin, spacing, direction) is taken from `a`.
#[derive(Debug, Clone, Default)]
pub struct BinaryOpFilter<Op: BinaryOp> {
    _op: core::marker::PhantomData<Op>,
}

impl<Op: BinaryOp> BinaryOpFilter<Op> {
    /// Create a new filter.
    pub fn new() -> Self {
        Self {
            _op: core::marker::PhantomData,
        }
    }

    /// Apply the binary operation to two co-registered images.
    pub fn apply<B: Backend>(
        &self,
        a: &Image<f32, B, 3>,
        b: &Image<f32, B, 3>,
    ) -> anyhow::Result<Image<f32, B, 3>> {
        check_shapes(a.shape(), b.shape())?;
        let (av, dims) = extract(a)?;
        let (bv, _) = extract(b)?;
        let out: Vec<f32> = av
            .iter()
            .zip(bv.iter())
            .map(|(&x, &y)| Op::apply(x, y))
            .collect();
        Ok(rebuild(out, dims, a))
    }

    /// Apply the binary operation to two Coeus-native images.
    pub fn apply_native<B>(
        &self,
        a: &NativeImage<f32, B, 3>,
        b: &NativeImage<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<NativeImage<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        check_shapes(a.shape(), b.shape())?;
        let output = a
            .data_slice()?
            .iter()
            .zip(b.data_slice()?)
            .map(|(&left, &right)| Op::apply(left, right))
            .collect();
        NativeImage::from_flat_on(
            output,
            a.shape(),
            *a.origin(),
            *a.spacing(),
            *a.direction(),
            backend,
        )
    }
}

// â”€â”€ Shape validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn check_shapes(a: [usize; 3], b: [usize; 3]) -> anyhow::Result<()> {
    anyhow::ensure!(
        a == b,
        "binary image filter: shape mismatch {:?} vs {:?}",
        a,
        b
    );
    Ok(())
}

// â”€â”€ Type aliases preserving public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Pixelwise addition of two images.
///
/// `out(x) = a(x) + b(x)`
///
/// # ITK Parity: `AddImageFilter`
pub type AddImageFilter = BinaryOpFilter<AddOp>;

/// Pixelwise subtraction of two images.
///
/// `out(x) = a(x) âˆ’ b(x)`
///
/// # ITK Parity: `SubtractImageFilter`
pub type SubtractImageFilter = BinaryOpFilter<SubtractOp>;

/// Pixelwise multiplication of two images.
///
/// `out(x) = a(x) Ã— b(x)`
///
/// # ITK Parity: `MultiplyImageFilter`
pub type MultiplyImageFilter = BinaryOpFilter<MultiplyOp>;

/// Pixelwise division of two images; division by zero yields 0.
///
/// `out(x) = a(x) / b(x)` (returns 0 where `b(x) = 0`)
///
/// # ITK Parity: `DivideImageFilter`
pub type DivideImageFilter = BinaryOpFilter<DivideOp>;

/// Pixelwise real division `a/b`, `f32::MAX` where `b = 0`.
/// # ITK Parity: `DivideRealImageFilter` (`sitk.DivideReal`)
pub type DivideRealImageFilter = BinaryOpFilter<DivideRealOp>;

/// Pixelwise floored division `âŒŠa/bâŒ‹`, `f32::MAX` where `b = 0`.
/// # ITK Parity: `DivideFloorImageFilter` (`sitk.DivideFloor`)
pub type DivideFloorImageFilter = BinaryOpFilter<DivideFloorOp>;

/// Pixelwise minimum of two images.
///
/// `out(x) = min(a(x), b(x))`
///
/// # ITK Parity: `MinimumImageFilter`
pub type ImageMinFilter = BinaryOpFilter<MinOp>;

/// Pixelwise maximum of two images.
///
/// `out(x) = max(a(x), b(x))`
///
/// # ITK Parity: `MaximumImageFilter`
pub type ImageMaxFilter = BinaryOpFilter<MaxOp>;

/// Pixelwise squared difference of two images.
///
/// `out(x) = (a(x) âˆ’ b(x))Â²`
///
/// # ITK Parity: `SquaredDifferenceImageFilter`
pub type SquaredDifferenceImageFilter = BinaryOpFilter<SquaredDifferenceOp>;

/// Pixelwise absolute difference of two images.
///
/// `out(x) = |a(x) âˆ’ b(x)|`
///
/// # ITK Parity: `AbsoluteValueDifferenceImageFilter`
pub type AbsoluteValueDifferenceImageFilter = BinaryOpFilter<AbsoluteValueDifferenceOp>;

/// Pixelwise four-quadrant arctangent of two images.
///
/// `out(x) = atan2(a(x), b(x))`
///
/// # ITK Parity: `Atan2ImageFilter`
pub type Atan2ImageFilter = BinaryOpFilter<Atan2Op>;

/// Pixelwise power of two images.
///
/// `out(x) = a(x)^{b(x)}`
///
/// # ITK Parity: `PowImageFilter`
pub type PowImageFilter = BinaryOpFilter<PowOp>;

/// Pixelwise magnitude of two images.
///
/// `out(x) = âˆš(a(x)Â² + b(x)Â²)`
///
/// # ITK Parity: `BinaryMagnitudeImageFilter`
pub type BinaryMagnitudeImageFilter = BinaryOpFilter<BinaryMagnitudeOp>;

/// Pixelwise equality mask. `out(x) = 1` where `a(x) == b(x)`, else `0`.
///
/// # ITK Parity: `EqualImageFilter`
pub type EqualImageFilter = BinaryOpFilter<EqualOp>;

/// Pixelwise inequality mask. # ITK Parity: `NotEqualImageFilter`
pub type NotEqualImageFilter = BinaryOpFilter<NotEqualOp>;

/// Binary-mask logical AND. # ITK Parity: `AndImageFilter` (binary inputs)
pub type AndImageFilter = BinaryOpFilter<AndOp>;

/// Binary-mask logical OR. # ITK Parity: `OrImageFilter` (binary inputs)
pub type OrImageFilter = BinaryOpFilter<OrOp>;

/// Binary-mask logical XOR. # ITK Parity: `XorImageFilter` (binary inputs)
pub type XorImageFilter = BinaryOpFilter<XorOp>;

/// Pixelwise greater-than mask. # ITK Parity: `GreaterImageFilter`
pub type GreaterImageFilter = BinaryOpFilter<GreaterOp>;

/// Pixelwise greater-or-equal mask. # ITK Parity: `GreaterEqualImageFilter`
pub type GreaterEqualImageFilter = BinaryOpFilter<GreaterEqualOp>;

/// Pixelwise less-than mask. # ITK Parity: `LessImageFilter`
pub type LessImageFilter = BinaryOpFilter<LessOp>;

/// Pixelwise less-or-equal mask. # ITK Parity: `LessEqualImageFilter`
pub type LessEqualImageFilter = BinaryOpFilter<LessEqualOp>;

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
mod tests_binary_ops;
