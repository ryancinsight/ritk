use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec_infallible as extract_vec, rebuild};

/// Bitwise-complement (`~x`) filter for integer-valued images.
///
/// # Mathematical Specification
///
/// The bitwise NOT of an integer pixel depends on its storage type's width and
/// signedness:
///
/// ```text
/// unsigned n-bit:  ~x = (2ⁿ − 1) − x
/// signed:          ~x = −x − 1            (two's complement)
/// ```
///
/// ritk's scalar-`f32` backend carries no integer type, so the width/signedness
/// are explicit parameters. Pixels are rounded to the nearest integer before
/// complementing. Matches `SimpleITK.BitwiseNot` for the corresponding pixel type
/// (e.g. `bits = 8, signed = false` ↔ `uint8`; `signed = true` ↔ `intN`).
///
/// Distinct from [`BinaryNotImageFilter`](super::binary_not::BinaryNotImageFilter)
/// (a `{0,1}` label flip) and `Not` (logical NOT to `0`/`1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitwiseNotImageFilter {
    /// Bit width of the integer type (used only for the unsigned complement).
    bits: u32,
    /// Two's-complement signed interpretation (`~x = −x − 1`).
    signed: bool,
}

impl Default for BitwiseNotImageFilter {
    fn default() -> Self {
        Self {
            bits: 8,
            signed: false,
        }
    }
}

impl BitwiseNotImageFilter {
    /// Construct for an unsigned `bits`-wide integer type.
    pub fn unsigned(bits: u32) -> Self {
        Self {
            bits,
            signed: false,
        }
    }

    /// Construct for a signed (two's-complement) integer type.
    pub fn signed() -> Self {
        Self {
            bits: 0,
            signed: true,
        }
    }

    /// Apply the bitwise NOT pixelwise (spatial metadata preserved).
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let (vals, dims) = extract_vec(image);
        // 2ⁿ − 1 as f64 to stay exact through 53-bit mantissa for n ≤ 32.
        let mask = ((1u64 << self.bits) - 1) as f64;
        let signed = self.signed;
        let out: Vec<f32> = vals
            .into_iter()
            .map(|v| {
                let x = (v as f64).round();
                let r = if signed { -x - 1.0 } else { mask - x };
                r as f32
            })
            .collect();
        rebuild(out, dims, image)
    }
    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::native::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        // 2ⁿ − 1 as f64 to stay exact through 53-bit mantissa for n ≤ 32.
        let mask = ((1u64 << self.bits) - 1) as f64;
        let signed = self.signed;
        let out: Vec<f32> = vals
            .into_iter()
            .map(|v| {
                let x = (v as f64).round();
                let r = if signed { -x - 1.0 } else { mask - x };
                r as f32
            })
            .collect();
        crate::native_support::rebuild_image(out, dims, image, backend)
    }
}

#[cfg(test)]
#[path = "tests_bitwise_not.rs"]
mod tests;
