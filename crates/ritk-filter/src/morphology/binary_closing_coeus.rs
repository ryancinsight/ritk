//! Coeus-native binary morphological closing.
//!
//! Atlas migration target: `super::binary_closing::BinaryMorphologicalClosing`
//! composes the two already substrate-agnostic cores
//! `dilate_binary_3d` then `erode_binary_3d` (pure `&[f32]`/`Vec<f32>`, no
//! Burn dependency) as `close = erode ∘ dilate`. This module reproduces that
//! exact composition on the flat buffer through
//! `crate::coeus_support::map_flat_image`, following the `binary_erode_coeus`
//! template: production Burn API stays, a Coeus-native equivalent is added
//! and verified against it, no algorithm is duplicated or rewritten.

use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::coeus::Image;

use crate::coeus_support::map_flat_image;

use super::binary_dilate::dilate_binary_3d;
use super::binary_erode::erode_binary_3d;
use super::types::ForegroundValue;

/// Binary morphological closing (`erode ∘ dilate`) on a Coeus-backed image.
///
/// See [`super::binary_closing::BinaryMorphologicalClosing`] for the
/// mathematical specification and ITK parity; this function computes the
/// identical contract via the identical core routines, only the image
/// boundary differs.
pub fn binary_closing_coeus<B>(
    image: &Image<f32, B, 3>,
    radius: usize,
    foreground_value: ForegroundValue,
    backend: &B,
) -> Result<Image<f32, B, 3>>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    map_flat_image(image, backend, |vals, dims| {
        let dilated = dilate_binary_3d(vals, dims, radius, foreground_value);
        erode_binary_3d(&dilated, dims, radius, foreground_value)
    })
}

#[cfg(test)]
#[path = "tests_binary_closing_coeus.rs"]
mod tests;
