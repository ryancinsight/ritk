//! Coeus-native binary morphological opening.
//!
//! Atlas migration target: `super::binary_opening::BinaryMorphologicalOpening`
//! composes the two already substrate-agnostic cores
//! `erode_binary_3d` then `dilate_binary_3d` (pure `&[f32]`/`Vec<f32>`, no
//! Burn dependency) as `open = dilate ∘ erode`. This module reproduces that
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

/// Binary morphological opening (`dilate ∘ erode`) on a Coeus-backed image.
///
/// See [`super::binary_opening::BinaryMorphologicalOpening`] for the
/// mathematical specification and ITK parity; this function computes the
/// identical contract via the identical core routines, only the image
/// boundary differs.
pub fn binary_opening_coeus<B>(
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
        let eroded = erode_binary_3d(vals, dims, radius, foreground_value);
        dilate_binary_3d(&eroded, dims, radius, foreground_value)
    })
}

#[cfg(test)]
#[path = "tests_binary_opening_coeus.rs"]
mod tests;
