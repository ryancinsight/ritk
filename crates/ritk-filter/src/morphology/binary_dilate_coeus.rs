//! Coeus-native binary dilation.
//!
//! Atlas migration target: `super::binary_dilate::dilate_binary_3d` (already
//! substrate-agnostic — pure `&[f32]` in, `Vec<f32>` out, separable 1-D
//! sweeps, no Burn dependency) is the same core the Burn-generic
//! `super::binary_dilate::BinaryDilateFilter::apply` calls. This module adds
//! a thin Coeus-`Image` boundary around that same core via
//! `crate::coeus_support::map_flat_image`, following the
//! `binary_erode_coeus` template: production Burn API stays, a Coeus-native
//! equivalent is added and verified against it, no algorithm is duplicated
//! or rewritten.

use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::coeus::Image;

use crate::coeus_support::map_flat_image;

use super::binary_dilate::dilate_binary_3d;
use super::types::ForegroundValue;

/// Binary dilation on a Coeus-backed image.
///
/// See [`super::binary_dilate::BinaryDilateFilter`] for the mathematical
/// specification, boundary handling, and ITK parity; this function computes
/// the identical contract via the identical core routine, only the image
/// boundary differs.
pub fn binary_dilate_coeus<B>(
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
        dilate_binary_3d(vals, dims, radius, foreground_value)
    })
}

#[cfg(test)]
#[path = "tests_binary_dilate_coeus.rs"]
mod tests;
