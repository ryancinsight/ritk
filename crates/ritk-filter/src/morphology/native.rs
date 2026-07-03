//! Atlas-native-substrate binary-morphology wrappers (transitional module:
//! plain end-state names, disambiguated from the Burn filters by module path
//! only; folds away when the Burn path is deleted — ADR 0002 A1).
//!
//! Each wrapper marshals a [`ritk_image::native::Image`] boundary around the
//! same substrate-agnostic core its Burn counterpart calls, via
//! `crate::native_support::map_flat_image` — generic over
//! `B: ComputeBackend`, statically dispatched, zero-cost.

use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;

use crate::native_support::map_flat_image;

use super::binary_dilate::dilate_binary_3d;
use super::binary_erode::erode_binary_3d;
use super::types::ForegroundValue;

/// Binary erosion on a Coeus-backed image.
///
/// See [`super::binary_erode::BinaryErodeFilter`] for the mathematical
/// specification, boundary handling, and ITK parity; this function computes
/// the identical contract via the identical core routine, only the image
/// boundary differs.
pub fn binary_erode<B>(
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
        erode_binary_3d(vals, dims, radius, foreground_value)
    })
}

/// Binary dilation on a Coeus-backed image.
///
/// See [`super::binary_dilate::BinaryDilateFilter`] for the mathematical
/// specification, boundary handling, and ITK parity; this function computes
/// the identical contract via the identical core routine, only the image
/// boundary differs.
pub fn binary_dilate<B>(
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

/// Binary morphological closing (`erode ∘ dilate`) on a Coeus-backed image.
///
/// See [`super::binary_closing::BinaryMorphologicalClosing`] for the
/// mathematical specification and ITK parity; this function computes the
/// identical contract via the identical core routines, only the image
/// boundary differs.
pub fn binary_closing<B>(
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

/// Binary morphological opening (`dilate ∘ erode`) on a Coeus-backed image.
///
/// See [`super::binary_opening::BinaryMorphologicalOpening`] for the
/// mathematical specification and ITK parity; this function computes the
/// identical contract via the identical core routines, only the image
/// boundary differs.
pub fn binary_opening<B>(
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
#[path = "tests_native.rs"]
mod tests;
