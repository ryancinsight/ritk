//! Coeus-native unsigned Euclidean distance transform.
//!
//! The private `euclidean_dt` function (the Meijster–Roerdink–Hesselink core,
//! `#![forbid(unsafe_code)]`, already substrate-agnostic with no Burn
//! dependency) is the same pure algorithm the Burn-generic
//! [`super::DistanceTransformImageFilter::apply`] calls. This module adds a
//! thin Coeus-`Image` boundary around that same core, following the
//! image boundary around that core. No algorithm is duplicated or rewritten.

use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;

use crate::native_support::map_flat_image;

use super::super::types::BinarizationThreshold;
use super::euclidean_dt;

/// Unsigned Euclidean distance transform on a Coeus-backed image.
///
/// See [`super::DistanceTransformImageFilter`] for the mathematical
/// specification and ITK parity; this function computes the identical
/// contract via the identical core routine, only the image boundary differs.
pub fn distance_transform<B>(
    image: &Image<f32, B, 3>,
    threshold: BinarizationThreshold,
    backend: &B,
) -> Result<Image<f32, B, 3>>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let sp = image.spacing();
    let spacing = [sp[0], sp[1], sp[2]];
    let threshold_f32: f32 = threshold.into();

    map_flat_image(image, backend, |vals, dims| {
        let fg: Vec<bool> = vals.iter().map(|&v| v > threshold_f32).collect();
        euclidean_dt(&fg, dims, spacing)
    })
}

/// Signed Euclidean distance transform on a Coeus-backed image.
///
/// Background voxels receive positive distance to foreground; foreground voxels
/// receive negative distance to background. This is the same voxel-centre
/// convention and Meijster core as [`super::SignedDistanceTransformImageFilter`].
pub fn signed_distance_transform<B>(
    image: &Image<f32, B, 3>,
    threshold: BinarizationThreshold,
    backend: &B,
) -> Result<Image<f32, B, 3>>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let spacing = {
        let spacing = image.spacing();
        [spacing[0], spacing[1], spacing[2]]
    };
    let threshold: f32 = threshold.into();

    map_flat_image(image, backend, |values, dims| {
        let foreground: Vec<bool> = values.iter().map(|&value| value > threshold).collect();
        let background: Vec<bool> = foreground.iter().map(|&value| !value).collect();
        let to_foreground = euclidean_dt(&foreground, dims, spacing);
        let to_background = euclidean_dt(&background, dims, spacing);

        foreground
            .iter()
            .zip(to_foreground)
            .zip(to_background)
            .map(|((&is_foreground, to_foreground), to_background)| {
                if is_foreground {
                    -to_background
                } else {
                    to_foreground
                }
            })
            .collect()
    })
}

#[cfg(test)]
#[path = "tests_native.rs"]
mod tests;
