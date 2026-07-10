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

#[cfg(test)]
#[path = "tests_native.rs"]
mod tests;
