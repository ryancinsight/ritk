//! Coeus-native unsigned Euclidean distance transform.
//!
//! Atlas migration target: [`super::euclidean_dt`] (the Meijster–Roerdink–
//! Hesselink core, `#![forbid(unsafe_code)]`, already substrate-agnostic —
//! no Burn dependency) is the same pure algorithm the Burn-generic
//! [`super::DistanceTransformImageFilter::apply`] calls. This module adds a
//! thin Coeus-`Image` boundary around that same core, following the
//! `read_jpeg_coeus`/`trilinear_interpolation_coeus` template: production
//! Burn API stays, a Coeus-native equivalent is added and verified against
//! it, no algorithm is duplicated or rewritten.

use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::coeus::Image;

use super::super::types::BinarizationThreshold;
use super::euclidean_dt;

/// Unsigned Euclidean distance transform on a Coeus-backed image.
///
/// See [`super::DistanceTransformImageFilter`] for the mathematical
/// specification and ITK parity; this function computes the identical
/// contract via the identical core routine, only the image boundary differs.
pub fn distance_transform_coeus<B>(
    image: &Image<f32, B, 3>,
    threshold: BinarizationThreshold,
    backend: &B,
) -> Result<Image<f32, B, 3>>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let dims = image.shape();
    let vals = image.data_slice()?;
    let threshold_f32: f32 = threshold.into();
    let fg: Vec<bool> = vals.iter().map(|&v| v > threshold_f32).collect();

    let sp = image.spacing();
    let spacing = [sp[0], sp[1], sp[2]];
    let result = euclidean_dt(&fg, dims, spacing);

    Image::from_flat_on(
        result,
        dims,
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        backend,
    )
}

#[cfg(test)]
#[path = "tests_unsigned_coeus.rs"]
mod tests;
