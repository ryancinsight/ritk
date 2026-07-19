//! Shared Coeus-native `Image` boundary helpers for this crate's flat-buffer
//! segmentation cores.

use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;

/// Apply a pure flat-buffer transform to a Coeus-native image, preserving its
/// shape and spatial metadata.
pub(crate) fn map_flat_image<B, F>(
    image: &Image<f32, B, 3>,
    backend: &B,
    f: F,
) -> Result<Image<f32, B, 3>>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    F: FnOnce(&[f32], [usize; 3]) -> Vec<f32>,
{
    let dims = image.shape();
    let vals = image.data_slice()?;
    let result = f(vals, dims);
    Image::from_flat_on(
        result,
        dims,
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        backend,
    )
}

/// Apply a pure two-input flat-buffer transform to a pair of shape-matched
/// Coeus-native images, preserving the primary image's metadata.
pub(crate) fn map_flat_pair<B, F>(
    primary: &Image<f32, B, 3>,
    secondary: &Image<f32, B, 3>,
    backend: &B,
    f: F,
) -> Result<Image<f32, B, 3>>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    F: FnOnce(&[f32], &[f32], [usize; 3]) -> Vec<f32>,
{
    let dims = primary.shape();
    anyhow::ensure!(
        dims == secondary.shape(),
        "native two-input filter: shape mismatch {:?} vs {:?}",
        dims,
        secondary.shape()
    );
    let a = primary.data_slice()?;
    let b = secondary.data_slice()?;
    let result = f(a, b, dims);
    Image::from_flat_on(
        result,
        dims,
        *primary.origin(),
        *primary.spacing(),
        *primary.direction(),
        backend,
    )
}
