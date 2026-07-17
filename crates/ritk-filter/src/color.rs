//! Per-component filtering for multi-component (RGB/vector) volumes.
//!
//! ITK's vector-image filters that support multi-component pixels apply the
//! corresponding scalar filter **independently to each component** (verified
//! bit-exact against `sitk.Median` on a `VectorFloat32` image). This module
//! provides the adaptor that realises that contract: it deinterleaves a
//! [`ColorVolume`] into its `C` scalar component buffers, wraps each as a scalar
//! [`Image`], applies the caller's scalar filter, and re-interleaves the
//! results — so the entire scalar filter library is reusable on color images
//! with no per-filter reimplementation.

use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::ColorVolume;
use ritk_image::native::Image;
use ritk_tensor_ops::native::extract_image_vec;

/// Apply a scalar 3-D image filter independently to each component of `vol`,
/// matching ITK's per-component vector-image filtering.
///
/// The closure receives a scalar [`Image`] view of one component (carrying the
/// volume's spatial metadata) and returns the filtered scalar image of the same
/// spatial shape. Component order and the `[depth, rows, cols, channel]` layout
/// are preserved.
///
/// # Errors
/// Propagates the recombination error from
/// [`ColorVolume::from_component_buffers`] (e.g. if a filter changed the spatial
/// shape, which per-component filters must not do).
pub fn map_color_components<B, const C: usize, F>(
    vol: &ColorVolume<f32, B, C>,
    mut f: F,
    backend: &B::default()) -> Result<ColorVolume<f32, B, C>>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    F: FnMut(&Image<f32, B, 3>) -> Image<f32, B, 3>,
{
    let spatial = vol.spatial_shape();
    let (origin, spacing, direction) = (*vol.origin(), *vol.spacing(), *vol.direction());

    let bufs = vol.into_component_buffers();
    let mut out_bufs: Vec<Vec<f32>> = Vec::with_capacity(C);
    for buf in bufs {
        let tensor = coeus_tensor::Tensor::<f32, B>::from_slice_on(spatial, &buf, backend);
        let img = Image::<f32, B, 3>::new(tensor, origin, spacing, direction)?;
        let result = f(&img);
        let (rvals, _) = extract_image_vec(&result)?;
        out_bufs.push(rvals);
    }
    ColorVolume::from_component_buffers(
        &out_bufs,
        spatial,
        origin,
        spacing,
        direction,
        backend,
    )
}

#[cfg(test)]
#[path = "tests_color.rs"]
mod tests_color;
