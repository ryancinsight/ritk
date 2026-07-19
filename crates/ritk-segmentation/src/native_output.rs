use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;

pub(crate) fn from_values<B, const D: usize>(
    source: &Image<f32, B, D>,
    values: Vec<f32>,
    backend: &B,
) -> anyhow::Result<Image<f32, B, D>>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    Image::from_flat_on(
        values,
        source.shape(),
        *source.origin(),
        *source.spacing(),
        *source.direction(),
        backend,
    )
}
