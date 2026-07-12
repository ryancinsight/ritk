use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;

pub(super) fn from_values<B>(
    source: &Image<f32, B, 3>,
    values: Vec<f32>,
    backend: &B,
) -> anyhow::Result<Image<f32, B, 3>>
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
