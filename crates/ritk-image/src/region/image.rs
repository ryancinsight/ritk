use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage, Scalar};

use super::voxel::VoxelRegion;
use crate::types::Image;

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    /// Borrow the whole image as a [`VoxelRegion`].
    ///
    /// The zero-copy entry point to the region seam: no voxel data is read,
    /// copied, or allocated, and the returned region carries the image's own
    /// physical metadata.
    ///
    /// Unlike [`Self::data_slice`], this accepts a strided image. A permuted or
    /// sliced tensor has no flat host slice, so the only previous route to its
    /// values was [`Self::data_cow`], which materialises the whole volume; a
    /// region reads the same values in place through the layout's strides.
    /// [`VoxelRegion::as_slice`] still returns `None` for such a region, so the
    /// distinction stays visible where it matters.
    ///
    /// # Errors
    ///
    /// Returns an error when the backing tensor's rank does not match `D`.
    /// Construction validates this, so a well-formed image never fails here.
    pub fn region(&self) -> Result<VoxelRegion<'_, T, D>> {
        let tensor = self.data();
        let layout = tensor.layout();
        let shape: [usize; D] = layout.shape().try_into().map_err(|_| {
            anyhow::anyhow!(
                "image tensor rank {} does not match D={D}",
                layout.shape().len()
            )
        })?;
        let strides: [usize; D] = layout
            .strides()
            .try_into()
            .map_err(|_| anyhow::anyhow!("image tensor strides do not match D={D}"))?;

        Ok(VoxelRegion {
            data: tensor.storage().as_slice(),
            shape,
            strides,
            offset: layout.offset(),
            origin: *self.origin(),
            spacing: *self.spacing(),
            direction: *self.direction(),
        })
    }
}
