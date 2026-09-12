//! Scalar-volume upload and device-limit handling.

use std::sync::Arc;

use wgpu::util::DeviceExt as _;

use crate::LoadedVolume;

use super::volume_limits::{volume_fits_device, volume_storage_size};
use super::GpuVolumeRenderer;

impl GpuVolumeRenderer {
    /// Upload `volume` to a GPU storage buffer if the volume has changed since
    /// the last call.
    ///
    /// Change detection uses the `Arc` raw pointer of `volume.data` and the
    /// shape dimensions. If both are identical to the cached values, the
    /// upload is skipped.
    ///
    /// # Zero-copy single-channel path
    ///
    /// When `volume.channels == 1` the raw `Arc<Vec<f32>>` slice is cast
    /// directly to bytes without any CPU extraction loop.
    ///
    /// # Multi-channel path
    ///
    /// When `volume.channels > 1` the first channel is extracted in parallel
    /// using Rayon before uploading.
    ///
    /// Returns `false` when the device cannot bind the complete scalar volume;
    /// callers then use the CPU renderer rather than reaching a wgpu validation
    /// panic during buffer creation.
    pub(super) fn ensure_volume_uploaded(&mut self, volume: &LoadedVolume) -> bool {
        let ptr = Arc::as_ptr(&volume.data) as usize;
        if self.vol_data_ptr == Some(ptr) && self.vol_size == Some(volume.shape) {
            return true;
        }

        let [depth, rows, cols] = volume.shape;
        let ch = volume.channels as usize;
        let Some((n_voxels, byte_count)) = volume_storage_size(volume.shape) else {
            tracing::error!(?volume.shape, "volume dimensions overflow GPU buffer size");
            return false;
        };
        let limits = self.ctx.device.limits();
        if !volume_fits_device(volume.shape, &limits) {
            tracing::warn!(
                ?volume.shape,
                bytes = byte_count,
                max_buffer_size = limits.max_buffer_size,
                max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size,
                "volume exceeds GPU storage-buffer limits; using CPU renderer"
            );
            return false;
        }
        let raw = &*volume.data;

        // Zero-copy path for single-channel volumes: the data is already in
        // [depth, rows, cols] row-major order — no extraction needed.
        let extracted: Option<Vec<f32>> = if ch == 1 && raw.len() >= n_voxels {
            None
        } else {
            // Multi-channel: extract first channel in parallel with Rayon.
            // Voxel at linear index `lin` has first-channel value at raw[lin * ch].
            Some(moirai::map_collect_index_with::<moirai::Adaptive, _, _>(
                n_voxels,
                |lin| *raw.get(lin * ch).unwrap_or(&0.0),
            ))
        };

        let slice: &[f32] = match extracted.as_deref() {
            Some(s) => s,
            None => &raw[..n_voxels],
        };

        let buf = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_vol_data"),
                contents: bytemuck::cast_slice(slice),
                usage: wgpu::BufferUsages::STORAGE,
            });

        self.vol_buffer = Some(buf);
        self.vol_size = Some(volume.shape);
        self.vol_data_ptr = Some(ptr);

        tracing::debug!(
            depth,
            rows,
            cols,
            ch,
            bytes = byte_count,
            "Volume uploaded to GPU"
        );
        true
    }
}
