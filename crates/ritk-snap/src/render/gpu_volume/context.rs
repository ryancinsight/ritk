//! wgpu device + queue initialization for headless GPU compute.
//!
//! `GpuContext::try_new()` acquires a wgpu adapter and device suitable for
//! compute work on native targets. Returns `None` when no compatible GPU is
//! available (headless CI, VM without compute, or driver error).
//!
//! # Contract
//!
//! - Does not require a window surface (headless compute only).
//! - Uses `pollster::block_on` for synchronous init.
//! - The resulting device supports `BufferBindingType::Storage` (base wgpu
//!   feature set, no additional features required).

use wgpu::{Device, InstanceDescriptor, Queue, RequestAdapterOptions};

/// Owned wgpu device and queue for GPU compute operations.
pub(super) struct GpuContext {
    pub device: Device,
    pub queue: Queue,
}

impl GpuContext {
    /// Attempt to create a compute-capable GPU context.
    ///
    /// Returns `None` on any failure (no GPU, feature mismatch, driver error).
    /// Callers must fall back to CPU rendering when this returns `None`.
    pub fn try_new() -> Option<Self> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            let adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .ok()?;
            Some(GpuContext { device, queue })
        })
    }
}
