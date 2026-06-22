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

use std::sync::{Arc, OnceLock};
use wgpu::{Device, InstanceDescriptor, Queue, RequestAdapterOptions};

static SHARED_DEVICE_QUEUE: OnceLock<Option<(Arc<Device>, Arc<Queue>)>> = OnceLock::new();

/// Fetch the shared global `wgpu::Device` and `wgpu::Queue` pair, initializing
/// them once on the first call. Subsequent calls reuse the same device context
/// to avoid driver resource limits and access violations.
pub(crate) fn get_shared_device_queue() -> Option<(Arc<Device>, Arc<Queue>)> {
    SHARED_DEVICE_QUEUE
        .get_or_init(|| {
            pollster::block_on(async {
                let backends = match std::env::var("WGPU_BACKEND").as_deref() {
                    Ok("dx12") => wgpu::Backends::DX12,
                    Ok("vulkan") => wgpu::Backends::VULKAN,
                    Ok("metal") => wgpu::Backends::METAL,
                    Ok("gl") => wgpu::Backends::GL,
                    Ok("browser") => wgpu::Backends::BROWSER_WEBGPU,
                    _ => wgpu::Backends::all(),
                };
                let instance = wgpu::Instance::new(InstanceDescriptor {
                    backends,
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
                Some((Arc::new(device), Arc::new(queue)))
            })
        })
        .clone()
}

/// Owned wgpu device and queue for GPU compute operations.
#[derive(Clone)]
pub(super) struct GpuContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

impl GpuContext {
    /// Attempt to create a compute-capable GPU context.
    ///
    /// Returns `None` on any failure (no GPU, feature mismatch, driver error).
    /// Callers must fall back to CPU rendering when this returns `None`.
    pub fn try_new() -> Option<Self> {
        let (device, queue) = get_shared_device_queue()?;
        Some(GpuContext { device, queue })
    }
}
