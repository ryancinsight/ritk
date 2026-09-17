//! Frame composition and capture encoding for the native Métis session.

use super::layout::{surface_frames, surface_frames_with_mip};
use super::{NativeViewport, RenderedProjection, RenderedView};
use crate::launch::NativePresentationMode;
use crate::tools::interaction::ViewportOffset;
use anyhow::{anyhow, Context, Result};
use metis_platform::Framebuffer;
use std::path::Path;

pub(super) fn compose_frames(
    views: &[RenderedView; 3],
    projection: Option<&RenderedProjection>,
    presentation_mode: NativePresentationMode,
    surface_width: u32,
    surface_height: u32,
    zoom: f32,
    pan_offset: ViewportOffset,
    cine_enabled: bool,
    cine_fps: f32,
    show_application_overlay: bool,
) -> Result<(Framebuffer, [NativeViewport; 3])> {
    match (presentation_mode, projection) {
        (NativePresentationMode::Orthogonal, None) => surface_frames(
            views,
            surface_width,
            surface_height,
            zoom,
            pan_offset,
            cine_enabled,
            cine_fps,
            show_application_overlay,
        ),
        (NativePresentationMode::OrthogonalWithMip, Some(projection)) => surface_frames_with_mip(
            views,
            projection,
            surface_width,
            surface_height,
            zoom,
            pan_offset,
            cine_enabled,
            cine_fps,
            show_application_overlay,
        ),
        (NativePresentationMode::Orthogonal, Some(_))
        | (NativePresentationMode::OrthogonalWithMip, None) => Err(anyhow!(
            "native presentation mode and projection state disagree"
        )),
    }
}

pub(super) fn save_capture(framebuffer: &Framebuffer, output: &Path) -> Result<()> {
    let pixel_count =
        usize::try_from(u64::from(framebuffer.width()) * u64::from(framebuffer.height()))
            .map_err(|_| anyhow!("native capture pixel count exceeds usize"))?;
    if framebuffer.pixels().len() != pixel_count {
        return Err(anyhow!(
            "native capture framebuffer storage is inconsistent"
        ));
    }
    let byte_count = pixel_count
        .checked_mul(4)
        .ok_or_else(|| anyhow!("native capture byte count overflows usize"))?;
    let mut rgba = Vec::new();
    rgba.try_reserve_exact(byte_count)
        .map_err(|_| anyhow!("unable to reserve native capture bytes"))?;
    for packed in framebuffer.pixels() {
        let [alpha, red, green, blue] = packed.to_be_bytes();
        rgba.extend_from_slice(&[red, green, blue, alpha]);
    }
    let pixels = image::RgbaImage::from_raw(framebuffer.width(), framebuffer.height(), rgba)
        .ok_or_else(|| anyhow!("native capture RGBA dimensions do not match the framebuffer"))?;
    pixels
        .save_with_format(output, image::ImageFormat::Png)
        .with_context(|| format!("write native Métis capture to {}", output.display()))
}
