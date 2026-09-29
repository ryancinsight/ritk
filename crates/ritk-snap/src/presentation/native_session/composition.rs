//! Frame composition and capture encoding for the native Métis session.

use super::layout::{
    surface_frames, surface_frames_responsive, surface_frames_with_oblique,
    surface_frames_with_projection,
};
use super::{NativeViewport, ObliqueView, RenderedProjection, RenderedView};
use crate::app::ObliqueViewport;
use crate::launch::{NativePresentationMode, NativePresentationSelection};
use crate::presentation::PaneLayout;
use crate::tools::interaction::ViewportOffset;
use anyhow::{anyhow, Context, Result};
use metis_platform::Framebuffer;
use std::path::Path;

pub(super) fn compose_frames(
    views: &[RenderedView; 3],
    projection: Option<&RenderedProjection>,
    oblique: Option<&ObliqueView>,
    presentation_mode: NativePresentationSelection,
    surface_width: u32,
    surface_height: u32,
    zoom: f32,
    pan_offset: ViewportOffset,
    cine_enabled: bool,
    cine_fps: f32,
    show_application_overlay: bool,
) -> Result<(Framebuffer, [NativeViewport; 3], Option<ObliqueViewport>)> {
    compose_selected_frames(
        views,
        projection,
        oblique,
        presentation_mode,
        surface_width,
        surface_height,
        zoom,
        pan_offset,
        cine_enabled,
        cine_fps,
        show_application_overlay,
    )
}

pub(super) fn compose_selected_frames(
    views: &[RenderedView; 3],
    projection: Option<&RenderedProjection>,
    oblique: Option<&ObliqueView>,
    presentation_mode: NativePresentationSelection,
    surface_width: u32,
    surface_height: u32,
    zoom: f32,
    pan_offset: ViewportOffset,
    cine_enabled: bool,
    cine_fps: f32,
    show_application_overlay: bool,
) -> Result<(Framebuffer, [NativeViewport; 3], Option<ObliqueViewport>)> {
    match presentation_mode {
        NativePresentationSelection::Responsive => {
            let projection = projection
                .ok_or_else(|| anyhow!("responsive native layout has no projection frame"))?;
            if oblique.is_some() {
                return Err(anyhow!(
                    "responsive native layout cannot include an oblique frame"
                ));
            }
            let (frame, viewports) = surface_frames_responsive(
                views,
                Some(projection),
                PaneLayout::responsive(surface_width, surface_height),
                surface_width,
                surface_height,
                zoom,
                pan_offset,
                cine_enabled,
                cine_fps,
                show_application_overlay,
            )?;
            Ok((frame, viewports, None))
        }
        NativePresentationSelection::Oblique => {
            let oblique = oblique
                .ok_or_else(|| anyhow!("oblique native layout has no physical reslice frame"))?;
            if projection.is_some() {
                return Err(anyhow!(
                    "oblique native layout cannot include a projection frame"
                ));
            }
            let plane = oblique.plane.as_ref();
            let (frame, viewports, oblique_viewport) = surface_frames_with_oblique(
                views,
                &oblique.frame,
                oblique.orientation.yaw_degrees(),
                oblique.orientation.pitch_degrees(),
                surface_width,
                surface_height,
                zoom,
                pan_offset,
                oblique.zoom,
                oblique.pan_offset,
                plane,
                cine_enabled,
                cine_fps,
                show_application_overlay,
            )?;
            if let (Some(plane), Some(viewport)) = (plane, oblique_viewport) {
                viewport
                    .validate_dimensions(plane.dimensions())
                    .map_err(|error| anyhow!("validate native oblique viewport: {error}"))?;
                Ok((frame, viewports, Some(viewport)))
            } else {
                Ok((frame, viewports, None))
            }
        }
        NativePresentationSelection::Fixed(mode) => match (mode, projection, oblique) {
            (NativePresentationMode::Orthogonal, None, None) => {
                let (frame, viewports) = surface_frames(
                    views,
                    surface_width,
                    surface_height,
                    zoom,
                    pan_offset,
                    cine_enabled,
                    cine_fps,
                    show_application_overlay,
                )?;
                Ok((frame, viewports, None))
            }
            (
                NativePresentationMode::OrthogonalWithMip
                | NativePresentationMode::OrthogonalWithMinip
                | NativePresentationMode::OrthogonalWithAverage,
                Some(projection),
                None,
            ) => {
                let (frame, viewports) = surface_frames_with_projection(
                    views,
                    projection,
                    surface_width,
                    surface_height,
                    zoom,
                    pan_offset,
                    cine_enabled,
                    cine_fps,
                    show_application_overlay,
                )?;
                Ok((frame, viewports, None))
            }
            _ => Err(anyhow!(
                "native presentation selection and rendered panels disagree"
            )),
        },
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
