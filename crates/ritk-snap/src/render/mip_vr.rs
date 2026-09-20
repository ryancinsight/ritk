//! Scalar axial maximum-intensity and volume-rendered projections.
//!
//! Projection math is host-neutral: RITK produces bounded RGBA storage and
//! the optional eframe shell adapts that storage to its image carrier.

use crate::render::{GrayscalePresentation, NamedColorMap, RgbaImage, WindowLevel};
use crate::LoadedVolume;
use iris::color::{ColorMap, Normalized};

/// Render a scalar axial maximum-intensity projection into owned RGBA storage.
pub(crate) fn render_mip_axial_rgba(
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: NamedColorMap,
) -> RgbaImage {
    let mut scratch = mnemosyne::AlignedVec::default();
    render_mip_axial_rgba_with_scratch(&mut scratch, volume, wl, colormap)
}

/// Render an axial maximum-intensity projection using caller-owned scratch.
pub(crate) fn render_mip_axial_rgba_with_scratch(
    scratch: &mut mnemosyne::AlignedVec<u8>,
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: NamedColorMap,
) -> RgbaImage {
    if volume.channels != 1 {
        return unsupported_projection_image(volume.channels);
    }
    let Some(presentation) = valid_presentation(volume) else {
        return invalid_image();
    };
    let [depth, rows, cols] = volume.shape;
    scratch.resize(rows.saturating_mul(cols).saturating_mul(4), 0);
    for row in 0..rows {
        for col in 0..cols {
            let mut max_val = f32::MIN;
            for z in 0..depth {
                max_val = max_val.max(volume.pixel_at(z, row, col));
            }
            let [red, green, blue, alpha] = map_scalar_value(max_val, presentation, wl, colormap);
            let index = (row * cols + col) * 4;
            scratch[index..index + 4].copy_from_slice(&[red, green, blue, alpha]);
        }
    }
    RgbaImage::new([cols, rows], scratch.to_vec())
}

pub(crate) fn map_scalar_value(
    value: f32,
    presentation: GrayscalePresentation,
    wl: WindowLevel,
    colormap: NamedColorMap,
) -> [u8; 4] {
    let norm = Normalized::from_u8(presentation.apply(wl, f64::from(value)));
    let [red, green, blue, _] = colormap.sample(norm).to_rgba8();
    [red, green, blue, 255]
}

/// Render a scalar axial front-to-back volume projection into RGBA storage.
#[cfg(feature = "eframe-shell")]
pub(crate) fn render_vr_axial_rgba(
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: NamedColorMap,
    alpha: f32,
) -> RgbaImage {
    let mut scratch = mnemosyne::AlignedVec::default();
    render_vr_axial_rgba_with_scratch(&mut scratch, volume, wl, colormap, alpha)
}

/// Render an axial volume projection using caller-owned scratch.
#[cfg(feature = "eframe-shell")]
pub(crate) fn render_vr_axial_rgba_with_scratch(
    scratch: &mut mnemosyne::AlignedVec<u8>,
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: NamedColorMap,
    alpha: f32,
) -> RgbaImage {
    if volume.channels != 1 {
        return unsupported_projection_image(volume.channels);
    }
    let Some(presentation) = valid_presentation(volume) else {
        return invalid_image();
    };
    let [depth, rows, cols] = volume.shape;
    scratch.resize(rows.saturating_mul(cols).saturating_mul(4), 0);
    for row in 0..rows {
        for col in 0..cols {
            let mut accum = [0.0_f32; 3];
            let mut accum_alpha = 0.0_f32;
            for z in 0..depth {
                let value = volume.pixel_at(z, row, col);
                let norm = Normalized::from_u8(presentation.apply(wl, f64::from(value)));
                let [red, green, blue, _] = colormap.sample(norm).to_rgba8();
                let sample_alpha = alpha * norm.get();
                let visibility = 1.0 - accum_alpha;
                accum[0] += visibility * (f32::from(red) / super::U8_MAX_F32) * sample_alpha;
                accum[1] += visibility * (f32::from(green) / super::U8_MAX_F32) * sample_alpha;
                accum[2] += visibility * (f32::from(blue) / super::U8_MAX_F32) * sample_alpha;
                accum_alpha += visibility * sample_alpha;
                if accum_alpha >= 0.99 {
                    break;
                }
            }
            let index = (row * cols + col) * 4;
            scratch[index] = channel_to_byte(accum[0]);
            scratch[index + 1] = channel_to_byte(accum[1]);
            scratch[index + 2] = channel_to_byte(accum[2]);
            scratch[index + 3] = channel_to_byte(accum_alpha);
        }
    }
    RgbaImage::new([cols, rows], scratch.to_vec())
}

fn valid_presentation(volume: &LoadedVolume) -> Option<GrayscalePresentation> {
    match GrayscalePresentation::for_volume(volume) {
        Ok(presentation) => Some(presentation),
        Err(error) => {
            tracing::error!(%error, "invalid DICOM grayscale presentation metadata");
            None
        }
    }
}

#[cfg(feature = "eframe-shell")]
fn channel_to_byte(value: f32) -> u8 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "value is clamped to byte range"
    )]
    let byte = (value.clamp(0.0, 1.0) * super::U8_MAX_F32) as u8;
    byte
}

fn unsupported_projection_image(channels: u8) -> RgbaImage {
    tracing::error!(channels, "3D projection requires a scalar volume");
    invalid_image()
}

fn invalid_image() -> RgbaImage {
    RgbaImage::new([1, 1], vec![255_u8, 0, 255, 255])
}

#[cfg(feature = "eframe-shell")]
/// Render a maximum-intensity projection for the eframe shell.
pub fn render_mip_axial(
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: NamedColorMap,
) -> egui::ColorImage {
    render_mip_axial_rgba(volume, wl, colormap).to_color_image()
}

#[cfg(feature = "eframe-shell")]
/// Render a maximum-intensity projection using eframe-owned scratch.
pub(crate) fn render_mip_axial_with_scratch(
    scratch: &mut mnemosyne::AlignedVec<u8>,
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: NamedColorMap,
) -> egui::ColorImage {
    render_mip_axial_rgba_with_scratch(scratch, volume, wl, colormap).to_color_image()
}

#[cfg(feature = "eframe-shell")]
/// Render a volume-rendered projection for the eframe shell.
pub fn render_vr_axial(
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: NamedColorMap,
    alpha: f32,
) -> egui::ColorImage {
    render_vr_axial_rgba(volume, wl, colormap, alpha).to_color_image()
}

#[cfg(feature = "eframe-shell")]
/// Render a volume-rendered projection using eframe-owned scratch.
pub(crate) fn render_vr_axial_with_scratch(
    scratch: &mut mnemosyne::AlignedVec<u8>,
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: NamedColorMap,
    alpha: f32,
) -> egui::ColorImage {
    render_vr_axial_rgba_with_scratch(scratch, volume, wl, colormap, alpha).to_color_image()
}
