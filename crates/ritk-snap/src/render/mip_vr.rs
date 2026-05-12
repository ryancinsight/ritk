//! 3D MIP/VR volume rendering for ritk-snap.
//!
//! Provides canonical maximum intensity projection (MIP) and simple volume rendering (VR)
//! for the MIP viewport slot. This module is SRP/SoC/SSOT and does not depend on UI code.
//!
//! - Input: 3D volume, window/level, colormap
//! - Output: 2D egui::ColorImage for display

use crate::render::{Colormap, WindowLevel};
use crate::LoadedVolume;
use egui::ColorImage;

/// Render a maximum intensity projection (MIP) along the axial axis (z, depth).
///
/// - `volume`: loaded 3D volume
/// - `wl`: window/level settings
/// - `colormap`: colormap for intensity mapping
/// - Returns: 2D ColorImage (width x height = cols x rows)
pub fn render_mip_axial(volume: &LoadedVolume, wl: WindowLevel, colormap: Colormap) -> ColorImage {
    let shape = volume.shape;
    let (depth, rows, cols) = (shape[0], shape[1], shape[2]);
    let center = wl.center as f32;
    let width = (wl.width as f32).max(1.0);
    let mut pixels = vec![0u8; rows * cols * 4];
    for row in 0..rows {
        for col in 0..cols {
            let mut max_val = f32::MIN;
            for z in 0..depth {
                let v = volume.pixel_at(z, row, col);
                if v > max_val {
                    max_val = v;
                }
            }
            let norm = ((max_val - (center - 0.5 * width)) / width).clamp(0.0, 1.0);
            let rgb = colormap.map(norm);
            let idx = (row * cols + col) * 4;
            pixels[idx] = rgb[0];
            pixels[idx + 1] = rgb[1];
            pixels[idx + 2] = rgb[2];
            pixels[idx + 3] = 255;
        }
    }
    ColorImage::from_rgba_unmultiplied([cols, rows], &pixels)
}

/// Render a simple alpha-blended VR (front-to-back) along the axial axis.
pub fn render_vr_axial(
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: Colormap,
    alpha: f32,
) -> ColorImage {
    let shape = volume.shape;
    let (depth, rows, cols) = (shape[0], shape[1], shape[2]);
    let center = wl.center as f32;
    let width = (wl.width as f32).max(1.0);
    let mut pixels = vec![0u8; rows * cols * 4];
    for row in 0..rows {
        for col in 0..cols {
            let mut accum = [0.0; 4];
            let mut accum_alpha = 0.0;
            for z in 0..depth {
                let v = volume.pixel_at(z, row, col);
                let norm = ((v - (center - 0.5 * width)) / width).clamp(0.0, 1.0);
                let rgb = colormap.map(norm);
                let a = alpha * norm;
                for i in 0..3 {
                    accum[i] += (1.0 - accum_alpha) * (rgb[i] as f32 / 255.0) * a;
                }
                accum_alpha += (1.0 - accum_alpha) * a;
                if accum_alpha >= 0.99 {
                    break;
                }
            }
            let idx = (row * cols + col) * 4;
            for i in 0..3 {
                pixels[idx + i] = (accum[i].clamp(0.0, 1.0) * 255.0) as u8;
            }
            pixels[idx + 3] = (accum_alpha.clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    ColorImage::from_rgba_unmultiplied([cols, rows], &pixels)
}
