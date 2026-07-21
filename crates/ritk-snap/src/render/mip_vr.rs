//! 3D MIP/VR volume rendering for ritk-snap.
//!
//! Provides canonical maximum intensity projection (MIP) and simple volume rendering (VR)
//! for the MIP viewport slot. This module is SRP/SoC/SSOT and does not depend on UI code.
//!
//! - Input: 3D volume, window/level, colormap
//! - Output: 2D egui::ColorImage for display
//!
//! # Zero-allocation variants
//!
//! `render_mip_axial_with_scratch` and `render_vr_axial_with_scratch` are the
//! zero-allocation core implementations. The public `render_mip_axial` and
//! `render_vr_axial` delegate to them with a locally-allocated scratch buffer
//! (used on code paths where no pool is available).
//!
//! # Differential equivalence invariant
//!
//! For all valid inputs:
//! ```text
//! render_mip_axial(v, wl, cm).pixels
//!   == render_mip_axial_with_scratch(&mut s, v, wl, cm).pixels
//! render_vr_axial(v, wl, cm, a).pixels
//!   == render_vr_axial_with_scratch(&mut s, v, wl, cm, a).pixels
//! ```

use crate::render::{NamedColorMap, WindowLevel};
use crate::LoadedVolume;
use egui::ColorImage;
use iris::color::{ColorMap, Normalized};

// ── MIP ───────────────────────────────────────────────────────────────────────

/// Render a maximum intensity projection (MIP) along the axial axis (z, depth).
///
/// Allocates a local scratch buffer internally. Use `render_mip_axial_with_scratch`
/// on hot paths where a pre-allocated scratch buffer is available.
pub fn render_mip_axial(
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: NamedColorMap,
) -> ColorImage {
    let mut scratch = Vec::new();
    render_mip_axial_with_scratch(&mut scratch, volume, wl, colormap)
}

/// Render a MIP into a caller-supplied scratch buffer, eliminating per-call
/// heap allocation on the hot render path.
///
/// `scratch` is resized (never shrunk in capacity) to `rows × cols × 4` bytes
/// before being filled. Callers may pass a `Vec` owned by a [`RenderBufferPool`].
pub(crate) fn render_mip_axial_with_scratch(
    scratch: &mut Vec<u8>,
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: NamedColorMap,
) -> ColorImage {
    let shape = volume.shape;
    let (depth, rows, cols) = (shape[0], shape[1], shape[2]);
    let len = rows * cols * 4;
    scratch.resize(len, 0);
    for row in 0..rows {
        for col in 0..cols {
            let mut max_val = f32::MIN;
            for z in 0..depth {
                let v = volume.pixel_at(z, row, col);
                if v > max_val {
                    max_val = v;
                }
            }
            let norm = Normalized::from_u8(wl.apply(f64::from(max_val)));
            let rgb = colormap.sample(norm).to_rgba8();
            let idx = (row * cols + col) * 4;
            scratch[idx] = rgb[0];
            scratch[idx + 1] = rgb[1];
            scratch[idx + 2] = rgb[2];
            scratch[idx + 3] = 255;
        }
    }
    ColorImage::from_rgba_unmultiplied([cols, rows], scratch)
}

// ── VR ────────────────────────────────────────────────────────────────────────

/// Render a simple alpha-blended VR (front-to-back) along the axial axis.
///
/// Allocates a local scratch buffer internally. Use `render_vr_axial_with_scratch`
/// on hot paths where a pre-allocated scratch buffer is available.
pub fn render_vr_axial(
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: NamedColorMap,
    alpha: f32,
) -> ColorImage {
    let mut scratch = Vec::new();
    render_vr_axial_with_scratch(&mut scratch, volume, wl, colormap, alpha)
}

/// Render a VR into a caller-supplied scratch buffer, eliminating per-call
/// heap allocation on the hot render path.
///
/// `scratch` is resized (never shrunk in capacity) to `rows × cols × 4` bytes
/// before being filled. Callers may pass a `Vec` owned by a [`RenderBufferPool`].
pub(crate) fn render_vr_axial_with_scratch(
    scratch: &mut Vec<u8>,
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: NamedColorMap,
    alpha: f32,
) -> ColorImage {
    let shape = volume.shape;
    let (depth, rows, cols) = (shape[0], shape[1], shape[2]);
    let len = rows * cols * 4;
    scratch.resize(len, 0);
    for row in 0..rows {
        for col in 0..cols {
            let mut accum = [0.0f32; 4];
            let mut accum_alpha = 0.0f32;
            for z in 0..depth {
                let v = volume.pixel_at(z, row, col);
                let norm = Normalized::from_u8(wl.apply(f64::from(v)));
                let rgb = colormap.sample(norm).to_rgba8();
                let a = alpha * norm.get();
                for i in 0..3 {
                    accum[i] += (1.0 - accum_alpha) * (rgb[i] as f32 / super::U8_MAX_F32) * a;
                }
                accum_alpha += (1.0 - accum_alpha) * a;
                if accum_alpha >= 0.99 {
                    break;
                }
            }
            let idx = (row * cols + col) * 4;
            for i in 0..3 {
                scratch[idx + i] = (accum[i].clamp(0.0, 1.0) * super::U8_MAX_F32) as u8;
            }
            scratch[idx + 3] = (accum_alpha.clamp(0.0, 1.0) * super::U8_MAX_F32) as u8;
        }
    }
    ColorImage::from_rgba_unmultiplied([cols, rows], scratch)
}
