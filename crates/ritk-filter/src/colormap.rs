//! Scalar-to-RGB colormap filter (`itk::ScalarToRGBColormapImageFilter`).
//!
//! Maps a scalar image to a 3-component RGB image by linearly normalising each
//! voxel to `[0, 1]` against the image's own min/max
//! (`UseInputImageExtremaForScaling = true`, the ITK default), applying a
//! per-channel colormap function, and quantising to `[0, 255]` by truncation:
//!
//! ```text
//! t      = clamp((v − min) / (max − min), 0, 1)
//! s      = floor(t · 255)                     (C++ uint8 cast truncates)
//! (R,G,B)= colormap(s)
//! ```
//!
//! Only the linear-LUT colormaps are implemented (`Grey`, `Red`, `Green`,
//! `Blue`); the perceptual maps (`Hot`, `Jet`, `Cool`, `HSV`, …) need ITK's
//! piecewise tables and are rejected explicitly rather than approximated.
//!
//! Verified against `sitk.ScalarToRGBColormap`: a `[10,20,30,40,50]` ramp →
//! `Grey` gives `[0,63,127,191,255]` per channel (`0.25·255 = 63.75 → 63`,
//! truncation, not rounding).

use anyhow::{bail, Result};
use burn::tensor::backend::Backend;
use ritk_image::{ColorVolume, Image};
use ritk_tensor_ops::extract_vec;

/// Linear-LUT colormaps (those expressible without a piecewise table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colormap {
    /// Greyscale — `(s, s, s)`. ITK default.
    Grey,
    /// Red ramp — `(s, 0, 0)`.
    Red,
    /// Green ramp — `(0, s, 0)`.
    Green,
    /// Blue ramp — `(0, 0, s)`.
    Blue,
}

impl Colormap {
    /// Parse a case-insensitive colormap name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "grey" | "gray" => Ok(Self::Grey),
            "red" => Ok(Self::Red),
            "green" => Ok(Self::Green),
            "blue" => Ok(Self::Blue),
            other => bail!(
                "ScalarToRGBColormap: colormap '{other}' is not a linear LUT; \
                 only grey/red/green/blue are supported"
            ),
        }
    }

    #[inline]
    fn rgb(self, s: f32) -> [f32; 3] {
        match self {
            Self::Grey => [s, s, s],
            Self::Red => [s, 0.0, 0.0],
            Self::Green => [0.0, s, 0.0],
            Self::Blue => [0.0, 0.0, s],
        }
    }
}

/// Scalar-to-RGB colormap filter.
#[derive(Debug, Clone, Copy)]
pub struct ScalarToRGBColormapFilter {
    colormap: Colormap,
}

impl ScalarToRGBColormapFilter {
    /// Construct with the given colormap.
    pub fn new(colormap: Colormap) -> Self {
        Self { colormap }
    }

    /// Apply the colormap, returning a 3-component RGB image (channel values in
    /// `[0, 255]`).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<ColorVolume<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let (min, max) = vals.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
        let range = max - min;

        let n = vals.len();
        let mut r = vec![0.0f32; n];
        let mut g = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        for (i, &v) in vals.iter().enumerate() {
            let t = if range > 0.0 {
                ((v - min) / range).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let s = (t * 255.0).floor();
            let [cr, cg, cb] = self.colormap.rgb(s);
            r[i] = cr;
            g[i] = cg;
            b[i] = cb;
        }

        ColorVolume::<B, 3>::from_component_buffers(
            &[r, g, b],
            dims,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            &image.data().device(),
        )
    }
}

/// ITK `LabelToRGBImageFilter` default 30-colour table (labels `1..=30`; label
/// `k ≥ 1` maps to `LABEL_COLORS[(k − 1) mod 30]`). Extracted from
/// `sitk.LabelToRGB`.
const LABEL_COLORS: [[f32; 3]; 30] = [
    [0.0, 205.0, 0.0],
    [0.0, 0.0, 255.0],
    [0.0, 255.0, 255.0],
    [255.0, 0.0, 255.0],
    [255.0, 127.0, 0.0],
    [0.0, 100.0, 0.0],
    [138.0, 43.0, 226.0],
    [139.0, 35.0, 35.0],
    [0.0, 0.0, 128.0],
    [139.0, 139.0, 0.0],
    [255.0, 62.0, 150.0],
    [139.0, 76.0, 57.0],
    [0.0, 134.0, 139.0],
    [205.0, 104.0, 57.0],
    [191.0, 62.0, 255.0],
    [0.0, 139.0, 69.0],
    [199.0, 21.0, 133.0],
    [205.0, 55.0, 0.0],
    [32.0, 178.0, 170.0],
    [106.0, 90.0, 205.0],
    [255.0, 20.0, 147.0],
    [69.0, 139.0, 116.0],
    [72.0, 118.0, 255.0],
    [205.0, 79.0, 57.0],
    [0.0, 0.0, 205.0],
    [139.0, 34.0, 82.0],
    [139.0, 0.0, 139.0],
    [238.0, 130.0, 238.0],
    [139.0, 0.0, 0.0],
    [255.0, 0.0, 0.0],
];

/// Map a label image to RGB using ITK's default label-colour table
/// (`itk::LabelToRGBImageFilter` / `sitk.LabelToRGB`).
///
/// Background voxels (those equal to `background`, default `0`) map to black;
/// every other label `k` maps to `LABEL_COLORS[(k − 1) mod 30]`, cycling through
/// the 30-colour table.
#[derive(Debug, Clone, Copy)]
pub struct LabelToRGBFilter {
    background: i64,
}

impl LabelToRGBFilter {
    /// Construct with the given background label (default ITK value `0`).
    pub fn new(background: i64) -> Self {
        Self { background }
    }

    /// Apply the label-to-RGB mapping, returning a 3-component RGB image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<ColorVolume<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let n = vals.len();
        let (mut r, mut g, mut b) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        for (i, &v) in vals.iter().enumerate() {
            let lbl = v.round() as i64;
            if lbl == self.background {
                continue; // black
            }
            let idx = (lbl - 1).rem_euclid(LABEL_COLORS.len() as i64) as usize;
            let [cr, cg, cb] = LABEL_COLORS[idx];
            r[i] = cr;
            g[i] = cg;
            b[i] = cb;
        }
        ColorVolume::<B, 3>::from_component_buffers(
            &[r, g, b],
            dims,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            &image.data().device(),
        )
    }
}

#[cfg(test)]
#[path = "tests_colormap.rs"]
mod tests_colormap;
