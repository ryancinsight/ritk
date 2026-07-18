//! Overlay composition state for annotation rendering.
//!
//! Mathematical specification:
//!   O = (image_overlays, contour_overlays, mask_overlays)
//! Invariants:
//! ImageOverlay : data.len() == dims\[0\]\*dims\[1\]\*dims\[2\]
//! ContourOverlay: every contour >= 2 points
//! MaskOverlay : data.len() == dims\[0\]\*dims\[1\]\*dims\[2\]

use super::color::RgbaLinear;
use super::types::LabelId;
use ritk_spatial::VolumeDims;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Display visibility for overlay layers.
///
/// - `Hidden`: overlay is not rendered.
/// - `Visible`: overlay is rendered (default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Visibility {
    /// Overlay is not rendered.
    Hidden,
    /// Overlay is rendered.
    #[default]
    Visible }

/// Normalized opacity value in the closed interval `[0.0, 1.0]`.
///
/// # Panics
/// [`Opacity::new`] panics if `v` is not in `[0.0, 1.0]` (including NaN).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Opacity(f32);

impl Opacity {
    /// Construct a valid opacity. Panics if `v < 0.0`, `v > 1.0`, or `v` is NaN.
    pub fn new(v: f32) -> Self {
        assert!(
            v.is_finite() && (0.0..=1.0).contains(&v),
            "Opacity must be in [0.0, 1.0], got {v}"
        );
        Self(v)
    }

    /// Construct without validation. Caller must guarantee `v âˆˆ [0.0, 1.0]`.
    ///
    /// # Safety contract (invariant)
    /// This function is safe; "unchecked" refers to domain invariant only.
    #[inline]
    pub fn new_unchecked(v: f32) -> Self {
        Self(v)
    }

    /// Raw `f32` value.
    #[inline]
    pub fn get(self) -> f32 {
        self.0
    }
}

impl Default for Opacity {
    fn default() -> Self {
        Self(1.0)
    }
}

/// Display colormap for image overlays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum Colormap {
    #[default]
    Grayscale,
    Hot,
    Cool,
    Jet,
    Custom(Vec<[f32; 4]>) }

/// Secondary image overlay. Invariant: data.len() == dims product.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageOverlay {
    pub name: String,
    pub data: Vec<f32>,
    pub dims: VolumeDims,
    pub opacity: Opacity,
    pub colormap: Colormap,
    pub visible: Visibility }
impl ImageOverlay {
    /// Panics when `data.len() != dims.total_voxels()`.
    pub fn new(name: impl Into<String>, data: Vec<f32>, dims: impl Into<VolumeDims>) -> Self {
        let dims = dims.into();
        let expected = dims.total_voxels();
        assert_eq!(
            data.len(),
            expected,
            "ImageOverlay data length {} does not match dims {:?} (product {})",
            data.len(),
            dims,
            expected
        );
        Self {
            name: name.into(),
            data,
            dims,
            opacity: Opacity::new(1.0),
            colormap: Colormap::Grayscale,
            visible: Visibility::Visible }
    }
    pub fn with_opacity(mut self, v: f32) -> Self {
        self.opacity = Opacity::new(v);
        self
    }
    pub fn with_colormap(mut self, v: Colormap) -> Self {
        self.colormap = v;
        self
    }
    pub fn with_visible(mut self, v: Visibility) -> Self {
        self.visible = v;
        self
    }
}

/// Contour overlay: closed 3-D polygons for one label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContourOverlay {
    pub name: String,
    pub label_id: LabelId,
    pub contours: Vec<Vec<[f64; 3]>>,
    pub color: RgbaLinear,
    pub line_width: f32,
    pub visible: Visibility }
impl ContourOverlay {
    pub fn new(name: impl Into<String>, label_id: impl Into<LabelId>) -> Self {
        Self {
            name: name.into(),
            label_id: label_id.into(),
            contours: Vec::new(),
            color: RgbaLinear::new(1.0, 1.0, 1.0, 1.0),
            line_width: 1.0,
            visible: Visibility::Visible }
    }
    /// Returns `Err` when `pts.len() < 2`.
    pub fn add_contour(&mut self, pts: Vec<[f64; 3]>) -> Result<(), String> {
        if pts.len() < 2 {
            return Err(format!("contour requires >= 2 points, got {}", pts.len()));
        }
        self.contours.push(pts);
        Ok(())
    }
    pub fn with_color(mut self, v: RgbaLinear) -> Self {
        self.color = v;
        self
    }
    pub fn with_visible(mut self, v: Visibility) -> Self {
        self.visible = v;
        self
    }
}

/// Mask overlay: dense label volume (0 = transparent).  Invariant: data.len() == dims product.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskOverlay {
    pub name: String,
    pub data: Vec<u32>,
    pub dims: VolumeDims,
    pub opacity: Opacity,
    pub visible: Visibility }
impl MaskOverlay {
    /// Panics when `data.len() != dims.total_voxels()`.
    pub fn new(name: impl Into<String>, data: Vec<u32>, dims: impl Into<VolumeDims>) -> Self {
        let dims = dims.into();
        let expected = dims.total_voxels();
        assert_eq!(
            data.len(),
            expected,
            "MaskOverlay data length {} does not match dims {:?} (product {})",
            data.len(),
            dims,
            expected
        );
        Self {
            name: name.into(),
            data,
            dims,
            opacity: Opacity::new(0.5),
            visible: Visibility::Visible }
    }
    pub fn with_opacity(mut self, v: f32) -> Self {
        self.opacity = Opacity::new(v);
        self
    }
    /// Theorem: label_count = |{ v : v in data, v != 0 }|
    pub fn label_count(&self) -> usize {
        self.data
            .iter()
            .filter(|&&v| v != 0)
            .copied()
            .collect::<HashSet<u32>>()
            .len()
    }
}

/// Composite overlay state O = (image_overlays, contour_overlays, mask_overlays).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OverlayState {
    pub image_overlays: Vec<ImageOverlay>,
    pub contour_overlays: Vec<ContourOverlay>,
    pub mask_overlays: Vec<MaskOverlay> }
impl OverlayState {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_image_overlay(&mut self, o: ImageOverlay) -> &mut Self {
        self.image_overlays.push(o);
        self
    }
    pub fn add_contour_overlay(&mut self, o: ContourOverlay) -> &mut Self {
        self.contour_overlays.push(o);
        self
    }
    pub fn add_mask_overlay(&mut self, o: MaskOverlay) -> &mut Self {
        self.mask_overlays.push(o);
        self
    }
    pub fn remove_image_overlay(&mut self, name: &str) -> bool {
        if let Some(pos) = self.image_overlays.iter().position(|o| o.name == name) {
            self.image_overlays.remove(pos);
            true
        } else {
            false
        }
    }
    pub fn remove_mask_overlay(&mut self, name: &str) -> bool {
        if let Some(pos) = self.mask_overlays.iter().position(|o| o.name == name) {
            self.mask_overlays.remove(pos);
            true
        } else {
            false
        }
    }
    pub fn visible_image_overlays(&self) -> Vec<&ImageOverlay> {
        self.image_overlays
            .iter()
            .filter(|o| o.visible == Visibility::Visible)
            .collect()
    }
    pub fn visible_mask_overlays(&self) -> Vec<&MaskOverlay> {
        self.mask_overlays
            .iter()
            .filter(|o| o.visible == Visibility::Visible)
            .collect()
    }
}

#[cfg(test)]
#[path = "tests_overlay.rs"]
mod tests;
