//! Overlay composition state for annotation rendering.
//!
//! Mathematical specification:
//!   O = (image_overlays, contour_overlays, mask_overlays)
//! Invariants:
//!   ImageOverlay  : data.len() == dims[0]*dims[1]*dims[2]
//!   ContourOverlay: every contour >= 2 points
//!   MaskOverlay   : data.len() == dims[0]*dims[1]*dims[2]

use std::collections::HashSet;
use serde::{Deserialize, Serialize};

/// Display colormap for image overlays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Colormap { Grayscale, Hot, Cool, Jet, Custom(Vec<[f32; 4]>) }

impl Default for Colormap {
    fn default() -> Self { Self::Grayscale }
}

/// Secondary image overlay.  Invariant: data.len() == dims product.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageOverlay {
    pub name: String,
    pub data: Vec<f32>,
    pub dims: [usize; 3],
    pub opacity: f32,
    pub colormap: Colormap,
    pub visible: bool,
}
impl ImageOverlay {
    /// # Panics
    /// Panics when `data.len() != dims[0]*dims[1]*dims[2]`.
    pub fn new(name: impl Into<String>, data: Vec<f32>, dims: [usize; 3]) -> Self {
        let expected = dims[0] * dims[1] * dims[2];
        assert_eq!(data.len(), expected,
            "ImageOverlay data length {} does not match dims {:?} (product {})",
            data.len(), dims, expected);
        Self { name: name.into(), data, dims, opacity: 1.0,
               colormap: Colormap::Grayscale, visible: true }
    }
    pub fn with_opacity(mut self, v: f32) -> Self { self.opacity = v; self }
    pub fn with_colormap(mut self, v: Colormap) -> Self { self.colormap = v; self }
    pub fn with_visible(mut self, v: bool) -> Self { self.visible = v; self }
}

/// Contour overlay: closed 3-D polygons for one label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContourOverlay {
    pub name: String,
    pub label_id: u32,
    pub contours: Vec<Vec<[f64; 3]>>,
    pub color: [f32; 4],
    pub line_width: f32,
    pub visible: bool,
}
impl ContourOverlay {
    pub fn new(name: impl Into<String>, label_id: u32) -> Self {
        Self { name: name.into(), label_id, contours: Vec::new(),
               color: [1.0, 1.0, 1.0, 1.0], line_width: 1.0, visible: true }
    }
    /// Returns `Err` when `pts.len() < 2`.
    pub fn add_contour(&mut self, pts: Vec<[f64; 3]>) -> Result<(), String> {
        if pts.len() < 2 {
            return Err(format!("contour requires >= 2 points, got {}", pts.len()));
        }
        self.contours.push(pts);
        Ok(())
    }
    pub fn with_color(mut self, v: [f32; 4]) -> Self { self.color = v; self }
    pub fn with_visible(mut self, v: bool) -> Self { self.visible = v; self }
}

/// Mask overlay: dense label volume (0 = transparent).  Invariant: data.len() == dims product.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskOverlay {
    pub name: String,
    pub data: Vec<u32>,
    pub dims: [usize; 3],
    pub opacity: f32,
    pub visible: bool,
}
impl MaskOverlay {
    /// # Panics
    /// Panics when `data.len() != dims[0]*dims[1]*dims[2]`.
    pub fn new(name: impl Into<String>, data: Vec<u32>, dims: [usize; 3]) -> Self {
        let expected = dims[0] * dims[1] * dims[2];
        assert_eq!(data.len(), expected,
            "MaskOverlay data length {} does not match dims {:?} (product {})",
            data.len(), dims, expected);
        Self { name: name.into(), data, dims, opacity: 0.5, visible: true }
    }
    pub fn with_opacity(mut self, v: f32) -> Self { self.opacity = v; self }
    /// Theorem: label_count = |{ v : v in data, v != 0 }|
    pub fn label_count(&self) -> usize {
        self.data.iter().filter(|&&v| v != 0).cloned()
            .collect::<HashSet<u32>>().len()
    }
}

/// Composite overlay state O = (image_overlays, contour_overlays, mask_overlays).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OverlayState {
    pub image_overlays: Vec<ImageOverlay>,
    pub contour_overlays: Vec<ContourOverlay>,
    pub mask_overlays: Vec<MaskOverlay>,
}
impl OverlayState {
    pub fn new() -> Self { Self::default() }
    pub fn add_image_overlay(&mut self, o: ImageOverlay) -> &mut Self {
        self.image_overlays.push(o); self
    }
    pub fn add_contour_overlay(&mut self, o: ContourOverlay) -> &mut Self {
        self.contour_overlays.push(o); self
    }
    pub fn add_mask_overlay(&mut self, o: MaskOverlay) -> &mut Self {
        self.mask_overlays.push(o); self
    }
    pub fn remove_image_overlay(&mut self, name: &str) -> bool {
        if let Some(pos) = self.image_overlays.iter().position(|o| o.name == name) {
            self.image_overlays.remove(pos); true
        } else { false }
    }
    pub fn remove_mask_overlay(&mut self, name: &str) -> bool {
        if let Some(pos) = self.mask_overlays.iter().position(|o| o.name == name) {
            self.mask_overlays.remove(pos); true
        } else { false }
    }
    pub fn visible_image_overlays(&self) -> Vec<&ImageOverlay> {
        self.image_overlays.iter().filter(|o| o.visible).collect()
    }
    pub fn visible_mask_overlays(&self) -> Vec<&MaskOverlay> {
        self.mask_overlays.iter().filter(|o| o.visible).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1a – correct length ok
    #[test]
    fn test_image_overlay_new_correct_length() {
        let o = ImageOverlay::new("layer", vec![0.0f32; 24], [2, 3, 4]);
        assert_eq!(o.data.len(), 24);
        assert_eq!(o.dims, [2, 3, 4]);
        assert_eq!(o.name, "layer");
        assert!(o.visible);
        assert_eq!(o.opacity, 1.0);
        assert_eq!(o.colormap, Colormap::Grayscale);
    }

    // Test 1b – mismatch panics
    #[test]
    #[should_panic(expected = "does not match dims")]
    fn test_image_overlay_new_panics_on_dims_mismatch() {
        ImageOverlay::new("bad", vec![0.0f32; 5], [2, 2, 2]);
    }

    // Test 2a – valid contour
    #[test]
    fn test_contour_overlay_add_contour_valid() {
        let mut c = ContourOverlay::new("c", 1);
        let pts = vec![[0.0f64,0.0,0.0],[1.0,0.0,0.0],[0.5,1.0,0.0]];
        assert!(c.add_contour(pts.clone()).is_ok());
        assert_eq!(c.contours.len(), 1);
        assert_eq!(c.contours[0], pts);
    }

    // Test 2b – single point returns Err
    #[test]
    fn test_contour_overlay_add_contour_single_point_returns_err() {
        let mut c = ContourOverlay::new("c", 1);
        let r = c.add_contour(vec![[0.0f64, 0.0, 0.0]]);
        assert!(r.is_err());
        let msg = r.unwrap_err();
        assert!(msg.contains("got 1"), "msg: {}", msg);
    }

    // Test 3 – label_count {0,1,2,2,3} -> 3
    #[test]
    fn test_mask_overlay_label_count() {
        let o = MaskOverlay::new("m", vec![0u32,1,2,2,3], [1,1,5]);
        assert_eq!(o.label_count(), 3);
    }

    // Test 4 – add_image_overlay len == 1
    #[test]
    fn test_overlay_state_add_image_overlay_len() {
        let mut s = OverlayState::new();
        s.add_image_overlay(ImageOverlay::new("img", vec![0.0f32; 8], [2,2,2]));
        assert_eq!(s.image_overlays.len(), 1);
    }

    // Test 5 – remove returns true/false
    #[test]
    fn test_overlay_state_remove_image_overlay() {
        let mut s = OverlayState::new();
        s.add_image_overlay(ImageOverlay::new("alpha", vec![0.0f32; 1], [1,1,1]));
        s.add_image_overlay(ImageOverlay::new("beta",  vec![0.0f32; 1], [1,1,1]));
        assert!(s.remove_image_overlay("alpha"));
        assert_eq!(s.image_overlays.len(), 1);
        assert_eq!(s.image_overlays[0].name, "beta");
        assert!(!s.remove_image_overlay("nonexistent"));
        assert_eq!(s.image_overlays.len(), 1);
    }

    // Test 6 – visible_image_overlays filters hidden
    #[test]
    fn test_overlay_state_visible_image_overlays() {
        let mut s = OverlayState::new();
        s.add_image_overlay(ImageOverlay::new("vis", vec![1.0f32; 4], [1,2,2]).with_visible(true));
        s.add_image_overlay(ImageOverlay::new("hid", vec![1.0f32; 4], [1,2,2]).with_visible(false));
        let r = s.visible_image_overlays();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].name, "vis");
    }

    // Test 7 – Colormap default
    #[test]
    fn test_colormap_default_is_grayscale() {
        assert_eq!(Colormap::default(), Colormap::Grayscale);
    }

    // Test 8 – JSON round-trip
    #[test]
    fn test_overlay_state_serde_round_trip() {
        let mut s = OverlayState::new();
        let mut img = ImageOverlay::new("img", vec![0.5f32, 0.75], [1,1,2]);
        img.opacity = 0.8;
        img.colormap = Colormap::Hot;
        s.add_image_overlay(img);
        let mut cnt = ContourOverlay::new("cnt", 42);
        cnt.add_contour(vec![[1.0,2.0,3.0],[4.0,5.0,6.0]]).unwrap();
        s.add_contour_overlay(cnt);
        let msk = MaskOverlay::new("msk", vec![0u32,1,2,0], [1,2,2]).with_opacity(0.6);
        s.add_mask_overlay(msk);
        let json = serde_json::to_string(&s).unwrap();
        let r: OverlayState = serde_json::from_str(&json).unwrap();
        assert_eq!(r.image_overlays.len(), 1);
        assert_eq!(r.image_overlays[0].name, "img");
        assert!((r.image_overlays[0].opacity - 0.8).abs() < 1e-6);
        assert_eq!(r.image_overlays[0].colormap, Colormap::Hot);
        assert_eq!(r.image_overlays[0].data, vec![0.5f32, 0.75]);
        assert_eq!(r.contour_overlays.len(), 1);
        assert_eq!(r.contour_overlays[0].label_id, 42);
        assert_eq!(r.contour_overlays[0].contours[0].len(), 2);
        assert_eq!(r.mask_overlays.len(), 1);
        assert_eq!(r.mask_overlays[0].name, "msk");
        assert!((r.mask_overlays[0].opacity - 0.6).abs() < 1e-6);
        assert_eq!(r.mask_overlays[0].data, vec![0u32,1,2,0]);
    }
}
