//! Viewport image orientation transforms (flip/rotate).
//!
//! # Mathematical Specification
//!
//! A `ViewTransform` defines an orientation transformation on a 2-D pixel grid:
//!
//! ## Flip Horizontal
//! For an image of width W and height H, pixel at column c is mapped to column
//! W − 1 − c. Formal: `f_h(r, c) = (r, W−1−c)`.
//!
//! ## Flip Vertical
//! Pixel at row r is mapped to row H − 1 − r. Formal: `f_v(r, c) = (H−1−r, c)`.
//!
//! ## Rotation (clockwise, 90° steps)
//! For n steps of 90° clockwise rotation on a W×H image:
//! - 0°:   `(r, c)` → `(r, c)`,   output size `(W, H)`
//! - 90°:  `(r, c)` → `(c, H−1−r)`, output size `(H, W)`
//! - 180°: `(r, c)` → `(H−1−r, W−1−c)`, output size `(W, H)`
//! - 270°: `(r, c)` → `(W−1−c, r)`, output size `(H, W)`
//!
//! Transforms are applied in the order: flip_h → flip_v → rotate.
//!
//! ## Invariants
//! - `apply_to_image(img, identity)` returns a pixel-identical image.
//! - Four 90° clockwise rotations compose to the identity.
//! - Two flip_h applications compose to the identity.

use egui::ColorImage;

/// Number of 90° clockwise rotation steps (0=0°, 1=90°, 2=180°, 3=270°).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum RotationSteps {
    #[default]
    Zero,
    Ninety,
    OneEighty,
    TwoSeventy,
}

impl RotationSteps {
    /// Advance by one 90° clockwise step.
    pub fn rotate_cw(self) -> Self {
        match self {
            Self::Zero => Self::Ninety,
            Self::Ninety => Self::OneEighty,
            Self::OneEighty => Self::TwoSeventy,
            Self::TwoSeventy => Self::Zero,
        }
    }

    /// Reverse by one 90° step (counter-clockwise).
    pub fn rotate_ccw(self) -> Self {
        match self {
            Self::Zero => Self::TwoSeventy,
            Self::Ninety => Self::Zero,
            Self::OneEighty => Self::Ninety,
            Self::TwoSeventy => Self::OneEighty,
        }
    }
}

/// Viewport image orientation state.
///
/// Encodes the sequence of flip and rotation transforms applied to each
/// rendered slice before display. The transform is stateless and deterministic:
/// equal `ViewTransform` values produce identical output for identical input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct ViewTransform {
    /// Mirror the image about its vertical axis (left↔right).
    pub flip_h: bool,
    /// Mirror the image about its horizontal axis (up↔down).
    pub flip_v: bool,
    /// Clockwise rotation applied after flips.
    pub rotation: RotationSteps,
}

impl ViewTransform {
    /// True when the transform is the identity (no flip, no rotation).
    pub fn is_identity(self) -> bool {
        !self.flip_h && !self.flip_v && self.rotation == RotationSteps::Zero
    }

    /// Toggle horizontal flip.
    pub fn toggle_flip_h(self) -> Self {
        Self { flip_h: !self.flip_h, ..self }
    }

    /// Toggle vertical flip.
    pub fn toggle_flip_v(self) -> Self {
        Self { flip_v: !self.flip_v, ..self }
    }

    /// Advance rotation by one 90° clockwise step.
    pub fn rotate_cw(self) -> Self {
        Self { rotation: self.rotation.rotate_cw(), ..self }
    }

    /// Advance rotation by one 90° counter-clockwise step.
    pub fn rotate_ccw(self) -> Self {
        Self { rotation: self.rotation.rotate_ccw(), ..self }
    }

    /// Reset to identity.
    pub fn reset(self) -> Self {
        Self::default()
    }
}

// ── Pixel-level transforms ────────────────────────────────────────────────────

/// Apply a horizontal flip (left↔right) to a `ColorImage`.
///
/// For width W, pixel at column c maps to column W−1−c.
/// Time complexity: O(W × H). No allocation beyond the output pixel buffer.
pub fn flip_h_image(img: &ColorImage) -> ColorImage {
    let [w, h] = img.size;
    let mut out = vec![egui::Color32::BLACK; w * h];
    for row in 0..h {
        for col in 0..w {
            out[row * w + (w - 1 - col)] = img.pixels[row * w + col];
        }
    }
    ColorImage { size: [w, h], pixels: out }
}

/// Apply a vertical flip (up↔down) to a `ColorImage`.
///
/// For height H, pixel at row r maps to row H−1−r.
/// Time complexity: O(W × H).
pub fn flip_v_image(img: &ColorImage) -> ColorImage {
    let [w, h] = img.size;
    let mut out = vec![egui::Color32::BLACK; w * h];
    for row in 0..h {
        for col in 0..w {
            out[(h - 1 - row) * w + col] = img.pixels[row * w + col];
        }
    }
    ColorImage { size: [w, h], pixels: out }
}

/// Rotate a `ColorImage` 90° clockwise.
///
/// Output size is `[H, W]` (swapped from input `[W, H]`).
/// Mapping: input `(row, col)` → output column `H−1−row`, output row `col`.
/// Formally: `out[col, H−1−row] = in[row, col]`.
/// Time complexity: O(W × H).
pub fn rotate_90_cw_image(img: &ColorImage) -> ColorImage {
    let [w, h] = img.size;
    // Output dimensions are [h, w] (width and height swapped).
    let ow = h;
    let oh = w;
    let mut out = vec![egui::Color32::BLACK; ow * oh];
    for row in 0..h {
        for col in 0..w {
            // 90° CW: output(col, h−1−row) = input(row, col)
            let orow = col;
            let ocol = h - 1 - row;
            out[orow * ow + ocol] = img.pixels[row * w + col];
        }
    }
    ColorImage { size: [ow, oh], pixels: out }
}

/// Apply a `ViewTransform` to a `ColorImage`.
///
/// Applies transforms in the canonical order: flip_h → flip_v → rotate.
/// When `transform.is_identity()`, returns a clone of the input unchanged.
pub fn apply_to_image(img: &ColorImage, transform: ViewTransform) -> ColorImage {
    if transform.is_identity() {
        return img.clone();
    }

    let mut result = img.clone();

    if transform.flip_h {
        result = flip_h_image(&result);
    }
    if transform.flip_v {
        result = flip_v_image(&result);
    }

    result = match transform.rotation {
        RotationSteps::Zero => result,
        RotationSteps::Ninety => rotate_90_cw_image(&result),
        RotationSteps::OneEighty => {
            // 180° = two 90° CW rotations.
            let r = rotate_90_cw_image(&result);
            rotate_90_cw_image(&r)
        }
        RotationSteps::TwoSeventy => {
            // 270° CW = three 90° CW rotations.
            let r1 = rotate_90_cw_image(&result);
            let r2 = rotate_90_cw_image(&r1);
            rotate_90_cw_image(&r2)
        }
    };

    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Color32, ColorImage};

    /// Build a 3×2 (width=3, height=2) test image where each pixel encodes
    /// its position uniquely: pixel(row, col) = Color32::from_rgb(row as u8, col as u8, 0).
    fn make_test_image() -> ColorImage {
        // size = [width, height] = [3, 2]
        let w = 3usize;
        let h = 2usize;
        let mut pixels = Vec::with_capacity(w * h);
        for row in 0..h {
            for col in 0..w {
                pixels.push(Color32::from_rgb(row as u8, col as u8, 0));
            }
        }
        ColorImage { size: [w, h], pixels }
    }

    fn px(img: &ColorImage, row: usize, col: usize) -> Color32 {
        img.pixels[row * img.size[0] + col]
    }

    // ── Identity ──────────────────────────────────────────────────────────────

    #[test]
    fn test_identity_no_change() {
        let img = make_test_image();
        let result = apply_to_image(&img, ViewTransform::default());
        assert_eq!(result.size, img.size, "identity must preserve dimensions");
        assert_eq!(result.pixels, img.pixels, "identity must preserve pixel values");
    }

    // ── Flip horizontal ───────────────────────────────────────────────────────

    #[test]
    fn test_flip_h_pixel_mapping() {
        // Original: pixel(0,0)=Color32::from_rgb(0,0,0), pixel(0,1)=Color32::from_rgb(0,1,0)
        // After flip_h on width=3: col 0→2, col 1→1, col 2→0
        let img = make_test_image(); // 3×2
        let flipped = flip_h_image(&img);

        assert_eq!(flipped.size, [3, 2], "flip_h must preserve dimensions");
        // Pixel originally at (row=0, col=0) must now be at (row=0, col=2)
        assert_eq!(
            px(&flipped, 0, 2),
            px(&img, 0, 0),
            "flip_h: col 0 maps to col W-1"
        );
        // Pixel originally at (row=0, col=2) must now be at (row=0, col=0)
        assert_eq!(
            px(&flipped, 0, 0),
            px(&img, 0, 2),
            "flip_h: col W-1 maps to col 0"
        );
        // Center column (col=1) is invariant for odd-width images.
        assert_eq!(
            px(&flipped, 0, 1),
            px(&img, 0, 1),
            "flip_h: center col is invariant"
        );
    }

    #[test]
    fn test_flip_h_involution() {
        // Two flip_h applications compose to identity.
        let img = make_test_image();
        let twice = flip_h_image(&flip_h_image(&img));
        assert_eq!(twice.pixels, img.pixels, "flip_h ∘ flip_h = identity");
    }

    // ── Flip vertical ─────────────────────────────────────────────────────────

    #[test]
    fn test_flip_v_pixel_mapping() {
        let img = make_test_image(); // 3×2 (height=2)
        let flipped = flip_v_image(&img);

        assert_eq!(flipped.size, [3, 2], "flip_v must preserve dimensions");
        // Row 0 ↔ row 1 (height = 2 → H-1-0 = 1).
        assert_eq!(
            px(&flipped, 0, 0),
            px(&img, 1, 0),
            "flip_v: row 0 maps from row H-1"
        );
        assert_eq!(
            px(&flipped, 1, 0),
            px(&img, 0, 0),
            "flip_v: row H-1 maps from row 0"
        );
    }

    #[test]
    fn test_flip_v_involution() {
        let img = make_test_image();
        let twice = flip_v_image(&flip_v_image(&img));
        assert_eq!(twice.pixels, img.pixels, "flip_v ∘ flip_v = identity");
    }

    // ── Rotation ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rotate_90_cw_dimensions() {
        // 3×2 (W=3, H=2) rotated 90° CW → 2×3 (W=H_orig, H=W_orig).
        let img = make_test_image();
        let r = rotate_90_cw_image(&img);
        assert_eq!(r.size, [2, 3], "90° CW rotation: size must swap W↔H");
    }

    #[test]
    fn test_rotate_90_cw_analytical_mapping() {
        // For 3×2 (W=3, H=2): pixel at (row=0, col=0) maps to output (orow=0, ocol=H-1-row=1).
        // For pixel at (row=1, col=2): output (orow=2, ocol=H-1-1=0).
        let img = make_test_image();
        let r = rotate_90_cw_image(&img);
        // Original (0,0) → output (0, 1)
        assert_eq!(px(&r, 0, 1), px(&img, 0, 0), "rotate_90_cw: (0,0)→(0,H-1)");
        // Original (0,2) → output (2, 1)
        assert_eq!(px(&r, 2, 1), px(&img, 0, 2), "rotate_90_cw: (0,W-1)→(W-1,H-1)");
        // Original (1,0) → output (0, 0)
        assert_eq!(px(&r, 0, 0), px(&img, 1, 0), "rotate_90_cw: (H-1,0)→(0,0)");
    }

    #[test]
    fn test_four_rotations_is_identity() {
        // Four 90° CW rotations must produce the original dimensions and pixel values.
        let img = make_test_image();
        let mut r = img.clone();
        for _ in 0..4 {
            r = rotate_90_cw_image(&r);
        }
        assert_eq!(r.size, img.size, "4×90° rotation: dimensions must be restored");
        assert_eq!(r.pixels, img.pixels, "4×90° rotation: pixels must be restored");
    }

    // ── ViewTransform helpers ─────────────────────────────────────────────────

    #[test]
    fn test_is_identity_default() {
        assert!(ViewTransform::default().is_identity());
    }

    #[test]
    fn test_toggle_flip_h_toggle_twice_is_identity() {
        let t = ViewTransform::default().toggle_flip_h().toggle_flip_h();
        assert!(t.is_identity());
    }

    #[test]
    fn test_rotate_cw_four_times_is_identity() {
        let t = ViewTransform::default()
            .rotate_cw()
            .rotate_cw()
            .rotate_cw()
            .rotate_cw();
        assert!(t.is_identity());
    }
}
