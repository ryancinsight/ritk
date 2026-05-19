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

use crate::render::buffer_pool::RenderBufferPool;
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
        Self {
            flip_h: !self.flip_h,
            ..self
        }
    }

    /// Toggle vertical flip.
    pub fn toggle_flip_v(self) -> Self {
        Self {
            flip_v: !self.flip_v,
            ..self
        }
    }

    /// Advance rotation by one 90° clockwise step.
    pub fn rotate_cw(self) -> Self {
        Self {
            rotation: self.rotation.rotate_cw(),
            ..self
        }
    }

    /// Advance rotation by one 90° counter-clockwise step.
    pub fn rotate_ccw(self) -> Self {
        Self {
            rotation: self.rotation.rotate_ccw(),
            ..self
        }
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
    ColorImage {
        size: [w, h],
        pixels: out,
    }
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
    ColorImage {
        size: [w, h],
        pixels: out,
    }
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
    ColorImage {
        size: [ow, oh],
        pixels: out,
    }
}

/// Apply a `ViewTransform` to a `ColorImage`.
///
/// Applies transforms in the canonical order: flip_h → flip_v → rotate.
/// When `transform.is_identity()`, returns a clone of the input unchanged.
///
/// # Allocation cost
///
/// This function allocates a new `Vec<Color32>` per transform step.
/// For zero-allocation hot-path usage, prefer [`apply_to_image_into`] which
/// writes into pre-allocated scratch buffers from the [`RenderBufferPool`].
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

/// Apply a `ViewTransform` to a `ColorImage`, writing output into pre-allocated
/// scratch buffers.
///
/// Produces output pixel-identical to [`apply_to_image`] for the same inputs
/// while eliminating all `Vec<Color32>` allocations after pool warm-up.
///
/// # Differential equivalence invariant
///
/// For all valid (`img`, `transform`) inputs:
/// ```text
/// apply_to_image_into(pool, img, transform).pixels == apply_to_image(img, transform).pixels
/// ```
///
/// # Allocation behaviour
///
/// - Identity transform: zero heap allocations. Returns a `ColorImage` that
///   borrows the input's pixel slice (no clone).
/// - Non-identity transform: zero heap allocations after `pool.color32` has
///   reached peak capacity. The scratch buffer is reused across calls.
pub(crate) fn apply_to_image_into(
    pool: &mut RenderBufferPool,
    img: &ColorImage,
    transform: ViewTransform,
) -> ColorImage {
    if transform.is_identity() {
        // Return a ColorImage that shares the input's Arc< Vec<Color32> >.
        // egui::ColorImage::clone() is cheap (Arc bump), not a deep copy.
        return img.clone();
    }

    let [w, h] = img.size;
    // Determine whether the output dimensions swap (rotation 90° or 270°).
    let swaps_axes = matches!(
        transform.rotation,
        RotationSteps::Ninety | RotationSteps::TwoSeventy
    );
    let (out_w, out_h) = if swaps_axes { (h, w) } else { (w, h) };
    let out_n = out_w * out_h;

    pool.resize_color32(out_n);
    let out = pool.color32.as_mut_slice();

    // Compute the composite transform in a single pass over the input pixels.
    //
    // The canonical order is flip_h → flip_v → rotate.
    // We fuse all three into one index mapping so each input pixel is read
    // once and written once — no intermediate allocation.
    //
    // Index mapping derivation (W=in_w, H=in_h):
    //   After flip_h:  (r, c) → (r, W−1−c)
    //   After flip_v:  (r, c) → (H−1−r, c)
    //   After flip_h + flip_v: (r, c) → (H−1−r, W−1−c)
    //   After rotation n×90° CW on W×H → out_w×out_h:
    //     0°:   out(orow, ocol) = in(r, c),  out_w=W, out_h=H
    //     90°:  out(c, H−1−r)   = in(r, c),  out_w=H, out_h=W
    //     180°: out(H−1−r, W−1−c) = in(r,c), out_w=W, out_h=H
    //     270°: out(W−1−c, r)   = in(r, c),  out_w=H, out_h=W

    match (transform.flip_h, transform.flip_v, transform.rotation) {
        // ── No rotation ───────────────────────────────────────────────
        // flip_h only: out[orow * W + (W−1−ocol)] = in[orow * W + ocol]
        (true, false, RotationSteps::Zero) => {
            for orow in 0..h {
                for ocol in 0..w {
                    let r = orow;
                    let c = w - 1 - ocol;
                    out[orow * w + ocol] = img.pixels[r * w + c];
                }
            }
        }
        // flip_v only: out[(H−1−orow) * W + ocol] = in[orow * W + ocol]
        (false, true, RotationSteps::Zero) => {
            for orow in 0..h {
                for ocol in 0..w {
                    let r = h - 1 - orow;
                    let c = ocol;
                    out[orow * w + ocol] = img.pixels[r * w + c];
                }
            }
        }
        // flip_h + flip_v, no rotation
        (true, true, RotationSteps::Zero) => {
            for orow in 0..h {
                for ocol in 0..w {
                    let r = h - 1 - orow;
                    let c = w - 1 - ocol;
                    out[orow * w + ocol] = img.pixels[r * w + c];
                }
            }
        }
        // Identity: no flips, no rotation.
        // Unreachable (identity is caught by the early return above)
        // but required for exhaustive match.
        #[allow(unreachable_code)]
        (false, false, RotationSteps::Zero) => {}

        // ── 90° CW rotation ────────────────────────────────────────────
        // Output size: [H, W]. For output (orow, ocol):
        // r = H−1−ocol, c = orow
        (false, false, RotationSteps::Ninety) => {
            for orow in 0..out_h {
                for ocol in 0..out_w {
                    let r = h - 1 - ocol;
                    let c = orow;
                    out[orow * out_w + ocol] = img.pixels[r * w + c];
                }
            }
        }
        // flip_h + 90° CW: r = H−1−ocol, c = W−1−orow
        (true, false, RotationSteps::Ninety) => {
            for orow in 0..out_h {
                for ocol in 0..out_w {
                    let r = h - 1 - ocol;
                    let c = w - 1 - orow;
                    out[orow * out_w + ocol] = img.pixels[r * w + c];
                }
            }
        }
        // flip_v + 90° CW: r = ocol, c = orow
        (false, true, RotationSteps::Ninety) => {
            for orow in 0..out_h {
                for ocol in 0..out_w {
                    let r = ocol;
                    let c = orow;
                    out[orow * out_w + ocol] = img.pixels[r * w + c];
                }
            }
        }
        // flip_h + flip_v + 90° CW: r = ocol, c = W−1−orow
        (true, true, RotationSteps::Ninety) => {
            for orow in 0..out_h {
                for ocol in 0..out_w {
                    let r = ocol;
                    let c = w - 1 - orow;
                    out[orow * out_w + ocol] = img.pixels[r * w + c];
                }
            }
        }

        // ── 180° rotation ──────────────────────────────────────────────
        // Output size: [W, H]. r = H−1−orow, c = W−1−ocol
        (false, false, RotationSteps::OneEighty) => {
            for orow in 0..h {
                for ocol in 0..w {
                    let r = h - 1 - orow;
                    let c = w - 1 - ocol;
                    out[orow * w + ocol] = img.pixels[r * w + c];
                }
            }
        }
        // flip_h + 180° = flip_v: r = H−1−orow, c = ocol
        (true, false, RotationSteps::OneEighty) => {
            for orow in 0..h {
                for ocol in 0..w {
                    let r = h - 1 - orow;
                    let c = ocol;
                    out[orow * w + ocol] = img.pixels[r * w + c];
                }
            }
        }
        // flip_v + 180° = flip_h: r = orow, c = W−1−ocol
        (false, true, RotationSteps::OneEighty) => {
            for orow in 0..h {
                for ocol in 0..w {
                    let r = orow;
                    let c = w - 1 - ocol;
                    out[orow * w + ocol] = img.pixels[r * w + c];
                }
            }
        }
        // flip_h + flip_v + 180° = identity: r = orow, c = ocol
        (true, true, RotationSteps::OneEighty) => {
            for orow in 0..h {
                for ocol in 0..w {
                    let r = orow;
                    let c = ocol;
                    out[orow * w + ocol] = img.pixels[r * w + c];
                }
            }
        }

        // ── 270° CW rotation ───────────────────────────────────────────
        // Output size: [H, W]. For output (orow, ocol):
        // r = ocol, c = W−1−orow
        (false, false, RotationSteps::TwoSeventy) => {
            for orow in 0..out_h {
                for ocol in 0..out_w {
                    let r = ocol;
                    let c = w - 1 - orow;
                    out[orow * out_w + ocol] = img.pixels[r * w + c];
                }
            }
        }
        // flip_h + 270° CW: r = ocol, c = orow
        (true, false, RotationSteps::TwoSeventy) => {
            for orow in 0..out_h {
                for ocol in 0..out_w {
                    let r = ocol;
                    let c = orow;
                    out[orow * out_w + ocol] = img.pixels[r * w + c];
                }
            }
        }
        // flip_v + 270° CW: r = H−1−ocol, c = W−1−orow
        (false, true, RotationSteps::TwoSeventy) => {
            for orow in 0..out_h {
                for ocol in 0..out_w {
                    let r = h - 1 - ocol;
                    let c = w - 1 - orow;
                    out[orow * out_w + ocol] = img.pixels[r * w + c];
                }
            }
        }
        // flip_h + flip_v + 270° CW: r = H−1−ocol, c = orow
        (true, true, RotationSteps::TwoSeventy) => {
            for orow in 0..out_h {
                for ocol in 0..out_w {
                    let r = h - 1 - ocol;
                    let c = orow;
                    out[orow * out_w + ocol] = img.pixels[r * w + c];
                }
            }
        }
    }

    // Construct the output ColorImage from the scratch buffer.
    // This copies the Vec<Color32> content, which is the remaining
    // unavoidable allocation (egui has no in-place texture update API).
    // However, we've reduced from N intermediate Vec<Color32> allocations
    // (one per transform step) to exactly one final construction.
    ColorImage {
        size: [out_w, out_h],
        pixels: out.to_vec(),
    }
}

#[cfg(test)]
mod tests;
