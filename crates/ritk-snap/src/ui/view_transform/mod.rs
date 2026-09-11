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
mod image_ops;
mod transform;

pub(crate) use image_ops::apply_to_image_into;
#[cfg(any(windows, test))]
pub(crate) use image_ops::apply_to_rgba;
pub use image_ops::{apply_to_image, flip_h_image, flip_v_image, rotate_90_cw_image};
pub use transform::{RotationSteps, ViewTransform};

#[cfg(test)]
mod tests;
