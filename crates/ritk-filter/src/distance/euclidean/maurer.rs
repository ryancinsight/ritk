//! Signed Maurer distance map filter (ITK `SignedMaurerDistanceMapImageFilter`).
//!
//! # Mathematical Specification
//!
//! ITK computes the signed distance to the **object border**, defined exactly as
//! in `itkSignedMaurerDistanceMapImageFilter.hxx`:
//!
//! 1. Binarize: foreground `S = { x : in(x) ≠ background_value }`.
//! 2. Border `B`: foreground voxels having at least one background voxel in their
//!    **fully-connected** neighbourhood (8-connected in 2-D, 26-connected in 3-D —
//!    ITK uses `BinaryContourImageFilter` with `FullyConnected = true`). Voxels
//!    outside the image are *not* treated as background.
//! 3. Distance: `d(x) = min_{b ∈ B} ‖x − b‖₂` (exact Euclidean, physical units),
//!    computed with the linear-time Meijster transform [`super::core::euclidean_dt`].
//! 4. Sign: with `inside_is_positive = false` (ITK default), foreground voxels get
//!    `−d`, background voxels `+d`; with `true`, the signs are swapped.
//!
//! `squared_distance` returns `d²` (signed); `use_image_spacing = false` ignores
//! the image spacing (unit voxels).
//!
//! Validated bit-exact (max-err 0.0) against `sitk.SignedMaurerDistanceMap` across
//! circle, L-shape, random-blob and signed-distance-threshold inputs.
//!
//! ## References
//! - Maurer, C., Qi, R., Raghavan, V. (2003). "A Linear Time Algorithm for
//!   Computing Exact Euclidean Distance Transforms of Binary Images in Arbitrary
//!   Dimensions." *IEEE TPAMI*, 25(2), 265–270.
//! - ITK `itkSignedMaurerDistanceMapImageFilter.hxx`.

use super::core::euclidean_dt;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Signed Maurer distance map (exact signed Euclidean distance to the object border).
///
/// Bit-exact to `sitk.SignedMaurerDistanceMap`.
///
/// # Defaults (match ITK)
/// - `background_value = 0.0`
/// - `inside_is_positive = false` (foreground negative)
/// - `squared_distance = true` (ITK default; `sitk` exposes both)
/// - `use_image_spacing = true`
#[derive(Debug, Clone)]
pub struct SignedMaurerDistanceMapImageFilter {
    /// Pixel value identifying background; foreground is everything `!=` this.
    pub background_value: f32,
    /// If `true`, inside (foreground) distances are positive.
    pub inside_is_positive: bool,
    /// If `true`, return signed squared distance `d²`; else `d`.
    pub squared_distance: bool,
    /// If `true`, use the image spacing; else unit voxels.
    pub use_image_spacing: bool,
}

impl Default for SignedMaurerDistanceMapImageFilter {
    fn default() -> Self {
        Self {
            background_value: 0.0,
            inside_is_positive: false,
            squared_distance: true,
            use_image_spacing: true,
        }
    }
}

impl SignedMaurerDistanceMapImageFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let [nz, ny, nx] = dims;
        let (vals, _) = extract_vec_infallible(image);
        let fg: Vec<bool> = vals.iter().map(|&v| v != self.background_value).collect();
        let sp = image.spacing();
        let spacing = if self.use_image_spacing {
            [sp[0], sp[1], sp[2]]
        } else {
            [1.0, 1.0, 1.0]
        };

        let signed = signed_maurer_core(
            &fg,
            dims,
            spacing,
            self.inside_is_positive,
            self.squared_distance,
        );

        Ok(rebuild(signed, [nz, ny, nx], image))
    }
}

/// Signed Maurer distance for a boolean foreground mask (reused by level-set
/// reinitialization). Returns signed distance to the object border, with
/// foreground negative when `inside_is_positive == false`.
pub(crate) fn signed_maurer_core(
    fg: &[bool],
    dims: [usize; 3],
    spacing: [f64; 3],
    inside_is_positive: bool,
    squared: bool,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let idx = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;

    // Border: foreground voxels with a background voxel in the full (8/26-conn)
    // neighbourhood. Image-exterior is NOT background (matches ITK contour).
    let mut border = vec![false; fg.len()];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let f = idx(z, y, x);
                if !fg[f] {
                    continue;
                }
                let mut is_border = false;
                'scan: for dz in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dz == 0 && dy == 0 && dx == 0 {
                                continue;
                            }
                            let (z2, y2, x2) = (z as i32 + dz, y as i32 + dy, x as i32 + dx);
                            if z2 < 0
                                || y2 < 0
                                || x2 < 0
                                || z2 >= nz as i32
                                || y2 >= ny as i32
                                || x2 >= nx as i32
                            {
                                continue;
                            }
                            if !fg[idx(z2 as usize, y2 as usize, x2 as usize)] {
                                is_border = true;
                                break 'scan;
                            }
                        }
                    }
                }
                border[f] = is_border;
            }
        }
    }

    let d = euclidean_dt(&border, dims, spacing);
    let inside_sign = if inside_is_positive { 1.0f32 } else { -1.0f32 };
    fg.iter()
        .zip(d.iter())
        .map(|(&is_fg, &dist)| {
            let v = if squared { dist * dist } else { dist };
            if is_fg {
                inside_sign * v
            } else {
                -inside_sign * v
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "../tests_maurer.rs"]
mod tests_maurer;
