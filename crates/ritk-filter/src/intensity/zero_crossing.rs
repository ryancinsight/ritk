//! Zero-crossing detection filter.
//!
//! # Mathematical Specification
//!
//! A zero crossing exists at voxel x if:
//!
//! 1. `I(x) == 0.0`, **or**
//! 2. There exists a 6-connected neighbour y such that `I(x) * I(y) < 0`
//!    (i.e. x and y have strictly opposite signs).
//!
//! This matches the ITK `ZeroCrossingImageFilter` definition:
//! a voxel is a zero crossing if it equals exactly 0 or if at least one
//! 6-connected neighbour has an opposite sign.
//!
//! ## Invariants
//!
//! - `foreground_value` is emitted at zero-crossing voxels.
//! - `background_value` is emitted at all other voxels.
//! - Boundary voxels (edges of the volume) can only cross with in-bounds neighbours;
//!   out-of-bounds neighbours are ignored (do not trigger a crossing).
//! - Spatial metadata (shape, origin, spacing, direction) is preserved exactly.
//!
//! # ITK Parity
//!
//! `itk::ZeroCrossingImageFilter` with `SetForegroundValue` / `SetBackgroundValue`.
//! Typical use: detect zero crossings of a Laplacian-of-Gaussian edge image.

use crate::morphology::types::ForegroundValue;
use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Detect zero crossings in a 3-D image.
///
/// Output voxels equal `foreground_value` where a sign change or exact zero
/// occurs in the 6-connected neighbourhood, and `background_value` elsewhere.
#[derive(Debug, Clone)]
pub struct ZeroCrossingImageFilter {
    /// Value assigned to zero-crossing voxels (default 1.0).
    pub foreground_value: ForegroundValue,
    /// Value assigned to non-crossing voxels (default 0.0).
    pub background_value: f32,
}

impl Default for ZeroCrossingImageFilter {
    fn default() -> Self {
        Self {
            foreground_value: ForegroundValue::ONE,
            background_value: 0.0,
        }
    }
}

impl ZeroCrossingImageFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_foreground(mut self, v: impl Into<ForegroundValue>) -> Self {
        self.foreground_value = v.into();
        self
    }

    pub fn with_background(mut self, v: f32) -> Self {
        self.background_value = v;
        self
    }

    /// Apply to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec_infallible(image);
        let vals = &vals_vec;
        let [nz, ny, nx] = dims;

        let idx = |iz: usize, iy: usize, ix: usize| iz * ny * nx + iy * nx + ix;

        let fg = f32::from(self.foreground_value);
        let bg = self.background_value;

        let mut out = vec![bg; nz * ny * nx];

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = vals[idx(iz, iy, ix)];
                    // Case 1: exact zero
                    if v == 0.0 {
                        out[idx(iz, iy, ix)] = fg;
                        continue;
                    }
                    // Case 2: opposite-sign 6-connected neighbour
                    let neighbours: &[(isize, isize, isize)] = &[
                        (-1, 0, 0),
                        (1, 0, 0),
                        (0, -1, 0),
                        (0, 1, 0),
                        (0, 0, -1),
                        (0, 0, 1),
                    ];
                    let crosses = neighbours.iter().any(|&(dz, dy, dx)| {
                        let nz_ = iz as isize + dz;
                        let ny_ = iy as isize + dy;
                        let nx_ = ix as isize + dx;
                        if nz_ < 0
                            || ny_ < 0
                            || nx_ < 0
                            || nz_ >= nz as isize
                            || ny_ >= ny as isize
                            || nx_ >= nx as isize
                        {
                            return false;
                        }
                        let nv = vals[idx(nz_ as usize, ny_ as usize, nx_ as usize)];
                        v * nv < 0.0
                    });
                    if crosses {
                        out[idx(iz, iy, ix)] = fg;
                    }
                }
            }
        }

        Ok(rebuild(out, dims, image))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_zero_crossing.rs"]
mod tests_zero_crossing;
