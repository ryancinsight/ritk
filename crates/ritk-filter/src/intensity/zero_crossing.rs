//! Zero-crossing detection filter.
//!
//! # Mathematical Specification
//!
//! Voxel `x` is marked a zero crossing if:
//!
//! 1. `I(x) == 0.0`, **or**
//! 2. some 6-connected neighbour `y` has the opposite sign (`I(x)·I(y) < 0`)
//!    **and** `x` is the side closer to zero. Per axis the forward (+) neighbour
//!    is accepted with `|I(x)| <= |I(y)|` and the backward (−) neighbour with the
//!    strict `|I(x)| < |I(y)|`, so an exact-magnitude tie is resolved toward the
//!    forward voxel.
//!
//! This matches ITK `ZeroCrossingImageFilter`, which marks only the voxel on the
//! near-zero side of each crossing — marking *both* sides (every voxel with any
//! opposite-sign neighbour) roughly doubles the detected crossings.
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
                    // Case 1: exact zero is always a crossing.
                    if v == 0.0 {
                        out[idx(iz, iy, ix)] = fg;
                        continue;
                    }
                    // Case 2: opposite-sign 6-connected neighbour, but mark this
                    // voxel only on the side *closer to zero* (matching ITK
                    // ZeroCrossingImageFilter — otherwise both sides of every
                    // crossing get marked, ~doubling the detections). Per axis the
                    // forward (+) neighbour uses `|v| <= |nv|` and the backward (−)
                    // neighbour uses the strict `|v| < |nv|`, so an exact-magnitude
                    // tie is resolved toward the forward voxel.
                    let in_b = |z: isize, y: isize, x: isize| {
                        z >= 0
                            && y >= 0
                            && x >= 0
                            && z < nz as isize
                            && y < ny as isize
                            && x < nx as isize
                    };
                    let av = v.abs();
                    let mut crosses = false;
                    // axis offsets: (dz, dy, dx)
                    for &(dz, dy, dx) in &[(1isize, 0isize, 0isize), (0, 1, 0), (0, 0, 1)] {
                        let (fz, fy, fx) = (iz as isize + dz, iy as isize + dy, ix as isize + dx);
                        if in_b(fz, fy, fx) {
                            let nv = vals[idx(fz as usize, fy as usize, fx as usize)];
                            if v * nv < 0.0 && av <= nv.abs() {
                                crosses = true;
                                break;
                            }
                        }
                        let (bz, by, bx) = (iz as isize - dz, iy as isize - dy, ix as isize - dx);
                        if in_b(bz, by, bx) {
                            let nv = vals[idx(bz as usize, by as usize, bx as usize)];
                            if v * nv < 0.0 && av < nv.abs() {
                                crosses = true;
                                break;
                            }
                        }
                    }
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
