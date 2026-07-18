//! Zero-crossing detection filter.
//!
//! # Mathematical Specification
//!
//! Voxel `x` is marked a zero crossing when some 6-connected neighbour `y` has a
//! **sign change** â€” opposite signs (`I(x)Â·I(y) < 0`) or exactly one of the two
//! is zero â€” **and** `x` is the side closer to zero. Per axis the forward (+)
//! neighbour is accepted with `|I(x)| <= |I(y)|` and the backward (âˆ’) neighbour
//! with the strict `|I(x)| < |I(y)|`, so an exact-magnitude tie is resolved
//! toward the forward voxel.
//!
//! This is float-exact to ITK `ZeroCrossingImageFilter`, which marks only the
//! voxel on the near-zero side of each crossing. An exact zero is a crossing only
//! when it has a non-zero neighbour â€” a zero with all-zero neighbours (e.g. the
//! Laplacian of a flat region) is **not** marked, otherwise whole constant
//! regions would be flagged.
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
use ritk_image::tensor::Backend;
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
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals_vec, dims) = extract_vec_infallible(image);
        let out = zero_crossing_vec(
            &vals_vec,
            dims,
            f32::from(self.foreground_value),
            self.background_value,
        );
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`ZeroCrossingImageFilter::apply`].
    ///
    /// Runs the identical 6-connected sign-change detection via the shared
    /// `zero_crossing_vec` host core on the image's contiguous host buffer, so
    /// the result is bitwise-identical to the Burn path. No Burn tensor is
    /// constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let fg = f32::from(self.foreground_value);
        let bg = self.background_value;
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            zero_crossing_vec(vals, dims, fg, bg)
        })
    }
}

/// Substrate-agnostic host core for [`ZeroCrossingImageFilter`].
///
/// Emits `fg` at every voxel that is the near-zero side of a 6-connected sign
/// change and `bg` elsewhere (see the module-level ITK-parity specification).
/// Out-of-bounds neighbours are ignored.
pub(crate) fn zero_crossing_vec(vals: &[f32], dims: [usize; 3], fg: f32, bg: f32) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let idx = |iz: usize, iy: usize, ix: usize| iz * ny * nx + iy * nx + ix;
    let mut out = vec![bg; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let v = vals[idx(iz, iy, ix)];
                // ITK ZeroCrossingImageFilter: a voxel is a crossing if its
                // sign differs from a 6-connected neighbour (opposite signs, or
                // exactly one of the two is zero) AND it is the side *closer to
                // zero* (`|v| < |nv|` strictly; an exact-magnitude tie is broken
                // toward the forward `+` neighbour via `<=`). An exact zero is
                // NOT a crossing on its own â€” only when it has a non-zero
                // neighbour (otherwise flat zero regions, e.g. the Laplacian of
                // constant background, would be marked wholesale).
                let in_b = |z: isize, y: isize, x: isize| {
                    z >= 0
                        && y >= 0
                        && x >= 0
                        && z < nz as isize
                        && y < ny as isize
                        && x < nx as isize
                };
                // Sign change: opposite signs, or exactly one operand is zero.
                let sign_change = |a: f32, b: f32| (a * b < 0.0) || ((a == 0.0) != (b == 0.0));
                let av = v.abs();
                let mut crosses = false;
                // axis offsets: (dz, dy, dx)
                for &(dz, dy, dx) in &[(1isize, 0isize, 0isize), (0, 1, 0), (0, 0, 1)] {
                    let (fz, fy, fx) = (iz as isize + dz, iy as isize + dy, ix as isize + dx);
                    if in_b(fz, fy, fx) {
                        let nv = vals[idx(fz as usize, fy as usize, fx as usize)];
                        if sign_change(v, nv) && av <= nv.abs() {
                            crosses = true;
                            break;
                        }
                    }
                    let (bz, by, bx) = (iz as isize - dz, iy as isize - dy, ix as isize - dx);
                    if in_b(bz, by, bx) {
                        let nv = vals[idx(bz as usize, by as usize, bx as usize)];
                        if sign_change(v, nv) && av < nv.abs() {
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
    out
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_zero_crossing.rs"]
mod tests_zero_crossing;
