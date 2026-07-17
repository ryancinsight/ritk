//! Fast chamfer distance propagation and the approximate signed distance map
//! (`itk::FastChamferDistanceImageFilter`, `itk::ApproximateSignedDistanceMapImageFilter`).
//!
//! # Mathematical Specification
//!
//! [`FastChamferDistanceFilter`] propagates a signed narrow-band distance field
//! outward with a two-pass chamfer sweep (Butt & Maragos weights). For 3-D the
//! per-neighbour-class weights are `[0.92644, 1.34065, 1.65849]` (face, edge,
//! corner). Pass 1 raster-scans forward propagating to the "ahead" neighbours,
//! pass 2 scans backward to the "behind" neighbours; voxels with `|value| ≥
//! maximum_distance` are not propagation sources. Updates take the minimum on the
//! positive side and the maximum on the negative side, so the field stays signed.
//!
//! [`ApproximateSignedDistanceMapFilter`] composes ITK's mini-pipeline: an
//! [`IsoContourDistanceFilter`] at level `(inside + outside)/2` and far value
//! `d+1` (`d = ⌊√Σ sizeᵢ²⌋`), then a chamfer sweep with `maximum_distance = d`.
//! ritk's iso-contour is inside-positive; ITK's signed-distance convention is
//! inside-negative, so the iso field is negated before the chamfer (the chamfer
//! is antisymmetric, `chamfer(−f) = −chamfer(f)`). Float-exact to SimpleITK.

use anyhow::Result;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

use crate::iso_contour::IsoContourDistanceFilter;

/// 3-D chamfer weights by neighbour class: `[face, edge, corner]`.
const WEIGHTS: [f32; 3] = [0.92644, 1.34065, 1.65849];

/// A neighbour offset `(dz, dy, dx, weight)` in the chamfer sweep.
type Neighbour = (i64, i64, i64, f32);

/// Fast chamfer distance propagation over a signed narrow band.
#[derive(Debug, Clone)]
pub struct FastChamferDistanceFilter {
    /// Voxels with `|value| ≥ maximum_distance` are not propagated (ITK default 10).
    pub maximum_distance: f64,
}

impl Default for FastChamferDistanceFilter {
    fn default() -> Self {
        Self {
            maximum_distance: 10.0,
        }
    }
}

/// Forward neighbour offsets (linear index > 13 in the 3×3×3 kernel).
///
/// Each entry is `(dz, dy, dx, weight)`. Linear index: `(dz+1)*9 + (dy+1)*3 + (dx+1)`.
/// Class: `|dz|+|dy|+|dx| - 1`; weights `[face=0.92644, edge=1.34065, corner=1.65849]`.
const FWD_NEIGHBOURS: &[Neighbour] = &[
    (0, 0, 1, 0.92644),   // lin=14 face  class=0
    (0, 1, -1, 1.34065),  // lin=15 edge  class=1
    (0, 1, 0, 0.92644),   // lin=16 face  class=0
    (0, 1, 1, 1.34065),   // lin=17 edge  class=1
    (1, -1, -1, 1.65849), // lin=18 corner class=2
    (1, -1, 0, 1.34065),  // lin=19 edge  class=1
    (1, -1, 1, 1.65849),  // lin=20 corner class=2
    (1, 0, -1, 1.34065),  // lin=21 edge  class=1
    (1, 0, 0, 0.92644),   // lin=22 face  class=0
    (1, 0, 1, 1.34065),   // lin=23 edge  class=1
    (1, 1, -1, 1.65849),  // lin=24 corner class=2
    (1, 1, 0, 1.34065),   // lin=25 edge  class=1
    (1, 1, 1, 1.65849),   // lin=26 corner class=2
];

/// Backward neighbour offsets (linear index < 13 in the 3×3×3 kernel).
///
/// Each entry is `(dz, dy, dx, weight)`. Linear index: `(dz+1)*9 + (dy+1)*3 + (dx+1)`.
/// Class: `|dz|+|dy|+|dx| - 1`; weights `[face=0.92644, edge=1.34065, corner=1.65849]`.
const BWD_NEIGHBOURS: &[Neighbour] = &[
    (-1, -1, -1, 1.65849), // lin=0  corner class=2
    (-1, -1, 0, 1.34065),  // lin=1  edge   class=1
    (-1, -1, 1, 1.65849),  // lin=2  corner class=2
    (-1, 0, -1, 1.34065),  // lin=3  edge   class=1
    (-1, 0, 0, 0.92644),   // lin=4  face   class=0
    (-1, 0, 1, 1.34065),   // lin=5  edge   class=1
    (-1, 1, -1, 1.65849),  // lin=6  corner class=2
    (-1, 1, 0, 1.34065),   // lin=7  edge   class=1
    (-1, 1, 1, 1.65849),   // lin=8  corner class=2
    (0, -1, -1, 1.34065),  // lin=9  edge   class=1
    (0, -1, 0, 0.92644),   // lin=10 face   class=0
    (0, -1, 1, 1.34065),   // lin=11 edge   class=1
    (0, 0, -1, 0.92644),   // lin=12 face   class=0
];

impl FastChamferDistanceFilter {
    /// Run the chamfer sweep over a signed field (modifies a copy of the values).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (mut out, dims) = extract_vec_infallible(image);
        self.run(&mut out, dims);
        rebuild(out, dims, image)
    }

    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B::default()) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (mut out, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        self.run(&mut out, dims);
        crate::native_support::rebuild_image(out, dims, image, backend)
    }

    /// In-place chamfer sweep over a flat `[z, y, x]` buffer.
    pub(crate) fn run(&self, out: &mut [f32], dims: [usize; 3]) {
        let [nz, ny, nx] = dims;
        let md = self.maximum_distance as f32;
        let w0 = WEIGHTS[0];
        let idx = |z: usize, y: usize, x: usize| (z * ny + y) * nx + x;

        let n = nz * ny * nx;
        let mut sweep = |reverse: bool, nbrs: &[Neighbour]| {
            for step in 0..n {
                let c = if reverse { n - 1 - step } else { step };
                let (z, y, x) = (c / (ny * nx), (c % (ny * nx)) / nx, c % nx);
                let v = out[idx(z, y, x)];
                if v >= md || v <= -md {
                    continue;
                }
                let pos = v > -w0;
                let neg = v < w0;
                for &(dz, dy, dx, w) in nbrs {
                    let (zi, yi, xi) = (z as i64 + dz, y as i64 + dy, x as i64 + dx);
                    if zi < 0
                        || yi < 0
                        || xi < 0
                        || zi >= nz as i64
                        || yi >= ny as i64
                        || xi >= nx as i64
                    {
                        continue;
                    }
                    let ni = idx(zi as usize, yi as usize, xi as usize);
                    if pos {
                        let val = v + w;
                        if val < out[ni] {
                            out[ni] = val;
                        }
                    }
                    if neg {
                        let val = v - w;
                        if val > out[ni] {
                            out[ni] = val;
                        }
                    }
                }
            }
        };

        sweep(false, FWD_NEIGHBOURS);
        sweep(true, BWD_NEIGHBOURS);
    }
}

/// Approximate signed distance map (`itk::ApproximateSignedDistanceMapImageFilter`).
#[derive(Debug, Clone)]
pub struct ApproximateSignedDistanceMapFilter {
    /// Pixel value denoting the inside region (ITK/sitk default 1.0).
    pub inside_value: f64,
    /// Pixel value denoting the outside region (ITK/sitk default 0.0).
    pub outside_value: f64,
}

impl Default for ApproximateSignedDistanceMapFilter {
    fn default() -> Self {
        Self {
            inside_value: 1.0,
            outside_value: 0.0,
        }
    }
}

impl ApproximateSignedDistanceMapFilter {
    /// Compute the approximate signed distance map (inside negative, outside positive).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let dims = image.shape();
        let diag2: f64 = dims.iter().map(|&s| (s * s) as f64).sum();
        let max_distance = diag2.sqrt().floor();
        let level = (self.inside_value + self.outside_value) / 2.0;

        let iso = IsoContourDistanceFilter::new(level, max_distance + 1.0).apply(image);
        let (iso_vals, _) = extract_vec_infallible(&iso);
        // ritk iso is inside-positive; negate to ITK's inside-negative convention.
        let mut out: Vec<f32> = iso_vals.iter().map(|&v| -v).collect();
        FastChamferDistanceFilter {
            maximum_distance: max_distance,
        }
        .run(&mut out, dims);
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B::default()) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let dims = image.shape();
        let diag2: f64 = dims.iter().map(|&s| (s * s) as f64).sum();
        let max_distance = diag2.sqrt().floor();
        let level = (self.inside_value + self.outside_value) / 2.0;

        let iso = IsoContourDistanceFilter::new(level, max_distance + 1.0).apply_native(image, backend)?;
        let (iso_vals, _) = ritk_tensor_ops::native::extract_image_vec(&iso)?;
        let mut out: Vec<f32> = iso_vals.iter().map(|&v| -v).collect();
        FastChamferDistanceFilter {
            maximum_distance: max_distance,
        }
        .run(&mut out, dims);
        crate::native_support::rebuild_image(out, dims, image, backend)
    }
}

#[cfg(test)]
#[path = "tests_fast_chamfer.rs"]
mod tests_fast_chamfer;
