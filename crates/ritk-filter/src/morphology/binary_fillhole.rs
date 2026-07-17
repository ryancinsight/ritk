//! Binary hole filling filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Given a binary image f with foreground `fg` and background `0`:
//!
//! 1. Compute the set of background voxels reachable from the image border
//!    via 6-connected BFS (external background):
//!    E = { x ∈ background(f) : x is 6-connected to any border voxel }
//!
//! 2. Background voxels not in E are interior holes:
//!    H = background(f) \ E
//!
//! 3. Output:
//!    output(x) = fg   if f(x) = fg  or  x ∈ H
//!    output(x) = 0    if x ∈ E
//!
//! # Properties
//!
//! - **Extensivity**: `output(x) ≥ f(x)` — holes are filled; no fg removed.
//! - **Topology preservation**: fills enclosed cavities only.
//! - All foreground voxels in f remain foreground in the output.
//! - Background voxels connected to the image border remain background.
//!
//! # ITK Parity
//!
//! Matches `itk::BinaryFillholeImageFilter` with:
//! - `SetForegroundValue(foreground_value)` (default 1.0)
//! - `SetFullyConnected(false)` (6-connectivity, ITK default)
//!
//! # Complexity
//!
//! O(N) for BFS flood fill + O(N) for output generation, where N is total
//! voxel count.
//!
//! # References
//!
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer §5.6.
//! - ITK Software Guide, Vol 2, §6.3.4 Binary Fillhole Image Filter.

use super::types::ForegroundValue;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};
use std::collections::VecDeque;

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Binary hole filling filter for 3-D images.
///
/// Fills enclosed cavities (background regions not connected to the image
/// border) by setting them to the foreground value.  Uses a 6-connected BFS
/// flood fill starting from all border voxels.
#[derive(Debug, Clone)]
pub struct BinaryFillholeFilter {
    /// Voxel value treated as foreground. Default: 1.0.
    foreground_value: ForegroundValue,
}

impl BinaryFillholeFilter {
    /// Create a hole-filling filter with default `foreground_value = 1.0`.
    pub fn new() -> Self {
        Self {
            foreground_value: ForegroundValue::ONE,
        }
    }

    /// Set the foreground value (ITK `SetForegroundValue`).
    pub fn with_foreground(mut self, v: impl Into<ForegroundValue>) -> Self {
        self.foreground_value = v.into();
        self
    }

    /// Apply binary hole filling to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata.
    /// Output voxels are `foreground_value` (foreground or hole) or `0.0` (external background).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec_infallible(image);

        let result = fill_holes_3d(&vals, dims, self.foreground_value);

        Ok(rebuild(result, dims, image))
    }

    /// Coeus-native sister of [`BinaryFillholeFilter::apply`].
    ///
    /// Runs the identical 6-connected BFS hole fill via the shared
    /// `fill_holes_3d` host core on the image's contiguous host buffer, so the
    /// result is bitwise-identical to the Burn path. No Burn tensor is
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
        let fg = self.foreground_value;
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            fill_holes_3d(vals, dims, fg)
        })
    }
}

impl Default for BinaryFillholeFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Core algorithm ────────────────────────────────────────────────────────────

/// Binary hole filling on a flat Z×Y×X volume via 6-connected BFS.
///
/// # Algorithm
///
/// 1. Seed the BFS queue with every background voxel on the 6 image faces.
/// 2. BFS propagates through background voxels using 6-connected adjacency.
/// 3. Any background voxel NOT reached by BFS is a hole → set to fg.
/// 4. All foreground voxels are preserved as fg.
///
/// # Invariants
///
/// - `output.len() == nz * ny * nx`.
/// - `output[i] ∈ {fg, 0.0}`.
/// - `f(i) == fg ⇒ output[i] == fg` (extensivity).
/// - `i ∈ E ⇒ output[i] == 0.0` (external bg preserved).
pub(crate) fn fill_holes_3d(data: &[f32], dims: [usize; 3], fg: ForegroundValue) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let fg: f32 = fg.into();

    // `reached[i]` = true if voxel i is external background (BFS-reachable from border).
    let mut reached = vec![false; n];
    let mut queue: VecDeque<usize> = VecDeque::new();

    // ── Seed: all border background voxels ─────────────────────────────────
    let seed = |iz: usize,
                iy: usize,
                ix: usize,
                data: &[f32],
                reached: &mut Vec<bool>,
                queue: &mut VecDeque<usize>| {
        let idx = iz * ny * nx + iy * nx + ix;
        if data[idx] != fg && !reached[idx] {
            reached[idx] = true;
            queue.push_back(idx);
        }
    };

    // Z faces
    for iy in 0..ny {
        for ix in 0..nx {
            seed(0, iy, ix, data, &mut reached, &mut queue);
            if nz > 1 {
                seed(nz - 1, iy, ix, data, &mut reached, &mut queue);
            }
        }
    }
    // Y faces
    for iz in 0..nz {
        for ix in 0..nx {
            seed(iz, 0, ix, data, &mut reached, &mut queue);
            if ny > 1 {
                seed(iz, ny - 1, ix, data, &mut reached, &mut queue);
            }
        }
    }
    // X faces
    for iz in 0..nz {
        for iy in 0..ny {
            seed(iz, iy, 0, data, &mut reached, &mut queue);
            if nx > 1 {
                seed(iz, iy, nx - 1, data, &mut reached, &mut queue);
            }
        }
    }

    // ── BFS through background using 6-connectivity ────────────────────────
    while let Some(idx) = queue.pop_front() {
        let iz = idx / (ny * nx);
        let rem = idx % (ny * nx);
        let iy = rem / nx;
        let ix = rem % nx;

        macro_rules! try_neighbor {
            ($nz:expr, $ny_:expr, $nx_:expr) => {
                let nidx = $nz * ny * nx + $ny_ * nx + $nx_;
                if data[nidx] != fg && !reached[nidx] {
                    reached[nidx] = true;
                    queue.push_back(nidx);
                }
            };
        }

        if iz > 0 {
            try_neighbor!(iz - 1, iy, ix);
        }
        if iz + 1 < nz {
            try_neighbor!(iz + 1, iy, ix);
        }
        if iy > 0 {
            try_neighbor!(iz, iy - 1, ix);
        }
        if iy + 1 < ny {
            try_neighbor!(iz, iy + 1, ix);
        }
        if ix > 0 {
            try_neighbor!(iz, iy, ix - 1);
        }
        if ix + 1 < nx {
            try_neighbor!(iz, iy, ix + 1);
        }
    }

    // ── Build output ───────────────────────────────────────────────────────
    // output[i] = fg if original fg OR unreached background (hole).
    // output[i] = 0.0 if external background (reached).
    (0..n)
        .map(|i| {
            if data[i] == fg || !reached[i] {
                fg
            } else {
                0.0
            }
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_binary_fillhole.rs"]
mod tests_binary_fillhole;
