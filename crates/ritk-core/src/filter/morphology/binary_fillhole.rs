//! Binary hole filling filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Given a binary image f with foreground `fg` and background `0`:
//!
//! 1. Compute the set of background voxels reachable from the image border
//!    via 6-connected BFS (external background):
//!       E = { x ∈ background(f) : x is 6-connected to any border voxel }
//!
//! 2. Background voxels not in E are interior holes:
//!       H = background(f) \ E
//!
//! 3. Output:
//!       output(x) = fg   if f(x) = fg  or  x ∈ H
//!                 = 0    if x ∈ E
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

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
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
    foreground_value: f32,
}

impl BinaryFillholeFilter {
    /// Create a hole-filling filter with default `foreground_value = 1.0`.
    pub fn new() -> Self {
        Self {
            foreground_value: 1.0,
        }
    }

    /// Set the foreground value (ITK `SetForegroundValue`).
    pub fn with_foreground(mut self, v: f32) -> Self {
        self.foreground_value = v;
        self
    }

    /// Apply binary hole filling to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata.
    /// Output voxels are `foreground_value` (foreground or hole) or `0.0` (external background).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let td = image.data().clone().into_data();
        let vals: &[f32] = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("BinaryFillholeFilter requires f32 data: {:?}", e))?;

        let result = fill_holes_3d(vals, dims, self.foreground_value);

        let device = image.data().device();
        let t = Tensor::<B, 3>::from_data(TensorData::new(result, Shape::new(dims)), &device);
        Ok(Image::new(
            t,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
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
fn fill_holes_3d(data: &[f32], dims: [usize; 3], fg: f32) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    // `reached[i]` = true if voxel i is external background (BFS-reachable from border).
    let mut reached = vec![false; n];
    let mut queue: VecDeque<usize> = VecDeque::new();

    // ── Seed: all border background voxels ─────────────────────────────────
    let seed = |iz: usize, iy: usize, ix: usize, data: &[f32], reached: &mut Vec<bool>, queue: &mut VecDeque<usize>| {
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

        if iz > 0         { try_neighbor!(iz - 1, iy, ix); }
        if iz + 1 < nz    { try_neighbor!(iz + 1, iy, ix); }
        if iy > 0         { try_neighbor!(iz, iy - 1, ix); }
        if iy + 1 < ny    { try_neighbor!(iz, iy + 1, ix); }
        if ix > 0         { try_neighbor!(iz, iy, ix - 1); }
        if ix + 1 < nx    { try_neighbor!(iz, iy, ix + 1); }
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
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let t = Tensor::<B, 3>::from_data(TensorData::new(vals, Shape::new(dims)), &device);
        Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn flat(img: &Image<B, 3>) -> Vec<f32> {
        img.data().clone().into_data().as_slice::<f32>().unwrap().to_vec()
    }

    /// T1: All-foreground image stays all-foreground.
    #[test]
    fn all_foreground_unchanged() {
        let img = make_image(vec![1.0; 8], [2, 2, 2]);
        let out = BinaryFillholeFilter::new().apply(&img).unwrap();
        assert!(flat(&out).iter().all(|&v| v == 1.0));
    }

    /// T2: All-background image — all voxels are external bg (reachable from
    ///     border), so output is all background.
    #[test]
    fn all_background_stays_background() {
        let img = make_image(vec![0.0; 8], [2, 2, 2]);
        let out = BinaryFillholeFilter::new().apply(&img).unwrap();
        assert!(flat(&out).iter().all(|&v| v == 0.0));
    }

    /// T3: Enclosed hole is filled.
    ///
    /// 3×3×3 volume: foreground shell (outer voxels) with a single background
    /// voxel at the centre (index 13 in ZYX order).
    ///
    /// The centre voxel at (1,1,1) is background and NOT reachable from any
    /// border voxel (all 6 face-neighbors are foreground), so it must be filled.
    ///
    /// Construction:
    ///   All 27 voxels = 1.0 except centre voxel index 13 = 0.0.
    #[test]
    fn enclosed_hole_filled() {
        let mut vals = vec![1.0_f32; 27];
        vals[13] = 0.0; // centre of 3×3×3 = (1,1,1), index = 1*9+1*3+1 = 13
        let img = make_image(vals, [3, 3, 3]);
        let out = BinaryFillholeFilter::new().apply(&img).unwrap();
        assert_eq!(flat(&out)[13], 1.0, "enclosed hole at centre must be filled");
    }

    /// T4: Interior background region in 5×5×5 volume is filled.
    ///
    /// 5×5×5 volume with fg outer shell and bg interior (iz∈{1..3}, iy∈{1..3}, ix∈{1..3}).
    /// The inner 3×3×3 = 27 voxels are bg and not reachable from any border face
    /// (all 6 immediate Z/Y/X face-neighbours of the inner region are fg → no bg path).
    /// After filling, all inner voxels must be fg.
    #[test]
    fn interior_bg_region_filled() {
        let mut vals = vec![1.0_f32; 125];
        for iz in 1..=3usize {
            for iy in 1..=3usize {
                for ix in 1..=3usize {
                    vals[iz * 25 + iy * 5 + ix] = 0.0;
                }
            }
        }
        let img = make_image(vals, [5, 5, 5]);
        let out = BinaryFillholeFilter::new().apply(&img).unwrap();
        let result = flat(&out);
        for iz in 1..=3usize {
            for iy in 1..=3usize {
                for ix in 1..=3usize {
                    let idx = iz * 25 + iy * 5 + ix;
                    assert_eq!(result[idx], 1.0, "inner voxel ({iz},{iy},{ix}) must be filled");
                }
            }
        }
    }

    /// T5: Extensivity — no foreground voxel is removed.
    #[test]
    fn extensivity_no_foreground_removed() {
        let vals: Vec<f32> = vec![
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        // 1×3×3 flat slice, centre bg at index 4.
        let img = make_image(vals.clone(), [1, 3, 3]);
        let out = BinaryFillholeFilter::new().apply(&img).unwrap();
        let result = flat(&out);
        for (i, &v) in vals.iter().enumerate() {
            if v == 1.0 {
                assert_eq!(result[i], 1.0, "fg voxel {i} was incorrectly removed");
            }
        }
    }

    /// T6: Custom foreground value fills enclosed bg with fg value.
    ///
    /// 3×3×3 shell of fg=255 with interior bg at (1,1,1) — same geometry as T3
    /// but with fg=255.  The centre voxel is enclosed and must be filled to 255.
    #[test]
    fn custom_foreground_value() {
        let mut vals = vec![255.0_f32; 27];
        vals[13] = 0.0; // centre (1,1,1) = bg
        let img = make_image(vals, [3, 3, 3]);
        let out = BinaryFillholeFilter::new().with_foreground(255.0).apply(&img).unwrap();
        assert_eq!(flat(&out)[13], 255.0, "enclosed bg centre must be filled to 255");
    }

    /// T7: Spatial metadata preserved.
    #[test]
    fn spatial_metadata_preserved() {
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let origin = Point::new([5.0, 3.0, 1.0]);
        let spacing = Spacing::new([0.8, 0.8, 1.2]);
        let direction = Direction::identity();
        let t = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0_f32; 8], Shape::new([2, 2, 2])),
            &device,
        );
        let img = Image::new(t, origin, spacing, direction);
        let out = BinaryFillholeFilter::new().apply(&img).unwrap();
        assert_eq!(*out.origin(), origin);
        assert_eq!(*out.spacing(), spacing);
        assert_eq!(*out.direction(), direction);
    }
}
