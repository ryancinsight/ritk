//! Grayscale fill-hole filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! A "hole" in a grayscale image is a dark regional minimum enclosed by a
//! brighter surrounding "wall" that is not connected to the image boundary.
//! The fill-hole operator raises each such hole to the level of the lowest
//! wall connecting it to the image border.
//!
//! Formally, the output H satisfies:
//!
//!   H(x) = min over all paths P from x to any border voxel of
//!           max_{q ∈ P} I(q)
//!
//! where I is the input image. This is the **widest-path** (minimax path)
//! from each voxel to the image boundary under the input intensity landscape.
//!
//! # Algorithm
//!
//! The minimax path from every voxel to the image border is computed in
//! O(N log N) via a Dijkstra-like priority-queue sweep:
//!
//! 1. Initialise:
//!    - `h[b] = I[b]` for every border voxel `b`; enqueue `(I[b], b)`.
//!    - `h[x] = +∞` for every interior voxel `x`.
//! 2. Pop the smallest-level voxel `(level, x)`.
//! 3. For each 6-connected neighbour `y` of `x`:
//!    - `new_level = max(level, I[y])`
//!    - If `new_level < h[y]`: set `h[y] = new_level` and enqueue `(new_level, y)`.
//! 4. Repeat until the queue is empty.
//!
//! # Boundary Definition
//!
//! A voxel is on the border if any of its coordinates equals 0 or the
//! corresponding dimension maximum: `iz = 0 OR iz = nz − 1 OR iy = 0 OR
//! iy = ny − 1 OR ix = 0 OR ix = nx − 1`.
//!
//! # ITK Parity
//!
//! Matches `itk::GrayscaleFillholeImageFilter` with:
//! - Fully symmetric (non-directional) fill.
//! - 6-connected boundary propagation.
//!
//! # Complexity
//!
//! O(N log N) where N = nz × ny × nx, dominated by the binary heap.
//!
//! # References
//!
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer,
//!   pp. 180–184.
//! - Vincent, L. (1993). Morphological grayscale reconstruction in image
//!   analysis: Applications and efficient algorithms. *IEEE Trans. Image
//!   Processing*, 2(2), 176–201.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Grayscale fill-hole filter for 3-D images.
///
/// Removes dark regional minima that are not connected to the image border.
/// Each enclosed dark pit is raised to the level of the lowest surrounding
/// "wall" connecting it to the image boundary (the minimax path level).
#[derive(Debug, Clone, Default)]
pub struct GrayscaleFillholeFilter;

impl GrayscaleFillholeFilter {
    /// Create a new grayscale fill-hole filter.
    pub fn new() -> Self {
        Self
    }

    /// Apply the fill-hole filter to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata.
    /// Output satisfies `h[x] ≥ I[x]` for all x.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let td = image.data().clone().into_data();
        let vals: &[f32] = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("GrayscaleFillholeFilter requires f32 data: {:?}", e))?;

        let filled = fill_holes_3d(vals, dims);

        let device = image.data().device();
        let out_td = TensorData::new(filled, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(out_td, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

// ── Core computation ──────────────────────────────────────────────────────────

/// Compute the grayscale fill-hole operation on a flat Z×Y×X volume.
///
/// # Algorithm
///
/// Dijkstra-like minimax-path sweep from all image border voxels.
/// Each output voxel h[x] = min over all border-connecting paths of
/// max(I[q]) along the path.
///
/// # Invariants
///
/// - `h[x] >= I[x]` for all x: holes can only be raised.
/// - `h[b] = I[b]` for all border voxels b.
/// - Output length = `nz * ny * nx`.
fn fill_holes_3d(data: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    // h[x] = minimax level from x to any border; initialise interior to +∞.
    let mut h = vec![f32::INFINITY; n];

    // Priority queue: min-heap on (level, flat_index).
    // `Reverse` on an `OrderedFloat`-compatible wrapper (f32 as u32 bits).
    // We compare levels as ordered floats via the total order on the u32 repr
    // of non-NaN, non-negative f32 values, which is monotone.
    let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();

    // Seed all border voxels.
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let on_border = iz == 0 || iz == nz - 1
                    || iy == 0 || iy == ny - 1
                    || ix == 0 || ix == nx - 1;
                if on_border {
                    let flat = iz * ny * nx + iy * nx + ix;
                    let v = data[flat];
                    h[flat] = v;
                    heap.push(Reverse((v.to_bits(), flat)));
                }
            }
        }
    }

    // 6-connected neighbour offsets in flat index space.
    let neighbours = |iz: usize, iy: usize, ix: usize| {
        let mut nbrs: [Option<usize>; 6] = [None; 6];
        let mut k = 0;
        if iz > 0       { nbrs[k] = Some((iz - 1) * ny * nx + iy * nx + ix); k += 1; }
        if iz + 1 < nz  { nbrs[k] = Some((iz + 1) * ny * nx + iy * nx + ix); k += 1; }
        if iy > 0       { nbrs[k] = Some(iz * ny * nx + (iy - 1) * nx + ix); k += 1; }
        if iy + 1 < ny  { nbrs[k] = Some(iz * ny * nx + (iy + 1) * nx + ix); k += 1; }
        if ix > 0       { nbrs[k] = Some(iz * ny * nx + iy * nx + (ix - 1)); k += 1; }
        if ix + 1 < nx  { nbrs[k] = Some(iz * ny * nx + iy * nx + (ix + 1)); k += 1; }
        let _ = k;
        nbrs
    };

    while let Some(Reverse((level_bits, flat))) = heap.pop() {
        let level = f32::from_bits(level_bits);
        // Stale entry: already processed at a lower level.
        if level > h[flat] + 1e-9 {
            continue;
        }
        let iz = flat / (ny * nx);
        let iy = (flat % (ny * nx)) / nx;
        let ix = flat % nx;
        for nb_opt in neighbours(iz, iy, ix) {
            if let Some(nb) = nb_opt {
                let new_level = level.max(data[nb]);
                if new_level < h[nb] {
                    h[nb] = new_level;
                    heap.push(Reverse((new_level.to_bits(), nb)));
                }
            }
        }
    }

    h
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
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    fn flat(iz: usize, iy: usize, ix: usize, ny: usize, nx: usize) -> usize {
        iz * ny * nx + iy * nx + ix
    }

    /// Constant image unchanged: no holes in a uniform field.
    ///
    /// **Proof**: every voxel has minimax path level = c. ∎
    #[test]
    fn constant_image_unchanged() {
        let c = 7.0_f32;
        let dims = [5, 5, 5];
        let img = make_image(vec![c; 125], dims);
        let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
        for &v in extract_vals(&out).iter() {
            assert!((v - c).abs() < 1e-6, "constant unchanged: got {v}");
        }
    }

    /// Output satisfies h[x] ≥ I[x] for all x (holes only raised, not lowered).
    #[test]
    fn output_ge_input_everywhere() {
        let dims = [6, 6, 6];
        let n = 216;
        let vals: Vec<f32> = (0..n as u32).map(|i| (i * 7919 % 128) as f32).collect();
        let img = make_image(vals.clone(), dims);
        let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
        let out_vals = extract_vals(&out);
        for (i, (&before, &after)) in vals.iter().zip(out_vals.iter()).enumerate() {
            assert!(
                after >= before - 1e-5,
                "output must be ≥ input at voxel {i}: before={before} after={after}"
            );
        }
    }

    /// Border voxels are never modified.
    ///
    /// **Proof**: border voxels are seeded with I[b]; they can only be updated
    /// by paths from other borders, which can never produce a lower level. ∎
    #[test]
    fn border_voxels_unchanged() {
        let [nz, ny, nx] = [5usize, 5, 5];
        let n = nz * ny * nx;
        let mut vals: Vec<f32> = (0..n as u32).map(|i| (i * 2017 % 64) as f32).collect();
        // Put a dark pit in the interior at (2,2,2)
        vals[flat(2, 2, 2, ny, nx)] = 0.0;
        let img = make_image(vals.clone(), [nz, ny, nx]);
        let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
        let out_vals = extract_vals(&out);
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let on_border = iz == 0 || iz == nz - 1
                        || iy == 0 || iy == ny - 1
                        || ix == 0 || ix == nx - 1;
                    if on_border {
                        let f = flat(iz, iy, ix, ny, nx);
                        assert!(
                            (out_vals[f] - vals[f]).abs() < 1e-6,
                            "border voxel [{iz},{iy},{ix}] changed"
                        );
                    }
                }
            }
        }
    }

    /// Enclosed dark pit filled to surrounding wall level.
    ///
    /// Volume: 3×3×3 (27 voxels). Only interior voxel is flat[13] at (1,1,1).
    /// I = 5 everywhere on border; I[1,1,1] = 0 (dark pit).
    ///
    /// Expected: h[1,1,1] = 5.0 (raised to border level).
    ///
    /// **Proof**: minimax path from (1,1,1) to any border passes through exactly
    /// one border voxel at level 5. max along path = max(0, 5) = 5. ∎
    #[test]
    fn enclosed_pit_filled_to_border_level() {
        let dims = [3usize, 3, 3];
        let n = 27;
        let mut vals = vec![5.0_f32; n];
        vals[flat(1, 1, 1, 3, 3)] = 0.0;
        let img = make_image(vals, dims);
        let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
        let out_vals = extract_vals(&out);
        let pit_out = out_vals[flat(1, 1, 1, 3, 3)];
        assert!(
            (pit_out - 5.0).abs() < 1e-5,
            "pit filled to 5.0; got {pit_out}"
        );
    }

    /// Enclosed pit filled to WALL level (not border level) when wall < border.
    ///
    /// Volume: 5×5×5. Outer shell (border) = 1.0. Inner shell at iz/iy/ix ∈ {1..3}
    /// = 8.0. Innermost voxel (2,2,2) = 2.0.
    ///
    /// The minimum-barrier path from (2,2,2) to the border must pass through
    /// the inner shell at level 8. Therefore h[2,2,2] = 8.0.
    ///
    /// **Proof**: any path from (2,2,2) to the border with |dx|+|dy|+|dz|=1 steps
    /// must pass through a voxel in the inner shell with value 8. The minimax
    /// path level is therefore min(8) = 8, not the border level 1. ∎
    #[test]
    fn pit_filled_to_wall_level_not_border_level() {
        let [nz, ny, nx] = [5usize, 5, 5];
        let n = nz * ny * nx;
        let mut vals = vec![0.0_f32; n]; // all outer = 0 (overwritten below)
        // Outer shell: value 1
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let on_border = iz == 0 || iz == nz - 1
                        || iy == 0 || iy == ny - 1
                        || ix == 0 || ix == nx - 1;
                    if on_border {
                        vals[flat(iz, iy, ix, ny, nx)] = 1.0;
                    }
                }
            }
        }
        // Inner shell iz/iy/ix ∈ {1..3}: value 8
        for iz in 1..4 {
            for iy in 1..4 {
                for ix in 1..4 {
                    vals[flat(iz, iy, ix, ny, nx)] = 8.0;
                }
            }
        }
        // Innermost pit at (2,2,2): value 2
        vals[flat(2, 2, 2, ny, nx)] = 2.0;

        let img = make_image(vals, [nz, ny, nx]);
        let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
        let out_vals = extract_vals(&out);
        let pit_out = out_vals[flat(2, 2, 2, ny, nx)];
        assert!(
            (pit_out - 8.0).abs() < 1e-5,
            "pit filled to wall level 8.0; got {pit_out}"
        );
    }

    /// Border-connected dark region NOT filled.
    ///
    /// Volume: 3×3×3 with value 0 everywhere. All voxels are on the border
    /// or connect to it through 0-valued paths. Fill should not increase anything.
    #[test]
    fn border_connected_dark_not_filled() {
        let dims = [3usize, 3, 3];
        let img = make_image(vec![0.0_f32; 27], dims);
        let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
        for &v in extract_vals(&out).iter() {
            assert!(v.abs() < 1e-6, "border-connected dark must stay 0, got {v}");
        }
    }

    /// Spatial metadata (origin, spacing, direction) is preserved.
    #[test]
    fn spatial_metadata_preserved() {
        let origin = Point::new([2.0, 3.0, 4.0]);
        let spacing = Spacing::new([0.75, 0.75, 1.5]);
        let direction = Direction::identity();
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let td = TensorData::new(vec![1.0_f32; 27], Shape::new([3, 3, 3]));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(tensor, origin, spacing, direction);
        let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
        assert_eq!(out.origin(), img.origin());
        assert_eq!(out.spacing(), img.spacing());
    }

    /// Multi-level landscape: deeper pit fills to higher wall than shallower pit.
    ///
    /// Volume: 1×1×7 image: I = [5, 5, 1, 5, 1, 5, 5].
    /// All voxels in a 1×1×N volume ARE on the border (nz=1,ny=1 → every voxel
    /// has iz=0=iz_max and iy=0=iy_max).  So output = input exactly.
    ///
    /// This test verifies that purely border-connected volumes are unchanged.
    #[test]
    fn all_border_volume_unchanged() {
        let vals = vec![5.0_f32, 5.0, 1.0, 5.0, 1.0, 5.0, 5.0];
        let dims = [1, 1, 7];
        let img = make_image(vals.clone(), dims);
        let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
        let out_vals = extract_vals(&out);
        for (i, (&a, &b)) in vals.iter().zip(out_vals.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "all-border voxel {i}: {a} ≠ {b}");
        }
    }
}
