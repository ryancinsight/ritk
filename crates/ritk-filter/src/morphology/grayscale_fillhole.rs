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

use ritk_core::filter::ops::extract_vec;
use ritk_image::Image;
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
        let (vals, dims) = extract_vec(image)?;

        let filled = fill_holes_3d(&vals, dims);

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
                let on_border =
                    iz == 0 || iz == nz - 1 || iy == 0 || iy == ny - 1 || ix == 0 || ix == nx - 1;
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
        if iz > 0 {
            nbrs[k] = Some((iz - 1) * ny * nx + iy * nx + ix);
            k += 1;
        }
        if iz + 1 < nz {
            nbrs[k] = Some((iz + 1) * ny * nx + iy * nx + ix);
            k += 1;
        }
        if iy > 0 {
            nbrs[k] = Some(iz * ny * nx + (iy - 1) * nx + ix);
            k += 1;
        }
        if iy + 1 < ny {
            nbrs[k] = Some(iz * ny * nx + (iy + 1) * nx + ix);
            k += 1;
        }
        if ix > 0 {
            nbrs[k] = Some(iz * ny * nx + iy * nx + (ix - 1));
            k += 1;
        }
        if ix + 1 < nx {
            nbrs[k] = Some(iz * ny * nx + iy * nx + (ix + 1));
            k += 1;
        }
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
        for nb in neighbours(iz, iy, ix).into_iter().flatten() {
            let new_level = level.max(data[nb]);
            if new_level < h[nb] {
                h[nb] = new_level;
                heap.push(Reverse((new_level.to_bits(), nb)));
            }
        }
    }

    h
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_grayscale_fillhole.rs"]
mod tests_grayscale_fillhole;
