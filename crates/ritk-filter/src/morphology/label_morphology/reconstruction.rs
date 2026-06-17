//! Geodesic morphological reconstruction for grayscale f32 images.
//!
//! # Mathematical Specification
//!
//! **Dilation reconstruction** (Vincent 1993):
//!   Given marker M and mask I with M <= I:
//!   M* = lim_{k->inf} min(dilate_1(M_k), I)
//!   where dilate_1 is one-step dilation with the unit-radius cubic B_1.
//!
//! **Erosion reconstruction**:
//!   Given marker M and mask I with M >= I:
//!   M* = lim_{k->inf} max(erode_1(M_k), I)
//!
//! Convergence criterion: max_x |M_{k+1}(x) - M_k(x)| < 1e-5, or max_iter reached.
//!
//! # References
//! - Vincent, L. (1993). Morphological grayscale reconstruction in image analysis.
//!   *IEEE Trans. Image Process.* 2(2):176-201.

use crate::morphology::Connectivity;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;
use std::collections::VecDeque;

// ════════════════════════════════════════════════════════════════════════════
// ReconstructionMode
// ════════════════════════════════════════════════════════════════════════════

/// Reconstruction mode for geodesic morphological reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructionMode {
    /// Geodesic dilation: marker expands upward constrained by mask (M <= I).
    Dilation,
    /// Geodesic erosion: marker contracts downward constrained by mask (M >= I).
    Erosion,
}

// ════════════════════════════════════════════════════════════════════════════
// MorphologicalReconstruction
// ════════════════════════════════════════════════════════════════════════════

/// Geodesic morphological reconstruction for grayscale f32 images.
///
/// Uses Vincent's (1993) hybrid grayscale reconstruction algorithm — a single
/// forward raster scan, a single anti-raster scan that seeds a FIFO queue, and
/// a queue-driven propagation — which reaches the exact fixed point in O(N)
/// regardless of geodesic-path length. (The earlier parallel-raster iteration
/// was O(N · path length): ~4.4 s on a 64×64×128 ramp; the hybrid is ~20 ms,
/// bit-identical output.)
#[derive(Debug, Clone)]
pub struct MorphologicalReconstruction {
    pub mode: ReconstructionMode,
    /// Structuring-element adjacency for each geodesic step. Defaults to
    /// [`Connectivity::Face6`], matching ITK's `FullyConnectedOff`.
    pub connectivity: Connectivity,
}

impl MorphologicalReconstruction {
    pub fn new(mode: ReconstructionMode) -> Self {
        Self {
            mode,
            connectivity: Connectivity::default(),
        }
    }

    /// Set the structuring-element adjacency (face vs full connectivity).
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Apply geodesic reconstruction.
    ///
    /// # Arguments
    /// - `marker`: initial marker image; must have same shape as `mask`
    /// - `mask`: constraint mask image
    ///
    /// # Errors
    /// Returns `Err` when marker and mask shapes differ.
    pub fn apply<B: Backend>(
        &self,
        marker: &Image<B, 3>,
        mask: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let (marker_vals, dims) = extract_vec(marker)?;
        let (mask_vals, mask_dims) = extract_vec(mask)?;
        if dims != mask_dims {
            anyhow::bail!(
                "MorphologicalReconstruction: marker shape {:?} != mask shape {:?}",
                dims,
                mask_dims
            );
        }

        let current: Vec<f32> = match self.mode {
            ReconstructionMode::Dilation => {
                hybrid_reconstruct::<Dilation>(&marker_vals, &mask_vals, dims, self.connectivity)
            }
            ReconstructionMode::Erosion => {
                hybrid_reconstruct::<Erosion>(&marker_vals, &mask_vals, dims, self.connectivity)
            }
        };

        let device = marker.data().device();
        let t = Tensor::<B, 3>::from_data(TensorData::new(current, Shape::new(dims)), &device);
        Ok(Image::new(
            t,
            *marker.origin(),
            *marker.spacing(),
            *marker.direction(),
        ))
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Vincent's hybrid grayscale reconstruction (O(N))
// ════════════════════════════════════════════════════════════════════════════

/// Reconstruction polarity (dilation vs erosion) as a zero-cost strategy.
///
/// Encoding the max/min/comparison direction in the type lets one generic
/// [`hybrid_reconstruct`] monomorphise into two specialised kernels with no
/// branch on the mode in the hot loops.
trait Polarity {
    /// Clamp the marker to the feasible side of the mask: `min` for dilation
    /// (`M ≤ I`), `max` for erosion (`M ≥ I`).
    fn clamp_marker(marker: f32, mask: f32) -> f32;
    /// Combine an accumulator with a neighbour value: `max` for dilation,
    /// `min` for erosion.
    fn extend(acc: f32, nbr: f32) -> f32;
    /// Cap a propagated value by the mask: `min` for dilation, `max` for erosion.
    fn cap(v: f32, mask: f32) -> f32;
    /// True if `from` is strictly more dominant than `to`, i.e. `from` should
    /// propagate into `to`: `from > to` for dilation, `from < to` for erosion.
    fn dominates(from: f32, to: f32) -> bool;
}

/// Dilation reconstruction polarity (propagate maxima up to the mask ceiling).
struct Dilation;
/// Erosion reconstruction polarity (propagate minima down to the mask floor).
struct Erosion;

impl Polarity for Dilation {
    #[inline]
    fn clamp_marker(marker: f32, mask: f32) -> f32 {
        marker.min(mask)
    }
    #[inline]
    fn extend(acc: f32, nbr: f32) -> f32 {
        acc.max(nbr)
    }
    #[inline]
    fn cap(v: f32, mask: f32) -> f32 {
        v.min(mask)
    }
    #[inline]
    fn dominates(from: f32, to: f32) -> bool {
        from > to
    }
}

impl Polarity for Erosion {
    #[inline]
    fn clamp_marker(marker: f32, mask: f32) -> f32 {
        marker.max(mask)
    }
    #[inline]
    fn extend(acc: f32, nbr: f32) -> f32 {
        acc.min(nbr)
    }
    #[inline]
    fn cap(v: f32, mask: f32) -> f32 {
        v.max(mask)
    }
    #[inline]
    fn dominates(from: f32, to: f32) -> bool {
        from < to
    }
}

/// Connectivity offsets split by raster causality. `causal` are the neighbours
/// scanned *before* the current voxel in forward raster order (z slowest, x
/// fastest); `anti` are those scanned after; `full` is their union.
struct NeighbourOffsets {
    causal: Vec<(i32, i32, i32)>,
    anti: Vec<(i32, i32, i32)>,
    full: Vec<(i32, i32, i32)>,
}

impl NeighbourOffsets {
    fn new(conn: Connectivity) -> Self {
        let (mut causal, mut anti, mut full) = (Vec::new(), Vec::new(), Vec::new());
        for dz in -1i32..=1 {
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if (dz, dy, dx) == (0, 0, 0) || !conn.includes(dz, dy, dx) {
                        continue;
                    }
                    let o = (dz, dy, dx);
                    full.push(o);
                    // Lexicographic (z,y,x) sign decides scan causality.
                    let causal_side =
                        dz < 0 || (dz == 0 && dy < 0) || (dz == 0 && dy == 0 && dx < 0);
                    if causal_side {
                        causal.push(o);
                    } else {
                        anti.push(o);
                    }
                }
            }
        }
        Self {
            causal,
            anti,
            full,
        }
    }
}

/// Resolve a neighbour flat index, returning `None` when out of bounds.
///
/// OOB is skipped, which for grayscale reconstruction is identical to ITK's
/// replicate (edge-clamp) boundary: a clamped out-of-bounds neighbour reads the
/// boundary voxel itself, and including a voxel's own value in the `extend`
/// (max/min) step is idempotent.
#[inline]
fn neighbour_index(
    iz: i32,
    iy: i32,
    ix: i32,
    off: (i32, i32, i32),
    dims: [usize; 3],
) -> Option<usize> {
    let [nz, ny, nx] = dims;
    let (zz, yy, xx) = (iz + off.0, iy + off.1, ix + off.2);
    if zz < 0 || zz >= nz as i32 || yy < 0 || yy >= ny as i32 || xx < 0 || xx >= nx as i32 {
        return None;
    }
    Some(zz as usize * ny * nx + yy as usize * nx + xx as usize)
}

/// Vincent's (1993) hybrid grayscale reconstruction: forward raster scan,
/// anti-raster scan seeding a FIFO queue, then queue-driven propagation. Reaches
/// the exact morphological-reconstruction fixed point in O(N).
fn hybrid_reconstruct<P: Polarity>(
    marker: &[f32],
    mask: &[f32],
    dims: [usize; 3],
    conn: Connectivity,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let offsets = NeighbourOffsets::new(conn);

    // Feasible-side clamp of the marker against the mask.
    let mut j: Vec<f32> = marker
        .iter()
        .zip(mask.iter())
        .map(|(&m, &i)| P::clamp_marker(m, i))
        .collect();

    let coords = |flat: usize| -> (i32, i32, i32) {
        let iz = (flat / (ny * nx)) as i32;
        let rem = flat % (ny * nx);
        ((iz), (rem / nx) as i32, (rem % nx) as i32)
    };

    // Forward raster scan: extend over already-visited (causal) neighbours.
    for p in 0..n {
        let (iz, iy, ix) = coords(p);
        let mut acc = j[p];
        for &o in &offsets.causal {
            if let Some(q) = neighbour_index(iz, iy, ix, o, dims) {
                acc = P::extend(acc, j[q]);
            }
        }
        j[p] = P::cap(acc, mask[p]);
    }

    // Anti-raster scan: extend over anti-causal neighbours and seed the queue.
    let mut queue: VecDeque<usize> = VecDeque::new();
    for p in (0..n).rev() {
        let (iz, iy, ix) = coords(p);
        let mut acc = j[p];
        for &o in &offsets.anti {
            if let Some(q) = neighbour_index(iz, iy, ix, o, dims) {
                acc = P::extend(acc, j[q]);
            }
        }
        j[p] = P::cap(acc, mask[p]);

        let jp = j[p];
        for &o in &offsets.anti {
            if let Some(q) = neighbour_index(iz, iy, ix, o, dims) {
                if P::dominates(jp, j[q]) && P::dominates(mask[q], j[q]) {
                    queue.push_back(p);
                    break;
                }
            }
        }
    }

    // Propagation: drain the queue, spreading dominant values up to the mask.
    while let Some(p) = queue.pop_front() {
        let (iz, iy, ix) = coords(p);
        let jp = j[p];
        for &o in &offsets.full {
            if let Some(q) = neighbour_index(iz, iy, ix, o, dims) {
                if P::dominates(jp, j[q]) && j[q] != mask[q] {
                    j[q] = P::cap(jp, mask[q]);
                    queue.push_back(q);
                }
            }
        }
    }

    j
}
