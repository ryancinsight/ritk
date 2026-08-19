//! Block-matching displacement estimation for speckle tracking.
//!
//! Speckle tracking for elastography and motion estimation: for each block of
//! the fixed image, search a bounded region of the moving image for the offset
//! that maximizes a similarity metric, and report that offset as the local
//! displacement.
//!
//! # Why this is not the registration engine
//!
//! The B-spline FFD and demons machinery in this crate solve for a *smooth,
//! globally parameterized* transform. Block matching solves a different
//! problem: many small, independent, locally rigid displacements, where the
//! answer is a discrete peak in a similarity surface rather than the minimum of
//! a differentiable objective. Ultrasound elastography needs the latter,
//! because tissue between two frames translates locally and the useful signal
//! is precisely the *spatial variation* of that translation.
//!
//! # Structure
//!
//! Four seams, matching ITKUltrasound's decomposition and the D2 batch layer:
//!
//! 1. A **metric image** — the similarity evaluated at every candidate integer
//!    offset in the search region ([`metric_image`]).
//! 2. A **displacement calculator** — how a peak in that surface becomes a
//!    displacement ([`SubpixelRefinement`]).
//! 3. A **block grid** — deterministic centres, volume matching, and an axial
//!    strain calculator ([`BlockGrid`] and [`DisplacementField`]).
//! 4. A **displacement regularizer** — a post-filter that rejects and replaces
//!    peak-hopped estimates against a strain plausibility bound
//!    ([`strain_window_filter`]).
//!
//! `track_volume` is deliberately a regular-grid primitive. It does not pad
//! image edges, silently clamp a block, or claim a result for a partial block.
//! Multi-resolution search policies and FFT-backed metrics remain follow-on
//! seams; the current D2 field supplies the deterministic batch foundation on
//! which those policies can build.
//!
//! The metric and refinement choices are closed sets fixed by the method, so
//! they are exhaustively matched enums rather than trait objects, and the
//! choice is made once per block rather than per candidate (atlas ADR 0041).
//!
//! # Why its own crate
//!
//! The algorithm is plain arithmetic over sample buffers: no image type, no
//! tensor, no backend. It first shipped inside `ritk-registration`, whose
//! manifest pulls `ritk-image` and with it the coeus autograd/nn/wgpu stack —
//! weight that a consumer wanting only speckle tracking should not inherit.
//! kwavers' elastography is exactly such a consumer, and could not take the
//! dependency at all without enabling an unrelated feature.
//!
//! # References
//! - `itkBlockMatchingNormalizedCrossCorrelationMetricImageFilter.hxx` and
//!   `itkBlockMatchingMaximumPixelDisplacementCalculator.hxx`,
//!   KitwareMedical/ITKUltrasound — the metric-image / displacement-calculator
//!   split and the peak convention.
//! - Céspedes, I., Huang, Y., Ophir, J., & Spratt, S. (1995). "Methods for
//!   estimation of subsample time delays of digitized echo signals."
//!   *Ultrason. Imaging* 17(2), 142–171 — the parabolic and cosine sub-sample
//!   estimators and their bias behaviour.

use anyhow::{bail, Result};

/// A sample type the matcher can correlate.
///
/// Correlation accumulates in `f64` regardless of the stored precision, because
/// the sums are over the whole block and the sub-sample peak estimate is
/// differenced from near-equal neighbours — both places where `f32`
/// accumulation loses exactly the precision the method exists to provide.
///
/// The stored type is nonetheless parameterized rather than fixed: image data
/// is `f32`, while RF and the displacement estimators built on it are `f64`,
/// and forcing either through the other's precision is a real loss. This is the
/// scalar variation dimension, abstracted rather than duplicated.
pub trait Sample: Copy {
    /// Widen to the accumulation type.
    fn to_f64(self) -> f64;
}

impl Sample for f32 {
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl Sample for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

mod metric;
mod refine;
mod regularize;

#[cfg(test)]
#[path = "tests_block_matching.rs"]
mod tests;

pub use metric::{metric_image, BlockMetric, MetricImage};
pub use refine::SubpixelRefinement;
pub use regularize::{strain_window_filter, StrainWindowParams, StrainWindowReport};

/// Geometry of a block-matching run, in voxels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockMatchingConfig {
    /// Half-extent of the fixed block, per axis. The block is
    /// `2·radius + 1` voxels across.
    pub block_radius: [usize; 3],
    /// Half-extent of the search region in the moving image, per axis. The
    /// largest displacement representable is `search_radius` voxels; a true
    /// displacement beyond it cannot be found, and the peak will sit on the
    /// search boundary.
    pub search_radius: [usize; 3],
}

impl BlockMatchingConfig {
    /// Validate the geometry.
    ///
    /// # Errors
    ///
    /// Returns an error when the block is a single voxel — it has no variance,
    /// so normalized correlation against it is undefined — or when the search
    /// region admits only the null displacement.
    ///
    /// A zero radius on *some* axis is valid and expected: a 2-D acquisition is
    /// a 3-D image with a singleton axis, and both the block and the search
    /// region are correctly flat on it. Requiring every axis to be positive
    /// would reject the dominant ultrasound geometry outright.
    pub fn validate(&self) -> Result<()> {
        if self.block_radius.iter().all(|&r| r == 0) {
            bail!(
                "block_radius is zero on every axis {:?}: a single-voxel block has no variance \
                 and normalized correlation against it is undefined",
                self.block_radius
            );
        }
        if self.search_radius.iter().all(|&r| r == 0) {
            bail!(
                "search_radius is zero on every axis {:?}: the search region would admit only \
                 the null displacement",
                self.search_radius
            );
        }
        Ok(())
    }
}

/// Displacement of one block, in voxels, plus the peak similarity that produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockDisplacement {
    /// Displacement per axis `[z, y, x]`, in voxels. Fractional when a
    /// sub-voxel refinement is applied.
    pub displacement: [f64; 3],
    /// Similarity at the peak, in `[-1, 1]` for normalized cross-correlation.
    ///
    /// Callers should treat a low peak as an unreliable displacement rather
    /// than a small one: decorrelated blocks still produce *some* maximum.
    pub peak_similarity: f64,
}

/// Estimate the displacement of the block centred at `centre` in the fixed
/// image, searching the moving image over the configured region.
///
/// Both images are flat row-major `[nz, ny, nx]` buffers of the same shape.
///
/// # Errors
///
/// Returns an error when the configuration is invalid, when the buffers do not
/// match `dims`, or when the block around `centre` leaves the image — the
/// caller chooses the block grid, so an out-of-bounds block is a caller error
/// rather than something to silently clamp, which would compare a different
/// block than the one requested.
pub fn match_block<T: Sample>(
    fixed: &[T],
    moving: &[T],
    dims: [usize; 3],
    centre: [usize; 3],
    config: BlockMatchingConfig,
    refinement: SubpixelRefinement,
) -> Result<BlockDisplacement> {
    config.validate()?;
    let expected = dims[0] * dims[1] * dims[2];
    if fixed.len() != expected || moving.len() != expected {
        bail!(
            "fixed ({}) and moving ({}) buffers must both hold {expected} voxels for dims {dims:?}",
            fixed.len(),
            moving.len()
        );
    }
    for axis in 0..3 {
        let lo = centre[axis].checked_sub(config.block_radius[axis]);
        let hi = centre[axis] + config.block_radius[axis];
        if lo.is_none() || hi >= dims[axis] {
            bail!(
                "block at {centre:?} with radius {:?} leaves the image on axis {axis} (extent {})",
                config.block_radius,
                dims[axis]
            );
        }
    }

    let surface = metric_image(
        fixed,
        moving,
        dims,
        centre,
        config,
        BlockMetric::NormalizedCrossCorrelation,
    )?;
    Ok(refine::displacement_from(&surface, refinement))
}

// ── Volume-level pipeline (US-023-D2) ────────────────────────────────────────

/// Layout of block centres across a volume.
///
/// The grid partitions the image into non-overlapping tiles; the centre of each
/// tile is the block centre. Axis `i` contributes
/// `n_blocks[i] = (dims[i] - 2*block_radius[i]) / stride[i]` centres, placed
/// at `block_radius[i] + k * stride[i]` for `k = 0 .. n_blocks[i]`.
///
/// Choosing `stride = 2 * block_radius + 1` gives non-overlapping, dense
/// coverage. A stride smaller than the block size gives overlapping tiles and
/// a denser displacement map at the cost of redundant correlation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockGrid {
    /// Step between adjacent block centres per axis, in voxels.
    pub stride: [usize; 3],
}

impl BlockGrid {
    /// Non-overlapping stride: `2 * block_radius + 1` per axis.
    #[must_use]
    pub fn dense(block_radius: [usize; 3]) -> Self {
        Self {
            stride: [
                2 * block_radius[0] + 1,
                2 * block_radius[1] + 1,
                2 * block_radius[2] + 1,
            ],
        }
    }

    /// Enumerate block centres within `dims` for `config`.
    fn centres(&self, dims: [usize; 3], config: &BlockMatchingConfig) -> Vec<[usize; 3]> {
        let r = config.block_radius;
        let mut out = Vec::new();
        let mut z = r[0];
        while z + r[0] < dims[0] {
            let mut y = r[1];
            while y + r[1] < dims[1] {
                let mut x = r[2];
                while x + r[2] < dims[2] {
                    out.push([z, y, x]);
                    x = match x.checked_add(self.stride[2]) {
                        Some(v) => v,
                        None => break,
                    };
                }
                y = match y.checked_add(self.stride[1]) {
                    Some(v) => v,
                    None => break,
                };
            }
            z = match z.checked_add(self.stride[0]) {
                Some(v) => v,
                None => break,
            };
        }
        out
    }
}

/// Displacement estimates for a grid of blocks across a volume.
///
/// Each entry `i` corresponds to the block whose centre is `centres[i]`.
/// Centres are in `[z, y, x]` image-index order. Displacements are in voxels
/// (sub-voxel when a sub-pixel refinement is applied).
///
/// Blocks whose fixed window has zero variance are recorded with
/// `peak_similarity = f64::NAN` and `displacement = [0.0; 3]`, so callers can
/// filter low-quality estimates by similarity threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplacementField {
    /// Block centre coordinates, in `[z, y, x]` voxel order.
    pub centres: Vec<[usize; 3]>,
    /// Displacement per block, in voxels.
    pub displacements: Vec<[f64; 3]>,
    /// Similarity at the correlation peak for each block (`[-1, 1]`).
    pub peak_similarities: Vec<f64>,
}

impl DisplacementField {
    /// Number of estimated blocks.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.centres.len()
    }

    /// Whether the field is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.centres.is_empty()
    }
}

/// Estimate displacement at every block centre in a volume.
///
/// Scans the fixed/moving buffer pair using the given `grid` layout and
/// `config`, reporting `BlockDisplacement` at each centre. Blocks whose fixed
/// window is constant (zero variance) are recorded with `peak_similarity = NAN`
/// and zero displacement rather than propagating an error.
///
/// Both buffers are flat row-major `[nz, ny, nx]` with `nz * ny * nx` voxels.
///
/// # Errors
///
/// Returns an error when `dims` product does not equal `fixed.len()`, when the
/// configuration is invalid, or when `grid.stride` is zero on any axis.
pub fn track_volume<T: Sample>(
    fixed: &[T],
    moving: &[T],
    dims: [usize; 3],
    config: BlockMatchingConfig,
    grid: BlockGrid,
    refinement: SubpixelRefinement,
) -> Result<DisplacementField> {
    config.validate()?;
    for axis in 0..3 {
        if grid.stride[axis] == 0 {
            bail!("grid stride is zero on axis {axis}; that would loop forever");
        }
    }
    let expected = dims[0] * dims[1] * dims[2];
    if fixed.len() != expected || moving.len() != expected {
        bail!(
            "fixed ({}) and moving ({}) buffers must both hold {expected} voxels for dims {dims:?}",
            fixed.len(),
            moving.len()
        );
    }

    let centres = grid.centres(dims, &config);
    let n = centres.len();
    let mut displacements = vec![[0.0f64; 3]; n];
    let mut peak_similarities = vec![f64::NAN; n];

    for (i, &centre) in centres.iter().enumerate() {
        if let Ok(bd) = match_block(fixed, moving, dims, centre, config, refinement) {
            displacements[i] = bd.displacement;
            peak_similarities[i] = bd.peak_similarity;
        }
        // constant block (Err) — leave NAN / zeros
    }

    Ok(DisplacementField {
        centres,
        displacements,
        peak_similarities,
    })
}

/// Estimate axial strain from a displacement field, in strain units (voxel/voxel).
///
/// For each block, the axial strain is estimated by central finite differences
/// over the neighbouring blocks' axial displacements, divided by the axial
/// stride between centres. Blocks at the axial boundary where a neighbour does
/// not exist use the one-sided (forward or backward) difference instead.
///
/// The returned vector is parallel to `field.centres`: `strain[i]` is the
/// axial strain at `field.centres[i]`.
///
/// # Panics
///
/// Panics when `axial_stride == 0` (would divide by zero). Use `grid.stride[0]`
/// or an equivalent positive value.
#[must_use]
pub fn strain_from_displacement(field: &DisplacementField, axial_stride: usize) -> Vec<f64> {
    assert!(axial_stride > 0, "axial_stride must be positive");
    let n = field.len();
    if n == 0 {
        return Vec::new();
    }

    // Group centre indices by their (y, x) lateral position, ordered by z.
    // We reconstruct per-line sequences by sorting on (y, x) then z.
    let mut indexed: Vec<(usize, usize, usize, usize)> = field
        .centres
        .iter()
        .enumerate()
        .map(|(i, &[z, y, x])| (y, x, z, i))
        .collect();
    indexed.sort_unstable();

    let mut strain = vec![0.0f64; n];

    // Walk each lateral position's axial sequence.
    let mut start = 0;
    while start < indexed.len() {
        let (y0, x0, _, _) = indexed[start];
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].0 == y0 && indexed[end].1 == x0 {
            end += 1;
        }
        let line = &indexed[start..end]; // sorted by z (key index 2)
        let m = line.len();
        for pos in 0..m {
            let i = line[pos].3;
            let disp_here = field.displacements[i][0]; // axial = axis 0
            let s = if m == 1 {
                0.0
            } else if pos == 0 {
                let j = line[1].3;
                (field.displacements[j][0] - disp_here) / axial_stride as f64
            } else if pos == m - 1 {
                let j = line[m - 2].3;
                (disp_here - field.displacements[j][0]) / axial_stride as f64
            } else {
                let j_prev = line[pos - 1].3;
                let j_next = line[pos + 1].3;
                (field.displacements[j_next][0] - field.displacements[j_prev][0])
                    / (2 * axial_stride) as f64
            };
            strain[i] = s;
        }
        start = end;
    }

    strain
}
