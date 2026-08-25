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
//! Seven seams, matching ITKUltrasound's decomposition and the D2 batch layer:
//!
//! 1. A **metric image** — the similarity evaluated at every candidate integer
//!    offset in the search region ([`metric_image`]).
//! 2. A **displacement calculator** — how a peak in that surface becomes a
//!    displacement ([`SubpixelRefinement`]).
//! 3. A **block grid** — deterministic centres, volume matching, and an axial
//!    strain calculator ([`BlockGrid`] and [`DisplacementField`]).
//! 4. A **coarse-to-fine search** — caller-owned pyramid levels with propagated
//!    moving centres ([`MultiResolutionSearch`]).
//! 5. An explicit **confidence regularizer** — a Bayesian post-process that
//!    pulls the final displacement toward a configured prior ([`BayesianDisplacementPrior`]).
//! 6. Acquisition-aware **radius calculators** — derive axial block support
//!    from signal correlation or transducer bandwidth ([`radius_from_bandwidth`]).
//! 7. A **rejection post-filter** — replaces peak-hopped estimates that violate
//!    a strain plausibility bound ([`strain_window_filter`]). Seams 5 and 7 are
//!    complements, not alternatives: 5 conditions every block toward a prior,
//!    7 discards the blocks whose measurement cannot be believed at all.
//!
//! `track_volume` is deliberately a regular-grid primitive. It does not pad
//! image edges, silently clamp a block, or claim a result for a partial block.
//! [`MultiResolutionSearch`] adds an explicit coarse-to-fine execution seam;
//! callers own the image pyramid and its resampling. Pyramid regularization is
//! deliberately a post-process: it uses the finest peak confidence and leaves
//! level diagnostics intact. The optional FFT-backed metric uses Apollo for
//! finite, zero-padded linear NCC; it does not silently switch the direct metric
//! to circular correlation. The FFT pyramid methods reuse the direct
//! propagation policy and are therefore an optimization choice, not a second
//! coordinate or boundary contract.
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

    /// Convert a computed pyramid sample back to the stored scalar type.
    ///
    /// This is used by caller-owned min/max pyramid construction. Floating
    /// conversion follows Rust's saturating float-to-float cast semantics.
    fn from_f64_saturating(value: f64) -> Self;
}

impl Sample for f32 {
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    #[inline]
    fn from_f64_saturating(value: f64) -> Self {
        value as f32
    }
}

impl Sample for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn from_f64_saturating(value: f64) -> Self {
        value
    }
}

#[cfg(feature = "fft")]
mod fft;
mod metric;
mod radius;
mod refine;
mod regularization;
mod regularize;
mod search;

#[cfg(test)]
#[path = "tests_block_matching.rs"]
mod tests;

#[cfg(feature = "fft")]
pub use fft::{match_block_fft, metric_image_fft, FftPadding};
pub use metric::{metric_image, BlockMetric, MetricImage};
pub use radius::{radius_from_axial_autocorrelation, radius_from_bandwidth};
pub use refine::SubpixelRefinement;
pub use regularization::{BayesianDisplacementPrior, LeastSquaresDisplacementPrior};
pub use regularize::{strain_window_filter, StrainWindowParams, StrainWindowReport};
pub use search::{
    MultiResolutionDisplacement, MultiResolutionSearch, OwnedPyramid, PyramidDisplacementField,
    PyramidLevel, PyramidLevelDisplacement, SearchRegion,
};

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
    /// Build a configuration from an axial radius and two transverse radii.
    ///
    /// `axial_axis` selects the axial component in `[z, y, x]` order. The two
    /// entries in `transverse_radius` are assigned to the remaining axes in
    /// ascending axis order. This makes the orientation explicit instead of
    /// assuming that every acquisition uses the same storage axis.
    ///
    /// The resulting configuration is validated before it is returned.
    ///
    /// # Errors
    ///
    /// Returns an error when `axial_axis` is not one of `0..3`, or when the
    /// resulting block/search geometry is invalid.
    pub fn with_axial_radius(
        axial_axis: usize,
        axial_radius: usize,
        transverse_radius: [usize; 2],
        search_radius: [usize; 3],
    ) -> Result<Self> {
        if axial_axis >= 3 {
            bail!("axial axis must be in [0, 3), got {axial_axis}");
        }
        // The two transverse radii fill the two non-axial axes in axis order,
        // so an axis below the axial one takes its own index and an axis above
        // takes the index shifted past the axial slot.
        let block_radius = std::array::from_fn(|axis| match axis.cmp(&axial_axis) {
            std::cmp::Ordering::Equal => axial_radius,
            std::cmp::Ordering::Less => transverse_radius[axis],
            std::cmp::Ordering::Greater => transverse_radius[axis - 1],
        });
        let config = Self {
            block_radius,
            search_radius,
        };
        config.validate()?;
        Ok(config)
    }

    /// Build a configuration using the first decorrelation lag of an axial line.
    ///
    /// This is the validated composition of [`radius_from_axial_autocorrelation`]
    /// and [`Self::with_axial_radius`].
    pub fn from_axial_autocorrelation(
        signal: &[f64],
        threshold: f64,
        axial_axis: usize,
        transverse_radius: [usize; 2],
        search_radius: [usize; 3],
    ) -> Result<Self> {
        let axial_radius = radius_from_axial_autocorrelation(signal, threshold)?;
        Self::with_axial_radius(axial_axis, axial_radius, transverse_radius, search_radius)
    }

    /// Build a configuration from transducer bandwidth and axial sample spacing.
    ///
    /// The axial radius is derived from the pulse-echo resolution estimate
    /// `c / (2 · f_c · BW)` and then mapped through [`Self::with_axial_radius`].
    pub fn from_transducer_bandwidth(
        speed_of_sound_m_s: f64,
        centre_frequency_hz: f64,
        fractional_bandwidth: f64,
        axial_sample_spacing_m: f64,
        axial_axis: usize,
        transverse_radius: [usize; 2],
        search_radius: [usize; 3],
    ) -> Result<Self> {
        let axial_radius = radius_from_bandwidth(
            speed_of_sound_m_s,
            centre_frequency_hz,
            fractional_bandwidth,
            axial_sample_spacing_m,
        )?;
        Self::with_axial_radius(axial_axis, axial_radius, transverse_radius, search_radius)
    }

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
    match_block_at(fixed, moving, dims, centre, centre, config, refinement)
}

/// Estimate a block using separate fixed and moving-image search centres.
///
/// This is the execution seam used by [`MultiResolutionSearch`]. The returned
/// displacement is the absolute moving-centre offset plus the local metric
/// peak, so it remains meaningful when the moving search centre was propagated
/// from a coarser pyramid level. No padding is performed: the fixed block and
/// the moving block at the search centre must fit; individual candidate blocks
/// outside the image are skipped by the finite-boundary metric.
///
/// This is crate-visible because the public pyramid API is the supported
/// coarse-to-fine entry point; keeping the lower-level centre distinction
/// private prevents callers from accidentally mixing coordinate systems.
pub(crate) fn match_block_at<T: Sample>(
    fixed: &[T],
    moving: &[T],
    dims: [usize; 3],
    fixed_centre: [usize; 3],
    moving_centre: [usize; 3],
    config: BlockMatchingConfig,
    refinement: SubpixelRefinement,
) -> Result<BlockDisplacement> {
    config.validate()?;
    let expected = dims[0]
        .checked_mul(dims[1])
        .and_then(|v| v.checked_mul(dims[2]))
        .ok_or_else(|| anyhow::anyhow!("dims {dims:?} overflow the buffer size calculation"))?;
    if fixed.len() != expected || moving.len() != expected {
        bail!(
            "fixed ({}) and moving ({}) buffers must both hold {expected} voxels for dims {dims:?}",
            fixed.len(),
            moving.len()
        );
    }

    for axis in 0..3 {
        let radius = config.block_radius[axis];
        let fixed_hi = fixed_centre[axis].checked_add(radius);
        if fixed_centre[axis].checked_sub(radius).is_none()
            || fixed_hi.is_none_or(|hi| hi >= dims[axis])
        {
            bail!(
                "fixed block at {fixed_centre:?} with radius {:?} leaves the image on axis {axis} (extent {})",
                config.block_radius,
                dims[axis]
            );
        }
        let moving_hi = moving_centre[axis].checked_add(config.block_radius[axis]);
        if moving_centre[axis]
            .checked_sub(config.block_radius[axis])
            .is_none()
            || moving_hi.is_none_or(|hi| hi >= dims[axis])
        {
            bail!(
                "moving block at {moving_centre:?} with radius {:?} leaves the image on axis {axis} (extent {})",
                config.block_radius,
                dims[axis]
            );
        }
    }

    let surface = metric::metric_image_at(
        fixed,
        moving,
        dims,
        fixed_centre,
        moving_centre,
        config,
        BlockMetric::NormalizedCrossCorrelation,
    )?;
    let mut result = refine::displacement_from(&surface, refinement);
    for axis in 0..3 {
        result.displacement[axis] += moving_centre[axis] as f64 - fixed_centre[axis] as f64;
    }
    Ok(result)
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
    ///
    /// # Panics
    ///
    /// Panics when a radius is too large to represent its dense stride. Use
    /// [`Self::try_dense`] when radii originate outside trusted configuration.
    #[must_use]
    pub fn dense(block_radius: [usize; 3]) -> Self {
        Self::try_dense(block_radius).expect("block radius overflows dense grid stride")
    }

    /// Build a non-overlapping grid with checked stride arithmetic.
    ///
    /// This is the fallible counterpart to [`Self::dense`].
    ///
    /// # Errors
    ///
    /// Returns an error when `2 * block_radius + 1` overflows on any axis.
    pub fn try_dense(block_radius: [usize; 3]) -> Result<Self> {
        let mut stride = [0; 3];
        for axis in 0..3 {
            stride[axis] = block_radius[axis]
                .checked_mul(2)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| anyhow::anyhow!("dense grid stride overflows on axis {axis}"))?;
        }
        Ok(Self { stride })
    }

    /// Validate the grid stride.
    ///
    /// # Errors
    ///
    /// Returns an error when any stride is zero, which would make centre
    /// enumeration fail to advance.
    pub fn validate(&self) -> Result<()> {
        for axis in 0..3 {
            if self.stride[axis] == 0 {
                bail!("grid stride is zero on axis {axis}; that would loop forever");
            }
        }
        Ok(())
    }

    /// Enumerate block centres within `dims` for `config`.
    fn centres(&self, dims: [usize; 3], config: &BlockMatchingConfig) -> Vec<[usize; 3]> {
        let r = config.block_radius;
        let mut out = Vec::new();
        let mut z = r[0];
        while z.checked_add(r[0]).is_some_and(|high| high < dims[0]) {
            let mut y = r[1];
            while y.checked_add(r[1]).is_some_and(|high| high < dims[1]) {
                let mut x = r[2];
                while x.checked_add(r[2]).is_some_and(|high| high < dims[2]) {
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

    /// Validate that all per-block arrays have the same length.
    ///
    /// Public field members remain available for convenient construction, so
    /// callers that assemble a field manually should validate it before
    /// passing it to strain or regularization code.
    ///
    /// # Errors
    ///
    /// Returns an error when centres, displacements, and peak similarities do
    /// not contain the same number of entries.
    pub fn validate(&self) -> Result<()> {
        let expected = self.centres.len();
        if self.displacements.len() != expected || self.peak_similarities.len() != expected {
            bail!(
                "displacement field arrays must have equal lengths: centres {}, displacements {}, peaks {}",
                expected,
                self.displacements.len(),
                self.peak_similarities.len()
            );
        }
        Ok(())
    }

    /// Return a confidence mask for blocks suitable for downstream estimation.
    ///
    /// A block is valid when its displacement is finite and its peak similarity
    /// is finite and at least `minimum_peak_similarity`. This makes the batch
    /// matcher's documented `NAN`/zero invalid-block convention explicit without
    /// changing the field's stored values. Similarity values above one are not
    /// rejected here because floating-point correlation can overshoot by a
    /// tiny amount; callers choose the acceptance threshold.
    ///
    /// # Errors
    ///
    /// Returns an error when the field arrays have different lengths or when
    /// `minimum_peak_similarity` is not finite and in `[0, 1]`.
    pub fn valid_mask(&self, minimum_peak_similarity: f64) -> Result<Vec<bool>> {
        self.validate()?;
        if !minimum_peak_similarity.is_finite() || !(0.0..=1.0).contains(&minimum_peak_similarity) {
            bail!("minimum peak similarity must be finite and in [0, 1]");
        }
        Ok(self
            .displacements
            .iter()
            .zip(&self.peak_similarities)
            .map(|(displacement, &peak)| {
                displacement.iter().all(|value| value.is_finite())
                    && peak.is_finite()
                    && peak >= minimum_peak_similarity
            })
            .collect())
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
    grid.validate()?;
    let expected = dims[0]
        .checked_mul(dims[1])
        .and_then(|v| v.checked_mul(dims[2]))
        .ok_or_else(|| anyhow::anyhow!("dims {dims:?} overflow the buffer size calculation"))?;
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
/// Panics when the field arrays are malformed or `axial_stride == 0`. Use
/// [`try_strain_from_displacement`] for untrusted fields or runtime input.
#[must_use]
pub fn strain_from_displacement(field: &DisplacementField, axial_stride: usize) -> Vec<f64> {
    try_strain_from_displacement(field, axial_stride)
        .expect("invalid displacement field for strain estimation")
}

/// Fallibly estimate axial strain from a displacement field.
///
/// This is the validated counterpart to [`strain_from_displacement`]. It uses
/// central finite differences in the interior and one-sided differences at
/// axial boundaries.
///
/// # Errors
///
/// Returns an error when the field arrays are not aligned or `axial_stride` is
/// zero.
pub fn try_strain_from_displacement(
    field: &DisplacementField,
    axial_stride: usize,
) -> Result<Vec<f64>> {
    field.validate()?;
    if axial_stride == 0 {
        bail!("axial_stride must be positive");
    }
    let n = field.len();
    if n == 0 {
        return Ok(Vec::new());
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
                    / (2.0 * axial_stride as f64)
            };
            strain[i] = s;
        }
        start = end;
    }

    Ok(strain)
}

/// Estimate axial strain using only confidence-qualified field entries.
///
/// Blocks below `minimum_peak_similarity`, with non-finite confidence, or with
/// non-finite displacement are assigned `NAN` in the returned vector and are
/// omitted from neighbouring finite differences. Valid blocks use the nearest
/// valid block on each side, so the denominator includes the number of skipped
/// grid gaps (`axial_stride * gap_count`). This prevents an invalid zero
/// displacement from manufacturing a strain spike.
///
/// Unlike [`strain_from_displacement`], this fallible variant validates the
/// field's parallel arrays and the confidence threshold before estimating.
///
/// # Errors
///
/// Returns an error when the field arrays have different lengths, when
/// `axial_stride` is zero, or when the confidence threshold is not finite and
/// in `[0, 1]`.
pub fn strain_from_displacement_filtered(
    field: &DisplacementField,
    axial_stride: usize,
    minimum_peak_similarity: f64,
) -> Result<Vec<f64>> {
    if axial_stride == 0 {
        bail!("axial_stride must be positive");
    }
    let valid = field.valid_mask(minimum_peak_similarity)?;
    let n = field.len();
    let mut strain = vec![f64::NAN; n];
    if n == 0 {
        return Ok(strain);
    }

    let mut indexed: Vec<(usize, usize, usize, usize)> = field
        .centres
        .iter()
        .enumerate()
        .map(|(i, &[z, y, x])| (y, x, z, i))
        .collect();
    indexed.sort_unstable();

    let mut start = 0;
    while start < indexed.len() {
        let (y0, x0, _, _) = indexed[start];
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].0 == y0 && indexed[end].1 == x0 {
            end += 1;
        }
        let line = &indexed[start..end];
        let valid_positions: Vec<usize> = line
            .iter()
            .enumerate()
            .filter_map(|(position, &(_, _, _, index))| valid[index].then_some(position))
            .collect();

        for (valid_index, &position) in valid_positions.iter().enumerate() {
            let current = line[position].3;
            let displacement = field.displacements[current][0];
            let estimate = if valid_positions.len() == 1 {
                0.0
            } else if valid_index == 0 {
                let next_position = valid_positions[1];
                let next = line[next_position].3;
                let gap = (next_position - position) as f64;
                (field.displacements[next][0] - displacement) / (gap * axial_stride as f64)
            } else if valid_index == valid_positions.len() - 1 {
                let previous_position = valid_positions[valid_index - 1];
                let previous = line[previous_position].3;
                let gap = (position - previous_position) as f64;
                (displacement - field.displacements[previous][0]) / (gap * axial_stride as f64)
            } else {
                let previous_position = valid_positions[valid_index - 1];
                let next_position = valid_positions[valid_index + 1];
                let previous = line[previous_position].3;
                let next = line[next_position].3;
                let gap = (next_position - previous_position) as f64;
                (field.displacements[next][0] - field.displacements[previous][0])
                    / (gap * axial_stride as f64)
            };
            strain[current] = estimate;
        }
        start = end;
    }

    Ok(strain)
}

// ── End-to-end pipeline (US-023-D2) ─────────────────────────────────────────

/// Which correlation metric the pipeline's per-block matcher uses.
///
/// The FFT variant is only available when the `fft` feature is enabled; it is
/// the Apollo-backed finite-boundary path, equivalent to the direct metric up
/// to the FFT kernel's floating-point error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PipelineMetric {
    /// Direct normalized cross-correlation over the finite candidate block.
    #[default]
    Direct,
    /// Apollo-FFT linear normalized cross-correlation with zero padding.
    #[cfg(feature = "fft")]
    Fft,
}

/// Optional post-processing stages applied after block matching.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PipelineStages {
    /// Apply a confidence-weighted Bayesian prior to the displacement field.
    pub bayesian_prior: Option<BayesianDisplacementPrior>,
    /// Smooth each axial line toward its local least-squares strain window.
    pub least_squares_prior: Option<LeastSquaresDisplacementPrior>,
    /// Optional minimum peak similarity for confidence-filtered pipeline strain.
    ///
    /// When set alongside `least_squares_prior`, invalid or low-confidence blocks are
    /// omitted from finite differences and reported as `NAN`. When `None`, the
    /// legacy unfiltered strain estimator is retained.
    pub minimum_peak_similarity: Option<f64>,
}

impl PipelineStages {
    /// Validate all manually configured post-processing stages.
    ///
    /// Constructors validate their own values, but the stage fields are public
    /// and can be assembled directly. Pipeline entry points call this before
    /// matching so malformed configuration fails before any metric work.
    ///
    /// # Errors
    ///
    /// Returns an error when a prior, least-squares window, or confidence threshold is
    /// invalid.
    pub fn validate(&self) -> Result<()> {
        if let Some(prior) = self.bayesian_prior {
            prior.validate()?;
        }
        if let Some(window) = self.least_squares_prior {
            window.validate()?;
        }
        if let Some(minimum_peak_similarity) = self.minimum_peak_similarity {
            if !minimum_peak_similarity.is_finite()
                || !(0.0..=1.0).contains(&minimum_peak_similarity)
            {
                bail!("minimum peak similarity must be finite and in [0, 1]");
            }
        }
        Ok(())
    }
}

/// Configuration for a complete block-matching run over a volume.
///
/// The fields are the closed set of choices the pipeline supports; there is no
/// trait-object seam (atlas ADR 0041). `metric` selects the correlation
/// backend, `refinement` the sub-sample peak estimator, `grid` the block
/// layout, and `stages` the optional regularization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplacementPipeline {
    /// Correlation metric.
    pub metric: PipelineMetric,
    /// Sub-sample peak refinement.
    pub refinement: SubpixelRefinement,
    /// Block-grid layout.
    pub grid: BlockGrid,
    /// Optional regularization / strain-window stages.
    pub stages: PipelineStages,
}

/// Result of a full pipeline run: the displacement field and, when requested,
/// the axial strain derived from it.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineResult {
    /// Regularized (or raw) displacement field over the block grid.
    pub field: DisplacementField,
    /// Axial strain per block centre, present only when the pipeline's
    /// [`PipelineStages`] requested it via a strain window.
    pub axial_strain: Option<Vec<f64>>,
}

/// Result of a pipeline run over a pyramid with raw per-level diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct PyramidPipelineResult {
    /// Regularized (or raw) displacement field over the block grid.
    pub field: DisplacementField,
    /// Raw direct/FFT pyramid evidence, retained independently of post-processing.
    pub diagnostics: PyramidDisplacementField,
    /// Axial strain per block centre, when requested by the pipeline stages.
    pub axial_strain: Option<Vec<f64>>,
}

impl DisplacementPipeline {
    /// Run the pipeline over a fixed/moving volume pair.
    ///
    /// The matcher is dispatched once per block (ADR 0041); the strain window
    /// is applied after matching, before the Bayesian prior, so the prior can
    /// pull any residual outliers toward the smoothed trend.
    ///
    /// # Errors
    ///
    /// Returns an error when `dims` is invalid, the buffers do not match, the
    /// grid stride is zero, the Bayesian prior configuration is invalid, or a
    /// configured confidence threshold is outside `[0, 1]`.
    pub fn run<T: Sample>(
        &self,
        fixed: &[T],
        moving: &[T],
        dims: [usize; 3],
        config: BlockMatchingConfig,
    ) -> Result<PipelineResult> {
        self.stages.validate()?;
        let mut field = match self.metric {
            PipelineMetric::Direct => {
                track_volume(fixed, moving, dims, config, self.grid, self.refinement)?
            }
            #[cfg(feature = "fft")]
            PipelineMetric::Fft => {
                track_volume_fft(fixed, moving, dims, config, self.grid, self.refinement)?
            }
        };

        if let Some(window) = self.stages.least_squares_prior {
            field = window.try_regularize(&field)?;
        }
        if let Some(prior) = self.stages.bayesian_prior {
            field = prior.try_regularize(&field)?;
        }

        let axial_strain = if self.stages.least_squares_prior.is_some() {
            Some(match self.stages.minimum_peak_similarity {
                Some(minimum_peak_similarity) => strain_from_displacement_filtered(
                    &field,
                    self.grid.stride[0],
                    minimum_peak_similarity,
                )?,
                None => try_strain_from_displacement(&field, self.grid.stride[0])?,
            })
        } else {
            None
        };

        Ok(PipelineResult {
            field,
            axial_strain,
        })
    }

    /// Run the same pipeline stages over a caller-owned coarse-to-fine pyramid.
    ///
    /// The search plan owns the per-level block and search radii; this pipeline
    /// owns metric selection, grid enumeration, subpixel refinement, and
    /// post-processing. Direct and FFT modes therefore differ only in the
    /// finite NCC implementation. The strain window is applied before the
    /// Bayesian prior, matching [`Self::run`].
    ///
    /// # Errors
    ///
    /// Returns pyramid, grid, buffer, metric, or configured confidence-threshold
    /// validation errors. As with the batch pyramid APIs, individual
    /// non-evaluable block centres are retained with zero displacement and
    /// `NAN` confidence.
    pub fn run_pyramid<T: Sample>(
        &self,
        search: &MultiResolutionSearch,
        pyramid: &[PyramidLevel<'_, T>],
    ) -> Result<PipelineResult> {
        let result = self.run_pyramid_with_diagnostics(search, pyramid)?;
        Ok(PipelineResult {
            field: result.field,
            axial_strain: result.axial_strain,
        })
    }

    /// Run a pyramid pipeline while retaining raw per-level diagnostics.
    ///
    /// Matching evidence is collected before strain-window or Bayesian stages;
    /// `diagnostics` therefore remains a faithful record of direct/FFT matching
    /// while `field` contains the configured post-processing result.
    ///
    /// # Errors
    ///
    /// Returns the same pyramid, grid, metric, stage, and threshold validation
    /// errors as [`Self::run_pyramid`].
    pub fn run_pyramid_with_diagnostics<T: Sample>(
        &self,
        search: &MultiResolutionSearch,
        pyramid: &[PyramidLevel<'_, T>],
    ) -> Result<PyramidPipelineResult> {
        self.stages.validate()?;
        let diagnostics = match self.metric {
            PipelineMetric::Direct => {
                search.track_volume_pyramid_diagnostics(pyramid, self.grid, self.refinement)?
            }
            #[cfg(feature = "fft")]
            PipelineMetric::Fft => search.track_volume_pyramid_fft_diagnostics(
                pyramid,
                self.grid,
                self.refinement,
                FftPadding::Zero,
            )?,
        };
        let mut field = diagnostics.try_as_field()?;

        if let Some(window) = self.stages.least_squares_prior {
            field = window.try_regularize(&field)?;
        }
        if let Some(prior) = self.stages.bayesian_prior {
            field = prior.try_regularize(&field)?;
        }

        let axial_strain = if self.stages.least_squares_prior.is_some() {
            Some(match self.stages.minimum_peak_similarity {
                Some(minimum_peak_similarity) => strain_from_displacement_filtered(
                    &field,
                    self.grid.stride[0],
                    minimum_peak_similarity,
                )?,
                None => try_strain_from_displacement(&field, self.grid.stride[0])?,
            })
        } else {
            None
        };

        Ok(PyramidPipelineResult {
            field,
            diagnostics,
            axial_strain,
        })
    }

    /// Run the pipeline directly from an [`OwnedPyramid`].
    ///
    /// This is a convenience adapter over [`Self::run_pyramid`]: it borrows
    /// the pyramid's caller-owned levels for the duration of the run and does
    /// not resample, copy, or alter the matching contract. Use this when the
    /// pyramid was constructed with [`OwnedPyramid::nearest`] or
    /// [`OwnedPyramid::min_max`].
    ///
    /// # Errors
    ///
    /// Returns the same plan, level, grid, and metric errors as
    /// [`Self::run_pyramid`].
    pub fn run_owned_pyramid<T: Sample>(
        &self,
        search: &MultiResolutionSearch,
        pyramid: &OwnedPyramid<T>,
    ) -> Result<PipelineResult> {
        let levels = pyramid.levels();
        self.run_pyramid(search, &levels)
    }

    /// Run an owned pyramid while retaining raw per-level diagnostics.
    ///
    /// This is the owned-pyramid adapter for
    /// [`Self::run_pyramid_with_diagnostics`]. It borrows the constructed levels
    /// only for the duration of the call and preserves the same post-processing
    /// and direct/FFT metric selection.
    pub fn run_owned_pyramid_with_diagnostics<T: Sample>(
        &self,
        search: &MultiResolutionSearch,
        pyramid: &OwnedPyramid<T>,
    ) -> Result<PyramidPipelineResult> {
        let levels = pyramid.levels();
        self.run_pyramid_with_diagnostics(search, &levels)
    }
}

/// FFT-backed volume tracking, mirroring [`track_volume`] with the Apollo
/// finite-boundary metric. Only compiled with the `fft` feature.
#[cfg(feature = "fft")]
fn track_volume_fft<T: Sample>(
    fixed: &[T],
    moving: &[T],
    dims: [usize; 3],
    config: BlockMatchingConfig,
    grid: BlockGrid,
    refinement: SubpixelRefinement,
) -> Result<DisplacementField> {
    use crate::fft::{match_block_fft, FftPadding};

    config.validate()?;
    grid.validate()?;
    let expected = dims[0]
        .checked_mul(dims[1])
        .and_then(|v| v.checked_mul(dims[2]))
        .ok_or_else(|| anyhow::anyhow!("dims {dims:?} overflow the buffer size calculation"))?;
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
        if let Ok(bd) = match_block_fft(
            fixed,
            moving,
            dims,
            centre,
            config,
            refinement,
            FftPadding::Zero,
        ) {
            displacements[i] = bd.displacement;
            peak_similarities[i] = bd.peak_similarity;
        }
    }

    Ok(DisplacementField {
        centres,
        displacements,
        peak_similarities,
    })
}
