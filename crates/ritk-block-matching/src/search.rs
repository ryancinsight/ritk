//! Coarse-to-fine search planning and execution.
//!
//! This module owns the policy that turns a finest-resolution search radius into
//! explicit pyramid levels. It does not resample images: callers provide the
//! fixed and moving buffers for each level, in coarse-to-fine order.

use anyhow::{bail, Result};

use super::{
    match_block_at, BayesianDisplacementPrior, BlockGrid, DisplacementField, Sample,
    SubpixelRefinement,
};

/// One level in a caller-owned image pyramid.
///
/// `scale` is the number of finest-resolution voxels represented by one voxel
/// at this level. Levels must be supplied in the same coarse-to-fine order as
/// [`MultiResolutionSearch::regions`]. The matcher validates buffer lengths but
/// deliberately leaves downsampling, interpolation, and image registration to
/// the caller.
pub struct PyramidLevel<'a, T> {
    /// Fixed-image samples at this resolution.
    pub fixed: &'a [T],
    /// Moving-image samples at this resolution.
    pub moving: &'a [T],
    /// Buffer dimensions in `[z, y, x]` order.
    pub dims: [usize; 3],
}

/// Search radius and block support at one pyramid scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchRegion {
    /// Finest-resolution voxels represented by one voxel at this level.
    pub scale: usize,
    /// Half-extent of the fixed block at this level, in level voxels.
    pub block_radius: [usize; 3],
    /// Half-extent of the moving search region at this level, in level voxels.
    pub search_radius: [usize; 3],
}

/// A deterministic coarse-to-fine search policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiResolutionSearch {
    regions: Vec<SearchRegion>,
}

impl MultiResolutionSearch {
    /// Build `levels` power-of-two regions from finest-resolution geometry.
    ///
    /// The first region is the coarsest and the final region is the finest
    /// (`scale == 1`). Radius conversion uses ceiling division, so physical
    /// coverage is never reduced by moving to a coarser grid. At least one axis
    /// of each radius must be non-zero, matching [`super::BlockMatchingConfig`]
    /// and supporting 1-D/2-D acquisitions with singleton axes.
    ///
    /// # Errors
    ///
    /// Rejects an empty plan, zero radii, more than `usize::BITS - 1` levels,
    /// or an overflowing power-of-two scale.
    pub fn new(block_radius: [usize; 3], search_radius: [usize; 3], levels: usize) -> Result<Self> {
        if levels == 0 {
            bail!("multi-resolution search requires at least one level");
        }
        if block_radius.iter().all(|&r| r == 0) {
            bail!("multi-resolution block radius cannot be zero on every axis");
        }
        if search_radius.iter().all(|&r| r == 0) {
            bail!("multi-resolution search radius cannot be zero on every axis");
        }
        if levels > usize::BITS as usize {
            bail!("multi-resolution level count {levels} is too large");
        }

        let mut regions = Vec::with_capacity(levels);
        for level in (0..levels).rev() {
            let scale = 1usize
                .checked_shl(level as u32)
                .ok_or_else(|| anyhow::anyhow!("pyramid scale overflows usize at level {level}"))?;
            regions.push(SearchRegion {
                scale,
                block_radius: ceil_div_axes(block_radius, scale),
                search_radius: ceil_div_axes(search_radius, scale),
            });
        }
        Ok(Self { regions })
    }

    /// The planned regions in coarse-to-fine order.
    #[must_use]
    pub fn regions(&self) -> &[SearchRegion] {
        &self.regions
    }

    /// Match caller-provided pyramid levels and propagate each displacement to
    /// the next finer resolution.
    ///
    /// The finest centre is expressed in finest-resolution coordinates. At
    /// each level it is converted to level coordinates, then the previous
    /// moving-image centre is scaled into the current level and used as the
    /// centre of the current search region. The returned displacement is in
    /// finest-resolution voxels. No image resampling or boundary padding is
    /// performed by this method.
    ///
    /// # Errors
    ///
    /// Returns an error when the number of images differs from the plan, when a
    /// level has invalid dimensions/buffers, or when a propagated moving block
    /// centre cannot fit in the moving image without padding.
    pub fn match_pyramid<T: Sample>(
        &self,
        pyramid: &[PyramidLevel<'_, T>],
        finest_centre: [usize; 3],
        refinement: SubpixelRefinement,
    ) -> Result<MultiResolutionDisplacement> {
        self.match_pyramid_with(
            pyramid,
            finest_centre,
            |level, fixed_centre, moving_centre, config| {
                match_block_at(
                    level.fixed,
                    level.moving,
                    level.dims,
                    fixed_centre,
                    moving_centre,
                    config,
                    refinement,
                )
            },
        )
    }

    /// Coarse-to-fine walk shared by every pyramid metric.
    ///
    /// Centre propagation, scaling and diagnostics are metric-independent, so
    /// they exist once here and the metric enters as the per-level match. A
    /// second copy of this walk is how the direct and FFT paths would drift
    /// apart in propagated centres while both still looked correct in
    /// isolation, which is precisely what their parity test checks.
    fn match_pyramid_with<T: Sample, F>(
        &self,
        pyramid: &[PyramidLevel<'_, T>],
        finest_centre: [usize; 3],
        mut match_level: F,
    ) -> Result<MultiResolutionDisplacement>
    where
        F: FnMut(
            &PyramidLevel<'_, T>,
            [usize; 3],
            [usize; 3],
            super::BlockMatchingConfig,
        ) -> Result<super::BlockDisplacement>,
    {
        if pyramid.len() != self.regions.len() {
            bail!(
                "pyramid has {} levels but search plan has {} regions",
                pyramid.len(),
                self.regions.len()
            );
        }

        let mut previous_moving: Option<[usize; 3]> = None;
        let mut previous_scale = 0usize;
        let mut diagnostics = Vec::with_capacity(pyramid.len());

        for (index, (level, region)) in pyramid.iter().zip(&self.regions).enumerate() {
            let fixed_centre = scale_coordinate(finest_centre, region.scale);
            let moving_centre = match previous_moving {
                None => fixed_centre,
                Some(previous) => {
                    let mut current = [0usize; 3];
                    for axis in 0..3 {
                        // The previous centre already includes the previous
                        // level's measured displacement. Convert that absolute
                        // moving coordinate, rather than scaling displacement
                        // alone, to avoid rounding drift around the origin.
                        let physical = previous[axis] as f64 * previous_scale as f64;
                        let coordinate = (physical / region.scale as f64).round();
                        if !coordinate.is_finite() || coordinate < 0.0 {
                            bail!("propagated moving centre is invalid at level {index}");
                        }
                        current[axis] = coordinate as usize;
                    }
                    current
                }
            };

            let config = super::BlockMatchingConfig {
                block_radius: region.block_radius,
                search_radius: region.search_radius,
            };
            let result = match_level(level, fixed_centre, moving_centre, config)?;

            previous_moving = Some(add_displacement(fixed_centre, result.displacement));
            previous_scale = region.scale;
            diagnostics.push(PyramidLevelDisplacement {
                scale: region.scale,
                fixed_centre,
                moving_centre,
                displacement: result.displacement,
                peak_similarity: result.peak_similarity,
            });
        }

        let finest = diagnostics
            .last()
            .ok_or_else(|| anyhow::anyhow!("multi-resolution search produced no levels"))?;
        Ok(MultiResolutionDisplacement {
            displacement: scale_displacement(finest.displacement, finest.scale),
            peak_similarity: finest.peak_similarity,
            levels: diagnostics,
        })
    }

    /// Match the pyramid and apply a confidence-weighted Bayesian prior to the
    /// final finest-resolution displacement.
    ///
    /// This is explicit composition: pyramid matching completes first, then
    /// [`BayesianDisplacementPrior`] uses the finest peak similarity as the
    /// observation confidence. Per-level diagnostics and peak metadata are not
    /// rewritten. Use [`Self::match_pyramid`] when the raw observation is
    /// required for a separate post-processing policy.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`Self::match_pyramid`]. The prior
    /// is already validated when constructed with
    /// [`BayesianDisplacementPrior::new`].
    pub fn match_pyramid_regularized<T: Sample>(
        &self,
        pyramid: &[PyramidLevel<'_, T>],
        finest_centre: [usize; 3],
        refinement: SubpixelRefinement,
        prior: &BayesianDisplacementPrior,
    ) -> Result<MultiResolutionDisplacement> {
        let raw = self.match_pyramid(pyramid, finest_centre, refinement)?;
        Ok(prior.regularize_pyramid(&raw))
    }

    /// Match every valid block centre across a caller-owned pyramid.
    ///
    /// The finest level supplies the grid geometry. Each centre is matched
    /// coarse-to-fine with the preceding level's absolute moving centre, then
    /// stored in a [`DisplacementField`] expressed in finest-resolution voxels.
    /// Fixed or propagated moving blocks that cannot fit are skipped with the
    /// same `NAN` confidence and zero displacement convention as
    /// [`super::track_volume`]. The caller owns pyramid construction and
    /// resampling; this method only executes the matching.
    ///
    /// # Errors
    ///
    /// Returns an error when the pyramid level count, dimensions, or buffer
    /// lengths are invalid, or when any grid stride is zero.
    pub fn track_volume_pyramid<T: Sample>(
        &self,
        pyramid: &[PyramidLevel<'_, T>],
        grid: BlockGrid,
        refinement: SubpixelRefinement,
    ) -> Result<DisplacementField> {
        self.validate_pyramid(pyramid)?;
        for axis in 0..3 {
            if grid.stride[axis] == 0 {
                bail!("grid stride is zero on axis {axis}; that would loop forever");
            }
        }

        let finest = self
            .regions
            .last()
            .ok_or_else(|| anyhow::anyhow!("multi-resolution search produced no levels"))?;
        let finest_dims = pyramid
            .last()
            .ok_or_else(|| anyhow::anyhow!("multi-resolution pyramid is empty"))?
            .dims;
        let config = super::BlockMatchingConfig {
            block_radius: finest.block_radius,
            search_radius: finest.search_radius,
        };
        let centres = grid.centres(finest_dims, &config);
        let mut displacements = vec![[0.0; 3]; centres.len()];
        let mut peak_similarities = vec![f64::NAN; centres.len()];

        for (index, &centre) in centres.iter().enumerate() {
            if let Ok(result) = self.match_pyramid(pyramid, centre, refinement) {
                displacements[index] = result.displacement;
                peak_similarities[index] = result.peak_similarity;
            }
        }

        Ok(DisplacementField {
            centres,
            displacements,
            peak_similarities,
        })
    }

    /// Match a pyramid volume and apply a confidence-weighted prior afterward.
    ///
    /// This is explicit composition: all raw pyramid observations are collected
    /// first, then [`BayesianDisplacementPrior`] uses each finest-level peak to
    /// regularize the resulting field. Centres and confidence metadata remain
    /// unchanged.
    pub fn track_volume_pyramid_regularized<T: Sample>(
        &self,
        pyramid: &[PyramidLevel<'_, T>],
        grid: BlockGrid,
        refinement: SubpixelRefinement,
        prior: &BayesianDisplacementPrior,
    ) -> Result<DisplacementField> {
        let field = self.track_volume_pyramid(pyramid, grid, refinement)?;
        Ok(prior.regularize(&field))
    }

    /// [`Self::match_pyramid`] with the FFT-backed metric.
    ///
    /// Same walk, same centre propagation, same diagnostics — only the
    /// per-level similarity differs, so any divergence from the direct path is
    /// a property of the metric rather than of the traversal.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::match_pyramid`], plus any raised by
    /// the FFT provider.
    #[cfg(feature = "fft")]
    pub fn match_pyramid_fft<T: Sample>(
        &self,
        pyramid: &[PyramidLevel<'_, T>],
        finest_centre: [usize; 3],
        refinement: SubpixelRefinement,
        padding: crate::FftPadding,
    ) -> Result<MultiResolutionDisplacement> {
        self.match_pyramid_with(
            pyramid,
            finest_centre,
            |level, fixed_centre, moving_centre, config| {
                crate::fft::match_block_fft_at(
                    level.fixed,
                    level.moving,
                    level.dims,
                    fixed_centre,
                    moving_centre,
                    config,
                    refinement,
                    padding,
                )
            },
        )
    }

    /// [`Self::track_volume_pyramid`] with the FFT-backed metric.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::track_volume_pyramid`], plus any
    /// raised by the FFT provider.
    #[cfg(feature = "fft")]
    pub fn track_volume_pyramid_fft<T: Sample>(
        &self,
        pyramid: &[PyramidLevel<'_, T>],
        grid: BlockGrid,
        refinement: SubpixelRefinement,
        padding: crate::FftPadding,
    ) -> Result<DisplacementField> {
        self.track_volume_pyramid_fft_diagnostics(pyramid, grid, refinement, padding)
            .map(|diagnostics| diagnostics.as_field())
    }

    /// [`Self::track_volume_pyramid_diagnostics`] with the FFT-backed metric.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::track_volume_pyramid_diagnostics`],
    /// plus any raised by the FFT provider.
    #[cfg(feature = "fft")]
    pub fn track_volume_pyramid_fft_diagnostics<T: Sample>(
        &self,
        pyramid: &[PyramidLevel<'_, T>],
        grid: BlockGrid,
        refinement: SubpixelRefinement,
        padding: crate::FftPadding,
    ) -> Result<PyramidDisplacementField> {
        self.track_volume_pyramid_with(pyramid, grid, |centre| {
            self.match_pyramid_fft(pyramid, centre, refinement, padding)
        })
    }

    /// Match every block of a grid, retaining each block's per-level evidence.
    ///
    /// Identical to [`Self::track_volume_pyramid`] in what it computes and in
    /// which blocks it skips; it differs only in keeping the coarse-to-fine
    /// diagnostics that the plain form discards. Projecting the result with
    /// [`PyramidDisplacementField::try_as_field`] yields exactly the field the
    /// plain form returns, so the two never disagree.
    ///
    /// # Errors
    ///
    /// Returns the same pyramid and grid validation errors as
    /// [`Self::track_volume_pyramid`].
    pub fn track_volume_pyramid_diagnostics<T: Sample>(
        &self,
        pyramid: &[PyramidLevel<'_, T>],
        grid: BlockGrid,
        refinement: SubpixelRefinement,
    ) -> Result<PyramidDisplacementField> {
        self.track_volume_pyramid_with(pyramid, grid, |centre| {
            self.match_pyramid(pyramid, centre, refinement)
        })
    }

    /// Shared body of the pyramid volume walks.
    ///
    /// The centre grid, the skip rule, and the arrays are identical whichever
    /// metric ran; only the per-centre match differs. Keeping one walk is what
    /// makes the direct and FFT diagnostics comparable — a divergence between
    /// them would otherwise be a difference in the walk rather than in the
    /// metric under test.
    fn track_volume_pyramid_with<T: Sample, F>(
        &self,
        pyramid: &[PyramidLevel<'_, T>],
        grid: BlockGrid,
        mut match_at: F,
    ) -> Result<PyramidDisplacementField>
    where
        F: FnMut([usize; 3]) -> Result<MultiResolutionDisplacement>,
    {
        self.validate_pyramid(pyramid)?;
        grid.validate()?;

        let finest = self
            .regions
            .last()
            .ok_or_else(|| anyhow::anyhow!("multi-resolution search produced no levels"))?;
        let finest_dims = pyramid
            .last()
            .ok_or_else(|| anyhow::anyhow!("multi-resolution pyramid is empty"))?
            .dims;
        let config = super::BlockMatchingConfig {
            block_radius: finest.block_radius,
            search_radius: finest.search_radius,
        };
        let centres = grid.centres(finest_dims, &config);
        let mut displacements = vec![[0.0; 3]; centres.len()];
        let mut peak_similarities = vec![f64::NAN; centres.len()];
        let mut level_diagnostics = vec![None; centres.len()];

        for (index, &centre) in centres.iter().enumerate() {
            // A failed block keeps its NaN peak and `None` diagnostics: it was
            // not measured, which is distinct from measuring zero displacement.
            if let Ok(result) = match_at(centre) {
                displacements[index] = result.displacement;
                peak_similarities[index] = result.peak_similarity;
                level_diagnostics[index] = Some(result.levels);
            }
        }

        Ok(PyramidDisplacementField {
            centres,
            displacements,
            peak_similarities,
            level_diagnostics,
        })
    }

    fn validate_pyramid<T: Sample>(&self, pyramid: &[PyramidLevel<'_, T>]) -> Result<()> {
        if pyramid.len() != self.regions.len() {
            bail!(
                "pyramid has {} levels but search plan has {} regions",
                pyramid.len(),
                self.regions.len()
            );
        }
        for (index, level) in pyramid.iter().enumerate() {
            if level.dims.contains(&0) {
                bail!("pyramid level {index} has a zero image dimension");
            }
            let expected = level
                .dims
                .iter()
                .try_fold(1usize, |size, &extent| size.checked_mul(extent))
                .ok_or_else(|| anyhow::anyhow!("pyramid level {index} dimensions overflow"))?;
            if level.fixed.len() != expected || level.moving.len() != expected {
                bail!(
                    "pyramid level {index} buffers must both hold {expected} voxels for dims {:?}",
                    level.dims
                );
            }
        }
        Ok(())
    }
}

/// Per-level diagnostic from a coarse-to-fine match.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyramidLevelDisplacement {
    /// Finest-resolution voxels per level voxel.
    pub scale: usize,
    /// Fixed-image centre at this level.
    pub fixed_centre: [usize; 3],
    /// Moving-image search centre before the local search at this level.
    pub moving_centre: [usize; 3],
    /// Displacement measured relative to `fixed_centre`, in level voxels.
    pub displacement: [f64; 3],
    /// Similarity at the level's correlation peak.
    pub peak_similarity: f64,
}

/// A displacement field that retains each block's coarse-to-fine evidence.
///
/// [`MultiResolutionSearch::track_volume_pyramid`] returns only the finest-level
/// answer, which is all most callers want. When a block's result is surprising,
/// though, the useful question is *where* in the pyramid it went wrong — a
/// displacement that was already wrong at the coarsest level failed for a
/// different reason than one that drifted at the last refinement, and the
/// finest-level peak alone cannot distinguish them.
///
/// The four arrays are parallel and indexed together. `level_diagnostics[i]` is
/// `None` exactly when block `i` was skipped, which is the same condition as a
/// non-finite `peak_similarities[i]`: a skipped block has no evidence rather
/// than empty evidence, and the two must not be confused.
#[derive(Debug, Clone, PartialEq)]
pub struct PyramidDisplacementField {
    /// Block centre coordinates at the finest level, in `[z, y, x]` voxel order.
    pub centres: Vec<[usize; 3]>,
    /// Displacement per block, in finest-level voxels.
    pub displacements: Vec<[f64; 3]>,
    /// Finest-level peak similarity per block; non-finite when skipped.
    pub peak_similarities: Vec<f64>,
    /// Per-level evidence for each block, coarse to fine; `None` when skipped.
    pub level_diagnostics: Vec<Option<Vec<PyramidLevelDisplacement>>>,
}

impl PyramidDisplacementField {
    /// Number of blocks.
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

    /// Check the invariants every consumer relies on.
    ///
    /// The matcher produces a valid field by construction, so this exists for
    /// fields assembled by hand or arriving from deserialization, where a
    /// length mismatch would index out of bounds or silently drop blocks.
    ///
    /// # Errors
    ///
    /// Returns an error when the four arrays disagree in length, when a
    /// retained diagnostic list is empty (a matched block has at least one
    /// level), or when a retained level carries a zero scale — scale is a
    /// divisor when mapping level coordinates to finest-level voxels, so zero
    /// is not a degenerate case to tolerate but an impossible one.
    pub fn validate(&self) -> Result<()> {
        let n = self.centres.len();
        if self.displacements.len() != n
            || self.peak_similarities.len() != n
            || self.level_diagnostics.len() != n
        {
            bail!(
                "pyramid displacement field arrays disagree: {n} centres, {} displacements, \
                 {} peak similarities, {} diagnostic entries",
                self.displacements.len(),
                self.peak_similarities.len(),
                self.level_diagnostics.len()
            );
        }
        for (index, entry) in self.level_diagnostics.iter().enumerate() {
            let Some(levels) = entry else { continue };
            if levels.is_empty() {
                bail!(
                    "block {index} retains an empty diagnostic list; a skipped block is \
                     represented by `None`, not by empty evidence"
                );
            }
            if let Some(level) = levels.iter().position(|level| level.scale == 0) {
                bail!("block {index} level {level} has scale 0, which cannot be a divisor");
            }
        }
        Ok(())
    }

    /// Which blocks carry a peak at or above `minimum_peak_similarity`.
    ///
    /// A non-finite peak marks a skipped block and is always `false`, so a
    /// caller thresholding at `0.0` still excludes skipped blocks rather than
    /// admitting them through a comparison that a `NaN` would fail silently.
    ///
    /// # Errors
    ///
    /// Returns an error when the field is invalid, or when the threshold is not
    /// a finite value in `[0, 1]` — normalized correlation has no other range,
    /// so a threshold outside it is a caller mistake rather than a filter that
    /// admits everything or nothing.
    pub fn valid_mask(&self, minimum_peak_similarity: f64) -> Result<Vec<bool>> {
        self.validate()?;
        if !minimum_peak_similarity.is_finite() || !(0.0..=1.0).contains(&minimum_peak_similarity) {
            bail!(
                "minimum peak similarity must be finite and in [0, 1], got \
                 {minimum_peak_similarity}"
            );
        }
        Ok(self
            .peak_similarities
            .iter()
            .zip(&self.level_diagnostics)
            .map(|(peak, levels)| {
                levels.is_some() && peak.is_finite() && *peak >= minimum_peak_similarity
            })
            .collect())
    }

    /// Project to a plain [`DisplacementField`], dropping the level evidence.
    ///
    /// # Errors
    ///
    /// Returns an error when [`Self::validate`] does, so an invalid field
    /// cannot be laundered into a valid-looking one by discarding the very
    /// evidence that shows it is malformed.
    pub fn try_as_field(&self) -> Result<DisplacementField> {
        self.validate()?;
        Ok(self.as_field_unchecked())
    }

    /// Project to a plain [`DisplacementField`].
    ///
    /// # Panics
    ///
    /// Panics when the field is invalid. Prefer this only for a field this
    /// crate produced, where validity holds by construction; use
    /// [`Self::try_as_field`] for anything assembled or deserialized.
    #[must_use]
    pub fn as_field(&self) -> DisplacementField {
        self.validate()
            .expect("invariant: a matcher-produced pyramid field is well-formed");
        self.as_field_unchecked()
    }

    fn as_field_unchecked(&self) -> DisplacementField {
        DisplacementField {
            centres: self.centres.clone(),
            displacements: self.displacements.clone(),
            peak_similarities: self.peak_similarities.clone(),
        }
    }
}

/// Result of a coarse-to-fine pyramid match.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiResolutionDisplacement {
    /// Final displacement in finest-resolution voxels.
    pub displacement: [f64; 3],
    /// Finest-level peak similarity.
    pub peak_similarity: f64,
    /// Coarse-to-fine diagnostics, one entry per planned level.
    pub levels: Vec<PyramidLevelDisplacement>,
}

/// A caller-owned coarse-to-fine pyramid for [`MultiResolutionSearch`].
///
/// Levels are ordered coarse-to-fine (the first entry has the largest scale and
/// the final entry has `scale == 1`), matching [`MultiResolutionSearch::regions`].
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedPyramid<T> {
    levels: Vec<OwnedLevel<T>>,
}

#[derive(Debug, Clone, PartialEq)]
struct OwnedLevel<T> {
    fixed: Vec<T>,
    moving: Vec<T>,
    dims: [usize; 3],
}

impl<T: Sample> OwnedPyramid<T> {
    /// Build a nearest-neighbour decimated pyramid.
    ///
    /// `scales` must be strictly decreasing coarse-to-fine, end at `1`, and
    /// divide every image extent. Nearest-neighbour decimation is only a valid
    /// RF pyramid when the source is already band-limited at the coarsened
    /// resolution; this function deliberately does not filter. Callers that
    /// need an anti-aliased pyramid for wideband RF should use
    /// [`Self::min_max`], which preserves local extrema instead of aliasing.
    ///
    /// # Errors
    ///
    /// Returns an error when `scales` is empty, not strictly decreasing, ends
    /// anywhere other than `1`, contains a zero, when any extent is not
    /// divisible by its scale, or when the buffers do not match `dims`.
    pub fn nearest(fixed: &[T], moving: &[T], dims: [usize; 3], scales: &[usize]) -> Result<Self> {
        validate_scales(dims, scales)?;
        let expected = dims[0] * dims[1] * dims[2];
        if fixed.len() != expected || moving.len() != expected {
            bail!(
                "fixed ({}) and moving ({}) buffers must both hold {expected} voxels for dims {dims:?}",
                fixed.len(),
                moving.len()
            );
        }
        let mut levels = Vec::with_capacity(scales.len());
        for scale in scales {
            let level_dims = [
                level_extent(dims[0], *scale),
                level_extent(dims[1], *scale),
                level_extent(dims[2], *scale),
            ];
            let level_len = level_dims[0] * level_dims[1] * level_dims[2];
            let mut fixed_level = Vec::with_capacity(level_len);
            let mut moving_level = Vec::with_capacity(level_len);
            for z in 0..level_dims[0] {
                for y in 0..level_dims[1] {
                    for x in 0..level_dims[2] {
                        let source = (z * scale * dims[1] + y * scale) * dims[2] + x * scale;
                        fixed_level.push(fixed[source]);
                        moving_level.push(moving[source]);
                    }
                }
            }
            levels.push(OwnedLevel {
                fixed: fixed_level,
                moving: moving_level,
                dims: level_dims,
            });
        }
        Ok(Self { levels })
    }

    /// Build a min/max pyramid that preserves the speckle envelope.
    ///
    /// Each `scale × scale × scale` source neighbourhood is reduced to its
    /// local minimum and maximum, stored as two adjacent samples along the
    /// fastest (x) axis at the coarse position. The level extent is therefore
    /// `[dims[0]/scale, dims[1]/scale, 2·dims[2]/scale]`: the min/max pair
    /// doubles only the axial extent, which is the axis speckle tracking
    /// refines. This is the RF-safer alternative to [`Self::nearest`] when the
    /// source is not band-limited at the coarsened resolution.
    ///
    /// # Errors
    ///
    /// Same validation as [`Self::nearest`].
    pub fn min_max(fixed: &[T], moving: &[T], dims: [usize; 3], scales: &[usize]) -> Result<Self> {
        validate_scales(dims, scales)?;
        let expected = dims[0] * dims[1] * dims[2];
        if fixed.len() != expected || moving.len() != expected {
            bail!(
                "fixed ({}) and moving ({}) buffers must both hold {expected} voxels for dims {dims:?}",
                fixed.len(),
                moving.len()
            );
        }
        let mut levels = Vec::with_capacity(scales.len());
        for scale in scales {
            let base_dims = [
                level_extent(dims[0], *scale),
                level_extent(dims[1], *scale),
                level_extent(dims[2], *scale),
            ];
            // The reduction window is the scale, except on a singleton axis,
            // which is not downsampled and has no second sample to read.
            let window = [
                reduction_window(dims[0], *scale),
                reduction_window(dims[1], *scale),
                reduction_window(dims[2], *scale),
            ];
            let level_dims = [base_dims[0], base_dims[1], 2 * base_dims[2]];
            let level_len = level_dims[0] * level_dims[1] * level_dims[2];
            let mut fixed_level = vec![f64::NAN; level_len];
            let mut moving_level = vec![f64::NAN; level_len];
            for z in 0..base_dims[0] {
                for y in 0..base_dims[1] {
                    for x in 0..base_dims[2] {
                        let mut block = [[f64::INFINITY, f64::NEG_INFINITY]; 2];
                        for dz in 0..window[0] {
                            for dy in 0..window[1] {
                                for dx in 0..window[2] {
                                    let sz = z * scale + dz;
                                    let sy = y * scale + dy;
                                    let sx = x * scale + dx;
                                    let source = (sz * dims[1] + sy) * dims[2] + sx;
                                    for (image, plane) in [(fixed, 0), (moving, 1)] {
                                        let v = image[source].to_f64();
                                        block[plane][0] = block[plane][0].min(v);
                                        block[plane][1] = block[plane][1].max(v);
                                    }
                                }
                            }
                        }
                        let level_index = (z * level_dims[1] + y) * level_dims[2] + 2 * x;
                        fixed_level[level_index] = block[0][0];
                        fixed_level[level_index + 1] = block[0][1];
                        moving_level[level_index] = block[1][0];
                        moving_level[level_index + 1] = block[1][1];
                    }
                }
            }
            let fixed_level: Vec<T> = fixed_level
                .into_iter()
                .map(T::from_f64_saturating)
                .collect();
            let moving_level: Vec<T> = moving_level
                .into_iter()
                .map(T::from_f64_saturating)
                .collect();
            levels.push(OwnedLevel {
                fixed: fixed_level,
                moving: moving_level,
                dims: level_dims,
            });
        }
        Ok(Self { levels })
    }

    /// The levels in coarse-to-fine order, as borrowed [`PyramidLevel`]s.
    #[must_use]
    pub fn levels(&self) -> Vec<PyramidLevel<'_, T>> {
        self.levels
            .iter()
            .map(|level| PyramidLevel {
                fixed: &level.fixed,
                moving: &level.moving,
                dims: level.dims,
            })
            .collect()
    }
}

/// Validate a scale list for pyramid construction.
///
/// Scales must be strictly decreasing coarse-to-fine, end at `1`, be non-zero,
/// and divide every image extent (a fractional level would silently drop data).
/// Size of the reduction window on one axis.
///
/// A singleton axis contributes exactly one sample: it is not downsampled, so
/// reading `scale` samples along it would index past the image.
fn reduction_window(extent: usize, scale: usize) -> usize {
    if extent == 1 {
        1
    } else {
        scale
    }
}

/// Extent of one axis at a given pyramid scale.
///
/// A singleton axis stays singleton: it carries no resolution to trade away,
/// and dividing it would produce a zero-extent level.
fn level_extent(extent: usize, scale: usize) -> usize {
    if extent == 1 {
        1
    } else {
        extent / scale
    }
}

fn validate_scales(dims: [usize; 3], scales: &[usize]) -> Result<()> {
    if scales.is_empty() {
        bail!("a pyramid needs at least one scale");
    }
    if dims.contains(&0) {
        bail!("all image dimensions must be positive, got {dims:?}");
    }
    for (index, &scale) in scales.iter().enumerate() {
        if scale == 0 {
            bail!("pyramid scale at level {index} is zero");
        }
        for (axis, &extent) in dims.iter().enumerate() {
            // A singleton axis is not downsampled, so it need not divide: a 2-D
            // acquisition is a volume with one out-of-plane sample, and
            // requiring `1 % scale == 0` would reject the dominant ultrasound
            // geometry outright — the same case `BlockMatchingConfig::validate`
            // already admits for a zero block radius.
            if extent != 1 && !extent.is_multiple_of(scale) {
                bail!(
                    "image extent {extent} on axis {axis} is not divisible by scale {scale} at level {index}"
                );
            }
        }
    }
    for pair in scales.windows(2) {
        if pair[0] <= pair[1] {
            bail!(
                "pyramid scales must be strictly decreasing coarse-to-fine, got {:?}",
                scales
            );
        }
    }
    if scales[scales.len() - 1] != 1 {
        bail!("the finest pyramid scale must be 1, got {:?}", scales);
    }
    Ok(())
}

fn ceil_div(value: usize, divisor: usize) -> usize {
    value / divisor + usize::from(!value.is_multiple_of(divisor))
}

fn ceil_div_axes(values: [usize; 3], divisor: usize) -> [usize; 3] {
    [
        ceil_div(values[0], divisor),
        ceil_div(values[1], divisor),
        ceil_div(values[2], divisor),
    ]
}

fn scale_coordinate(coordinate: [usize; 3], scale: usize) -> [usize; 3] {
    [
        coordinate[0] / scale,
        coordinate[1] / scale,
        coordinate[2] / scale,
    ]
}

fn scale_displacement(displacement: [f64; 3], scale: usize) -> [f64; 3] {
    [
        displacement[0] * scale as f64,
        displacement[1] * scale as f64,
        displacement[2] * scale as f64,
    ]
}

fn add_displacement(centre: [usize; 3], displacement: [f64; 3]) -> [usize; 3] {
    [
        (centre[0] as f64 + displacement[0]).round().max(0.0) as usize,
        (centre[1] as f64 + displacement[1]).round().max(0.0) as usize,
        (centre[2] as f64 + displacement[2]).round().max(0.0) as usize,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_coarse_to_fine_ceiling_radii() {
        let plan = MultiResolutionSearch::new([0, 5, 4], [0, 9, 7], 3).unwrap();
        assert_eq!(
            plan.regions(),
            &[
                SearchRegion {
                    scale: 4,
                    block_radius: [0, 2, 1],
                    search_radius: [0, 3, 2],
                },
                SearchRegion {
                    scale: 2,
                    block_radius: [0, 3, 2],
                    search_radius: [0, 5, 4],
                },
                SearchRegion {
                    scale: 1,
                    block_radius: [0, 5, 4],
                    search_radius: [0, 9, 7],
                },
            ]
        );
    }

    #[test]
    fn rejects_empty_and_featureless_plans() {
        assert!(MultiResolutionSearch::new([1, 1, 1], [1, 1, 1], 0).is_err());
        assert!(MultiResolutionSearch::new([0, 0, 0], [1, 1, 1], 1).is_err());
        assert!(MultiResolutionSearch::new([1, 1, 1], [0, 0, 0], 1).is_err());
    }

    /// Deterministic ramp texture: value increases with the axial (x) index, so
    /// a decimated level samples every `scale`-th ramp value exactly.
    fn ramp(dims: [usize; 3]) -> Vec<f32> {
        let mut out = vec![0.0_f32; dims[0] * dims[1] * dims[2]];
        for z in 0..dims[0] {
            for y in 0..dims[1] {
                for x in 0..dims[2] {
                    out[(z * dims[1] + y) * dims[2] + x] = (z * 100 + y * 10 + x) as f32;
                }
            }
        }
        out
    }

    #[test]
    fn nearest_pyramid_samples_every_scale_th_voxel() {
        let dims = [8, 8, 8];
        let fixed = ramp(dims);
        let moving = ramp(dims);
        let pyramid =
            OwnedPyramid::nearest(&fixed, &moving, dims, &[4, 2, 1]).expect("valid pyramid");
        let levels = pyramid.levels();
        assert_eq!(levels.len(), 3);

        // Coarse level (scale 4): extent 2³, and its (0,0,1) sample is the
        // source voxel (0,0,4), which has ramp value 4.
        assert_eq!(levels[0].dims, [2, 2, 2]);
        assert_eq!(levels[0].fixed.len(), 8);
        assert_eq!(levels[0].fixed[1], 4.0_f32);

        // Finest level (scale 1) is a copy of the source.
        assert_eq!(levels[2].dims, dims);
        assert_eq!(levels[2].fixed, fixed);
    }

    #[test]
    fn min_max_pyramid_preserves_local_extrema() {
        let dims = [4, 4, 4];
        // A single bright spike at the source centre (2,2,2).
        let mut fixed = ramp(dims);
        let centre = (2 * dims[1] + 2) * dims[2] + 2;
        fixed[centre] = 1000.0;
        let moving = ramp(dims);
        let pyramid = OwnedPyramid::min_max(&fixed, &moving, dims, &[2, 1]).expect("valid pyramid");

        // Scale-2 level: min/max pair along x, extent [2, 2, 4].
        let levels = pyramid.levels();
        assert_eq!(levels[0].dims, [2, 2, 4]);
        // The source spike is inside the z=1,y=1,x=1 block; its min/max pair
        // lives at level (1, 1, 2*x=2 → min, 2*x+1=3 → max). The ramp minimum
        // of that block is 222, but the spike replaced (2,2,2)=222, so the
        // remaining minimum is (2,2,3)=223.
        let pair = 3 * 4 + 2; // (z*2 + y)*4 + 2*x for z=y=x=1
        assert_eq!(levels[0].fixed[pair], 223.0);
        assert_eq!(levels[0].fixed[pair + 1], 1000.0); // the spike
    }

    #[test]
    fn pyramid_validates_scale_lists_and_buffer_lengths() {
        let dims = [8, 8, 8];
        let fixed = ramp(dims);
        let moving = ramp(dims);
        // Scales must end at 1.
        assert!(OwnedPyramid::nearest(&fixed, &moving, dims, &[4, 2]).is_err());
        // Scales must be strictly decreasing.
        assert!(OwnedPyramid::nearest(&fixed, &moving, dims, &[2, 4, 1]).is_err());
        // Extents must be divisible.
        assert!(OwnedPyramid::nearest(&fixed, &moving, dims, &[3, 1]).is_err());
        // Empty scale list.
        assert!(OwnedPyramid::nearest(&fixed, &moving, dims, &[]).is_err());
        // Buffer length mismatch.
        assert!(OwnedPyramid::nearest(&fixed, &moving[..10], dims, &[2, 1]).is_err());
    }

    /// Deterministic white-noise texture (splitmix64 finalizer), so correlation
    /// has a unique peak instead of the ambiguity a monotonic ramp produces.
    fn texture(z: usize, y: isize, x: isize) -> f32 {
        let seed = (z as i64)
            .wrapping_mul(7919)
            .wrapping_add((y as i64).wrapping_mul(104_729))
            .wrapping_add((x as i64).wrapping_mul(15_485_863)) as u64;
        let mut v = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        v ^= v >> 31;
        ((v >> 11) as f64 / (1_u64 << 53) as f64) as f32
    }

    #[test]
    fn pyramid_and_search_recover_a_coarse_to_fine_shift() {
        // Moving is the fixed texture shifted by (0, 0, +4): content at x in
        // moving came from x-4 in fixed. The coarse level (scale 2) sees the
        // shift as 2 level-voxels, within its search radius.
        let dims = [4, 16, 32];
        let mut fixed = vec![0.0_f32; dims[0] * dims[1] * dims[2]];
        let mut moving = vec![0.0_f32; dims[0] * dims[1] * dims[2]];
        for z in 0..dims[0] {
            for y in 0..dims[1] {
                for x in 0..dims[2] {
                    fixed[(z * dims[1] + y) * dims[2] + x] = texture(z, y as isize, x as isize);
                    moving[(z * dims[1] + y) * dims[2] + x] =
                        texture(z, y as isize, x as isize - 4);
                }
            }
        }

        let pyramid = OwnedPyramid::nearest(&fixed, &moving, dims, &[2, 1]).expect("pyramid");
        let plan = MultiResolutionSearch::new([0, 0, 2], [0, 0, 8], 2).expect("plan");
        let result = plan
            .match_pyramid(&pyramid.levels(), [2, 8, 16], SubpixelRefinement::None)
            .expect("pyramid match");

        // The coarse level recovers the axial shift and the fine level refines it.
        assert_eq!(result.displacement, [0.0, 0.0, 4.0]);
        assert_eq!(result.levels.len(), 2);
    }
}
