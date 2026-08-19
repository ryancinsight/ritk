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
//! Two seams, matching ITKUltrasound's decomposition:
//!
//! 1. A **metric image** — the similarity evaluated at every candidate integer
//!    offset in the search region ([`metric_image`]).
//! 2. A **displacement calculator** — how a peak in that surface becomes a
//!    displacement ([`SubpixelRefinement`]).
//!
//! Both are closed sets fixed by the method, so they are exhaustively matched
//! enums rather than trait objects, and the choice is made once per block
//! rather than per candidate (atlas ADR 0041).
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

#[cfg(test)]
#[path = "tests_block_matching.rs"]
mod tests;

pub use metric::{metric_image, BlockMetric, MetricImage};
pub use refine::SubpixelRefinement;

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
