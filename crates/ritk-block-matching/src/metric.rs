//! The metric image: similarity at every candidate integer offset.

use anyhow::{bail, Result};

use super::{BlockMatchingConfig, MovingSamples, Sample};

/// Similarity measure evaluated between the fixed block and a candidate moving
/// block.
///
/// A closed set: the choice is made once per block and then the candidate loop
/// runs monomorphically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlockMetric {
    /// Zero-mean normalized cross-correlation.
    ///
    /// ```text
    /// NCC = Σ(F − F̄)(M − M̄) / sqrt( Σ(F − F̄)² · Σ(M − M̄)² )
    /// ```
    ///
    /// Subtracting the means makes it invariant to a brightness offset and the
    /// normalization makes it invariant to gain, which is what lets it track
    /// speckle through the depth-dependent amplitude changes ultrasound has.
    /// The result lies in `[-1, 1]`.
    #[default]
    NormalizedCrossCorrelation,
}

/// Similarity sampled over the search region.
///
/// `values` is row-major over the candidate offsets, with extent
/// `2·search_radius + 1` per axis; the centre entry is the null displacement.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricImage {
    /// Similarity per candidate offset.
    pub values: Vec<f64>,
    /// Extent of the candidate grid per axis.
    pub extent: [usize; 3],
    /// Search half-extent per axis, so offset `i` on an axis is
    /// `i as isize − search_radius`.
    pub search_radius: [usize; 3],
}

impl MetricImage {
    /// Similarity at a candidate grid position.
    #[inline]
    #[must_use]
    pub fn at(&self, z: usize, y: usize, x: usize) -> f64 {
        self.values[(z * self.extent[1] + y) * self.extent[2] + x]
    }
}

/// Evaluate `metric` at every integer offset in a same-centre search region.
///
/// `centre` locates both the fixed block and the moving search. The
/// coarse-to-fine search keeps those centres distinct internally after
/// propagating displacement from a coarser level.
///
/// Moving candidates touching unavailable [`MovingSamples`] entries are left
/// at negative infinity.
///
/// # Errors
///
/// Returns an error for invalid configuration, mismatched buffer geometry, a
/// fixed or moving centre whose block leaves the image, or a non-finite or
/// featureless fixed block.
pub fn metric_image<T: Sample>(
    fixed: &[T],
    moving: MovingSamples<'_, T>,
    dims: [usize; 3],
    centre: [usize; 3],
    config: BlockMatchingConfig,
    metric: BlockMetric,
) -> Result<MetricImage> {
    metric_image_at(fixed, moving, dims, centre, centre, config, metric)
}

/// Evaluate `metric` around `moving_centre` for a block fixed at
/// `fixed_centre`.
///
/// Candidate offsets that leave the image are not padded or clamped. They are
/// left at negative infinity, preserving finite-edge semantics and preventing
/// padding values from becoming correlation evidence.
pub(crate) fn metric_image_at<T: Sample>(
    fixed: &[T],
    moving: MovingSamples<'_, T>,
    dims: [usize; 3],
    fixed_centre: [usize; 3],
    moving_centre: [usize; 3],
    config: BlockMatchingConfig,
    _metric: BlockMetric,
) -> Result<MetricImage> {
    validate_inputs(fixed, moving, dims, fixed_centre, moving_centre, config)?;
    let radius = config.block_radius;
    let search = config.search_radius;
    let extent = [
        search[0]
            .checked_mul(2)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("search extent overflows on axis 0"))?,
        search[1]
            .checked_mul(2)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("search extent overflows on axis 1"))?,
        search[2]
            .checked_mul(2)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("search extent overflows on axis 2"))?,
    ];
    let value_count = extent[0]
        .checked_mul(extent[1])
        .and_then(|v| v.checked_mul(extent[2]))
        .ok_or_else(|| anyhow::anyhow!("metric image extent {extent:?} overflows"))?;

    // Fixed block, mean-subtracted once: it is reused for every candidate.
    let block = gather_block(fixed, dims, fixed_centre, radius);
    if block.iter().any(|value| !value.is_finite()) {
        bail!(
            "fixed block at {fixed_centre:?} contains a non-finite sample; every candidate would depend on unavailable data"
        );
    }
    let block_mean = block.iter().sum::<f64>() / block.len() as f64;
    let fixed_centred: Vec<f64> = block.iter().map(|&v| v - block_mean).collect();
    let fixed_energy: f64 = fixed_centred.iter().map(|v| v * v).sum();
    if fixed_energy <= 0.0 {
        bail!(
            "fixed block at {fixed_centre:?} has zero variance; normalized correlation is undefined \
             and any peak would be an artefact of iteration order"
        );
    }
    let fixed_norm = fixed_energy.sqrt();

    let mut values = vec![f64::NEG_INFINITY; value_count];
    // One scratch buffer for every candidate. A speckle tracker calls this per
    // depth sample of every line, so allocating per candidate would put tens of
    // allocations into the inner loop of a volume-wide sweep.
    let mut candidate = Vec::with_capacity(block.len());
    for (oz, dz) in (-(search[0] as isize)..=search[0] as isize).enumerate() {
        for (oy, dy) in (-(search[1] as isize)..=search[1] as isize).enumerate() {
            for (ox, dx) in (-(search[2] as isize)..=search[2] as isize).enumerate() {
                let shifted = [
                    moving_centre[0] as isize + dz,
                    moving_centre[1] as isize + dy,
                    moving_centre[2] as isize + dx,
                ];
                // A candidate whose block leaves the image is not evaluated:
                // padding it would invent data and bias the correlation toward
                // the padding value. The centre-aware caller validates the
                // centre block, but retaining this guard makes the metric safe
                // when the search reaches a finite image boundary.
                let inside = (0..3).all(|axis| {
                    let r = radius[axis] as isize;
                    let extent = dims[axis] as isize;
                    shifted[axis] - r >= 0 && shifted[axis] + r < extent
                });
                if !inside {
                    continue;
                }
                let candidate_centre = [
                    shifted[0] as usize,
                    shifted[1] as usize,
                    shifted[2] as usize,
                ];
                if !candidate_is_valid(moving.validity(), dims, candidate_centre, radius) {
                    continue;
                }
                gather_block_into(
                    moving.values(),
                    dims,
                    candidate_centre,
                    radius,
                    &mut candidate,
                );
                if candidate.iter().any(|value| !value.is_finite()) {
                    continue;
                }
                let mean = candidate.iter().sum::<f64>() / candidate.len() as f64;
                let mut cross = 0.0;
                let mut energy = 0.0;
                for (&c, &f) in candidate.iter().zip(fixed_centred.iter()) {
                    let centred = c - mean;
                    cross += centred * f;
                    energy += centred * centred;
                }
                // A candidate block with no variance never correlates; leaving
                // it at -inf keeps it out of the peak search rather than
                // producing a 0/0.
                if energy > 0.0 {
                    values[(oz * extent[1] + oy) * extent[2] + ox] =
                        cross / (fixed_norm * energy.sqrt());
                }
            }
        }
    }

    Ok(MetricImage {
        values,
        extent,
        search_radius: search,
    })
}

fn validate_inputs<T: Sample>(
    fixed: &[T],
    moving: MovingSamples<'_, T>,
    dims: [usize; 3],
    fixed_centre: [usize; 3],
    moving_centre: [usize; 3],
    config: BlockMatchingConfig,
) -> Result<()> {
    config.validate()?;
    let expected = dims[0]
        .checked_mul(dims[1])
        .and_then(|value| value.checked_mul(dims[2]))
        .ok_or_else(|| anyhow::anyhow!("dims {dims:?} overflow the buffer size calculation"))?;
    if fixed.len() != expected || moving.values().len() != expected {
        bail!(
            "fixed ({}) and moving ({}) buffers must both hold {expected} voxels for dims {dims:?}",
            fixed.len(),
            moving.values().len()
        );
    }
    for axis in 0..3 {
        for (label, centre) in [("fixed", fixed_centre), ("moving", moving_centre)] {
            let radius = config.block_radius[axis];
            let high = centre[axis].checked_add(radius);
            if centre[axis].checked_sub(radius).is_none()
                || high.is_none_or(|value| value >= dims[axis])
            {
                bail!(
                    "{label} block at {centre:?} with radius {:?} leaves the image on axis {axis} (extent {})",
                    config.block_radius,
                    dims[axis]
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn candidate_is_valid(
    validity: Option<&[bool]>,
    dims: [usize; 3],
    centre: [usize; 3],
    radius: [usize; 3],
) -> bool {
    let Some(validity) = validity else {
        return true;
    };
    for z in centre[0] - radius[0]..=centre[0] + radius[0] {
        for y in centre[1] - radius[1]..=centre[1] + radius[1] {
            for x in centre[2] - radius[2]..=centre[2] + radius[2] {
                if !validity[(z * dims[1] + y) * dims[2] + x] {
                    return false;
                }
            }
        }
    }
    true
}

/// Copy the block centred at `centre` into a fresh buffer.
fn gather_block<T: Sample>(
    buf: &[T],
    dims: [usize; 3],
    centre: [usize; 3],
    radius: [usize; 3],
) -> Vec<f64> {
    let mut out =
        Vec::with_capacity((2 * radius[0] + 1) * (2 * radius[1] + 1) * (2 * radius[2] + 1));
    gather_block_into(buf, dims, centre, radius, &mut out);
    out
}

/// Copy the block centred at `centre` into `out`, reusing its allocation.
fn gather_block_into<T: Sample>(
    buf: &[T],
    dims: [usize; 3],
    centre: [usize; 3],
    radius: [usize; 3],
    out: &mut Vec<f64>,
) {
    out.clear();
    for z in centre[0] - radius[0]..=centre[0] + radius[0] {
        for y in centre[1] - radius[1]..=centre[1] + radius[1] {
            for x in centre[2] - radius[2]..=centre[2] + radius[2] {
                out.push(buf[(z * dims[1] + y) * dims[2] + x].to_f64());
            }
        }
    }
}
