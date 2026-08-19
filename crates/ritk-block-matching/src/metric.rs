//! The metric image: similarity at every candidate integer offset.

use anyhow::{bail, Result};

use super::BlockMatchingConfig;

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

/// Evaluate `metric` at every integer offset in the search region.
///
/// # Errors
///
/// Returns an error when the fixed block has no variance. A constant block
/// correlates equally with everything, so its peak is an artefact of iteration
/// order rather than a measurement; reporting a displacement there would be
/// indistinguishable from a real match at the API boundary.
pub fn metric_image(
    fixed: &[f32],
    moving: &[f32],
    dims: [usize; 3],
    centre: [usize; 3],
    config: BlockMatchingConfig,
    metric: BlockMetric,
) -> Result<MetricImage> {
    let BlockMetric::NormalizedCrossCorrelation = metric;

    let radius = config.block_radius;
    let search = config.search_radius;
    let extent = [2 * search[0] + 1, 2 * search[1] + 1, 2 * search[2] + 1];

    // Fixed block, mean-subtracted once: it is reused for every candidate.
    let block = gather_block(fixed, dims, centre, radius);
    let block_mean = block.iter().sum::<f64>() / block.len() as f64;
    let fixed_centred: Vec<f64> = block.iter().map(|&v| v - block_mean).collect();
    let fixed_energy: f64 = fixed_centred.iter().map(|v| v * v).sum();
    if fixed_energy <= 0.0 {
        bail!(
            "fixed block at {centre:?} has zero variance; normalized correlation is undefined \
             and any peak would be an artefact of iteration order"
        );
    }
    let fixed_norm = fixed_energy.sqrt();

    let mut values = vec![f64::NEG_INFINITY; extent[0] * extent[1] * extent[2]];
    for (oz, dz) in (-(search[0] as isize)..=search[0] as isize).enumerate() {
        for (oy, dy) in (-(search[1] as isize)..=search[1] as isize).enumerate() {
            for (ox, dx) in (-(search[2] as isize)..=search[2] as isize).enumerate() {
                let shifted = [
                    centre[0] as isize + dz,
                    centre[1] as isize + dy,
                    centre[2] as isize + dx,
                ];
                // A candidate whose block leaves the image is not evaluated:
                // padding it would invent data and bias the correlation toward
                // the padding value.
                let inside = (0..3).all(|axis| {
                    let r = radius[axis] as isize;
                    let extent = dims[axis] as isize;
                    shifted[axis] - r >= 0 && shifted[axis] + r < extent
                });
                if !inside {
                    continue;
                }
                let candidate = gather_block(
                    moving,
                    dims,
                    [
                        shifted[0] as usize,
                        shifted[1] as usize,
                        shifted[2] as usize,
                    ],
                    radius,
                );
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

/// Copy the block centred at `centre` into a flat buffer.
fn gather_block(buf: &[f32], dims: [usize; 3], centre: [usize; 3], radius: [usize; 3]) -> Vec<f64> {
    let mut out =
        Vec::with_capacity((2 * radius[0] + 1) * (2 * radius[1] + 1) * (2 * radius[2] + 1));
    for z in centre[0] - radius[0]..=centre[0] + radius[0] {
        for y in centre[1] - radius[1]..=centre[1] + radius[1] {
            for x in centre[2] - radius[2]..=centre[2] + radius[2] {
                out.push(f64::from(buf[(z * dims[1] + y) * dims[2] + x]));
            }
        }
    }
    out
}
