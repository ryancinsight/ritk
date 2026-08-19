//! Strain-window filtering: removing peak-hopping artefacts from a field.
//!
//! Block matching reports the maximum of a correlation surface, and a
//! decorrelated block still has a maximum. When speckle decorrelates — through
//! out-of-plane motion, shadowing, or simply too small a kernel — the reported
//! peak can jump to a neighbouring correlation lobe. That is *peak hopping*:
//! the displacement is not slightly wrong, it is wrong by roughly a wavelength.
//!
//! Strain is the spatial derivative of displacement, so a single hopped block
//! produces two large spurious strain values, one on each side. Elastography
//! reads exactly that derivative, which is why an unfiltered field is unusable
//! even when most of its blocks are correct.
//!
//! # Method
//!
//! A block whose local axial strain exceeds a physically plausible bound is
//! marked unreliable, and its displacement is replaced by linear interpolation
//! between the nearest reliable blocks above and below it on the same axial
//! line, or by the nearest reliable value where only one side exists. Because
//! removing one outlier changes its neighbours' strain, the pass repeats until
//! nothing is replaced or the iteration budget is spent.
//!
//! Blocks whose fixed window had no variance arrive carrying a non-finite
//! similarity from [`super::track_volume`]. They are unreliable by
//! construction, since a constant block correlates equally with everything.
//!
//! # What this cannot do
//!
//! The bound cannot distinguish a peak-hop from genuinely large strain. A real
//! deformation exceeding `max_abs_strain` is rejected exactly like an artefact,
//! so the bound must sit above the largest strain the study expects: it is a
//! plausibility limit, not a denoiser. That is inherent to the method, and is
//! why the filter reports how many blocks it replaced instead of quietly
//! returning a smoother field.
//!
//! # References
//! - `itkBlockMatchingStrainWindowDisplacementCalculator.h`,
//!   KitwareMedical/ITKUltrasound — the strain-bound criterion, the
//!   interpolate-or-extrapolate replacement, and the iteration budget.

use anyhow::{bail, Result};

use super::DisplacementField;

/// Tuning for [`strain_window_filter`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrainWindowParams {
    /// Largest absolute axial strain (voxel/voxel) considered physical.
    ///
    /// Soft-tissue elastography works in the low percent range, so a bound near
    /// `0.1` admits real deformation while rejecting the wavelength-scale jumps
    /// that peak hopping produces.
    pub max_abs_strain: f64,
    /// Maximum passes. Removing one outlier changes its neighbours' strain, so
    /// a single pass can leave a newly exposed artefact behind.
    pub max_iterations: usize,
}

impl Default for StrainWindowParams {
    fn default() -> Self {
        Self {
            max_abs_strain: 0.1,
            max_iterations: 3,
        }
    }
}

impl StrainWindowParams {
    /// Validate the parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when `max_abs_strain` is not finite and positive. A
    /// non-positive bound would reject every block, correct ones included.
    pub fn validate(&self) -> Result<()> {
        if !self.max_abs_strain.is_finite() || self.max_abs_strain <= 0.0 {
            bail!(
                "max_abs_strain must be finite and positive, got {}",
                self.max_abs_strain
            );
        }
        Ok(())
    }
}

/// What a filtering run did.
#[derive(Debug, Clone, PartialEq)]
pub struct StrainWindowReport {
    /// The filtered field.
    pub field: DisplacementField,
    /// Number of block replacements performed, summed over passes.
    pub replaced: usize,
    /// Passes actually run; fewer than the budget means it converged.
    pub iterations: usize,
    /// Blocks still implausible at the end because their axial line offered no
    /// reliable neighbour. Their displacement is left untouched rather than
    /// invented, so a caller can exclude them.
    pub unrecoverable: Vec<usize>,
}

/// Replace peak-hopped displacements using a strain plausibility bound.
///
/// `axial_stride` is the block-centre spacing along the axial axis, the same
/// value passed to [`crate::strain_from_displacement`].
///
/// # Errors
///
/// Returns an error when the parameters are invalid, or when `axial_stride` is
/// zero — it divides the strain estimate.
pub fn strain_window_filter(
    field: &DisplacementField,
    axial_stride: usize,
    params: StrainWindowParams,
) -> Result<StrainWindowReport> {
    params.validate()?;
    if axial_stride == 0 {
        bail!("axial_stride must be positive; it divides the strain estimate");
    }

    let mut field = field.clone();
    let mut replaced = 0usize;
    let mut iterations = 0usize;
    // A block whose value has already been substituted is not a candidate
    // again: its displacement is now an interpolant, so re-testing it against
    // the strain bound would either loop forever (a constant block keeps its
    // non-finite similarity) or re-derive the same answer.
    let mut recovered = vec![false; field.len()];

    // Strain is an axial derivative, so both the plausibility test and the
    // replacement run along axial lines.
    let lines = axial_lines(&field);

    for _ in 0..params.max_iterations {
        iterations += 1;
        let reliable = reliability(&field, &lines, axial_stride, params.max_abs_strain);

        let mut changed = 0usize;
        for line in &lines {
            for (position, &index) in line.iter().enumerate() {
                if reliable[index] || recovered[index] {
                    continue;
                }
                // Donors are measured blocks only. An interpolated value never
                // becomes the basis for another interpolation, which would let
                // one artefact spread along the line.
                if let Some(value) = replacement(line, position, index, &reliable, &field) {
                    field.displacements[index] = value;
                    recovered[index] = true;
                    changed += 1;
                }
            }
        }

        replaced += changed;
        if changed == 0 {
            break;
        }
    }

    let reliable = reliability(&field, &lines, axial_stride, params.max_abs_strain);
    let unrecoverable = (0..field.len())
        .filter(|&i| !reliable[i] && !recovered[i])
        .collect();

    Ok(StrainWindowReport {
        field,
        replaced,
        iterations,
        unrecoverable,
    })
}

/// Per-block reliability: a finite correlation peak, and a plausible gradient
/// to each immediate axial neighbour.
///
/// The gradient is deliberately one-sided on *both* sides rather than the
/// central difference [`crate::strain_from_displacement`] reports. A central
/// difference at the hopped block skips that block entirely — it differences
/// its two neighbours — so a single-block spike cancels exactly where it is
/// largest, and the artefact reads as plausible while its neighbours take the
/// blame. Requiring both one-sided gradients to be plausible flags the block
/// that jumped along with the two it corrupted, which is the honest suspect
/// set: all three displacements depend on the bad peak.
fn reliability(
    field: &DisplacementField,
    lines: &[Vec<usize>],
    axial_stride: usize,
    max_abs_strain: f64,
) -> Vec<bool> {
    let stride = axial_stride as f64;
    // A non-finite similarity marks a zero-variance window, which correlates
    // equally with everything and carries no displacement information.
    let mut ok: Vec<bool> = field
        .peak_similarities
        .iter()
        .map(|s| s.is_finite())
        .collect();

    // Ordered explicitly rather than by a negated `<=`: a non-finite gradient —
    // from a NaN displacement or a NaN neighbour — has no ordering against the
    // bound at all, and must read as implausible rather than silently passing.
    let implausible = |delta: f64| {
        (delta / stride)
            .abs()
            .partial_cmp(&max_abs_strain)
            .is_none_or(std::cmp::Ordering::is_gt)
    };

    for line in lines {
        for (pos, &i) in line.iter().enumerate() {
            let here = field.displacements[i][0];
            let below = pos
                .checked_sub(1)
                .is_some_and(|p| implausible(here - field.displacements[line[p]][0]));
            let above = line
                .get(pos + 1)
                .is_some_and(|&j| implausible(field.displacements[j][0] - here));
            if below || above || !here.is_finite() {
                ok[i] = false;
            }
        }
    }
    ok
}

/// Centre indices grouped into axial lines, each ordered by increasing depth.
fn axial_lines(field: &DisplacementField) -> Vec<Vec<usize>> {
    let mut indexed: Vec<(usize, usize, usize, usize)> = field
        .centres
        .iter()
        .enumerate()
        .map(|(i, &[z, y, x])| (y, x, z, i))
        .collect();
    indexed.sort_unstable();

    let mut lines = Vec::new();
    let mut start = 0;
    while start < indexed.len() {
        let (y0, x0, _, _) = indexed[start];
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].0 == y0 && indexed[end].1 == x0 {
            end += 1;
        }
        lines.push(indexed[start..end].iter().map(|&(_, _, _, i)| i).collect());
        start = end;
    }
    lines
}

/// Displacement to substitute at `position` on `line`.
///
/// Linear interpolation between the nearest reliable blocks on either side; the
/// nearest reliable value where only one side exists; `None` when the line
/// carries no reliable block at all, which is reported rather than invented.
fn replacement(
    line: &[usize],
    position: usize,
    index: usize,
    reliable: &[bool],
    field: &DisplacementField,
) -> Option<[f64; 3]> {
    let below = line[..position].iter().rposition(|&i| reliable[i]);
    let above = line[position + 1..].iter().position(|&i| reliable[i]);

    match (below, above) {
        (Some(b), Some(a)) => {
            let lo = line[b];
            let hi = line[position + 1 + a];
            // Interpolate against block-centre depth, so an uneven grid stays
            // correct rather than assuming a constant stride.
            let z_lo = field.centres[lo][0] as f64;
            let z_hi = field.centres[hi][0] as f64;
            let z = field.centres[index][0] as f64;
            let span = z_hi - z_lo;
            if span.abs() <= f64::EPSILON {
                return Some(field.displacements[lo]);
            }
            let t = (z - z_lo) / span;
            let (lo_d, hi_d) = (field.displacements[lo], field.displacements[hi]);
            Some(std::array::from_fn(|axis| {
                lo_d[axis] + t * (hi_d[axis] - lo_d[axis])
            }))
        }
        (Some(b), None) => Some(field.displacements[line[b]]),
        (None, Some(a)) => Some(field.displacements[line[position + 1 + a]]),
        (None, None) => None,
    }
}

#[cfg(test)]
#[path = "tests_regularize.rs"]
mod tests;
