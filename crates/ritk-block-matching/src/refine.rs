//! Turning a peak in the metric image into a displacement.

use super::{BlockDisplacement, MetricImage};

/// How the integer peak is refined to sub-voxel precision.
///
/// Speckle tracking needs far finer resolution than the sampling grid: an
/// elastography strain of 1% over a 20-voxel block is a 0.2-voxel displacement
/// difference, so an integer-only estimate would quantize the entire signal
/// away. All three estimators fit the peak and its two neighbours per axis,
/// independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubpixelRefinement {
    /// Report the integer peak. Honest but coarse; useful when the caller only
    /// needs a coarse displacement or is about to refine by another route.
    None,
    /// Fit a parabola through the peak and its neighbours.
    ///
    /// ```text
    /// δ = ½·(s₋ − s₊) / (s₋ − 2·s₀ + s₊)
    /// ```
    ///
    /// The standard estimator, and what the existing kwavers speckle tracker
    /// uses. It carries a known bias toward integer positions ("peak-locking")
    /// that grows as the correlation peak narrows.
    #[default]
    Parabolic,
    /// Fit a cosine through the peak and its neighbours.
    ///
    /// ```text
    /// ω = acos( (s₋ + s₊) / (2·s₀) ),   δ = −atan( (s₋ − s₊) / (2·s₀·sin ω) ) / ω
    /// ```
    ///
    /// Céspedes et al. show this is the better-matched shape for a correlation
    /// peak from band-limited RF, with less peak-locking bias than the
    /// parabola. It degenerates where the three samples cannot describe a
    /// cosine, and falls back to the parabolic estimate there rather than
    /// producing a NaN.
    Cosine,
}

/// Locate the metric peak and apply `refinement`.
pub(super) fn displacement_from(
    surface: &MetricImage,
    refinement: SubpixelRefinement,
) -> BlockDisplacement {
    let [ez, ey, ex] = surface.extent;
    let mut best = (0_usize, 0_usize, 0_usize);
    let mut best_value = f64::NEG_INFINITY;
    for z in 0..ez {
        for y in 0..ey {
            for x in 0..ex {
                let value = surface.at(z, y, x);
                if value > best_value {
                    best_value = value;
                    best = (z, y, x);
                }
            }
        }
    }

    let peak = [best.0, best.1, best.2];
    let integer = [
        peak[0] as f64 - surface.search_radius[0] as f64,
        peak[1] as f64 - surface.search_radius[1] as f64,
        peak[2] as f64 - surface.search_radius[2] as f64,
    ];

    if matches!(refinement, SubpixelRefinement::None) || !best_value.is_finite() {
        return BlockDisplacement {
            displacement: integer,
            peak_similarity: best_value,
        };
    }

    let mut displacement = integer;
    for axis in 0..3 {
        // A peak on the search boundary has no neighbour on one side, so it
        // cannot be refined. It also means the true displacement may lie
        // outside the search region entirely, which the caller detects from the
        // displacement sitting exactly on the boundary.
        if peak[axis] == 0 || peak[axis] + 1 >= surface.extent[axis] {
            continue;
        }
        let mut lo = peak;
        let mut hi = peak;
        lo[axis] -= 1;
        hi[axis] += 1;
        let s_minus = surface.at(lo[0], lo[1], lo[2]);
        let s_zero = best_value;
        let s_plus = surface.at(hi[0], hi[1], hi[2]);
        if !s_minus.is_finite() || !s_plus.is_finite() {
            continue;
        }
        displacement[axis] += offset(s_minus, s_zero, s_plus, refinement);
    }

    BlockDisplacement {
        displacement,
        peak_similarity: best_value,
    }
}

/// Sub-sample offset in `(-1, 1)` from three consecutive similarity samples.
fn offset(s_minus: f64, s_zero: f64, s_plus: f64, refinement: SubpixelRefinement) -> f64 {
    match refinement {
        SubpixelRefinement::None => 0.0,
        SubpixelRefinement::Parabolic => parabolic(s_minus, s_zero, s_plus),
        SubpixelRefinement::Cosine => {
            cosine(s_minus, s_zero, s_plus).unwrap_or_else(|| parabolic(s_minus, s_zero, s_plus))
        }
    }
}

fn parabolic(s_minus: f64, s_zero: f64, s_plus: f64) -> f64 {
    let denominator = s_minus - 2.0 * s_zero + s_plus;
    // Zero curvature means the three samples are collinear and the parabola has
    // no vertex; the integer peak is the best available answer.
    if denominator.abs() <= f64::EPSILON {
        return 0.0;
    }
    let delta = 0.5 * (s_minus - s_plus) / denominator;
    // The vertex of a parabola through a true local maximum lies within half a
    // sample. Anything beyond that means the peak was not a maximum, so the
    // refinement is discarded rather than moved outside its own bracket.
    if delta.abs() <= 0.5 {
        delta
    } else {
        0.0
    }
}

fn cosine(s_minus: f64, s_zero: f64, s_plus: f64) -> Option<f64> {
    if s_zero.abs() <= f64::EPSILON {
        return None;
    }
    let ratio = (s_minus + s_plus) / (2.0 * s_zero);
    // Outside [-1, 1] the samples do not describe a cosine peak.
    if !(-1.0..=1.0).contains(&ratio) {
        return None;
    }
    let omega = ratio.acos();
    let sin_omega = omega.sin();
    if omega.abs() <= f64::EPSILON || sin_omega.abs() <= f64::EPSILON {
        return None;
    }
    let delta = -((s_minus - s_plus) / (2.0 * s_zero * sin_omega)).atan() / omega;
    if delta.is_finite() && delta.abs() <= 0.5 {
        Some(delta)
    } else {
        None
    }
}
