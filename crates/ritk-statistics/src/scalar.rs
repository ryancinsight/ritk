//! Shared scalar helpers for the statistics kernels.

/// Sort a mutable slice of `f32` values using total ordering (NaN sorted last).
#[inline]
pub(crate) fn sort_floats(values: &mut [f32]) {
    values.sort_by(f32::total_cmp);
}

/// Binary-mask foreground threshold: voxels with mask value strictly above
/// this threshold are treated as foreground; those at or below are background.
pub(crate) const FOREGROUND_THRESHOLD: f32 = 0.5;
