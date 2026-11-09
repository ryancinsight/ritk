//! Measurement annotation geometry shared by native overlays.

/// Offset of `distance` perpendicular to the line `p1 → p2`, in `[x, y]`.
///
/// The perpendicular is rotated 90° counter-clockwise from the line direction.
/// A degenerate line returns the straight-up offset `[0, -distance]`.
pub(crate) fn perpendicular_offset(p1: [f32; 2], p2: [f32; 2], distance: f32) -> [f32; 2] {
    let delta = [p2[0] - p1[0], p2[1] - p1[1]];
    let length = delta[0].hypot(delta[1]);
    if length < 1e-4 {
        return [0.0, -distance];
    }
    [-delta[1] / length * distance, delta[0] / length * distance]
}

#[cfg(test)]
mod tests;
