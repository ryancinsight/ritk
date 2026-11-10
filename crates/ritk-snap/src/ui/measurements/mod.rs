//! Measurement annotation geometry.

/// Offset of `distance` perpendicular to the line `p1 → p2`, in `[x, y]`.
///
/// The perpendicular is the line direction rotated 90° counter-clockwise in
/// `[x, y]` order: `(-dy, dx) / |d| · distance`. A zero-length or non-finite
/// line has no direction and returns the straight-up offset `[0, -distance]`.
pub(crate) fn perpendicular_offset(p1: [f64; 2], p2: [f64; 2], distance: f64) -> [f64; 2] {
    let delta = [p2[0] - p1[0], p2[1] - p1[1]];
    let length = delta[0].hypot(delta[1]);
    if length > 0.0 && length.is_finite() {
        let scale = distance / length;
        [-delta[1] * scale, delta[0] * scale]
    } else {
        [0.0, -distance]
    }
}

#[cfg(test)]
mod tests;
