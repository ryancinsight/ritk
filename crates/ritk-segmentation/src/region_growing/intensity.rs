#[inline]
pub(super) fn within_finite_bounds(value: f32, lower: f32, upper: f32) -> bool {
    value.is_finite() && value >= lower && value <= upper
}
