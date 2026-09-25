use ritk_spatial::Vector;

#[inline]
pub(super) fn normalized_or_axis(vector: Vector<3>, norm: f64, fallback: Vector<3>) -> Vector<3> {
    if norm > 1e-9 {
        vector / norm
    } else {
        fallback
    }
}
