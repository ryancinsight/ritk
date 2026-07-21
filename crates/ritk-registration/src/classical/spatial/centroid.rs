//! Point-cloud centroid computation and centering.

use leto::{Array2, FixedVector};

type Vector3 = FixedVector<f64, 3>;

/// Compute the centroid of an (N×3) point set.
pub(crate) fn compute_centroid(points: &Array2<f64>) -> Vector3 {
    let shape = points.shape();
    let n = shape[0] as f64;
    let mut sum = Vector3::zeros();
    for i in 0..shape[0] {
        let px = *points.get([i, 0]).expect("valid index");
        let py = *points.get([i, 1]).expect("valid index");
        let pz = *points.get([i, 2]).expect("valid index");
        sum += Vector3::new([px, py, pz]);
    }
    sum / n
}

/// Subtract centroid from every row of an (N×3) point set.
pub(crate) fn center_points(points: &Array2<f64>, centroid: &Vector3) -> Array2<f64> {
    let mut centered = points.clone();
    let shape = centered.shape();
    for i in 0..shape[0] {
        *centered.get_mut([i, 0]).expect("valid index") -= centroid[0];
        *centered.get_mut([i, 1]).expect("valid index") -= centroid[1];
        *centered.get_mut([i, 2]).expect("valid index") -= centroid[2];
    }
    centered
}
