//! Point-cloud centroid computation and centering.

use nalgebra::Vector3;
use ndarray::Array2;

/// Compute the centroid of an (N×3) point set.
pub(crate) fn compute_centroid(points: &Array2<f64>) -> Vector3<f64> {
    let n = points.nrows() as f64;
    let mut sum = Vector3::zeros();
    for i in 0..points.nrows() {
        let row = points.row(i);
        sum += Vector3::new(row[0], row[1], row[2]);
    }
    sum / n
}

/// Subtract centroid from every row of an (N×3) point set.
pub(crate) fn center_points(points: &Array2<f64>, centroid: &Vector3<f64>) -> Array2<f64> {
    let mut centered = points.clone();
    for mut row in centered.rows_mut() {
        row[0] -= centroid[0];
        row[1] -= centroid[1];
        row[2] -= centroid[2];
    }
    centered
}
