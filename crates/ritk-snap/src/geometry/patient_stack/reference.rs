//! Intersection of a patient plane with another image plane.
use super::PatientStack;

/// Signed distance to the source plane at the target center, plus its change
/// per target column and row. Its zero set is the cross-reference line.
pub(crate) struct PlaneTrace {
    center: f64,
    gradient: [f64; 2],
}

impl PatientStack {
    pub(crate) fn trace_on(
        &self,
        index: usize,
        target: &Self,
        target_index: usize,
    ) -> Option<PlaneTrace> {
        // A pair of manual slice anchors does not register two patient spaces.
        if self.frame != target.frame || self.normal.cross(&target.normal).norm() == 0.0 {
            return None;
        }
        let [dy, dx] = *target.pixel_spacing.get(target_index)?;
        Some(PlaneTrace {
            center: (*target.centers.get(target_index)? - *self.centers.get(index)?)
                .dot(&self.normal),
            gradient: [
                self.normal.dot(&target.column) * dx,
                self.normal.dot(&target.row) * dy,
            ],
        })
    }
}

impl PlaneTrace {
    pub(crate) fn distance(&self, centered_pixel: [f64; 2]) -> f64 {
        self.center + self.gradient[0] * centered_pixel[0] + self.gradient[1] * centered_pixel[1]
    }

    /// An affine signed distance varies linearly on each edge. Two boundary
    /// crossings define the visible segment. Count each zero corner once.
    pub(crate) fn clip(corners: [[f64; 2]; 4], distances: [f64; 4]) -> Option<[[f64; 2]; 2]> {
        if !distances.iter().all(|d| d.is_finite()) || distances == [0.; 4] {
            return None;
        }
        let mut first = None;
        for i in 0..4 {
            let j = (i + 1) % 4;
            let a = distances[i];
            let b = distances[j];
            let point = if a == 0.0 {
                corners[i]
            } else if b != 0.0 && a.is_sign_positive() != b.is_sign_positive() {
                // Scale first so opposite finite distances cannot overflow on addition.
                let scale = a.abs().max(b.abs());
                let t = (a.abs() / scale) / ((a.abs() / scale) + (b.abs() / scale));
                std::array::from_fn(|k| corners[i][k] + t * (corners[j][k] - corners[i][k]))
            } else {
                continue;
            };
            if let Some(start) = first {
                return Some([start, point]);
            }
            first = Some(point);
        }
        None
    }
}
