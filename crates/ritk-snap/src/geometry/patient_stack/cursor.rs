//! Continuous patient cursor coordinates in the original acquisition planes.
use super::{PatientStack, Vector3};

#[derive(Clone, Copy, Debug)]
pub(crate) struct CursorProjection {
    pub(crate) index: usize,
    pub(crate) pixel: [f64; 2],
    pub(crate) distant: bool,
}

impl PatientStack {
    /// DICOM IPP denotes the first pixel center, at edge coordinates (½, ½).
    pub(crate) fn patient_point(&self, index: usize, [x, y]: [f64; 2]) -> Option<Vector3> {
        let [dy, dx] = *self.pixel_spacing.get(index)?;
        if !self.contains([x, y]) {
            return None;
        }
        Some(
            self.centers[index]
                + self.column * ((x - self.half_extent[0]) * dx)
                + self.row * ((y - self.half_extent[1]) * dy),
        )
    }

    fn contains(&self, pixel: [f64; 2]) -> bool {
        pixel
            .into_iter()
            .zip(self.half_extent)
            .all(|(v, h)| v >= 0.0 && v < 2.0 * h)
    }

    /// Orthogonal projection solves the Gram system for the two stored IOP
    /// directions, which may be rounded and need not be exactly orthonormal.
    /// Nearest-plane ties retain the lower displayed index. Outside a sampled
    /// slab we mark the nearest image red, including singleton acquisitions.
    pub(crate) fn project_cursor(&self, point: Vector3, target: &Self) -> Option<CursorProjection> {
        if self.frame != target.frame {
            return None;
        }
        let distance = (point - target.centers[0]).dot(&target.normal);
        if !distance.is_finite() {
            return None;
        }
        let index = target.closest(distance);
        let delta = point - target.centers[index];
        let (a, b) = (target.column, target.row);
        let (aa, ab, bb) = (a.dot(&a), a.dot(&b), b.dot(&b));
        let (x, y) = (a.dot(&delta), b.dot(&delta));
        let det = aa * bb - ab * ab;
        let [dy, dx] = target.pixel_spacing[index];
        let pixel = [
            target.half_extent[0] + (bb * x - ab * y) / (det * dx),
            target.half_extent[1] + (aa * y - ab * x) / (det * dy),
        ];
        if !target.contains(pixel) {
            return None;
        }
        let residual = (distance - target.distances[index]).abs();
        let distant = target.half_thickness[index].map_or_else(
            || {
                if target.distances.len() == 1 {
                    residual > 0.0
                } else {
                    target.nearest(distance).is_none()
                }
            },
            |half| residual > half,
        );
        Some(CursorProjection {
            index,
            pixel,
            distant,
        })
    }
}
