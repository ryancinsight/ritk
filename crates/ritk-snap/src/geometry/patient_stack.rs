//! Physical planes of a classic DICOM acquisition in displayed slice order.

use crate::loaded_volume::{LoadedVolume, READER_ORIENTATION_DIRECTION_TOLERANCE};
use ritk_spatial::Vector3;

/// Automatic linking uses the inclusive five-degree limit of RadiAnt's
/// automatic synchronization mode. Opposite normal directions describe
/// the same plane orientation.
const MAX_ANGLE: f64 = 5.0_f64.to_radians();

#[derive(Debug)]
pub(crate) struct PatientStack {
    study: Box<str>,
    frame: Box<str>,
    normal: Vector3,
    centers: Vec<Vector3>,
    distances: Vec<f64>,
}

impl PatientStack {
    pub(crate) fn from_volume(volume: &LoadedVolume) -> Option<Self> {
        let meta = volume.metadata.as_ref()?;
        let (study, frame) = meta.common_spatial_reference()?;
        let [depth, rows, cols] = volume.shape;
        if depth < 2
            || depth != meta.slices.len()
            || rows == 0
            || cols == 0
            || meta.enhanced_pet_selection.is_some()
        {
            return None;
        }
        let orientation = meta.slices[0].image_orientation_patient?;
        let col = Vector3::new([orientation[0], orientation[1], orientation[2]]);
        let row = Vector3::new([orientation[3], orientation[4], orientation[5]]);
        // Use the reader's direction-cosine tolerance for rounded DICOM DS values.
        if (col.norm() - 1.0).abs() > READER_ORIENTATION_DIRECTION_TOLERANCE
            || (row.norm() - 1.0).abs() > READER_ORIENTATION_DIRECTION_TOLERANCE
            || col.dot(&row).abs() > READER_ORIENTATION_DIRECTION_TOLERANCE
        {
            return None;
        }
        let mut normal = col.cross(&row).normalized()?;
        let half_rows = f64::from(u32::try_from(rows - 1).ok()?) / 2.0;
        let half_cols = f64::from(u32::try_from(cols - 1).ok()?) / 2.0;
        let mut centers = Vec::with_capacity(depth);
        for slice in &meta.slices {
            if slice.image_orientation_patient? != orientation
                || slice.gantry_tilt_degrees().is_some_and(|v| v != 0.0)
            {
                return None;
            }
            let [dy, dx] = slice.pixel_spacing_mm()?;
            if ![dy, dx].iter().all(|v| v.is_finite() && *v > 0.0) {
                return None;
            }
            let center = Vector3::new(slice.image_position_mm()?)
                + col * (dx * half_cols)
                + row * (dy * half_rows);
            if !center.as_slice().iter().all(|v| v.is_finite()) {
                return None;
            }
            centers.push(center);
        }
        // Orient the search coordinate along displayed indices, including reversed stacks.
        if (centers[1] - centers[0]).dot(&normal) < 0.0 {
            normal = -normal;
        }
        let distances: Vec<_> = centers
            .iter()
            .map(|c| (*c - centers[0]).dot(&normal))
            .collect();
        if !distances
            .windows(2)
            .all(|p| p[1].is_finite() && p[1] > p[0])
        {
            return None;
        }
        Some(Self {
            study: study.into(),
            frame: frame.into(),
            normal,
            centers,
            distances,
        })
    }

    pub(crate) fn corresponding_slice(&self, index: usize, target: &Self) -> Option<usize> {
        if self.study != target.study || self.frame != target.frame {
            return None;
        }
        let angle = self
            .normal
            .cross(&target.normal)
            .norm()
            .atan2(self.normal.dot(&target.normal).abs());
        if angle > MAX_ANGLE {
            return None;
        }
        let position = (*self.centers.get(index)? - target.centers[0]).dot(&target.normal);
        target.nearest(position)
    }

    /// Each slice owns the interval between adjacent midplanes. At equal
    /// distance the lower displayed index wins. Do not extrapolate beyond
    /// the outer half gap at either end of the acquisition.
    fn nearest(&self, position: f64) -> Option<usize> {
        let distances = &self.distances;
        let last = distances.len() - 1;
        if !position.is_finite()
            || position < -distances[1] / 2.0
            || position - distances[last] > (distances[last] - distances[last - 1]) / 2.0
        {
            return None;
        }
        let upper = distances.partition_point(|&d| d < position);
        if upper == 0 {
            return Some(0);
        }
        if upper == distances.len()
            || position - distances[upper - 1] <= distances[upper] - position
        {
            Some(upper - 1)
        } else {
            Some(upper)
        }
    }
}

#[cfg(test)]
mod tests;
