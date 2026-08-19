//! Single-point coordinate transforms between index and physical space.
//!
//! Both directions dispatch on the image's [`CoordinateMap`], so a beam-space
//! acquisition maps through its own geometry rather than the Cartesian
//! formula. Coordinates here are indexed by spatial axis; the batch forms in
//! [`super::batch`] apply the same maps to whole tensors with innermost-first
//! columns.

use anyhow::anyhow;
use coeus_core::{ComputeBackend, Scalar};
use ritk_spatial::{CoordinateMap, Point};

use crate::types::Image;

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
{
    /// Convert a physical-space point to a continuous image index.
    ///
    /// For [`CoordinateMap::Cartesian`] the mapping is `S^-1 D^-1 (point -
    /// origin)`, where `D` is the direction cosine matrix and `S` is the
    /// diagonal spacing matrix.
    ///
    /// Point coordinates here are indexed by spatial axis, unlike the
    /// innermost-first columns of the batch
    /// [`Self::world_to_index_native_on`]; both apply the same coordinate map.
    ///
    /// # Errors
    ///
    /// Returns an error when the direction matrix is singular, or when a
    /// non-Cartesian map is set and the point lies outside the acquisition
    /// (where it denotes no index at all — the batch form emits NaN there,
    /// which this single-point form reports as an error instead).
    pub fn physical_point_to_continuous_index(&self, point: &Point<D>) -> anyhow::Result<Point<D>> {
        if let CoordinateMap::CurvilinearArray(geometry) = &self.map {
            let (sample, beam) = geometry
                .index_from_cartesian(point[D - 1], point[D - 2])
                .ok_or_else(|| {
                    anyhow!(
                        "physical point ({}, {}) lies outside the curvilinear acquisition",
                        point[D - 1],
                        point[D - 2]
                    )
                })?;
            let mut index = Point::origin();
            index[D - 1] = sample;
            index[D - 2] = beam;
            return Ok(index);
        }
        if let CoordinateMap::PhasedArray3D(geometry) = &self.map {
            // World → probe frame: Direction^-1 · (world - origin).
            let inverse = self
                .direction
                .try_inverse()
                .ok_or_else(|| anyhow!("image direction matrix is singular"))?;
            let probe = inverse * (*point - self.origin);
            // probe axis order: [depth, elevation, azimuth] = [0, 1, 2]
            let (azimuth_index, elevation_index, sample) = geometry
                .index_from_cartesian(probe[2], probe[1], probe[0])
                .ok_or_else(|| {
                    anyhow!(
                        "physical point ({}, {}, {}) lies outside the phased-array acquisition",
                        point[D - 1],
                        point[D - 2],
                        point[D - 3]
                    )
                })?;
            let mut index = Point::origin();
            index[D - 1] = azimuth_index;
            index[D - 2] = elevation_index;
            index[D - 3] = sample;
            return Ok(index);
        }
        if let CoordinateMap::SliceSeries(sweep) = &self.map {
            let world = [point[D - 1], point[D - 2], point[D - 3]];
            let idx = sweep.index_from_world(world).ok_or_else(|| {
                anyhow!(
                    "physical point ({}, {}, {}) lies outside the slice-series sweep",
                    point[D - 1],
                    point[D - 2],
                    point[D - 3]
                )
            })?;
            let mut index = Point::origin();
            index[D - 1] = idx[0];
            index[D - 2] = idx[1];
            index[D - 3] = idx[2];
            return Ok(index);
        }
        let inverse = self
            .direction
            .try_inverse()
            .ok_or_else(|| anyhow!("image direction matrix is singular"))?;
        let rotated = inverse * (*point - self.origin);
        let mut index = Point::origin();
        for axis in 0..D {
            index[axis] = rotated[axis] / self.spacing[axis];
        }
        Ok(index)
    }

    /// Convert a continuous image index to a physical-space point.
    ///
    /// For [`CoordinateMap::Cartesian`] the mapping is `origin + D S index`.
    /// Index coordinates here are indexed by spatial axis, unlike the
    /// innermost-first columns of [`Self::index_to_world_native_on`]; both
    /// apply the same coordinate map.
    #[must_use]
    pub fn continuous_index_to_physical_point(&self, index: &Point<D>) -> Point<D> {
        if let CoordinateMap::CurvilinearArray(geometry) = &self.map {
            let (radius, angle) = geometry.polar_from_index(index[D - 1], index[D - 2]);
            let mut point = Point::origin();
            point[D - 1] = radius * angle.sin();
            point[D - 2] = radius * angle.cos();
            return point;
        }
        if let CoordinateMap::PhasedArray3D(geometry) = &self.map {
            // Probe frame → world: origin + Direction · probe_point.
            let mut point = Point::origin();
            if let Some((azimuth_axis, elevation_axis, depth)) =
                geometry.cartesian_from_index(index[D - 1], index[D - 2], index[D - 3])
            {
                // probe axis order: [depth=0, elevation=1, azimuth=2]
                let probe = [depth, elevation_axis, azimuth_axis];
                let d = self.direction;
                for c in 0..D {
                    let mut acc = self.origin[c];
                    for r in 0..3 {
                        acc += d[(c, r)] * probe[r];
                    }
                    point[c] = acc;
                }
            } else {
                for axis in 0..D {
                    point[axis] = f64::NAN;
                }
            }
            return point;
        }
        if let CoordinateMap::SliceSeries(sweep) = &self.map {
            let world = sweep.world_from_index(index[D - 1], index[D - 2], index[D - 3]);
            let mut point = Point::origin();
            point[D - 1] = world[0];
            point[D - 2] = world[1];
            point[D - 3] = world[2];
            return point;
        }
        let mut scaled = ritk_spatial::Vector::zeros();
        for axis in 0..D {
            scaled[axis] = index[axis] * self.spacing[axis];
        }
        self.origin + self.direction * scaled
    }
}
