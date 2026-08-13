//! Acquisition coordinate maps: how an image's index space relates to physical
//! space.
//!
//! A Cartesian image maps index to physical point by the affine
//! `origin + Direction · (index ⊙ spacing)`. Ultrasound acquisitions do not: a
//! convex (curvilinear) array samples along beams that fan out from an apex, so
//! its index space is `(sample along beam, beam number)` and the map into
//! physical space is polar.
//!
//! Carrying the map on the image — rather than converting to a Cartesian raster
//! first — is what lets every resampler, filter and registration method operate
//! on beam-space data unchanged; scan conversion then becomes an ordinary
//! resample onto a Cartesian grid rather than a bespoke step.
//!
//! The variant set is closed and fixed by acquisition physics, so it is an
//! exhaustively matched enum rather than a trait object (atlas ADR 0042, and
//! ADR 0041 for the dispatch precedent). Callers match once at an operation
//! boundary and then run a monomorphic per-voxel loop; a per-voxel match would
//! defeat the point.
//!
//! # References
//! - `itkCurvilinearArraySpecialCoordinatesImage.h`, KitwareMedical/ITKUltrasound
//!   (`TransformContinuousIndexToPhysicalPoint` and its inverse) — the formulas
//!   and the lateral-centering convention below are taken from that source.

use anyhow::{bail, Result};

/// Convex (curvilinear) array acquisition geometry.
///
/// Beams fan out from an apex at equal angular steps. Index column 0 (the
/// innermost/fastest axis) is the sample along a beam; index column 1 is the
/// beam number.
///
/// # Mathematical specification
///
/// With `maxLateral = lateral_count − 1`, for index `(s, b)`:
///
/// ```text
/// r = s · radius_sample_size + first_sample_distance
/// θ = (b − maxLateral/2) · lateral_angular_separation
/// ```
///
/// The beams are centered on `θ = 0`, so the fan is symmetric about the axial
/// axis regardless of beam count. The physical pair is `(r·sin θ, r·cos θ)`,
/// placed on the two innermost spatial axes (see
/// [`crate::types::Image::index_to_world_native_on`] for the column
/// conventions). Any further axes use the ordinary affine row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvilinearArray {
    radius_sample_size: f64,
    first_sample_distance: f64,
    lateral_angular_separation: f64,
}

impl CurvilinearArray {
    /// Construct a curvilinear geometry.
    ///
    /// `radius_sample_size` is the range increment per sample along a beam and
    /// `first_sample_distance` the apex-to-first-sample radius, both in the
    /// image's physical length unit; `lateral_angular_separation` is the angle
    /// between adjacent beams in radians.
    ///
    /// # Errors
    ///
    /// Returns an error when any parameter is non-finite, when
    /// `radius_sample_size` or `lateral_angular_separation` is not strictly
    /// positive (either would collapse the map and make it non-invertible), or
    /// when `first_sample_distance` is negative (the apex offset is a radius).
    pub fn try_new(
        radius_sample_size: f64,
        first_sample_distance: f64,
        lateral_angular_separation: f64,
    ) -> Result<Self> {
        if !radius_sample_size.is_finite()
            || !first_sample_distance.is_finite()
            || !lateral_angular_separation.is_finite()
        {
            bail!(
                "curvilinear geometry parameters must be finite: \
                 radius_sample_size={radius_sample_size}, \
                 first_sample_distance={first_sample_distance}, \
                 lateral_angular_separation={lateral_angular_separation}"
            );
        }
        if radius_sample_size <= 0.0 {
            bail!("curvilinear radius_sample_size must be > 0, got {radius_sample_size}");
        }
        if lateral_angular_separation <= 0.0 {
            bail!(
                "curvilinear lateral_angular_separation must be > 0, \
                 got {lateral_angular_separation}"
            );
        }
        if first_sample_distance < 0.0 {
            bail!("curvilinear first_sample_distance must be >= 0, got {first_sample_distance}");
        }
        Ok(Self {
            radius_sample_size,
            first_sample_distance,
            lateral_angular_separation,
        })
    }

    /// Range increment per sample along a beam.
    #[inline]
    #[must_use]
    pub fn radius_sample_size(&self) -> f64 {
        self.radius_sample_size
    }

    /// Apex-to-first-sample radius.
    #[inline]
    #[must_use]
    pub fn first_sample_distance(&self) -> f64 {
        self.first_sample_distance
    }

    /// Angle between adjacent beams, in radians.
    #[inline]
    #[must_use]
    pub fn lateral_angular_separation(&self) -> f64 {
        self.lateral_angular_separation
    }

    /// Polar `(radius, angle)` for a `(sample, beam)` index pair.
    ///
    /// `lateral_count` is the number of beams, which centers the fan.
    #[inline]
    #[must_use]
    pub fn polar_from_index(&self, sample: f64, beam: f64, lateral_count: usize) -> (f64, f64) {
        let max_lateral = lateral_count.saturating_sub(1) as f64;
        let radius = sample.mul_add(self.radius_sample_size, self.first_sample_distance);
        let angle = (beam - max_lateral / 2.0) * self.lateral_angular_separation;
        (radius, angle)
    }

    /// `(sample, beam)` index pair for a physical `(lateral, axial)` pair.
    ///
    /// Returns `None` outside the acquisition half-plane (`axial <= 0`), where
    /// the fan is not defined and the angle would be ambiguous. ITK returns
    /// `+π/2` there regardless of the lateral sign; rejecting is preferred to
    /// producing an index that silently denotes the wrong beam.
    #[inline]
    #[must_use]
    pub fn index_from_cartesian(
        &self,
        lateral: f64,
        axial: f64,
        lateral_count: usize,
    ) -> Option<(f64, f64)> {
        // Non-finite is rejected explicitly: `axial <= 0.0` alone is false for
        // NaN, so a NaN would otherwise pass into the polar inverse.
        if !axial.is_finite() || !lateral.is_finite() || axial <= 0.0 {
            return None;
        }
        let max_lateral = lateral_count.saturating_sub(1) as f64;
        let radius = lateral.hypot(axial);
        let angle = (lateral / axial).atan();
        let sample = (radius - self.first_sample_distance) / self.radius_sample_size;
        let beam = angle / self.lateral_angular_separation + max_lateral / 2.0;
        Some((sample, beam))
    }
}

/// How an image's index space maps into physical space.
///
/// See the module documentation for why this is a closed enum.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum CoordinateMap {
    /// Affine `origin + Direction · (index ⊙ spacing)` — the ordinary raster.
    #[default]
    Cartesian,
    /// Convex/curvilinear array beam space.
    CurvilinearArray(CurvilinearArray),
}

impl CoordinateMap {
    /// Whether this map is the plain Cartesian affine.
    #[inline]
    #[must_use]
    pub fn is_cartesian(&self) -> bool {
        matches!(self, Self::Cartesian)
    }

    /// Validate that this map is meaningful for a `D`-dimensional image.
    ///
    /// # Errors
    ///
    /// Returns an error when a non-Cartesian map is attached to an image with
    /// fewer than two dimensions — every acquisition geometry here needs a
    /// lateral axis in addition to the axis along the beam.
    pub fn validate_dimensionality(&self, d: usize) -> Result<()> {
        match self {
            Self::Cartesian => Ok(()),
            Self::CurvilinearArray(_) => {
                if d < 2 {
                    bail!("curvilinear coordinate map requires a 2-D or higher image, got {d}-D");
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> CurvilinearArray {
        // 0.1 mm range sampling, 60 mm apex offset, 0.5 degree beam pitch.
        CurvilinearArray::try_new(1.0e-4, 0.06, 0.5_f64.to_radians()).expect("valid geometry")
    }

    #[test]
    fn rejects_non_positive_and_non_finite_parameters() {
        assert!(CurvilinearArray::try_new(0.0, 0.06, 0.01).is_err());
        assert!(CurvilinearArray::try_new(-1.0e-4, 0.06, 0.01).is_err());
        assert!(CurvilinearArray::try_new(1.0e-4, -0.01, 0.01).is_err());
        assert!(CurvilinearArray::try_new(1.0e-4, 0.06, 0.0).is_err());
        assert!(CurvilinearArray::try_new(f64::NAN, 0.06, 0.01).is_err());
        assert!(CurvilinearArray::try_new(1.0e-4, f64::INFINITY, 0.01).is_err());
    }

    #[test]
    fn fan_is_centred_on_the_axial_axis() {
        let g = geometry();
        let lateral_count = 129;
        // The middle beam of an odd-count fan sits exactly on theta = 0.
        let (_, angle) = g.polar_from_index(0.0, 64.0, lateral_count);
        assert!(
            angle.abs() < 1.0e-15,
            "centre beam must be axial, got {angle}"
        );
        // Outermost beams are symmetric about it.
        let (_, first) = g.polar_from_index(0.0, 0.0, lateral_count);
        let (_, last) = g.polar_from_index(0.0, 128.0, lateral_count);
        assert!(
            (first + last).abs() < 1.0e-15,
            "fan must be symmetric: {first} vs {last}"
        );
    }

    #[test]
    fn radius_starts_at_the_apex_offset() {
        let g = geometry();
        let (radius, _) = g.polar_from_index(0.0, 0.0, 65);
        assert!((radius - 0.06).abs() < 1.0e-15, "got {radius}");
        let (radius, _) = g.polar_from_index(100.0, 0.0, 65);
        assert!(
            (radius - (0.06 + 100.0 * 1.0e-4)).abs() < 1.0e-15,
            "got {radius}"
        );
    }

    /// Round-trip over the whole fan: index -> Cartesian -> index.
    ///
    /// Tolerance is set by the conditioning of the polar inverse rather than
    /// bare machine epsilon: `hypot`/`atan` each contribute O(eps) relative
    /// error, and dividing by `radius_sample_size = 1e-4` amplifies the radius
    /// error by 1e4 when converting back to a sample index. With
    /// `radius <= 0.08 m`, the absolute sample error is bounded by roughly
    /// `0.08 * 4 * eps / 1e-4 ~ 7e-12`; 1e-9 is a comfortable bound on that.
    #[test]
    fn index_round_trips_through_cartesian_over_the_fan() {
        let g = geometry();
        let lateral_count = 129;
        for beam_i in 0..lateral_count {
            for sample_i in [0_usize, 1, 37, 128, 199] {
                let sample = sample_i as f64;
                let beam = beam_i as f64;
                let (radius, angle) = g.polar_from_index(sample, beam, lateral_count);
                let (lateral, axial) = (radius * angle.sin(), radius * angle.cos());
                let (s, b) = g
                    .index_from_cartesian(lateral, axial, lateral_count)
                    .expect("fan point must invert");
                assert!(
                    (s - sample).abs() < 1.0e-9,
                    "sample {sample} -> {s} (beam {beam})"
                );
                assert!((b - beam).abs() < 1.0e-9, "beam {beam} -> {b}");
            }
        }
    }

    #[test]
    fn points_behind_the_apex_plane_are_rejected() {
        let g = geometry();
        assert!(g.index_from_cartesian(0.01, 0.0, 65).is_none());
        assert!(g.index_from_cartesian(0.01, -0.05, 65).is_none());
        assert!(g.index_from_cartesian(0.0, 0.07, 65).is_some());
    }

    #[test]
    fn non_cartesian_maps_require_two_dimensions() {
        let map = CoordinateMap::CurvilinearArray(geometry());
        assert!(map.validate_dimensionality(1).is_err());
        assert!(map.validate_dimensionality(2).is_ok());
        assert!(map.validate_dimensionality(3).is_ok());
        assert!(CoordinateMap::Cartesian.validate_dimensionality(1).is_ok());
    }

    #[test]
    fn default_is_cartesian() {
        assert!(CoordinateMap::default().is_cartesian());
        assert!(!CoordinateMap::CurvilinearArray(geometry()).is_cartesian());
    }
}
