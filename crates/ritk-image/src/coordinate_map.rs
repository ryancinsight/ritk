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

/// Three-dimensional phased-array acquisition geometry.
///
/// A 3-D phased array steers in two angles from a single apex. Index column 0
/// is the azimuth beam, column 1 the elevation beam, and column 2 the sample
/// along the ray.
///
/// # Mathematical specification
///
/// With `maxAz = azimuth_count − 1` and `maxEl = elevation_count − 1`, for
/// index `(a, e, s)`:
///
/// ```text
/// azimuth   = (a − maxAz/2) · azimuth_angular_separation
/// elevation = (e − maxEl/2) · elevation_angular_separation
/// r         = s · radius_sample_size + first_sample_distance
///
/// depth = r / √(1 + tan²azimuth + tan²elevation)
/// lateral   (azimuth axis)   = depth · tan azimuth
/// elevation axis             = depth · tan elevation
/// ```
///
/// Note this is *not* a spherical polar map: `azimuth` and `elevation` are
/// independent tangent steering angles, so the depth term carries both
/// tangents. Beams are centred on the boresight in both angles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhasedArray3D {
    radius_sample_size: f64,
    first_sample_distance: f64,
    azimuth_angular_separation: f64,
    elevation_angular_separation: f64,
}

impl PhasedArray3D {
    /// Construct a 3-D phased-array geometry.
    ///
    /// # Errors
    ///
    /// Returns an error when any parameter is non-finite, when
    /// `radius_sample_size` or either angular separation is not strictly
    /// positive, or when `first_sample_distance` is negative.
    pub fn try_new(
        radius_sample_size: f64,
        first_sample_distance: f64,
        azimuth_angular_separation: f64,
        elevation_angular_separation: f64,
    ) -> Result<Self> {
        if !radius_sample_size.is_finite()
            || !first_sample_distance.is_finite()
            || !azimuth_angular_separation.is_finite()
            || !elevation_angular_separation.is_finite()
        {
            bail!("phased-array geometry parameters must be finite");
        }
        if radius_sample_size <= 0.0 {
            bail!("phased-array radius_sample_size must be > 0, got {radius_sample_size}");
        }
        if azimuth_angular_separation <= 0.0 {
            bail!(
                "phased-array azimuth_angular_separation must be > 0, \
                 got {azimuth_angular_separation}"
            );
        }
        if elevation_angular_separation <= 0.0 {
            bail!(
                "phased-array elevation_angular_separation must be > 0, \
                 got {elevation_angular_separation}"
            );
        }
        if first_sample_distance < 0.0 {
            bail!("phased-array first_sample_distance must be >= 0, got {first_sample_distance}");
        }
        Ok(Self {
            radius_sample_size,
            first_sample_distance,
            azimuth_angular_separation,
            elevation_angular_separation,
        })
    }

    /// Range increment per sample along a ray.
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

    /// Angle between adjacent azimuth beams, in radians.
    #[inline]
    #[must_use]
    pub fn azimuth_angular_separation(&self) -> f64 {
        self.azimuth_angular_separation
    }

    /// Angle between adjacent elevation beams, in radians.
    #[inline]
    #[must_use]
    pub fn elevation_angular_separation(&self) -> f64 {
        self.elevation_angular_separation
    }

    /// Cartesian `(azimuth_axis, elevation_axis, depth)` for an index triple.
    ///
    /// Returns `None` when either steering angle reaches or passes a quarter
    /// turn, where the ray no longer points into the forward half-space.
    ///
    /// The check is on the angle, not on `tan`: `tan` is finite for every
    /// representable `f64` (`tan(π/2) ≈ 1.6e16`, since `π/2` is not exactly
    /// representable), and past a quarter turn it silently changes sign, which
    /// would place the ray on the opposite side of the array with no other
    /// symptom.
    #[inline]
    #[must_use]
    pub fn cartesian_from_index(
        &self,
        azimuth_index: f64,
        elevation_index: f64,
        sample: f64,
        azimuth_count: usize,
        elevation_count: usize,
    ) -> Option<(f64, f64, f64)> {
        let max_azimuth = azimuth_count.saturating_sub(1) as f64;
        let max_elevation = elevation_count.saturating_sub(1) as f64;
        let azimuth = (azimuth_index - max_azimuth / 2.0) * self.azimuth_angular_separation;
        let elevation = (elevation_index - max_elevation / 2.0) * self.elevation_angular_separation;
        let radius = sample.mul_add(self.radius_sample_size, self.first_sample_distance);

        if !azimuth.is_finite()
            || !elevation.is_finite()
            || azimuth.abs() >= std::f64::consts::FRAC_PI_2
            || elevation.abs() >= std::f64::consts::FRAC_PI_2
        {
            return None;
        }
        let tan_azimuth = azimuth.tan();
        let tan_elevation = elevation.tan();
        let depth = radius
            / tan_elevation
                .mul_add(tan_elevation, tan_azimuth.mul_add(tan_azimuth, 1.0))
                .sqrt();
        if !depth.is_finite() {
            return None;
        }
        Some((depth * tan_azimuth, depth * tan_elevation, depth))
    }

    /// Index triple for a Cartesian `(azimuth_axis, elevation_axis, depth)`.
    ///
    /// Returns `None` outside the acquisition half-space (`depth <= 0`) or for
    /// non-finite input, where no ray is defined.
    #[inline]
    #[must_use]
    pub fn index_from_cartesian(
        &self,
        azimuth_axis: f64,
        elevation_axis: f64,
        depth: f64,
        azimuth_count: usize,
        elevation_count: usize,
    ) -> Option<(f64, f64, f64)> {
        if !azimuth_axis.is_finite()
            || !elevation_axis.is_finite()
            || !depth.is_finite()
            || depth <= 0.0
        {
            return None;
        }
        let max_azimuth = azimuth_count.saturating_sub(1) as f64;
        let max_elevation = elevation_count.saturating_sub(1) as f64;

        let azimuth = (azimuth_axis / depth).atan();
        let elevation = (elevation_axis / depth).atan();
        let radius =
            (azimuth_axis * azimuth_axis + elevation_axis * elevation_axis + depth * depth).sqrt();

        Some((
            azimuth / self.azimuth_angular_separation + max_azimuth / 2.0,
            elevation / self.elevation_angular_separation + max_elevation / 2.0,
            (radius - self.first_sample_distance) / self.radius_sample_size,
        ))
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
    /// Three-dimensional phased-array beam space.
    PhasedArray3D(PhasedArray3D),
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
    /// Returns an error when the map's required axes are absent: a curvilinear
    /// array needs a lateral axis in addition to the axis along the beam, and a
    /// 3-D phased array steers in two angles, so it is defined only at `D == 3`.
    pub fn validate_dimensionality(&self, d: usize) -> Result<()> {
        match self {
            Self::Cartesian => Ok(()),
            Self::CurvilinearArray(_) => {
                if d < 2 {
                    bail!("curvilinear coordinate map requires a 2-D or higher image, got {d}-D");
                }
                Ok(())
            }
            Self::PhasedArray3D(_) => {
                if d != 3 {
                    bail!("phased-array coordinate map requires a 3-D image, got {d}-D");
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

    fn phased_geometry() -> PhasedArray3D {
        // 0.1 mm range sampling, 10 mm apex offset, 0.75 deg azimuth / 1.5 deg
        // elevation beam pitch.
        PhasedArray3D::try_new(1.0e-4, 0.01, 0.75_f64.to_radians(), 1.5_f64.to_radians())
            .expect("valid geometry")
    }

    #[test]
    fn phased_array_rejects_invalid_parameters() {
        assert!(PhasedArray3D::try_new(0.0, 0.01, 0.01, 0.01).is_err());
        assert!(PhasedArray3D::try_new(1.0e-4, -0.01, 0.01, 0.01).is_err());
        assert!(PhasedArray3D::try_new(1.0e-4, 0.01, 0.0, 0.01).is_err());
        assert!(PhasedArray3D::try_new(1.0e-4, 0.01, 0.01, 0.0).is_err());
        assert!(PhasedArray3D::try_new(f64::NAN, 0.01, 0.01, 0.01).is_err());
    }

    /// The boresight ray (centre azimuth and elevation beams) must run straight
    /// down the depth axis with zero lateral offset, and its depth must be the
    /// full radius — the tangent denominator is 1 there.
    #[test]
    fn phased_array_boresight_is_pure_depth() {
        let g = phased_geometry();
        let (az, el, depth) = g
            .cartesian_from_index(32.0, 16.0, 100.0, 65, 33)
            .expect("boresight is representable");
        assert!(az.abs() < 1.0e-15, "azimuth offset {az}");
        assert!(el.abs() < 1.0e-15, "elevation offset {el}");
        let expected = 0.01 + 100.0 * 1.0e-4;
        assert!((depth - expected).abs() < 1.0e-15, "depth {depth}");
    }

    /// Steering must be independent per angle and symmetric about boresight.
    #[test]
    fn phased_array_steering_is_symmetric_per_angle() {
        let g = phased_geometry();
        let (az_left, el_left, d_left) = g.cartesian_from_index(0.0, 16.0, 100.0, 65, 33).unwrap();
        let (az_right, el_right, d_right) =
            g.cartesian_from_index(64.0, 16.0, 100.0, 65, 33).unwrap();
        assert!((az_left + az_right).abs() < 1.0e-15, "azimuth must mirror");
        assert!((d_left - d_right).abs() < 1.0e-15, "depth must match");
        // Steering azimuth alone leaves elevation on boresight.
        assert!(el_left.abs() < 1.0e-15 && el_right.abs() < 1.0e-15);
    }

    /// Round-trip across the steered volume.
    ///
    /// Tolerance follows the same reasoning as the curvilinear case: `atan` and
    /// the radius `sqrt` each carry O(eps) relative error, and recovering the
    /// sample index divides the radius error by `radius_sample_size = 1e-4`.
    /// With radius <= 0.03 m that bounds the sample error near 1e-11.
    #[test]
    fn phased_array_index_round_trips() {
        let g = phased_geometry();
        let (az_count, el_count) = (65_usize, 33_usize);
        for &a in &[0.0_f64, 1.0, 32.0, 47.5, 64.0] {
            for &e in &[0.0_f64, 8.0, 16.0, 32.0] {
                for &sample in &[0.0_f64, 25.0, 199.0] {
                    let (x, y, z) = g
                        .cartesian_from_index(a, e, sample, az_count, el_count)
                        .expect("steered ray is representable");
                    let (a2, e2, s2) = g
                        .index_from_cartesian(x, y, z, az_count, el_count)
                        .expect("forward point must invert");
                    assert!((a2 - a).abs() < 1.0e-9, "azimuth {a} -> {a2}");
                    assert!((e2 - e).abs() < 1.0e-9, "elevation {e} -> {e2}");
                    assert!((s2 - sample).abs() < 1.0e-8, "sample {sample} -> {s2}");
                }
            }
        }
    }

    #[test]
    fn phased_array_rejects_points_behind_the_array() {
        let g = phased_geometry();
        assert!(g.index_from_cartesian(0.001, 0.001, 0.0, 65, 33).is_none());
        assert!(g
            .index_from_cartesian(0.001, 0.001, -0.02, 65, 33)
            .is_none());
        assert!(g
            .index_from_cartesian(f64::NAN, 0.001, 0.02, 65, 33)
            .is_none());
        assert!(g.index_from_cartesian(0.001, 0.001, 0.02, 65, 33).is_some());
    }

    /// A steering angle at or past a quarter turn leaves the forward half-space
    /// and must be rejected.
    ///
    /// `tan` cannot detect this: it is finite for every representable `f64`,
    /// and past a quarter turn it flips sign, so an unguarded map would place
    /// the ray on the opposite side of the array and look plausible. This test
    /// exists to pin the angle-domain guard, not a finiteness check.
    #[test]
    fn phased_array_rejects_degenerate_steering() {
        // 90 degrees per beam: 3 beams centre on index 1, so index 2 is exactly
        // +pi/2 and index 3 is past it (where tan turns negative).
        let g = PhasedArray3D::try_new(1.0e-4, 0.01, std::f64::consts::FRAC_PI_2, 0.01)
            .expect("valid geometry");
        assert!(
            g.cartesian_from_index(2.0, 0.0, 10.0, 3, 1).is_none(),
            "a quarter-turn steer must be rejected"
        );
        assert!(
            g.cartesian_from_index(3.0, 0.0, 10.0, 3, 1).is_none(),
            "steering past a quarter turn must be rejected, not sign-flipped"
        );
        // Just inside the limit remains representable.
        let g = PhasedArray3D::try_new(1.0e-4, 0.01, 89.0_f64.to_radians(), 0.01)
            .expect("valid geometry");
        assert!(g.cartesian_from_index(2.0, 0.0, 10.0, 3, 1).is_some());
    }

    #[test]
    fn phased_array_map_requires_exactly_three_dimensions() {
        let map = CoordinateMap::PhasedArray3D(phased_geometry());
        assert!(map.validate_dimensionality(2).is_err());
        assert!(map.validate_dimensionality(3).is_ok());
        assert!(map.validate_dimensionality(4).is_err());
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
