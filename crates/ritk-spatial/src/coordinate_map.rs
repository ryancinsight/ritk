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

use crate::Direction;

/// Why a coordinate map or its geometry was rejected.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum InvalidCoordinateMap {
    /// A geometry parameter was NaN or infinite.
    #[error("{parameter} must be finite, got {value}")]
    NotFinite {
        /// The offending parameter's name.
        parameter: &'static str,
        /// The value supplied.
        value: f64,
    },
    /// A geometry parameter that must be strictly positive was not.
    #[error("{parameter} must be greater than zero, got {value}")]
    NotPositive {
        /// The offending parameter's name.
        parameter: &'static str,
        /// The value supplied.
        value: f64,
    },
    /// A radius-like parameter was negative.
    #[error("{parameter} must be zero or greater, got {value}")]
    Negative {
        /// The offending parameter's name.
        parameter: &'static str,
        /// The value supplied.
        value: f64,
    },
    /// The map is not defined at the image's dimensionality.
    #[error("{map} coordinate map requires {expected}, got a {actual}-D image")]
    Dimensionality {
        /// The map variant's name.
        map: &'static str,
        /// The dimensionality the variant requires.
        expected: &'static str,
        /// The image dimensionality supplied.
        actual: usize,
    },
    /// A [`SliceSeries`] must carry at least one transform.
    #[error("slice series must have at least one transform, got zero")]
    TooFewSlices,
}

/// Convex (curvilinear) array acquisition geometry.
///
/// Beams fan out from an apex at equal angular steps. Index column 0 (the
/// innermost/fastest axis) is the sample along a beam; index column 1 is the
/// beam number.
///
/// # Mathematical specification
///
/// For index `(s, b)`:
///
/// ```text
/// r = s · radius_sample_size + first_sample_distance
/// θ = b · lateral_angular_separation + first_lateral_angle
/// ```
///
/// The fan origin is explicit, so an acquisition whose beams are not symmetric
/// about the axial axis is representable. ITK centres its fan on boresight;
/// that is the special case `first_lateral_angle = −(n−1)/2 · Δ`, which
/// [`CurvilinearArray::centred`] constructs. The physical pair is `(r·sin θ, r·cos θ)`,
/// placed on the two innermost spatial axes. Index columns are innermost-first
/// (column `c` is spatial axis `D-1-c`) and physical points are axis-major,
/// matching the batch transforms in `ritk-image`; any further axes use the
/// ordinary affine row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvilinearArray {
    radius_sample_size: f64,
    first_sample_distance: f64,
    lateral_angular_separation: f64,
    first_lateral_angle: f64,
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
        first_lateral_angle: f64,
    ) -> Result<Self, InvalidCoordinateMap> {
        finite("radius_sample_size", radius_sample_size)?;
        finite("first_sample_distance", first_sample_distance)?;
        finite("lateral_angular_separation", lateral_angular_separation)?;
        finite("first_lateral_angle", first_lateral_angle)?;
        positive("radius_sample_size", radius_sample_size)?;
        positive("lateral_angular_separation", lateral_angular_separation)?;
        non_negative("first_sample_distance", first_sample_distance)?;
        Ok(Self {
            radius_sample_size,
            first_sample_distance,
            lateral_angular_separation,
            first_lateral_angle,
        })
    }

    /// Construct a fan centred on the axial axis, ITK's convention.
    ///
    /// Equivalent to [`Self::try_new`] with
    /// `first_lateral_angle = -(lateral_count - 1)/2 · lateral_angular_separation`.
    /// Provided because the centred fan is the common case and because it is the
    /// parameterization `itkCurvilinearArraySpecialCoordinatesImage.h` uses.
    ///
    /// # Errors
    ///
    /// As [`Self::try_new`].
    pub fn centred(
        radius_sample_size: f64,
        first_sample_distance: f64,
        lateral_angular_separation: f64,
        lateral_count: usize,
    ) -> Result<Self, InvalidCoordinateMap> {
        let max_lateral = lateral_count.saturating_sub(1) as f64;
        Self::try_new(
            radius_sample_size,
            first_sample_distance,
            lateral_angular_separation,
            -max_lateral / 2.0 * lateral_angular_separation,
        )
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

    /// Angle of beam zero, in radians. Zero is the axial axis.
    #[inline]
    #[must_use]
    pub fn first_lateral_angle(&self) -> f64 {
        self.first_lateral_angle
    }

    /// Polar `(radius, angle)` for a `(sample, beam)` index pair.
    #[inline]
    #[must_use]
    pub fn polar_from_index(&self, sample: f64, beam: f64) -> (f64, f64) {
        let radius = sample.mul_add(self.radius_sample_size, self.first_sample_distance);
        let angle = beam.mul_add(self.lateral_angular_separation, self.first_lateral_angle);
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
    pub fn index_from_cartesian(&self, lateral: f64, axial: f64) -> Option<(f64, f64)> {
        // Non-finite is rejected explicitly: `axial <= 0.0` alone is false for
        // NaN, so a NaN would otherwise pass into the polar inverse.
        if !axial.is_finite() || !lateral.is_finite() || axial <= 0.0 {
            return None;
        }
        let radius = lateral.hypot(axial);
        let angle = (lateral / axial).atan();
        let sample = (radius - self.first_sample_distance) / self.radius_sample_size;
        let beam = (angle - self.first_lateral_angle) / self.lateral_angular_separation;
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
/// For index `(a, e, s)`:
///
/// ```text
/// azimuth   = a · azimuth_angular_separation + first_azimuth_angle
/// elevation = e · elevation_angular_separation + first_elevation_angle
/// r         = s · radius_sample_size + first_sample_distance
///
/// depth = r / √(1 + tan²azimuth + tan²elevation)
/// lateral   (azimuth axis)   = depth · tan azimuth
/// elevation axis             = depth · tan elevation
/// ```
///
/// Note this is *not* a spherical polar map: `azimuth` and `elevation` are
/// independent tangent steering angles, so the depth term carries both
/// tangents. Both angular origins are explicit, as for
/// [`CurvilinearArray`]; [`PhasedArray3D::centred`] builds ITK's
/// boresight-centred volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhasedArray3D {
    radius_sample_size: f64,
    first_sample_distance: f64,
    azimuth_angular_separation: f64,
    elevation_angular_separation: f64,
    first_azimuth_angle: f64,
    first_elevation_angle: f64,
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
        first_azimuth_angle: f64,
        first_elevation_angle: f64,
    ) -> Result<Self, InvalidCoordinateMap> {
        finite("radius_sample_size", radius_sample_size)?;
        finite("first_sample_distance", first_sample_distance)?;
        finite("azimuth_angular_separation", azimuth_angular_separation)?;
        finite("elevation_angular_separation", elevation_angular_separation)?;
        finite("first_azimuth_angle", first_azimuth_angle)?;
        finite("first_elevation_angle", first_elevation_angle)?;
        positive("radius_sample_size", radius_sample_size)?;
        positive("azimuth_angular_separation", azimuth_angular_separation)?;
        positive("elevation_angular_separation", elevation_angular_separation)?;
        non_negative("first_sample_distance", first_sample_distance)?;
        Ok(Self {
            radius_sample_size,
            first_sample_distance,
            azimuth_angular_separation,
            elevation_angular_separation,
            first_azimuth_angle,
            first_elevation_angle,
        })
    }

    /// Construct a volume centred on the boresight in both angles, ITK's
    /// convention (`itkPhasedArray3DSpecialCoordinatesImage.h`).
    ///
    /// # Errors
    ///
    /// As [`Self::try_new`].
    pub fn centred(
        radius_sample_size: f64,
        first_sample_distance: f64,
        azimuth_angular_separation: f64,
        elevation_angular_separation: f64,
        azimuth_count: usize,
        elevation_count: usize,
    ) -> Result<Self, InvalidCoordinateMap> {
        let max_azimuth = azimuth_count.saturating_sub(1) as f64;
        let max_elevation = elevation_count.saturating_sub(1) as f64;
        Self::try_new(
            radius_sample_size,
            first_sample_distance,
            azimuth_angular_separation,
            elevation_angular_separation,
            -max_azimuth / 2.0 * azimuth_angular_separation,
            -max_elevation / 2.0 * elevation_angular_separation,
        )
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

    /// Angle of azimuth beam zero, in radians. Zero is the boresight.
    #[inline]
    #[must_use]
    pub fn first_azimuth_angle(&self) -> f64 {
        self.first_azimuth_angle
    }

    /// Angle of elevation beam zero, in radians. Zero is the boresight.
    #[inline]
    #[must_use]
    pub fn first_elevation_angle(&self) -> f64 {
        self.first_elevation_angle
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
    ) -> Option<(f64, f64, f64)> {
        let azimuth =
            azimuth_index.mul_add(self.azimuth_angular_separation, self.first_azimuth_angle);
        let elevation = elevation_index.mul_add(
            self.elevation_angular_separation,
            self.first_elevation_angle,
        );
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
    ) -> Option<(f64, f64, f64)> {
        if !azimuth_axis.is_finite()
            || !elevation_axis.is_finite()
            || !depth.is_finite()
            || depth <= 0.0
        {
            return None;
        }
        let azimuth = (azimuth_axis / depth).atan();
        let elevation = (elevation_axis / depth).atan();
        let radius =
            (azimuth_axis * azimuth_axis + elevation_axis * elevation_axis + depth * depth).sqrt();

        Some((
            (azimuth - self.first_azimuth_angle) / self.azimuth_angular_separation,
            (elevation - self.first_elevation_angle) / self.elevation_angular_separation,
            (radius - self.first_sample_distance) / self.radius_sample_size,
        ))
    }
}

/// A rigid transform placing one 2-D slice of a wobbler/freehand sweep in
/// 3-D world space.
///
/// The rotation matrix columns are the world-space directions of the probe
/// frame: column 0 = in-plane right, column 1 = in-plane up, column 2 = slice
/// normal (pointing from this slice toward the next slice). The columns encode
/// both orientation **and** in-plane pixel spacing (mm/pixel), so no separate
/// spacing field is needed.
///
/// Multiplying in-plane index `[j_x, j_y]` by the first two columns and adding
/// the translation gives the world-space position of that sample.
#[derive(Clone, Debug, PartialEq)]
pub struct SliceTransform {
    rotation: Direction<3>,
    translation: [f64; 3],
}

impl SliceTransform {
    /// Construct a slice transform.
    ///
    /// `rotation` columns are: in-plane right (0), in-plane up (1), and slice
    /// normal (2). `translation` is the world position of the slice origin
    /// (`j_x = j_y = 0`).
    #[must_use]
    pub fn new(rotation: Direction<3>, translation: [f64; 3]) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// The rotation matrix (columns = in-plane right, in-plane up, slice normal).
    #[inline]
    #[must_use]
    pub fn rotation(&self) -> &Direction<3> {
        &self.rotation
    }

    /// World-space position of the slice origin.
    #[inline]
    #[must_use]
    pub fn translation(&self) -> [f64; 3] {
        self.translation
    }

    /// Map in-plane indices `(j_x, j_y)` to a world-space 3-D point.
    #[inline]
    #[must_use]
    pub(crate) fn apply(&self, j_x: f64, j_y: f64) -> [f64; 3] {
        [
            self.rotation[(0, 0)] * j_x + self.rotation[(0, 1)] * j_y + self.translation[0],
            self.rotation[(1, 0)] * j_x + self.rotation[(1, 1)] * j_y + self.translation[1],
            self.rotation[(2, 0)] * j_x + self.rotation[(2, 1)] * j_y + self.translation[2],
        ]
    }

    /// Signed distance of a world-space point from this slice's plane.
    ///
    /// Positive means the point is on the leading side (toward the next slice);
    /// negative means it is behind this slice.
    #[inline]
    #[must_use]
    pub(crate) fn signed_distance(&self, world: [f64; 3]) -> f64 {
        let rel = [
            world[0] - self.translation[0],
            world[1] - self.translation[1],
            world[2] - self.translation[2],
        ];
        self.rotation[(0, 2)] * rel[0]
            + self.rotation[(1, 2)] * rel[1]
            + self.rotation[(2, 2)] * rel[2]
    }
}

/// Wobbler or freehand 3-D sweep: a stack of 2-D acquisitions, each placed in
/// 3-D by its own [`SliceTransform`].
///
/// Index column 0 is the in-plane right axis (`j_x`), column 1 the in-plane
/// up axis (`j_y`), and column 2 (outermost for a 3-D image) is the slice
/// index.
///
/// # Forward map (index → world)
///
/// For a continuous index `(j_x, j_y, slice_f)`:
///
/// 1. Clamp `slice_f` to `[0, n−1]`; extract `i0 = ⌊slice_f⌋`,
///    `i1 = ⌈slice_f⌉`, `t = slice_f − i0`.
/// 2. `p0 = T_{i0} · [j_x, j_y]^T + origin_{i0}`,
///    `p1 = T_{i1} · [j_x, j_y]^T + origin_{i1}`.
/// 3. `world = lerp(p0, p1, t)`.
///
/// This matches the ITK `GetSliceTransform` forward convention (forward clamp,
/// not reject).
///
/// # Inverse map (world → index)
///
/// Finds the consecutive slice pair whose planes bracket the point, then
/// interpolates to a sub-integer slice coordinate and projects onto the
/// in-plane axes. Returns `None` when the point is outside the sweep, matching
/// the rejection convention of [`CurvilinearArray`] and [`PhasedArray3D`].
///
/// # References
/// - `itkSliceSeriesSpecialCoordinatesImage.h`, KitwareMedical/ITKUltrasound
#[derive(Clone, Debug, PartialEq)]
pub struct SliceSeries {
    transforms: Vec<SliceTransform>,
}

impl SliceSeries {
    /// Construct a slice series from an ordered per-slice transform list.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidCoordinateMap::TooFewSlices`] when `transforms` is
    /// empty.
    pub fn try_new(transforms: Vec<SliceTransform>) -> Result<Self, InvalidCoordinateMap> {
        if transforms.is_empty() {
            return Err(InvalidCoordinateMap::TooFewSlices);
        }
        Ok(Self { transforms })
    }

    /// The per-slice transforms.
    #[inline]
    #[must_use]
    pub fn transforms(&self) -> &[SliceTransform] {
        &self.transforms
    }

    /// Number of slices.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Whether the series has no transforms (always `false` for a valid instance).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    /// Map in-plane indices and a continuous slice coordinate to a world-space
    /// point.
    ///
    /// The slice coordinate is clamped to `[0, n−1]` (ITK forward-clamp
    /// convention).
    #[must_use]
    pub fn world_from_index(&self, j_x: f64, j_y: f64, slice_f: f64) -> [f64; 3] {
        let n = self.transforms.len();
        let sc = slice_f.clamp(0.0, (n - 1) as f64);
        let i0 = (sc.floor() as usize).min(n - 1);
        let i1 = (sc.ceil() as usize).min(n - 1);
        let t = sc - i0 as f64;

        let p0 = self.transforms[i0].apply(j_x, j_y);
        let p1 = self.transforms[i1].apply(j_x, j_y);
        [
            p0[0] + t * (p1[0] - p0[0]),
            p0[1] + t * (p1[1] - p0[1]),
            p0[2] + t * (p1[2] - p0[2]),
        ]
    }

    /// Map a world-space point to in-plane indices and a continuous slice
    /// coordinate.
    ///
    /// Returns `None` when the point lies outside the sweep extent.
    #[must_use]
    pub fn index_from_world(&self, world: [f64; 3]) -> Option<[f64; 3]> {
        let n = self.transforms.len();

        let d0 = self.transforms[0].signed_distance(world);
        let dn = self.transforms[n - 1].signed_distance(world);

        // d0 < 0: the point is behind the first slice (not within the sweep).
        // dn > 0: the point is past the last slice (not within the sweep).
        if d0 < 0.0 || dn > 0.0 {
            return None;
        }

        // Find the consecutive pair that brackets the point.
        let mut i0 = n - 1;
        let mut i1 = n - 1;
        let mut t = 0.0_f64;
        let mut d_cur = d0;

        for i in 0..n - 1 {
            let d_next = self.transforms[i + 1].signed_distance(world);
            if d_cur >= 0.0 && d_next <= 0.0 {
                i0 = i;
                i1 = i + 1;
                let denom = d_cur - d_next;
                t = if denom.abs() < f64::EPSILON * 1.0e3 {
                    0.5
                } else {
                    (d_cur / denom).clamp(0.0, 1.0)
                };
                break;
            }
            d_cur = d_next;
        }

        let slice_f = i0 as f64 + t;

        // Interpolated slice origin.
        let tf0 = &self.transforms[i0];
        let tf1 = &self.transforms[i1];
        let orig = [
            tf0.translation[0] + t * (tf1.translation[0] - tf0.translation[0]),
            tf0.translation[1] + t * (tf1.translation[1] - tf0.translation[1]),
            tf0.translation[2] + t * (tf1.translation[2] - tf0.translation[2]),
        ];
        let rel = [world[0] - orig[0], world[1] - orig[1], world[2] - orig[2]];

        // Interpolated in-plane column vectors; project rel onto each.
        let j_x = {
            let rx = tf0.rotation[(0, 0)] * (1.0 - t) + tf1.rotation[(0, 0)] * t;
            let ry = tf0.rotation[(1, 0)] * (1.0 - t) + tf1.rotation[(1, 0)] * t;
            let rz = tf0.rotation[(2, 0)] * (1.0 - t) + tf1.rotation[(2, 0)] * t;
            let ns = rx.mul_add(rx, ry.mul_add(ry, rz * rz));
            (rx * rel[0] + ry * rel[1] + rz * rel[2]) / ns
        };
        let j_y = {
            let rx = tf0.rotation[(0, 1)] * (1.0 - t) + tf1.rotation[(0, 1)] * t;
            let ry = tf0.rotation[(1, 1)] * (1.0 - t) + tf1.rotation[(1, 1)] * t;
            let rz = tf0.rotation[(2, 1)] * (1.0 - t) + tf1.rotation[(2, 1)] * t;
            let ns = rx.mul_add(rx, ry.mul_add(ry, rz * rz));
            (rx * rel[0] + ry * rel[1] + rz * rel[2]) / ns
        };

        Some([j_x, j_y, slice_f])
    }
}

/// How an image's index space maps into physical space.
///
/// See the module documentation for why this is a closed enum.
///
/// Note: this enum is `Clone` but not `Copy` because the [`SliceSeries`]
/// variant owns a heap-allocated transform list. Existing callers use the map
/// by reference or clone-on-attach; this is therefore a mechanical
/// (non-functional) breaking change.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum CoordinateMap {
    /// Affine `origin + Direction · (index ⊙ spacing)` — the ordinary raster.
    #[default]
    Cartesian,
    /// Convex/curvilinear array beam space.
    CurvilinearArray(CurvilinearArray),
    /// Three-dimensional phased-array beam space.
    PhasedArray3D(PhasedArray3D),
    /// Wobbler or freehand 3-D sweep (slice series).
    SliceSeries(SliceSeries),
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
    pub fn validate_dimensionality(&self, d: usize) -> Result<(), InvalidCoordinateMap> {
        match self {
            Self::Cartesian => Ok(()),
            Self::CurvilinearArray(_) if d < 2 => Err(InvalidCoordinateMap::Dimensionality {
                map: "curvilinear",
                expected: "a 2-D or higher image",
                actual: d,
            }),
            Self::CurvilinearArray(_) => Ok(()),
            Self::PhasedArray3D(_) if d != 3 => Err(InvalidCoordinateMap::Dimensionality {
                map: "phased-array",
                expected: "a 3-D image",
                actual: d,
            }),
            Self::PhasedArray3D(_) => Ok(()),
            Self::SliceSeries(_) if d != 3 => Err(InvalidCoordinateMap::Dimensionality {
                map: "slice-series",
                expected: "a 3-D image",
                actual: d,
            }),
            Self::SliceSeries(_) => Ok(()),
        }
    }
}

fn finite(parameter: &'static str, value: f64) -> Result<(), InvalidCoordinateMap> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(InvalidCoordinateMap::NotFinite { parameter, value })
    }
}

fn positive(parameter: &'static str, value: f64) -> Result<(), InvalidCoordinateMap> {
    if value > 0.0 {
        Ok(())
    } else {
        Err(InvalidCoordinateMap::NotPositive { parameter, value })
    }
}

fn non_negative(parameter: &'static str, value: f64) -> Result<(), InvalidCoordinateMap> {
    if value >= 0.0 {
        Ok(())
    } else {
        Err(InvalidCoordinateMap::Negative { parameter, value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> CurvilinearArray {
        // 0.1 mm range sampling, 60 mm apex offset, 0.5 degree beam pitch.
        CurvilinearArray::centred(1.0e-4, 0.06, 0.5_f64.to_radians(), 129).expect("valid geometry")
    }

    #[test]
    fn rejects_non_positive_and_non_finite_parameters() {
        assert!(CurvilinearArray::try_new(0.0, 0.06, 0.01, 0.0).is_err());
        assert!(CurvilinearArray::try_new(-1.0e-4, 0.06, 0.01, 0.0).is_err());
        assert!(CurvilinearArray::try_new(1.0e-4, -0.01, 0.01, 0.0).is_err());
        assert!(CurvilinearArray::try_new(1.0e-4, 0.06, 0.0, 0.0).is_err());
        assert!(CurvilinearArray::try_new(f64::NAN, 0.06, 0.01, 0.0).is_err());
        assert!(CurvilinearArray::try_new(1.0e-4, f64::INFINITY, 0.01, 0.0).is_err());
    }

    #[test]
    fn fan_is_centred_on_the_axial_axis() {
        let g = geometry();
        // The middle beam of an odd-count fan sits exactly on theta = 0.
        let (_, angle) = g.polar_from_index(0.0, 64.0);
        assert!(
            angle.abs() < 1.0e-15,
            "centre beam must be axial, got {angle}"
        );
        // Outermost beams are symmetric about it.
        let (_, first) = g.polar_from_index(0.0, 0.0);
        let (_, last) = g.polar_from_index(0.0, 128.0);
        assert!(
            (first + last).abs() < 1.0e-15,
            "fan must be symmetric: {first} vs {last}"
        );
    }

    #[test]
    fn radius_starts_at_the_apex_offset() {
        let g = geometry();
        let (radius, _) = g.polar_from_index(0.0, 0.0);
        assert!((radius - 0.06).abs() < 1.0e-15, "got {radius}");
        let (radius, _) = g.polar_from_index(100.0, 0.0);
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
        for beam_i in 0..129 {
            for sample_i in [0_usize, 1, 37, 128, 199] {
                let sample = sample_i as f64;
                let beam = beam_i as f64;
                let (radius, angle) = g.polar_from_index(sample, beam);
                let (lateral, axial) = (radius * angle.sin(), radius * angle.cos());
                let (s, b) = g
                    .index_from_cartesian(lateral, axial)
                    .expect("fan point must invert");
                assert!(
                    (s - sample).abs() < 1.0e-9,
                    "sample {sample} -> {s} (beam {beam})"
                );
                assert!((b - beam).abs() < 1.0e-9, "beam {beam} -> {b}");
            }
        }
    }

    /// The fan origin is explicit, so an acquisition whose beams are not
    /// symmetric about the axial axis round-trips exactly.
    ///
    /// This is what the ITK-inherited centred convention could not express, and
    /// it is the geometry kwavers' `ScanGeometry::angle_min` already supports.
    #[test]
    fn asymmetric_fan_round_trips() {
        // Beams run from +5 deg to +25 deg: entirely off-axis, no symmetry.
        let g = CurvilinearArray::try_new(1.0e-4, 0.06, 0.5_f64.to_radians(), 5.0_f64.to_radians())
            .expect("valid geometry");
        for beam_i in 0..41 {
            for sample_i in [0_usize, 63, 250] {
                let (sample, beam) = (sample_i as f64, beam_i as f64);
                let (radius, angle) = g.polar_from_index(sample, beam);
                assert!(
                    angle > 0.0,
                    "every beam of this fan is off-axis, got {angle}"
                );
                let (lateral, axial) = (radius * angle.sin(), radius * angle.cos());
                let (s, b) = g
                    .index_from_cartesian(lateral, axial)
                    .expect("fan point must invert");
                assert!((s - sample).abs() < 1.0e-9, "sample {sample} -> {s}");
                assert!((b - beam).abs() < 1.0e-9, "beam {beam} -> {b}");
            }
        }
    }

    /// `centred` must reproduce ITK's convention exactly, so the generalization
    /// loses nothing: beam zero sits at `-(n-1)/2 · Delta`.
    #[test]
    fn centred_matches_the_itk_convention() {
        let separation = 0.5_f64.to_radians();
        let g = CurvilinearArray::centred(1.0e-4, 0.06, separation, 129).expect("valid geometry");
        assert!((g.first_lateral_angle() - (-64.0 * separation)).abs() < 1.0e-15);
        let (_, centre) = g.polar_from_index(0.0, 64.0);
        assert!(
            centre.abs() < 1.0e-15,
            "centre beam must be axial, got {centre}"
        );

        let p = PhasedArray3D::centred(1.0e-4, 0.01, separation, separation, 65, 33)
            .expect("valid geometry");
        assert!((p.first_azimuth_angle() - (-32.0 * separation)).abs() < 1.0e-15);
        assert!((p.first_elevation_angle() - (-16.0 * separation)).abs() < 1.0e-15);
    }

    #[test]
    fn points_behind_the_apex_plane_are_rejected() {
        let g = geometry();
        assert!(g.index_from_cartesian(0.01, 0.0).is_none());
        assert!(g.index_from_cartesian(0.01, -0.05).is_none());
        assert!(g.index_from_cartesian(0.0, 0.07).is_some());
    }

    fn phased_geometry() -> PhasedArray3D {
        // 0.1 mm range sampling, 10 mm apex offset, 0.75 deg azimuth / 1.5 deg
        // elevation beam pitch.
        PhasedArray3D::centred(
            1.0e-4,
            0.01,
            0.75_f64.to_radians(),
            1.5_f64.to_radians(),
            65,
            33,
        )
        .expect("valid geometry")
    }

    #[test]
    fn phased_array_rejects_invalid_parameters() {
        assert!(PhasedArray3D::try_new(0.0, 0.01, 0.01, 0.01, 0.0, 0.0).is_err());
        assert!(PhasedArray3D::try_new(1.0e-4, -0.01, 0.01, 0.01, 0.0, 0.0).is_err());
        assert!(PhasedArray3D::try_new(1.0e-4, 0.01, 0.0, 0.01, 0.0, 0.0).is_err());
        assert!(PhasedArray3D::try_new(1.0e-4, 0.01, 0.01, 0.0, 0.0, 0.0).is_err());
        assert!(PhasedArray3D::try_new(f64::NAN, 0.01, 0.01, 0.01, 0.0, 0.0).is_err());
    }

    /// The boresight ray (centre azimuth and elevation beams) must run straight
    /// down the depth axis with zero lateral offset, and its depth must be the
    /// full radius — the tangent denominator is 1 there.
    #[test]
    fn phased_array_boresight_is_pure_depth() {
        let g = phased_geometry();
        let (az, el, depth) = g
            .cartesian_from_index(32.0, 16.0, 100.0)
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
        let (az_left, el_left, d_left) = g.cartesian_from_index(0.0, 16.0, 100.0).unwrap();
        let (az_right, el_right, d_right) = g.cartesian_from_index(64.0, 16.0, 100.0).unwrap();
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
        for &a in &[0.0_f64, 1.0, 32.0, 47.5, 64.0] {
            for &e in &[0.0_f64, 8.0, 16.0, 32.0] {
                for &sample in &[0.0_f64, 25.0, 199.0] {
                    let (x, y, z) = g
                        .cartesian_from_index(a, e, sample)
                        .expect("steered ray is representable");
                    let (a2, e2, s2) = g
                        .index_from_cartesian(x, y, z)
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
        assert!(g.index_from_cartesian(0.001, 0.001, 0.0).is_none());
        assert!(g.index_from_cartesian(0.001, 0.001, -0.02).is_none());
        assert!(g.index_from_cartesian(f64::NAN, 0.001, 0.02).is_none());
        assert!(g.index_from_cartesian(0.001, 0.001, 0.02).is_some());
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
        let g = PhasedArray3D::centred(1.0e-4, 0.01, std::f64::consts::FRAC_PI_2, 0.01, 3, 1)
            .expect("valid geometry");
        assert!(
            g.cartesian_from_index(2.0, 0.0, 10.0).is_none(),
            "a quarter-turn steer must be rejected"
        );
        assert!(
            g.cartesian_from_index(3.0, 0.0, 10.0).is_none(),
            "steering past a quarter turn must be rejected, not sign-flipped"
        );
        // Just inside the limit remains representable.
        let g = PhasedArray3D::centred(1.0e-4, 0.01, 89.0_f64.to_radians(), 0.01, 3, 1)
            .expect("valid geometry");
        assert!(g.cartesian_from_index(2.0, 0.0, 10.0).is_some());
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

    // --- SliceSeries tests ---

    /// Build a pure-translation sweep: n slices, each translated by `dz` along
    /// the z axis, with identity in-plane rotation scaled by 1 mm/pixel.
    fn translation_sweep(n: usize, dz: f64) -> SliceSeries {
        let transforms = (0..n)
            .map(|i| {
                // rotation: identity (columns = [x, y, z] directions, 1 mm/pixel)
                let rot = Direction::from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
                SliceTransform::new(rot, [0.0, 0.0, i as f64 * dz])
            })
            .collect();
        SliceSeries::try_new(transforms).expect("valid sweep")
    }

    #[test]
    fn slice_series_rejects_empty_transform_list() {
        assert!(SliceSeries::try_new(vec![]).is_err());
    }

    #[test]
    fn pure_translation_sweep_round_trips() {
        let dz = 2.0_f64; // 2 mm slice spacing
        let sweep = translation_sweep(8, dz);

        for s in 0..8_usize {
            for j_x in [0.0_f64, 3.0, 10.5] {
                for j_y in [0.0_f64, 5.0, -2.0] {
                    let slice_f = s as f64;
                    let world = sweep.world_from_index(j_x, j_y, slice_f);

                    // Forward check: z = slice * dz, x = j_x, y = j_y.
                    assert!(
                        (world[0] - j_x).abs() < 1.0e-12,
                        "x: expected {j_x}, got {}",
                        world[0]
                    );
                    assert!(
                        (world[1] - j_y).abs() < 1.0e-12,
                        "y: expected {j_y}, got {}",
                        world[1]
                    );
                    assert!(
                        (world[2] - slice_f * dz).abs() < 1.0e-12,
                        "z: expected {}, got {}",
                        slice_f * dz,
                        world[2]
                    );

                    // Round-trip.
                    let back = sweep
                        .index_from_world(world)
                        .expect("on-sweep point must invert");
                    assert!(
                        (back[0] - j_x).abs() < 1.0e-9,
                        "j_x: expected {j_x}, got {}",
                        back[0]
                    );
                    assert!(
                        (back[1] - j_y).abs() < 1.0e-9,
                        "j_y: expected {j_y}, got {}",
                        back[1]
                    );
                    assert!(
                        (back[2] - slice_f).abs() < 1.0e-9,
                        "slice_f: expected {slice_f}, got {}",
                        back[2]
                    );
                }
            }
        }
    }

    #[test]
    fn forward_clamps_out_of_range_slice_indices() {
        let sweep = translation_sweep(4, 1.0);

        // Beyond last slice clamps to last slice's position.
        let world_clamped = sweep.world_from_index(1.0, 1.0, 10.0);
        let world_last = sweep.world_from_index(1.0, 1.0, 3.0);
        assert!(
            (world_clamped[2] - world_last[2]).abs() < 1.0e-12,
            "forward map must clamp to last slice, got z={}",
            world_clamped[2]
        );

        // Before first slice clamps to first slice's position.
        let world_before = sweep.world_from_index(1.0, 1.0, -5.0);
        let world_first = sweep.world_from_index(1.0, 1.0, 0.0);
        assert!(
            (world_before[2] - world_first[2]).abs() < 1.0e-12,
            "forward map must clamp to first slice, got z={}",
            world_before[2]
        );
    }

    #[test]
    fn inverse_rejects_out_of_range_world_points() {
        let sweep = translation_sweep(4, 1.0); // z ∈ [0, 3]

        // Before sweep start.
        assert!(
            sweep.index_from_world([0.0, 0.0, -0.1]).is_none(),
            "inverse must reject points before the first slice"
        );
        // Past sweep end.
        assert!(
            sweep.index_from_world([0.0, 0.0, 3.1]).is_none(),
            "inverse must reject points past the last slice"
        );
        // Within sweep is accepted.
        assert!(sweep.index_from_world([0.0, 0.0, 1.5]).is_some());
    }

    #[test]
    fn single_slice_degenerate_case() {
        // A single-slice sweep with identity rotation: the map is the 2-D
        // index→world formula for the z=0 plane.
        let rot = Direction::from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let sweep = SliceSeries::try_new(vec![SliceTransform::new(rot, [0.0, 0.0, 0.0])])
            .expect("valid single-slice sweep");

        let world = sweep.world_from_index(3.0, 5.0, 0.0);
        assert!((world[0] - 3.0).abs() < 1.0e-12);
        assert!((world[1] - 5.0).abs() < 1.0e-12);
        assert!(world[2].abs() < 1.0e-12);

        // Inverse: only z = 0 is within the sweep.
        assert!(sweep.index_from_world([3.0, 5.0, 0.0]).is_some());
        assert!(sweep.index_from_world([3.0, 5.0, 0.5]).is_none());
        assert!(sweep.index_from_world([3.0, 5.0, -0.1]).is_none());
    }

    #[test]
    fn slice_series_map_requires_exactly_three_dimensions() {
        let sweep = translation_sweep(2, 1.0);
        let map = CoordinateMap::SliceSeries(sweep);
        assert!(map.validate_dimensionality(2).is_err());
        assert!(map.validate_dimensionality(3).is_ok());
        assert!(map.validate_dimensionality(4).is_err());
    }
}
