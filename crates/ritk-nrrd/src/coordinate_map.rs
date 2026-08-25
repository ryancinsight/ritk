//! Persisting an acquisition coordinate map in a NRRD header.
//!
//! An ultrasound acquisition's geometry is part of what the image *is*: beam
//! data written without it reloads as a Cartesian raster, and every downstream
//! measurement — scan conversion, spectra, block matching — then silently
//! refers to the wrong physical points. ITK has `UltrasoundImageFileReader` for
//! exactly this reason; this is the same contract on ritk's NRRD path.
//!
//! # Encoding
//!
//! The map travels as a NRRD **key/value field**, written `key:=value`, which
//! is the mechanism the NRRD specification reserves for data outside its own
//! field set. Readers that do not know the key preserve or ignore it; using a
//! plain `key: value` field instead would present unknown *header* fields to
//! conformant readers such as ITK and 3D Slicer, which is not what that form
//! means.
//!
//! The value is a tag followed by named parameters:
//!
//! ```text
//! ritk_coordinate_map:=curvilinear radius_sample_size=1e-4 first_sample_distance=0.06 ...
//! ```
//!
//! Named rather than positional, so a parameter added later cannot silently
//! shift the meaning of an older file's fields.
//!
//! A Cartesian map is **not** written. Its absence is what every existing NRRD
//! already means, so omitting it keeps those files byte-identical and makes
//! "no key" and "Cartesian" the same statement rather than two.

use anyhow::{anyhow, bail, Result};
use ritk_spatial::{CoordinateMap, CurvilinearArray, PhasedArray3D};

/// The NRRD key/value key under which the map travels.
pub const COORDINATE_MAP_KEY: &str = "ritk_coordinate_map";

/// Encode a map as a NRRD key/value payload.
///
/// Returns `None` for [`CoordinateMap::Cartesian`], which is written by
/// omission — see the module documentation.
#[must_use]
pub fn encode(map: &CoordinateMap) -> Option<String> {
    match map {
        CoordinateMap::Cartesian => None,
        CoordinateMap::CurvilinearArray(g) => Some(format!(
            "curvilinear radius_sample_size={} first_sample_distance={} \
             lateral_angular_separation={} first_lateral_angle={}",
            g.radius_sample_size(),
            g.first_sample_distance(),
            g.lateral_angular_separation(),
            g.first_lateral_angle()
        )),
        CoordinateMap::PhasedArray3D(g) => Some(format!(
            "phased_array_3d radius_sample_size={} first_sample_distance={} \
             azimuth_angular_separation={} elevation_angular_separation={} \
             first_azimuth_angle={} first_elevation_angle={}",
            g.radius_sample_size(),
            g.first_sample_distance(),
            g.azimuth_angular_separation(),
            g.elevation_angular_separation(),
            g.first_azimuth_angle(),
            g.first_elevation_angle()
        )),
        // SliceSeries carries a variable-length per-slice transform list; the
        // NRRD key/value format is defined only for the fixed-parameter variants.
        // Writing is not yet supported; the caller should save to a format that
        // can represent the full transform list.
        CoordinateMap::SliceSeries(_) => None,
    }
}

/// Decode a NRRD key/value payload into a map.
///
/// # Errors
///
/// Returns an error when the tag is unknown, a named parameter is missing or
/// unparseable, or the geometry rejects the values. A malformed map is an
/// error rather than a silent fallback to Cartesian: falling back would hand
/// the caller beam data labelled as a raster, which is the exact failure this
/// field exists to prevent.
pub fn decode(value: &str) -> Result<CoordinateMap> {
    let mut parts = value.split_whitespace();
    let tag = parts
        .next()
        .ok_or_else(|| anyhow!("empty {COORDINATE_MAP_KEY} value"))?;
    let params: Vec<(&str, &str)> = parts
        .map(|token| {
            token
                .split_once('=')
                .ok_or_else(|| anyhow!("malformed {COORDINATE_MAP_KEY} parameter '{token}'"))
        })
        .collect::<Result<_>>()?;

    let get = |name: &str| -> Result<f64> {
        params
            .iter()
            .find(|(key, _)| *key == name)
            .ok_or_else(|| anyhow!("{COORDINATE_MAP_KEY} '{tag}' is missing '{name}'"))
            .and_then(|(_, raw)| {
                raw.parse::<f64>().map_err(|error| {
                    anyhow!("{COORDINATE_MAP_KEY} '{name}' is not a number: {error}")
                })
            })
    };

    match tag {
        "cartesian" => Ok(CoordinateMap::Cartesian),
        "curvilinear" => Ok(CoordinateMap::CurvilinearArray(CurvilinearArray::try_new(
            get("radius_sample_size")?,
            get("first_sample_distance")?,
            get("lateral_angular_separation")?,
            get("first_lateral_angle")?,
        )?)),
        "phased_array_3d" => Ok(CoordinateMap::PhasedArray3D(PhasedArray3D::try_new(
            get("radius_sample_size")?,
            get("first_sample_distance")?,
            get("azimuth_angular_separation")?,
            get("elevation_angular_separation")?,
            get("first_azimuth_angle")?,
            get("first_elevation_angle")?,
        )?)),
        other => bail!(
            "unknown {COORDINATE_MAP_KEY} tag '{other}'; this file was written by a newer ritk \
             and its geometry cannot be interpreted here"
        ),
    }
}

/// Read the map out of an already-parsed NRRD header map.
///
/// Absence means [`CoordinateMap::Cartesian`].
///
/// # Errors
///
/// Propagates [`decode`] failures.
pub fn from_header<S: ::std::hash::BuildHasher>(
    headers: &std::collections::HashMap<String, String, S>,
) -> Result<CoordinateMap> {
    match headers.get(COORDINATE_MAP_KEY) {
        None => Ok(CoordinateMap::Cartesian),
        Some(value) => decode(value),
    }
}

#[cfg(test)]
#[path = "tests_coordinate_map.rs"]
mod tests;
