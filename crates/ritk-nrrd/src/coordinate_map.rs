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
//! A slice series uses `slice_series count=N transforms=...`; each semicolon-
//! separated transform contains its row-major 3×3 direction followed by the
//! three translation coordinates.
//!
//! Named rather than positional, so a parameter added later cannot silently
//! shift the meaning of an older file's fields.
//!
//! A Cartesian map is **not** written. Its absence is what every existing NRRD
//! already means, so omitting it keeps those files byte-identical and makes
//! "no key" and "Cartesian" the same statement rather than two.

use anyhow::{anyhow, bail, Result};
use ritk_spatial::{
    CoordinateMap, CurvilinearArray, Direction, PhasedArray3D, SliceSeries, SliceTransform,
};
use std::fmt::Write as _;

/// The NRRD key/value key under which the map travels.
pub const COORDINATE_MAP_KEY: &str = "ritk_coordinate_map";

const MAX_COORDINATE_MAP_PARAMETERS: usize = 6;

/// Encode a map as a NRRD key/value payload.
///
/// Returns `None` for [`CoordinateMap::Cartesian`], which is written by
/// omission. Other maps use the RITK key/value extension, including every
/// per-slice direction and translation in a slice series.
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
        CoordinateMap::SliceSeries(series) => {
            let mut transforms = String::new();
            for (index, transform) in series.transforms().iter().enumerate() {
                if index != 0 {
                    transforms.push(';');
                }
                for row in 0..3 {
                    for column in 0..3 {
                        if row != 0 || column != 0 {
                            transforms.push(',');
                        }
                        write!(transforms, "{}", transform.rotation()[(row, column)])
                            .expect("invariant: writing to a String cannot fail");
                    }
                }
                let translation = transform.translation();
                for component in translation {
                    transforms.push(',');
                    write!(transforms, "{component}")
                        .expect("invariant: writing to a String cannot fail");
                }
            }
            Some(format!(
                "slice_series count={} transforms={transforms}",
                series.len()
            ))
        }
    }
}

pub(crate) fn write_key_value(
    writer: &mut impl std::io::Write,
    map: &CoordinateMap,
) -> std::io::Result<()> {
    match map {
        CoordinateMap::Cartesian => Ok(()),
        CoordinateMap::CurvilinearArray(geometry) => writeln!(
            writer,
            "{COORDINATE_MAP_KEY}:=curvilinear radius_sample_size={} first_sample_distance={} lateral_angular_separation={} first_lateral_angle={}",
            geometry.radius_sample_size(),
            geometry.first_sample_distance(),
            geometry.lateral_angular_separation(),
            geometry.first_lateral_angle()
        ),
        CoordinateMap::PhasedArray3D(geometry) => writeln!(
            writer,
            "{COORDINATE_MAP_KEY}:=phased_array_3d radius_sample_size={} first_sample_distance={} azimuth_angular_separation={} elevation_angular_separation={} first_azimuth_angle={} first_elevation_angle={}",
            geometry.radius_sample_size(),
            geometry.first_sample_distance(),
            geometry.azimuth_angular_separation(),
            geometry.elevation_angular_separation(),
            geometry.first_azimuth_angle(),
            geometry.first_elevation_angle()
        ),
        CoordinateMap::SliceSeries(series) => {
            write!(writer, "{COORDINATE_MAP_KEY}:=slice_series count={} transforms=", series.len())?;
            for (index, transform) in series.transforms().iter().enumerate() {
                if index != 0 {
                    writer.write_all(b";")?;
                }
                for row in 0..3 {
                    for column in 0..3 {
                        if row != 0 || column != 0 {
                            writer.write_all(b",")?;
                        }
                        write!(writer, "{}", transform.rotation()[(row, column)])?;
                    }
                }
                for component in transform.translation() {
                    write!(writer, ",{}", component)?;
                }
            }
            writer.write_all(b"\n")
        }
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
    decode_with_depth(value, None)
}

fn decode_with_depth(value: &str, expected_depth: Option<usize>) -> Result<CoordinateMap> {
    let mut parts = value.split_whitespace();
    let tag = parts
        .next()
        .ok_or_else(|| anyhow!("empty {COORDINATE_MAP_KEY} value"))?;
    let mut params = [("", ""); MAX_COORDINATE_MAP_PARAMETERS];
    let mut param_count = 0;
    for token in parts {
        let (name, raw) = token
            .split_once('=')
            .ok_or_else(|| anyhow!("malformed {COORDINATE_MAP_KEY} parameter '{token}'"))?;
        if name.is_empty() || raw.is_empty() {
            bail!("malformed {COORDINATE_MAP_KEY} parameter '{token}'");
        }
        if params[..param_count]
            .iter()
            .any(|(existing, _)| *existing == name)
        {
            bail!("duplicate {COORDINATE_MAP_KEY} parameter '{name}'");
        }
        if param_count == params.len() {
            bail!("{COORDINATE_MAP_KEY} has more than {MAX_COORDINATE_MAP_PARAMETERS} parameters");
        }
        params[param_count] = (name, raw);
        param_count += 1;
    }
    let params = &params[..param_count];

    let get = |name: &str| -> Result<f64> {
        parameter(params, tag, name)?
            .parse::<f64>()
            .map_err(|error| anyhow!("{COORDINATE_MAP_KEY} '{name}' is not a number: {error}"))
    };

    match tag {
        "cartesian" => {
            reject_unrecognized_parameters(params, tag, &[])?;
            Ok(CoordinateMap::Cartesian)
        }
        "curvilinear" => {
            reject_unrecognized_parameters(
                params,
                tag,
                &[
                    "radius_sample_size",
                    "first_sample_distance",
                    "lateral_angular_separation",
                    "first_lateral_angle",
                ],
            )?;
            Ok(CoordinateMap::CurvilinearArray(CurvilinearArray::try_new(
                get("radius_sample_size")?,
                get("first_sample_distance")?,
                get("lateral_angular_separation")?,
                get("first_lateral_angle")?,
            )?))
        }
        "phased_array_3d" => {
            reject_unrecognized_parameters(
                params,
                tag,
                &[
                    "radius_sample_size",
                    "first_sample_distance",
                    "azimuth_angular_separation",
                    "elevation_angular_separation",
                    "first_azimuth_angle",
                    "first_elevation_angle",
                ],
            )?;
            Ok(CoordinateMap::PhasedArray3D(PhasedArray3D::try_new(
                get("radius_sample_size")?,
                get("first_sample_distance")?,
                get("azimuth_angular_separation")?,
                get("elevation_angular_separation")?,
                get("first_azimuth_angle")?,
                get("first_elevation_angle")?,
            )?))
        }
        "slice_series" => {
            reject_unrecognized_parameters(params, tag, &["count", "transforms"])?;
            decode_slice_series(
                parameter(params, tag, "count")?,
                parameter(params, tag, "transforms")?,
                expected_depth,
            )
        }
        other => bail!(
            "unknown {COORDINATE_MAP_KEY} tag '{other}'; this file was written by a newer ritk \
             and its geometry cannot be interpreted here"
        ),
    }
}

fn reject_unrecognized_parameters(
    params: &[(&str, &str)],
    tag: &str,
    allowed: &[&str],
) -> Result<()> {
    if let Some((name, _)) = params.iter().find(|(name, _)| !allowed.contains(name)) {
        bail!("{COORDINATE_MAP_KEY} '{tag}' has unrecognized parameter '{name}'");
    }
    Ok(())
}

fn parameter<'a>(params: &'a [(&str, &str)], tag: &str, name: &str) -> Result<&'a str> {
    params
        .iter()
        .find(|(key, _)| *key == name)
        .ok_or_else(|| anyhow!("{COORDINATE_MAP_KEY} '{tag}' is missing '{name}'"))
        .map(|(_, raw)| *raw)
}

fn decode_slice_series(
    count_text: &str,
    encoded: &str,
    expected_depth: Option<usize>,
) -> Result<CoordinateMap> {
    let count = count_text.parse::<usize>().map_err(|error| {
        anyhow!("{COORDINATE_MAP_KEY} slice-series count is not an integer: {error}")
    })?;
    let group_count = encoded.split(';').count();
    if group_count != count {
        return Err(anyhow!(
            "{COORDINATE_MAP_KEY} declares {count} slice transforms but encodes {group_count}"
        ));
    }
    if let Some(expected) = expected_depth
        && count != expected
    {
        return Err(anyhow!(
            "{COORDINATE_MAP_KEY} has {count} slice transforms but volume depth is {expected}"
        ));
    }
    let mut groups = encoded.split(';');
    let mut transforms = Vec::new();
    transforms
        .try_reserve_exact(count)
        .map_err(|error| anyhow!("cannot allocate {count} slice transforms: {error}"))?;
    for index in 0..count {
        let group = groups.next().ok_or_else(|| {
            anyhow!("{COORDINATE_MAP_KEY} has fewer transforms than its declared count {count}")
        })?;
        let mut values = group.split(',');
        let mut matrix = [[0.0; 3]; 3];
        for row in &mut matrix {
            for value in row {
                *value = parse_slice_transform_component(values.next(), index)?;
            }
        }
        let translation = [
            parse_slice_transform_component(values.next(), index)?,
            parse_slice_transform_component(values.next(), index)?,
            parse_slice_transform_component(values.next(), index)?,
        ];
        if values.next().is_some() {
            return Err(anyhow!(
                "{COORDINATE_MAP_KEY} slice transform {index} has more than 12 components"
            ));
        }
        let rotation = Direction::from_rows(matrix);
        transforms.push(SliceTransform::new(rotation, translation));
    }
    if groups.next().is_some() {
        return Err(anyhow!(
            "{COORDINATE_MAP_KEY} has more transforms than its declared count {count}"
        ));
    }
    Ok(CoordinateMap::SliceSeries(SliceSeries::try_new(
        transforms,
    )?))
}

fn parse_slice_transform_component(value: Option<&str>, index: usize) -> Result<f64> {
    let raw = value.ok_or_else(|| {
        anyhow!("{COORDINATE_MAP_KEY} slice transform {index} has fewer than 12 components")
    })?;
    let component = raw.parse::<f64>().map_err(|error| {
        anyhow!("{COORDINATE_MAP_KEY} slice transform {index} has invalid component: {error}")
    })?;
    if !component.is_finite() {
        return Err(anyhow!(
            "{COORDINATE_MAP_KEY} slice transform {index} contains non-finite component {raw}"
        ));
    }
    Ok(component)
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

pub(crate) fn from_header_for_depth<S: ::std::hash::BuildHasher>(
    headers: &std::collections::HashMap<String, String, S>,
    depth: usize,
) -> Result<CoordinateMap> {
    match headers.get(COORDINATE_MAP_KEY) {
        None => Ok(CoordinateMap::Cartesian),
        Some(value) => decode_with_depth(value, Some(depth)),
    }
}

#[cfg(test)]
#[path = "tests_coordinate_map.rs"]
mod tests;
