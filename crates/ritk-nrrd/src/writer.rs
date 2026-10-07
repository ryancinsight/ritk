use anyhow::{anyhow, Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::{Image, ImageMetadata};
use ritk_image_io::{validate_coordinate_map, validate_physical_geometry, SeriesAxis};
use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::spatial::file_space_directions_from_internal;

mod compute;
mod formatting;
mod stored;

pub(super) use compute::write_nrrd_flat;
pub use compute::{write_nrrd, write_nrrd_with_data, NrrdWriter};
pub(super) use formatting::HeaderBuffer;
use formatting::{direction_row_major, format_nrrd_vector, write_float_payload};
pub(crate) use stored::{
    nrrd_type_name, validate_intensity_semantics, validate_series_axis,
    validate_series_header_entries, validate_series_intensity_unit, write_sample_payload,
};
pub use stored::{write_nrrd_stored, write_nrrd_stored_series, NrrdStoredWriteError};

#[derive(Clone, Copy)]
pub(super) enum SeriesLayout {
    AcquisitionFastest,
    AcquisitionSlowest,
}

pub(super) fn write_nrrd_header(
    writer: &mut impl Write,
    shape: [usize; 3],
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    element_type: &str,
    coordinate_map: &CoordinateMap,
) -> std::io::Result<()> {
    write_nrrd_header_with_metadata(
        writer,
        shape,
        spacing,
        origin,
        direction,
        element_type,
        coordinate_map,
        None,
        &[],
        &[],
    )
}

pub(super) fn write_nrrd_header_with_metadata(
    writer: &mut impl Write,
    shape: [usize; 3],
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    element_type: &str,
    coordinate_map: &CoordinateMap,
    sample_units: Option<&str>,
    comments: &[String],
    records: &[(String, String)],
) -> std::io::Result<()> {
    let [nz, ny, nx] = shape;
    let file_directions = file_space_directions_from_internal(
        [spacing[0], spacing[1], spacing[2]],
        direction_row_major(direction),
    );
    writeln!(writer, "NRRD0004")?;
    writeln!(writer, "# Complete NRRD file written by ritk")?;
    writeln!(writer, "type: {element_type}")?;
    if let Some(sample_units) = sample_units {
        writeln!(writer, "sample units: {sample_units}")?;
    }
    writeln!(writer, "dimension: 3")?;
    // RITK and ITK NRRDs use LPS physical coordinates; declaring RAS would
    // invert the first two physical axes when ITK-compatible readers load it.
    writeln!(writer, "space: left-posterior-superior")?;
    writeln!(writer, "space units: \"mm\" \"mm\" \"mm\"")?;
    writeln!(writer, "sizes: {nx} {ny} {nz}")?;
    writeln!(
        writer,
        "space directions: {} {} {}",
        format_nrrd_vector(file_directions[0]),
        format_nrrd_vector(file_directions[1]),
        format_nrrd_vector(file_directions[2])
    )?;
    writeln!(writer, "kinds: domain domain domain")?;
    writeln!(writer, "endian: little")?;
    writeln!(writer, "encoding: raw")?;
    writeln!(
        writer,
        "space origin: ({},{},{})",
        origin[0], origin[1], origin[2]
    )?;
    crate::coordinate_map::write_key_value(writer, coordinate_map)?;
    for comment in comments {
        writeln!(writer, "{comment}")?;
    }
    for (key, value) in records {
        writeln!(
            writer,
            "{}:={}",
            escape_key_value(key),
            escape_key_value(value)
        )?;
    }
    writeln!(writer)?;
    Ok(())
}

fn escape_key_value(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\n', "\\n")
}

pub(super) fn write_nrrd_series_header(
    writer: &mut impl Write,
    shape: [usize; 3],
    volume_count: usize,
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    element_type: &str,
    coordinate_map: &CoordinateMap,
    layout: SeriesLayout,
    axis: &SeriesAxis,
) -> std::io::Result<()> {
    write_nrrd_series_header_with_metadata(
        writer,
        shape,
        volume_count,
        spacing,
        origin,
        direction,
        element_type,
        coordinate_map,
        layout,
        axis,
        None,
        &[],
        &[],
    )
}

pub(super) fn write_nrrd_series_header_with_metadata(
    writer: &mut impl Write,
    shape: [usize; 3],
    volume_count: usize,
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    element_type: &str,
    coordinate_map: &CoordinateMap,
    layout: SeriesLayout,
    axis: &SeriesAxis,
    sample_units: Option<&str>,
    comments: &[String],
    records: &[(String, String)],
) -> std::io::Result<()> {
    let [nz, ny, nx] = shape;
    let file_directions = file_space_directions_from_internal(
        [spacing[0], spacing[1], spacing[2]],
        direction_row_major(direction),
    );
    let format_version = if matches!(axis, SeriesAxis::Diffusion(_)) {
        "NRRD0005"
    } else {
        "NRRD0004"
    };
    writeln!(writer, "{format_version}")?;
    writeln!(writer, "# Complete NRRD file written by ritk")?;
    writeln!(writer, "type: {element_type}")?;
    if let Some(sample_units) = sample_units {
        writeln!(writer, "sample units: {sample_units}")?;
    }
    writeln!(writer, "dimension: 4")?;
    writeln!(writer, "space: left-posterior-superior")?;
    writeln!(writer, "space units: \"mm\" \"mm\" \"mm\"")?;
    match layout {
        SeriesLayout::AcquisitionFastest => {
            writeln!(writer, "sizes: {volume_count} {nx} {ny} {nz}")?;
            writeln!(
                writer,
                "space directions: none {} {} {}",
                format_nrrd_vector(file_directions[0]),
                format_nrrd_vector(file_directions[1]),
                format_nrrd_vector(file_directions[2])
            )?;
            write_series_axis(writer, axis, layout)?;
        }
        SeriesLayout::AcquisitionSlowest => {
            writeln!(writer, "sizes: {nx} {ny} {nz} {volume_count}")?;
            writeln!(
                writer,
                "space directions: {} {} {} none",
                format_nrrd_vector(file_directions[0]),
                format_nrrd_vector(file_directions[1]),
                format_nrrd_vector(file_directions[2])
            )?;
            write_series_axis(writer, axis, layout)?;
        }
    }
    writeln!(writer, "endian: little")?;
    writeln!(writer, "encoding: raw")?;
    writeln!(
        writer,
        "space origin: ({},{},{})",
        origin[0], origin[1], origin[2]
    )?;
    if matches!(axis, SeriesAxis::Diffusion(_)) {
        writeln!(writer, "measurement frame: (1,0,0) (0,1,0) (0,0,1)")?;
    }
    crate::coordinate_map::write_key_value(writer, coordinate_map)?;
    if let SeriesAxis::Diffusion(scheme) = axis {
        let nominal = scheme
            .directions()
            .iter()
            .map(|entry| entry.weighting().seconds_per_square_millimeter())
            .fold(0.0_f64, f64::max);
        writeln!(writer, "modality:=DWMRI")?;
        writeln!(writer, "DWMRI_b-value:={nominal}")?;
        for (index, entry) in scheme.directions().iter().enumerate() {
            let weighting = entry.weighting().seconds_per_square_millimeter();
            let [x, y, z] = entry.direction().to_array();
            let scale = if weighting == 0.0 {
                0.0
            } else {
                (weighting / nominal).sqrt()
            };
            writeln!(
                writer,
                "DWMRI_gradient_{index:04}:={} {} {}",
                x * scale,
                y * scale,
                z * scale
            )?;
        }
    }
    for comment in comments {
        writeln!(writer, "{comment}")?;
    }
    for (key, value) in records {
        writeln!(
            writer,
            "{}:={}",
            escape_key_value(key),
            escape_key_value(value)
        )?;
    }
    writeln!(writer)?;
    Ok(())
}

fn write_series_axis(
    writer: &mut impl Write,
    axis: &SeriesAxis,
    layout: SeriesLayout,
) -> std::io::Result<()> {
    match axis {
        SeriesAxis::SingleVolume => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "single-volume data cannot use a four-dimensional series header",
        )),
        SeriesAxis::Unspecified => Ok(()),
        SeriesAxis::List | SeriesAxis::Diffusion(_) => match layout {
            SeriesLayout::AcquisitionFastest => {
                writeln!(writer, "kinds: list domain domain domain")
            }
            SeriesLayout::AcquisitionSlowest => {
                writeln!(writer, "kinds: domain domain domain list")
            }
        },
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "NRRD cannot encode this acquisition-axis meaning",
        )),
    }
}

/// Write an acquisition series to a NRRD file.
///
/// # Acquisition axis
///
/// The gradient/time axis is emitted **first**, as `kinds: list domain domain
/// domain` with a leading `none` in `space directions` — the NA-MIC convention
/// Slicer and DTIPrep produce and the one diffusion tooling expects. That axis
/// varies fastest, so volumes are interleaved voxel-by-voxel in the payload.
/// [`read_nrrd_series`](crate::read_nrrd_series) reads either that layout or a
/// trailing acquisition axis.
///
/// A one-volume series writes as an ordinary `dimension: 3` file, identical to
/// [`write_nrrd`], because that is the canonical form for a single volume.
///
/// # Errors
///
/// Returns an error when `volumes` is empty, when any volume's grid differs
/// from the first, or when writing fails.
pub fn write_nrrd_series<B, P>(path: P, volumes: &[Image<f32, B, 3>], backend: &B) -> Result<()>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
    let Some((first, rest)) = volumes.split_first() else {
        return Err(anyhow!(
            "write_nrrd_series: a series requires at least one volume"
        ));
    };

    let shape = first.shape();
    for (index, volume) in rest.iter().enumerate() {
        let position = index + 1;
        if volume.shape() != shape {
            return Err(anyhow!(
                "write_nrrd_series: volume {position} shape {:?} differs from volume 0 \
                 {shape:?}; a NRRD series has one spatial grid",
                volume.shape()
            ));
        }
        if volume.origin() != first.origin()
            || volume.spacing() != first.spacing()
            || volume.direction() != first.direction()
        {
            return Err(anyhow!(
                "write_nrrd_series: volume {position} physical geometry differs from \
                 volume 0; a NRRD series has one spatial grid"
            ));
        }
        if volume.coordinate_map() != first.coordinate_map() {
            return Err(anyhow!(
                "write_nrrd_series: volume {position} coordinate map differs from \
                 volume 0; a NRRD series has one coordinate map"
            ));
        }
    }

    let payloads: Vec<_> = volumes
        .iter()
        .map(|volume| volume.data_cow_on(backend))
        .collect();

    write_nrrd_series_flat(
        path.as_ref(),
        shape,
        first.spacing(),
        first.origin(),
        first.direction(),
        &payloads,
        first.coordinate_map(),
    )
}

fn write_nrrd_series_flat(
    path: &Path,
    shape: [usize; 3],
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    payloads: &[impl std::ops::Deref<Target = [f32]>],
    coordinate_map: &CoordinateMap,
) -> Result<()> {
    // One volume has no acquisition axis to declare, so it takes the ordinary
    // rank-3 path and stays byte-identical to `write_nrrd`.
    if let [single] = payloads {
        return write_nrrd_flat(
            path,
            shape,
            spacing,
            origin,
            direction,
            single,
            coordinate_map,
        );
    }

    let [nz, ny, nx] = shape;
    let voxels_per_volume = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .ok_or_else(|| anyhow!("NRRD shape [{nz}, {ny}, {nx}] voxel count overflows usize"))?;
    for (position, payload) in payloads.iter().enumerate() {
        if payload.len() != voxels_per_volume {
            return Err(anyhow!(
                "write_nrrd_series: volume {position} has {} voxels but shape \
                 [{nz}, {ny}, {nx}] requires {voxels_per_volume}",
                payload.len()
            ));
        }
    }

    validate_physical_geometry(&ImageMetadata::new(*origin, *spacing, *direction))?;
    validate_coordinate_map(coordinate_map, shape)?;

    let mut header = HeaderBuffer::new();
    let header_result = write_nrrd_series_header(
        &mut header,
        shape,
        payloads.len(),
        spacing,
        origin,
        direction,
        "float",
        coordinate_map,
        SeriesLayout::AcquisitionFastest,
        &SeriesAxis::List,
    );
    if header.exceeded_limit() {
        let maximum_bytes = crate::reader::MAX_HEADER_BYTES;
        anyhow::bail!("NRRD output header is larger than {maximum_bytes} bytes");
    }
    header_result?;

    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create NRRD file {:?}", path))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(header.bytes())?;

    // The acquisition axis varies fastest, so voxel i of every volume is
    // written before voxel i+1 of any of them. A bulk per-volume write is not
    // available in this layout; the interleave is the format's own ordering.
    let mut interleaved = Vec::with_capacity(payloads.len() * voxels_per_volume);
    for voxel in 0..voxels_per_volume {
        for payload in payloads {
            interleaved.push(payload[voxel]);
        }
    }
    write_float_payload(&mut writer, &interleaved)?;

    writer.flush().context("Failed to flush NRRD output file")?;
    Ok(())
}

#[cfg(test)]
#[path = "tests/writer_geometry.rs"]
mod geometry_tests;
