use anyhow::{anyhow, bail, Context, Result};
use coeus_core::ComputeBackend;
use consus_core::ByteOrder;
use flate2::bufread::MultiGzDecoder;
use ritk_codecs::parse_header_values;
use ritk_codecs::sample::{
    count_payload_bytes, validate_remaining_payload, Conversion, Sample, SampleBuffer, SampleType,
};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use super::decode::{
    parse_endian, parse_nrrd_point, parse_nrrd_point_planar, parse_space_direction_slots,
    parse_space_directions, parse_space_directions_planar,
};
use super::header::parse_nrrd_header_map_from_reader;
use crate::axes::{locate_acquisition_axis, AcquisitionAxis};
use crate::spatial::{metadata_from_file_space_directions, metadata_from_file_spacings};
use crate::types::sample_type_from_name;

/// Decode of a NRRD file into one flat `[Z, Y, X]` volume per acquisition,
/// sharing one spatial grid.
struct DecodedNrrd<T> {
    volumes: Vec<Vec<T>>,
    dims: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
    /// Acquisition geometry from the header's key/value field; `Cartesian`
    /// when absent, which is what every pre-existing NRRD means.
    coordinate_map: ritk_spatial::CoordinateMap,
}

impl<T> DecodedNrrd<T> {
    /// Take the sole volume, rejecting a series.
    ///
    /// The single-volume reader carries a `[nz, ny, nx]` contract, so a series
    /// has no correct representation through it; returning volume 0 would
    /// discard the rest of the acquisition while reporting success.
    fn into_single_volume(mut self) -> Result<Self> {
        if self.volumes.len() != 1 {
            return Err(anyhow!(
                "NRRD file declares {} volumes along its acquisition axis; this reader \
                 returns one 3-D volume. Use the series reader to decode an acquisition \
                 series (diffusion, time series) without discarding {} of its volumes.",
                self.volumes.len(),
                self.volumes.len() - 1
            ));
        }
        self.volumes.truncate(1);
        Ok(self)
    }
}

/// Read a NRRD (Nearly Raw Raster Data) file into a 3-D `Image` of `T`.
///
/// # Axis convention
/// NRRD files produced by ITK-compatible tools store voxels in `[X, Y, Z]`
/// order with X as the fastest-varying raw axis. That flat raw order is the
/// same byte sequence as a RITK tensor shaped `[Z, Y, X]`, so the returned
/// tensor is constructed directly with shape `[nz, ny, nx]`.
///
/// # Spatial metadata
/// Direction and spacing are derived from `space directions` when that field
/// is present. NRRD file-axis vectors `[x,y,z]` are reordered into RITK
/// metadata columns `[depth,row,col] = [z,y,x]`. If only `spacings` is present,
/// the scalar spacings follow the same axis reorder with axis-aligned
/// directions.
///
/// # Encoding
/// `raw` and `gzip` (`gz`) encodings are supported; any other encoding returns
/// an error with an actionable message.
///
/// # Supported types
/// Every numeric type of the NRRD specification, under each of its names:
/// `signed char`, `unsigned char`, `short`, `unsigned short`, `int`,
/// `unsigned int`, `long long int`, `unsigned long long int`, `float`, and
/// `double`. The samples decode in the stored type, then convert to `T` under
/// `conversion`: [`Exact`](ritk_codecs::sample::Exact) accepts the stored type
/// or a type it widens to and refuses anything else, and
/// [`Cast`](ritk_codecs::sample::Cast) converts with a warning. `block` is not
/// a numeric type and is rejected.
///
/// # Byte order
/// The `endian` field is `big` or `little`. It is required when the sample
/// type is wider than one byte, because the payload is raw or gzip-compressed
/// binary whose order the file must state; the specification (section 5,
/// field `endian`) requires it exactly then, and Teem `formatNRRD.c` refuses
/// such a file with "require endian info". An absent field is accepted for a
/// one-byte type, which has no byte order. Any other value is an error.
///
/// # Inline vs. detached data
/// * Inline: no `data file` field (or `data file: INTERNAL`) — binary data
///   follows the blank header-terminator line in the same file.
/// * Detached: `data file: <filename>` — binary data is in a separate file
///   resolved relative to the NRRD header file's directory.
///
/// # Errors
///
/// Returns an error when the file cannot be opened or read, the header is
/// invalid, the `type`, `endian`, or `encoding` value is unsupported, the
/// `endian` field is absent for a sample type wider than one byte, a detached
/// `data file` cannot be opened, the payload is shorter than the header
/// declares, the file declares more than one acquisition volume, or
/// `conversion` refuses the stored type.
pub fn read_nrrd<T, C, B, P>(path: P, backend: &B, conversion: C) -> Result<Image<T, B, 3>>
where
    T: Sample,
    C: Conversion,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let decoded = decode_nrrd(path, conversion)?.into_single_volume()?;
    let DecodedNrrd {
        dims,
        origin,
        spacing,
        direction,
        coordinate_map,
        volumes,
    } = decoded;
    Image::from_flat_on(
        volumes
            .into_iter()
            .next()
            .expect("single_volume guaranteed"),
        dims,
        origin,
        spacing,
        direction,
        backend,
    )?
    .with_coordinate_map(coordinate_map)
}

/// Read a NRRD acquisition series as one image of `T` per volume.
///
/// # Acquisition axis
///
/// A 4-D NRRD carries one non-spatial axis — the diffusion gradient index of a
/// DWI file, a functional timepoint. Unlike NIfTI, NRRD does not fix its
/// position: the NA-MIC convention Slicer and DTIPrep emit places it first
/// (fastest, volumes interleaved voxel-by-voxel), while other tools place it
/// last (slowest, volumes contiguous). Both are read here, located through
/// `kinds` or the `none` slot in `space directions`.
///
/// Every returned image shares the file's single spatial grid, in acquisition
/// order. A 2-D or 3-D file is a one-volume series, so this reader accepts an
/// ordinary volume; [`read_nrrd`] does not accept the converse, rejecting a
/// series rather than returning its first volume.
///
/// The stored samples convert to `T` under `conversion`, as in [`read_nrrd`].
///
/// # Errors
///
/// Returns an error when the file cannot be opened or read, when the header is
/// invalid, when the `type`, `endian`, or `encoding` value is unsupported, when
/// the acquisition axis is absent or in an unsupported position on a 4-D file,
/// when the `endian` field is absent for a sample type wider than one byte,
/// when a detached `data file` cannot be opened, when the payload is shorter
/// than the declared sizes require, when a gzip checksum is invalid, when a
/// gzip stream expands beyond the declared sizes, or when `conversion` refuses
/// the stored type. Raw bytes beyond the declared sizes are not read.
pub fn read_nrrd_series<T, C, B, P>(
    path: P,
    backend: &B,
    conversion: C,
) -> Result<Vec<Image<T, B, 3>>>
where
    T: Sample,
    C: Conversion,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let DecodedNrrd {
        volumes,
        dims,
        origin,
        spacing,
        direction,
        coordinate_map,
    } = decode_nrrd(path, conversion)?;

    volumes
        .into_iter()
        .map(|data| {
            Image::from_flat_on(data, dims, origin, spacing, direction, backend)?
                .with_coordinate_map(coordinate_map.clone())
        })
        .collect()
}

fn decode_nrrd<T: Sample, C: Conversion, P: AsRef<Path>>(
    path: P,
    conversion: C,
) -> Result<DecodedNrrd<T>> {
    let path = path.as_ref();

    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open NRRD file {:?}", path))?;
    let mut reader = BufReader::new(file);

    let headers = parse_nrrd_header_map_from_reader(&mut reader)?;

    let element_type = headers
        .get("type")
        .ok_or_else(|| anyhow!("Missing 'type' in NRRD header"))?
        .clone();

    let dimension: usize = headers
        .get("dimension")
        .ok_or_else(|| anyhow!("Missing 'dimension' in NRRD header"))?
        .parse()
        .context("NRRD 'dimension' is not a valid integer")?;

    if !(2..=4).contains(&dimension) {
        return Err(anyhow!(
            "Expected dimension between 2 and 4 for a NRRD file, found {}",
            dimension
        ));
    }

    let direction_slots = if dimension == 2 {
        None
    } else {
        headers
            .get("space directions")
            .map(|s| parse_space_direction_slots(s))
            .transpose()?
    };
    let direction_flags: Option<Vec<bool>> = direction_slots
        .as_ref()
        .map(|slots| slots.iter().map(Option::is_some).collect());
    let acquisition = locate_acquisition_axis(
        dimension,
        headers.get("kinds").map(String::as_str),
        direction_flags.as_deref(),
    )?;

    let sizes_str = headers
        .get("sizes")
        .ok_or_else(|| anyhow!("Missing 'sizes' in NRRD header"))?;
    let sizes = parse_header_values::<usize>(sizes_str, "sizes", dimension)?;

    let (volumes, spatial_sizes): (usize, &[usize]) = match acquisition {
        AcquisitionAxis::Absent => (1, &sizes[..]),
        AcquisitionAxis::Fastest => (sizes[0], &sizes[1..]),
        AcquisitionAxis::Slowest => (sizes[3], &sizes[..3]),
    };
    if volumes == 0 {
        return Err(anyhow!(
            "NRRD acquisition axis declares zero volumes; 'sizes' must be positive"
        ));
    }

    let nx = spatial_sizes[0];
    let ny = spatial_sizes[1];
    let nz = if spatial_sizes.len() >= 3 {
        spatial_sizes[2]
    } else {
        1
    };

    let encoding = headers
        .get("encoding")
        .map(|s| s.to_lowercase())
        .unwrap_or_else(|| "raw".to_string());

    let gzipped = match encoding.as_str() {
        "raw" => false,
        "gzip" | "gz" => true,
        other => {
            return Err(anyhow!(
                "Unsupported NRRD encoding '{}'. Supported: 'raw', 'gzip'.",
                other
            ));
        }
    };

    let spatial = if let Some(sd_str) = headers.get("space directions") {
        let dirs = if dimension == 2 {
            parse_space_directions_planar(sd_str)?
        } else {
            parse_space_directions(sd_str)?
        };
        metadata_from_file_space_directions(dirs)?
    } else if let Some(sp_str) = headers.get("spacings") {
        let sp = parse_header_values::<f64>(sp_str, "spacings", dimension)?;
        let sp: Vec<f64> = match acquisition {
            AcquisitionAxis::Absent => sp,
            AcquisitionAxis::Fastest => sp[1..].to_vec(),
            AcquisitionAxis::Slowest => sp[..3].to_vec(),
        };
        let sz = if sp.len() >= 3 { sp[2] } else { 1.0 };
        metadata_from_file_spacings([sp[0], sp[1], sz])?
    } else {
        metadata_from_file_spacings([1.0, 1.0, 1.0])?
    };

    let origin = if let Some(so_str) = headers.get("space origin") {
        if dimension == 2 {
            parse_nrrd_point_planar(so_str)?
        } else {
            parse_nrrd_point(so_str)?
        }
    } else {
        Point::new([0.0, 0.0, 0.0])
    };

    let voxels_per_volume = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .ok_or_else(|| anyhow!("NRRD sizes [{nx}, {ny}, {nz}] voxel count overflows usize"))?;
    let total_voxels = voxels_per_volume
        .checked_mul(volumes)
        .ok_or_else(|| anyhow!("NRRD series element count overflows usize"))?;
    let sample_type = sample_type_from_name(&element_type)?;
    let element_size = sample_type.byte_width();
    let byte_order = match headers.get("endian") {
        Some(value) => parse_endian(value)?,
        None if element_size == 1 => ByteOrder::LittleEndian,
        None => bail!(
            "NRRD type '{element_type}' is {element_size} bytes wide, so the 'endian' field is required to read its '{encoding}' payload"
        ),
    };
    let expected_payload_bytes = total_voxels.checked_mul(element_size).ok_or_else(|| {
        anyhow!("NRRD byte count overflows usize: {total_voxels} voxels x {element_size} bytes")
    })?;

    let payload = Payload {
        sample_type,
        byte_order,
        count: total_voxels,
        expected_bytes: expected_payload_bytes,
        gzipped,
    };
    let buffer = match headers.get("data file") {
        Some(data_file) if !data_file.eq_ignore_ascii_case("INTERNAL") => {
            let raw_path = path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(data_file);
            let file = std::fs::File::open(&raw_path)
                .with_context(|| format!("Cannot read NRRD data file {:?}", raw_path))?;
            payload.read(BufReader::new(file))
        }
        _ => payload.read(reader),
    }
    .with_context(|| {
        format!(
            "Cannot read {encoding}-encoded NRRD payload: {total_voxels} {sample_type} \
             samples need {expected_payload_bytes} bytes"
        )
    })?;
    let samples: Vec<T> = conversion.convert(buffer)?;

    let mut volume_data = Vec::new();
    volume_data
        .try_reserve_exact(volumes)
        .context("cannot allocate NRRD volume table")?;
    for _ in 0..volumes {
        let mut volume = Vec::new();
        volume
            .try_reserve_exact(voxels_per_volume)
            .context("cannot allocate decoded NRRD volume")?;
        volume_data.push(volume);
    }
    for (flat_index, value) in samples.into_iter().enumerate() {
        let volume = match acquisition {
            AcquisitionAxis::Fastest => flat_index % volumes,
            AcquisitionAxis::Absent | AcquisitionAxis::Slowest => flat_index / voxels_per_volume,
        };
        volume_data[volume].push(value);
    }

    Ok(DecodedNrrd {
        volumes: volume_data,
        dims: [nz, ny, nx],
        origin,
        spacing: spatial.spacing,
        direction: spatial.direction,
        coordinate_map: crate::coordinate_map::from_header(&headers)?,
    })
}

/// How a NRRD payload is laid out in its stream.
struct Payload {
    sample_type: SampleType,
    byte_order: ByteOrder,
    /// Samples across every volume.
    count: usize,
    /// Exact number of bytes the header's count and stored type require.
    expected_bytes: usize,
    gzipped: bool,
}

impl Payload {
    /// Decode `count` samples from `stream`, inflating it first when gzipped.
    ///
    /// Validate the payload length before allocating or decoding typed samples.
    ///
    /// Compressed input is inflated into a bounded sink on the first pass; the
    /// second pass decodes into the typed buffer and verifies the gzip trailer.
    fn read<R: BufRead + Seek>(&self, mut stream: R) -> Result<SampleBuffer> {
        let start = stream.stream_position()?;
        if self.gzipped {
            let actual = {
                let mut decoder = MultiGzDecoder::new(&mut stream);
                count_payload_bytes(&mut decoder, self.expected_bytes)?
            };
            match actual {
                None => bail!("gzip NRRD payload expands beyond the declared sample count"),
                Some(actual) if actual != self.expected_bytes => {
                    bail!(
                        "gzip NRRD payload expands to {actual} bytes; the declared sample count needs {} bytes",
                        self.expected_bytes
                    );
                }
                Some(_) => {}
            }
            stream.seek(SeekFrom::Start(start))?;
            let mut decoder = MultiGzDecoder::new(stream);
            let samples = SampleBuffer::read_from(
                &mut decoder,
                self.sample_type,
                self.byte_order,
                self.count,
            )?;
            let mut excess = [0_u8; 1];
            if decoder.read(&mut excess)? != 0 {
                bail!("gzip NRRD payload expands beyond the declared sample count");
            }
            Ok(samples)
        } else {
            validate_remaining_payload(&mut stream, self.expected_bytes)?;
            stream.seek(SeekFrom::Start(start))?;
            SampleBuffer::read_from(&mut stream, self.sample_type, self.byte_order, self.count)
                .map_err(Into::into)
        }
    }
}

/// Thin reader struct for NRRD files.
///
/// The backend `B` and device are supplied per-call so a single `NrrdReader`
/// instance can serve multiple backends.
pub struct NrrdReader;

impl NrrdReader {
    /// Read a NRRD file at `path` into an [`Image`] of `T` on `backend`,
    /// converting the stored samples under `conversion` (see [`read_nrrd`]).
    ///
    /// # Errors
    ///
    /// Returns the errors of [`read_nrrd`]: a header or detached data file that
    /// cannot be opened or read, an invalid header, an unsupported
    /// `type`, `encoding`, or `endian`, a missing `endian` on a type wider than
    /// one byte, a payload shorter than the header declares, a file with more
    /// than one volume, or a stored type `conversion` refuses.
    pub fn read<T, C, B, P>(&self, path: P, backend: &B, conversion: C) -> Result<Image<T, B, 3>>
    where
        T: Sample,
        C: Conversion,
        B: ComputeBackend,
        P: AsRef<Path>,
    {
        read_nrrd(path, backend, conversion)
    }
}
