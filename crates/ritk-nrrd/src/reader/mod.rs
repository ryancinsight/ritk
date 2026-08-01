mod decode;
mod diffusion;
use decode::*;

use crate::axes::{locate_acquisition_axis, AcquisitionAxis};
use crate::spatial::{metadata_from_file_space_directions, metadata_from_file_spacings};
use anyhow::{anyhow, Context, Result};
use coeus_core::ComputeBackend;
use ritk_codecs::{parse_f64_vec, parse_usize_vec, ByteOrder};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Decode of a NRRD file into one flat `[Z, Y, X]` volume per acquisition,
/// sharing one spatial grid.
struct DecodedNrrd {
    volumes: Vec<Vec<f32>>,
    dims: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
}

impl DecodedNrrd {
    /// Take the sole volume, rejecting a series.
    ///
    /// The single-volume reader carries a `[nz, ny, nx]` contract, so a series
    /// has no correct representation through it; returning volume 0 would
    /// discard the rest of the acquisition while reporting success.
    fn into_single_volume(mut self) -> Result<DecodedNrrd> {
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

    /// The sole volume's voxels, after [`Self::into_single_volume`].
    fn single_volume_data(mut self) -> Vec<f32> {
        self.volumes
            .pop()
            .expect("invariant: single_volume_data follows into_single_volume")
    }
}

/// Read a NRRD (Nearly Raw Raster Data) file into a 3-D `Image`.
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
/// `float`, `double`, `short`, `unsigned short`, `int`, `unsigned int`,
/// `uchar` / `unsigned char`, `char` / `signed char`.
/// All are converted to `f32` in the tensor.
///
/// # Inline vs. detached data
/// * Inline: no `data file` field (or `data file: INTERNAL`) — binary data
///   follows the blank header-terminator line in the same file.
/// * Detached: `data file: <filename>` — binary data is in a separate file
///   resolved relative to the NRRD header file's directory.
pub fn read_nrrd<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<Image<f32, B, 3>> {
    let decoded = decode_nrrd(path)?.into_single_volume()?;
    let (dims, origin, spacing, direction) = (
        decoded.dims,
        decoded.origin,
        decoded.spacing,
        decoded.direction,
    );

    // NRRD raw order is X-fastest. RITK [Z,Y,X] row-major tensors are also
    // X-fastest in flat memory, so the decoded payload is shaped directly.
    Image::from_flat_on(
        decoded.single_volume_data(),
        dims,
        origin,
        spacing,
        direction,
        backend,
    )
}

/// Read a NRRD acquisition series as one image per volume.
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
/// # Errors
///
/// Returns an error when the header is invalid, when the acquisition axis is
/// absent or in an unsupported position on a 4-D file, or when the payload does
/// not match the declared sizes.
pub fn read_nrrd_series<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<Vec<Image<f32, B, 3>>> {
    let DecodedNrrd {
        volumes,
        dims,
        origin,
        spacing,
        direction,
    } = decode_nrrd(path)?;

    volumes
        .into_iter()
        .map(|data| Image::from_flat_on(data, dims, origin, spacing, direction, backend))
        .collect()
}

fn decode_nrrd<P: AsRef<Path>>(path: P) -> Result<DecodedNrrd> {
    let path = path.as_ref();

    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open NRRD file {:?}", path))?;
    let mut reader = BufReader::new(file);

    let headers = parse_nrrd_header_map_from_reader(&mut reader)?;

    // ── Required fields ───────────────────────────────────────────────────
    let element_type = headers
        .get("type")
        .ok_or_else(|| anyhow!("Missing 'type' in NRRD header"))?
        .clone();

    let dimension: usize = headers
        .get("dimension")
        .ok_or_else(|| anyhow!("Missing 'dimension' in NRRD header"))?
        .parse()
        .context("'dimension' is not a valid integer")?;

    // 2-D NRRD files are promoted to a degenerate `[1, Y, X]` (z = 1) volume,
    // since ritk's `Image` is 3-D. A 4-D file carries three spatial axes plus
    // one acquisition axis.
    if !(2..=4).contains(&dimension) {
        return Err(anyhow!(
            "Expected dimension between 2 and 4 for a NRRD file, found {}",
            dimension
        ));
    }

    // The acquisition axis must be located before `sizes` and `space
    // directions` can be split into spatial and non-spatial parts.
    let direction_slots = headers
        .get("space directions")
        .map(|s| parse_space_direction_slots(s))
        .transpose()?;
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
    let sizes = parse_usize_vec(sizes_str, "sizes", dimension)?;

    // Split `sizes` into the acquisition extent and the three spatial extents,
    // which stay in file order `[x, y, z]` whichever side the acquisition axis
    // sits on.
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

    // ── Encoding ──────────────────────────────────────────────────────────
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
            ))
        }
    };

    // ── Endianness ────────────────────────────────────────────────────────
    // Delegates to the shared `ByteOrder::from_nrrd` constructor in
    // `ritk-codecs::byte_decode`. Unknown / misspelled byte-order strings
    // fall back to little-endian (pre-refactor behavior preserved).
    let endian_str = headers
        .get("endian")
        .map(String::as_str)
        .unwrap_or("little");
    let byte_order = ByteOrder::from_nrrd(endian_str);

    // ── Spacing and direction ─────────────────────────────────────────────
    // 2-D files carry 2-component directions/spacings/origin, promoted with an
    // identity through-plane z-axis (unit z-spacing, zero z-origin).
    let spatial = if let Some(sd_str) = headers.get("space directions") {
        // The acquisition axis contributes a `none` slot, so the spatial
        // vectors are what remains after dropping it.
        let dirs = if dimension == 2 {
            parse_space_directions_planar(sd_str)?
        } else {
            parse_space_directions(sd_str)?
        };
        metadata_from_file_space_directions(dirs)
    } else if let Some(sp_str) = headers.get("spacings") {
        let sp = parse_f64_vec(sp_str, "spacings", dimension)?;
        // `spacings` covers every axis, so drop the acquisition slot the same
        // way `space directions` drops its `none`.
        let sp: Vec<f64> = match acquisition {
            AcquisitionAxis::Absent => sp,
            AcquisitionAxis::Fastest => sp[1..].to_vec(),
            AcquisitionAxis::Slowest => sp[..3].to_vec(),
        };
        let sz = if sp.len() >= 3 { sp[2] } else { 1.0 };
        metadata_from_file_spacings([sp[0], sp[1], sz])
    } else {
        // Neither field present: unit spacing with canonical file-axis order.
        metadata_from_file_spacings([1.0, 1.0, 1.0])
    };

    // ── Origin ────────────────────────────────────────────────────────────
    // `space origin` is a point in the file's physical space, so it always has
    // one component per *space* dimension and never a slot for the acquisition
    // axis — unlike `sizes`, `kinds`, and `space directions`, which are
    // per-axis.
    let origin = if let Some(so_str) = headers.get("space origin") {
        if dimension == 2 {
            parse_nrrd_point_planar(so_str)?
        } else {
            parse_nrrd_point(so_str)?
        }
    } else {
        Point::new([0.0, 0.0, 0.0])
    };

    // ── Binary data ───────────────────────────────────────────────────────
    let voxels_per_volume = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .ok_or_else(|| anyhow!("NRRD sizes [{nx}, {ny}, {nz}] voxel count overflows usize"))?;
    let total_voxels = voxels_per_volume
        .checked_mul(volumes)
        .ok_or_else(|| anyhow!("NRRD series element count overflows usize"))?;
    let (element_size, _, _) = element_type_spec(&element_type)?;
    let expected_payload_bytes = total_voxels.checked_mul(element_size).ok_or_else(|| {
        anyhow!("NRRD byte count overflows usize: {total_voxels} voxels x {element_size} bytes")
    })?;
    let data_file_field = headers.get("data file").cloned();

    // Read the payload bytes (still compressed for gzip encoding) from the
    // inline stream or the detached data file, then gunzip if needed.
    let payload: Vec<u8> = match &data_file_field {
        None => {
            let mut bytes = Vec::new();
            reader
                .read_to_end(&mut bytes)
                .context("Failed to read inline NRRD binary data")?;
            bytes
        }
        Some(df) if df.to_uppercase() == "INTERNAL" => {
            let mut bytes = Vec::new();
            reader
                .read_to_end(&mut bytes)
                .context("Failed to read inline NRRD binary data (INTERNAL)")?;
            bytes
        }
        Some(df) => {
            let raw_path = path.parent().unwrap_or_else(|| Path::new(".")).join(df);
            std::fs::read(&raw_path)
                .with_context(|| format!("Cannot read NRRD data file {:?}", raw_path))?
        }
    };

    let raw_bytes: Vec<u8> = if gzipped {
        let output_limit = u64::try_from(expected_payload_bytes)
            .context("NRRD payload length exceeds u64")?
            .checked_add(1)
            .ok_or_else(|| anyhow!("NRRD payload read limit overflows u64"))?;
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(&payload[..])
            .take(output_limit)
            .read_to_end(&mut out)
            .context("Failed to inflate gzip-encoded NRRD payload")?;
        out
    } else {
        payload
    };

    let f32_data: Vec<f32> =
        decode_element_bytes(&raw_bytes, &element_type, total_voxels, byte_order)?;
    drop(raw_bytes);

    if f32_data.len() != total_voxels {
        return Err(anyhow!(
            "NRRD voxel count mismatch: sizes implies {} voxels but {} were decoded",
            total_voxels,
            f32_data.len()
        ));
    }

    // NRRD raw order is X-fastest. RITK [Z,Y,X] row-major tensors are also
    // X-fastest in flat memory, so each volume's voxels map directly to shape
    // [nz, ny, nx] with no permutation — only the acquisition stride separates
    // one volume's voxels from the next.
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
    for (flat_index, value) in f32_data.into_iter().enumerate() {
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
    })
}

// ── Public reader struct ──────────────────────────────────────────────────────

/// Thin reader struct for NRRD files.
///
/// The backend `B` and device are supplied per-call so a single `NrrdReader`
/// instance can serve multiple backends.
pub struct NrrdReader;

impl NrrdReader {
    /// Read a NRRD file at `path` into an [`Image`] on `device`.
    pub fn read<B: ComputeBackend, P: AsRef<Path>>(
        &self,
        path: P,
        backend: &B,
    ) -> Result<Image<f32, B, 3>> {
        read_nrrd(path, backend)
    }
}

// ── Header-map extraction ────────────────────────────────────────────────────

/// Parse the NRRD header into a key-value map without decoding the payload.
///
/// This is the shared header-parsing path used by [`read_nrrd`],
/// [`read_nrrd_series`], and [`read_nrrd_gradient_scheme`]. Keys are
/// lowercased for case-insensitive lookup. Comment lines (starting with `#`)
/// are skipped.
///
/// # Errors
///
/// Returns an error when the file cannot be opened, when the magic line is
/// absent or invalid, or when header lines cannot be read.
pub fn read_nrrd_header_map<P: AsRef<Path>>(path: P) -> Result<HashMap<String, String>> {
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open NRRD file {:?}", path))?;
    let mut reader = BufReader::new(file);
    parse_nrrd_header_map_from_reader(&mut reader)
}

/// Parse the NRRD header from an already-opened reader.
fn parse_nrrd_header_map_from_reader<R: BufRead>(
    reader: &mut R,
) -> Result<HashMap<String, String>> {
    let mut magic = String::new();
    reader
        .read_line(&mut magic)
        .context("Failed to read NRRD magic line")?;
    if !magic.trim_start().starts_with("NRRD") {
        return Err(anyhow!(
            "Not a valid NRRD file: magic line does not start with 'NRRD' (got '{}')",
            magic.trim()
        ));
    }

    let mut headers: HashMap<String, String> = HashMap::new();
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .context("Error reading NRRD header line")?;
        if n == 0 {
            break; // EOF without blank-line terminator
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if trimmed.starts_with('#') {
            continue;
        }
        if let Some(colon_pos) = trimmed.find(':') {
            let key = trimmed[..colon_pos].trim().to_lowercase();
            let value = trimmed[colon_pos + 1..].trim().to_string();
            headers.insert(key, value);
        }
    }
    Ok(headers)
}

/// Read the diffusion gradient scheme from a NRRD header.
///
/// Extracts `DWMRI_gradient_NNNN` direction keys and `DWMRI_b-value` from the
/// NRRD header and returns a validated [`ritk_diffusion_scheme::GradientScheme`].
/// Directions are in the image axis frame per the NRRD DWI convention.
///
/// # Errors
///
/// Returns an error when the file cannot be opened, the header is missing
/// required DWMRI fields, or the gradient table fails validation.
pub fn read_nrrd_gradient_scheme<P: AsRef<Path>>(
    path: P,
) -> Result<ritk_diffusion_scheme::GradientScheme> {
    let headers = read_nrrd_header_map(path)?;
    diffusion::scheme_from_headers(&headers)
}
