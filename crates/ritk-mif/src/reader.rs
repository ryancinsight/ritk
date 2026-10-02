//! MRtrix `.mif` image reader.
//!
//! Reads the MRtrix3 `.mif` container format — a text header followed by raw
//! binary voxel data.  Both inline (single-file) and detached (`file:` key
//! pointing to `.mif.dat`) layouts are supported.
//!
//! # Data offset
//!
//! The `file` key is `file: <name> <offset>`, and `<offset>` counts bytes from
//! the beginning of the file that holds the data, not from the end of the
//! header.  For an inline file (`<name>` is `.`) the data file is the `.mif`
//! itself, and the offset must lie at or after the end of the `END` line:
//! MRtrix writes it rounded up to a multiple of four, with zero padding
//! between `END` and the data, and refuses an inline offset of 0.  Sources are
//! the MRtrix3 documentation `getting_started/image_data.rst` (key `file`),
//! `core/formats/mrtrix.cpp` (`MRtrix::create`), and
//! `core/formats/mrtrix_utils.cpp` (`get_mrtrix_file_path`).

use crate::header::{
    parse_datatype, parse_dim, parse_layout, parse_mif_header_from_path, parse_transform, parse_vox,
};
use anyhow::{anyhow, bail, Context, Result};
use coeus_core::ComputeBackend;
use consus_core::ByteOrder;
use ritk_codecs::sample::{Conversion, Sample, SampleBuffer, SampleType};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing, Vector};
use std::io::{self, Read, Seek};
use std::path::Path;

/// Decoded `.mif` voxel data: one flat `[Z, Y, X]` volume per frame,
/// sharing one spatial grid.
struct DecodedMif<T> {
    volumes: Vec<Vec<T>>,
    dims: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
}

impl<T> DecodedMif<T> {
    fn into_single_volume(mut self) -> Result<Self> {
        if self.volumes.len() != 1 {
            return Err(anyhow!(
                ".mif file has {} frames; this reader returns one 3-D volume. \
                 Use the series reader for multi-frame files.",
                self.volumes.len()
            ));
        }
        self.volumes.truncate(1);
        Ok(self)
    }

    fn single_volume_data(mut self) -> Vec<T> {
        self.volumes
            .pop()
            .expect("invariant: single_volume_data follows into_single_volume")
    }
}

// ── Public API ──────────────────────────────────────────────────────────

/// Read a `.mif` file into a single 3‑D [`Image`] of `T`.
///
/// The voxels decode in the type and byte order the `datatype` key names
/// (`Int8` through `UInt64`, `Float32`, `Float64`), then convert to `T` under
/// `conversion`: [`Exact`](ritk_codecs::sample::Exact) accepts the stored type
/// or a type it widens to, and [`Cast`](ritk_codecs::sample::Cast) converts
/// with a warning. The header's `scaling` key is not interpreted.
///
/// Rejects multi-frame files; use [`read_mif_series`] for diffusion or
/// time‑series data.
///
/// # Errors
///
/// Returns an error when the file cannot be opened or read, when the header
/// ends before its `END` line, when the header lacks `dim` or `datatype`, when
/// `dim` is malformed or has an axis with no extent, when the voxel count
/// overflows `usize`, when a `vox`,
/// `transform`, or `layout` value is malformed, when the `datatype` is `Bit`,
/// complex, or unknown, when the `datatype` is a one-byte type with a
/// byte-order suffix (`UInt8LE`), when the `file` key is missing or malformed,
/// when it names a detached file by an absolute path or through `..`, or one
/// that cannot be opened, when an inline `file` offset lies
/// inside the header (including the MRtrix-invalid `file: . 0`) or past the end
/// of the file, when the payload is shorter than `dim` requires, when
/// `conversion` refuses the stored type, or when the file has more than one
/// frame.
///
/// # Spatial convention
///
/// The `.mif` `transform` is a 4×4 voxel→scanner (world) affine.  RITK
/// stores the equivalent as `origin` + `spacing` + `direction` through the
/// same decomposition the other format crates use.  When no `transform` key
/// is present, the `vox` sizes produce axis‑aligned spacing and the origin is
/// zero.
pub fn read_mif<T, C, B, P>(path: P, backend: &B, conversion: C) -> Result<Image<T, B, 3>>
where
    T: Sample,
    C: Conversion,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let decoded = decode_mif(path, conversion)?.into_single_volume()?;
    // Extract fields before consuming `decoded` for its data.
    let dims = decoded.dims;
    let origin = decoded.origin;
    let spacing = decoded.spacing;
    let direction = decoded.direction;
    let data = decoded.single_volume_data();
    Image::from_flat_on(data, dims, origin, spacing, direction, backend)
}

/// Read a `.mif` acquisition series as one image per volume.
///
/// Multi‑frame `.mif` files (diffusion, time series) carry one non‑spatial
/// axis — by convention axis 3 — whose extent is the frame count.  A
/// single‑frame file is a one‑volume series.
///
/// Every returned image shares the file's single spatial grid. The voxels
/// decode and convert as [`read_mif`] describes.
///
/// # Errors
///
/// Returns the errors of [`read_mif`] other than the frame-count rejection.
pub fn read_mif_series<T, C, B, P>(
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
    let DecodedMif {
        volumes,
        dims,
        origin,
        spacing,
        direction,
    } = decode_mif(path, conversion)?;

    volumes
        .into_iter()
        .map(|data| Image::from_flat_on(data, dims, origin, spacing, direction, backend))
        .collect()
}

// ── Internal decode ─────────────────────────────────────────────────────

fn decode_mif<T: Sample, C: Conversion, P: AsRef<Path>>(
    path: P,
    conversion: C,
) -> Result<DecodedMif<T>> {
    let path = path.as_ref();
    let (header, mut reader) = parse_mif_header_from_path(path)?;

    // ── Required fields ──────────────────────────────────────────────────
    let dim_str = header
        .entries
        .get("dim")
        .ok_or_else(|| anyhow!("Missing 'dim' in .mif header"))?
        .as_line();
    let dim = parse_dim(dim_str, 3)?;

    let nx = dim[0];
    let ny = if dim.len() > 1 { dim[1] } else { 1 };
    let nz = if dim.len() > 2 { dim[2] } else { 1 };
    let nframes = if dim.len() > 3 { dim[3] } else { 1 };

    // Every axis, not just the frame count: a zero spatial extent makes
    // `voxels_per_volume` zero, and `Vec::chunks(0)` panics. The payload check
    // downstream cannot catch it either, since zero voxels expects zero bytes
    // and a truncated file satisfies that.
    for (axis, extent) in [("x", nx), ("y", ny), ("z", nz), ("frame", nframes)] {
        if extent == 0 {
            return Err(anyhow!(
                ".mif 'dim' declares a {axis} extent of 0; every axis must span at least one voxel"
            ));
        }
    }

    // ── Datatype ──────────────────────────────────────────────────────────
    let dt_str = header
        .entries
        .get("datatype")
        .ok_or_else(|| anyhow!("Missing 'datatype' in .mif header"))?
        .as_line();
    let (sample_type, byte_order) = parse_datatype(dt_str)?;

    // ── Layout ────────────────────────────────────────────────────────────
    let layout = if let Some(layout_val) = header.entries.get("layout") {
        parse_layout(layout_val.as_line())?
    } else {
        // Default contiguous layout: [+0,+1,+2,+3] for 4-D, etc.
        let ndim = if nframes > 1 { 4 } else { 3 };
        (0..ndim).map(|i| i as isize).collect()
    };

    // ── Voxel sizes ──────────────────────────────────────────────────────
    let vox_sizes: Vec<f64> = if let Some(vox_val) = header.entries.get("vox") {
        parse_vox(vox_val.as_line())?
    } else {
        vec![1.0, 1.0, 1.0]
    };

    // ── Spatial metadata (transform) ─────────────────────────────────────
    let (origin, spacing, direction) = if let Some(transform_val) = header.entries.get("transform")
    {
        let matrix = parse_transform(transform_val.as_block())?;
        decompose_transform_affine(&matrix, &vox_sizes)
    } else {
        // No transform: axis-aligned identity direction, zero origin.
        (
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([vox_sizes[0], vox_sizes[1], vox_sizes[2]]),
            Direction::identity(),
        )
    };

    // ── Binary data ──────────────────────────────────────────────────────
    let voxels_per_volume = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .ok_or_else(|| anyhow!(".mif dim [{nx},{ny},{nz}] voxel count overflows usize"))?;
    let total_voxels = voxels_per_volume
        .checked_mul(nframes)
        .ok_or_else(|| anyhow!(".mif series element count overflows usize"))?;

    // `file: <name> [N]`, which MRtrix requires: N counts bytes from the
    // start of the file that holds the data and defaults to 0, as
    // `File::MRtrix::get_mrtrix_file_path` reads it. `.` names this file,
    // whose header occupies the first `header_end` bytes; any other name is a
    // detached file beside this one, read from its byte N.
    let header_end = reader
        .stream_position()
        .context("Cannot locate the end of the .mif header")?;
    let file_spec = header
        .entries
        .get("file")
        .map(|value| value.as_line().to_owned());
    let samples = match file_spec.as_deref().map(str::split_whitespace) {
        None => {
            return Err(anyhow!(
                ".mif header has no 'file' key; MRtrix requires one naming the data file and offset"
            ))
        }
        Some(mut parts) => {
            let Some(name) = parts.next() else {
                return Err(anyhow!("Invalid 'file' key in .mif header: no file name"));
            };
            let offset: u64 = parts
                .next()
                .map(str::parse)
                .transpose()
                .context("Invalid offset in .mif 'file' key")?
                .unwrap_or(0);
            if name == "." {
                read_payload(
                    &mut reader,
                    header_end,
                    offset,
                    sample_type,
                    byte_order,
                    total_voxels,
                )?
            } else {
                let detached = Path::new(name);
                if !detached
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_)))
                {
                    return Err(anyhow!(
                        ".mif 'file' key names '{name}'; a detached data file must be a relative path without '..'"
                    ));
                }
                let data_path = path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(detached);
                let file = std::fs::File::open(&data_path)
                    .with_context(|| format!("Cannot read .mif data file {data_path:?}"))?;
                read_payload(
                    &mut io::BufReader::new(file),
                    0,
                    offset,
                    sample_type,
                    byte_order,
                    total_voxels,
                )?
            }
        }
    };
    let samples = conversion.convert::<T>(samples)?;

    // ── De-interleave frames ─────────────────────────────────────────────
    // MRtrix data is stored with the fastest-varying axis determined by
    // the layout.  For a contiguous 4-D file with layout +0,+1,+2,+3
    // this means axis 3 varies fastest (interleaved by frame).
    let mut volume_data: Vec<Vec<T>> = Vec::with_capacity(nframes);
    for _ in 0..nframes {
        volume_data.push(Vec::with_capacity(voxels_per_volume));
    }

    // Determine the frame stride from the layout.
    // By convention, the non-spatial axis (index 3) has the slowest or
    // fastest stride.  For contiguous data, layout +0,+1,+2,+3 means
    // axes 0,1,2,3 in order, so frame index is the innermost loop
    // (frames are interleaved per-voxel).
    if nframes > 1 && layout.len() >= 4 {
        // axis-3 (frame) varies fastest: for each voxel, iterate frames.
        for chunk in samples.chunks(nframes) {
            for (fi, &val) in chunk.iter().enumerate() {
                volume_data[fi].push(val);
            }
        }
    } else {
        // Single frame or frames-outermost: contiguous volumes.
        for (fi, chunk) in samples.chunks(voxels_per_volume).enumerate() {
            volume_data[fi].extend_from_slice(chunk);
        }
    }

    Ok(DecodedMif {
        volumes: volume_data,
        dims: [nz, ny, nx],
        origin,
        spacing,
        direction,
    })
}

/// Read `count` samples of `sample_type` stored in `byte_order`, starting at
/// the absolute byte `offset` of the file `reader` is positioned in.
///
/// `position` is where `reader` currently stands in that file: the end of the
/// header for the inline file, 0 for a detached one. An offset before
/// `position` points into the header and is refused.
///
/// Both the offset and the count are numbers on header lines, not facts about
/// the file, so neither sizes an allocation: the gap to the offset is discarded
/// through a sink and the samples are read in bounded steps, which makes an
/// overstated value fail as the truncation it is.
fn read_payload<R: Read>(
    reader: &mut R,
    position: u64,
    offset: u64,
    sample_type: SampleType,
    byte_order: ByteOrder,
    count: usize,
) -> Result<SampleBuffer> {
    let Some(gap) = offset.checked_sub(position) else {
        bail!(
            ".mif 'file' key declares offset {offset}, inside the {position}-byte header; \
             the offset counts from the start of the file and must lie after the END line"
        );
    };
    if gap > 0 {
        let skipped = io::copy(&mut reader.by_ref().take(gap), &mut io::sink())
            .context("Failed to skip to the .mif data offset")?;
        if skipped != gap {
            let length = position + skipped;
            bail!(
                ".mif 'file' key declares offset {offset}, past the end of the \
                 {length}-byte file"
            );
        }
    }
    SampleBuffer::read_from(reader, sample_type, byte_order, count).map_err(|error| {
        match error.kind() {
            io::ErrorKind::UnexpectedEof => anyhow!(".mif voxel payload is truncated: {error}"),
            _ => anyhow::Error::new(error).context("Failed to read .mif voxel data"),
        }
    })
}

// ── Transform decomposition ──────────────────────────────────────────────

/// Decompose a 4×4 voxel→scanner affine `[row][col]` into RITK
/// `origin`, `spacing`, and `direction`.
///
/// The transform maps homogeneous voxel coords `[x, y, z, 1]` to scanner
/// coords `[sx, sy, sz, 1]`.  RITK's internal convention is ZYX, so
/// the first three columns are reordered to `(col_z, col_y, col_x)` before
/// decomposition.
fn decompose_transform_affine(
    matrix: &[[f64; 4]; 4],
    vox_sizes: &[f64],
) -> (Point<3>, Spacing<3>, Direction<3>) {
    // Extract the 3×3 linear part and the translation.
    // matrix[row][col]: row 0-2 are the scanner axes, col 0-2 are voxel axes.
    let linear = [
        [matrix[0][0], matrix[0][1], matrix[0][2]], // scanner-x from [vx, vy, vz]
        [matrix[1][0], matrix[1][1], matrix[1][2]], // scanner-y
        [matrix[2][0], matrix[2][1], matrix[2][2]], // scanner-z
    ];

    // Column norms are the spacings.
    let sx = (linear[0][0].powi(2) + linear[1][0].powi(2) + linear[2][0].powi(2)).sqrt();
    let sy = (linear[0][1].powi(2) + linear[1][1].powi(2) + linear[2][1].powi(2)).sqrt();
    let sz = (linear[0][2].powi(2) + linear[1][2].powi(2) + linear[2][2].powi(2)).sqrt();

    // Direction cosines (unit column vectors), reordered ZYX.
    let dz = if sz > 0.0 {
        [linear[0][2] / sz, linear[1][2] / sz, linear[2][2] / sz]
    } else {
        [0.0, 0.0, 1.0]
    };
    let dy = if sy > 0.0 {
        [linear[0][1] / sy, linear[1][1] / sy, linear[2][1] / sy]
    } else {
        [0.0, 1.0, 0.0]
    };
    let dx = if sx > 0.0 {
        [linear[0][0] / sx, linear[1][0] / sx, linear[2][0] / sx]
    } else {
        [1.0, 0.0, 0.0]
    };

    // RITK direction matrix: columns are (dz, dy, dx) in scanner coords.
    let direction = Direction::from_columns([Vector::new(dz), Vector::new(dy), Vector::new(dx)]);

    // Origin: the transform maps voxel [0,0,0,1] to the corner, but
    // RITK origin maps to voxel centre.  The translation column
    // [matrix[0][3], matrix[1][3], matrix[2][3]] maps voxel (0,0,0)
    // directly — this is the corner.  RITK centre origin = corner.
    let origin = Point::new([matrix[0][3], matrix[1][3], matrix[2][3]]);

    let spacing = Spacing::new([
        if sz > 0.0 {
            sz
        } else {
            vox_sizes.get(2).copied().unwrap_or(1.0)
        },
        if sy > 0.0 {
            sy
        } else {
            vox_sizes.get(1).copied().unwrap_or(1.0)
        },
        if sx > 0.0 {
            sx
        } else {
            vox_sizes.first().copied().unwrap_or(1.0)
        },
    ]);

    (origin, spacing, direction)
}

// ── Public reader struct ────────────────────────────────────────────────────

/// Thin reader struct for `.mif` files.
pub struct MifReader;

impl MifReader {
    /// Read a `.mif` file at `path` into an [`Image`] of `T` on `backend`.
    ///
    /// # Errors
    ///
    /// Returns the error of [`read_mif`].
    pub fn read<T, C, B, P>(&self, path: P, backend: &B, conversion: C) -> Result<Image<T, B, 3>>
    where
        T: Sample,
        C: Conversion,
        B: ComputeBackend,
        P: AsRef<Path>,
    {
        read_mif(path, backend, conversion)
    }
}

#[cfg(test)]
#[path = "tests_reader.rs"]
mod tests;
