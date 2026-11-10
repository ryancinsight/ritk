//! MRtrix `.mif` image reader.
//!
//! Reads the MRtrix3 `.mif` container format — a text header followed by raw
//! binary voxel data.  Both inline (single-file) and detached (`file:` key
//! pointing to `.mif.dat`) layouts are supported.

use crate::decode;
use crate::header::{
    parse_datatype, parse_dim, parse_f64_vec, parse_layout, parse_mif_header, parse_transform,
    HeaderValue, MifHeader,
};
use crate::MifError;
use coeus_core::ComputeBackend;
use ritk_codecs::{ImageReadBudget, ImageReadResource};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing, Vector};
use std::io::{BufReader, Read};
use std::path::Path;

/// Decoded `.mif` voxel data: one flat `[Z, Y, X]` volume per frame,
/// sharing one spatial grid.
struct DecodedMif {
    volumes: Vec<Vec<f32>>,
    dims: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
}

impl DecodedMif {
    fn into_single_volume(mut self) -> Result<DecodedMif, MifError> {
        if self.volumes.len() != 1 {
            return Err(MifError::MultiFrame {
                frames: self.volumes.len(),
            });
        }
        self.volumes.truncate(1);
        Ok(self)
    }

    fn single_volume_data(mut self) -> Vec<f32> {
        self.volumes
            .pop()
            .expect("invariant: single_volume_data follows into_single_volume")
    }
}

// ── Public API ──────────────────────────────────────────────────────────

/// Read a `.mif` file into a single 3‑D [`Image`].
///
/// Rejects multi-frame files; use [`read_mif_series`] for diffusion or
/// time‑series data.
///
/// # Spatial convention
///
/// The `.mif` `transform` is a 4×4 voxel→scanner (world) affine.  RITK
/// stores the equivalent as `origin` + `spacing` + `direction` through the
/// same decomposition the other format crates use.  When no `transform` key
/// is present, the `vox` sizes produce axis‑aligned spacing and the origin is
/// zero.
///
/// The frame count, the payload size, and the `f32` voxel storage the header
/// implies are checked against `budget` before the payload is read.
///
/// # Errors
///
/// Returns [`MifError::Budget`] when a declared size exceeds `budget`,
/// [`MifError::MultiFrame`] for a multi-frame file, and a header variant for
/// each malformed or missing key.
pub fn read_mif<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
    budget: ImageReadBudget,
) -> Result<Image<f32, B, 3>, MifError> {
    let decoded = decode_mif(path.as_ref(), budget)?.into_single_volume()?;
    // Extract fields before consuming `decoded` for its data.
    let dims = decoded.dims;
    let origin = decoded.origin;
    let spacing = decoded.spacing;
    let direction = decoded.direction;
    let data = decoded.single_volume_data();
    Image::from_flat_on(data, dims, origin, spacing, direction, backend)
        .map_err(|source| MifError::Image { source })
}

/// Read a `.mif` acquisition series as one image per volume.
///
/// Multi‑frame `.mif` files (diffusion, time series) carry one non‑spatial
/// axis — by convention axis 3 — whose extent is the frame count.  A
/// single‑frame file is a one‑volume series.
///
/// Every returned image shares the file's single spatial grid.
///
/// The frame count is checked against [`ImageReadBudget::max_series_volumes`],
/// and the payload and `f32` voxel storage against the encoded and decoded
/// byte ceilings, before the payload is read.
///
/// # Errors
///
/// Returns [`MifError::Budget`] when a declared size exceeds `budget`, and a
/// header variant for each malformed or missing key.
pub fn read_mif_series<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
    budget: ImageReadBudget,
) -> Result<Vec<Image<f32, B, 3>>, MifError> {
    let DecodedMif {
        volumes,
        dims,
        origin,
        spacing,
        direction,
    } = decode_mif(path.as_ref(), budget)?;

    volumes
        .into_iter()
        .map(|data| {
            Image::from_flat_on(data, dims, origin, spacing, direction, backend)
                .map_err(|source| MifError::Image { source })
        })
        .collect()
}

// ── Internal decode ─────────────────────────────────────────────────────

fn decode_mif(path: &Path, budget: ImageReadBudget) -> Result<DecodedMif, MifError> {
    let file = std::fs::File::open(path).map_err(|source| MifError::Open {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let header = parse_mif_header(&mut reader).map_err(|source| MifError::Header { source })?;

    // ── Required fields ──────────────────────────────────────────────────
    let dim_str = require_entry(&header, "dim")?.as_line();
    let dim = parse_dim(dim_str, 3).map_err(|source| MifError::InvalidField {
        field: "dim",
        source,
    })?;

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
            return Err(MifError::ZeroExtent { axis });
        }
    }

    // ── Datatype ──────────────────────────────────────────────────────────
    let dt_str = require_entry(&header, "datatype")?.as_line();
    let (elem_size, _signed, _float) =
        parse_datatype(dt_str).map_err(|source| MifError::InvalidField {
            field: "datatype",
            source,
        })?;

    // Determine endianness from the datatype string suffix.
    let is_big_endian = dt_str.trim().to_lowercase().ends_with("be");

    // ── Layout ────────────────────────────────────────────────────────────
    let layout = if let Some(layout_val) = header.entries.get("layout") {
        parse_layout(layout_val.as_line()).map_err(|source| MifError::InvalidField {
            field: "layout",
            source,
        })?
    } else {
        // Default contiguous layout: [+0,+1,+2,+3] for 4-D, etc.
        let ndim = if nframes > 1 { 4 } else { 3 };
        (0..ndim).map(|i| i as isize).collect()
    };

    // ── Voxel sizes ──────────────────────────────────────────────────────
    let vox_sizes: Vec<f64> = if let Some(vox_val) = header.entries.get("vox") {
        let v = parse_f64_vec(vox_val.as_line()).map_err(|source| MifError::InvalidField {
            field: "vox",
            source,
        })?;
        if v.len() < 3 {
            return Err(MifError::TooFewVoxelSizes { found: v.len() });
        }
        v
    } else {
        vec![1.0, 1.0, 1.0]
    };

    // ── Spatial metadata (transform) ─────────────────────────────────────
    let (origin, spacing, direction) = if let Some(transform_val) = header.entries.get("transform")
    {
        let matrix =
            parse_transform(transform_val.as_block()).map_err(|source| MifError::InvalidField {
                field: "transform",
                source,
            })?;
        decompose_transform_affine(&matrix, &vox_sizes)
    } else {
        // No transform: axis-aligned identity direction, zero origin.
        //
        // The `vox:` field is X, Y, Z (MRtrix3), so it arrives as
        // `[Δcol, Δrow, Δdepth]` and must be reversed into RITK
        // `[Δdepth, Δrow, Δcol]` — the same order `decompose_transform_affine`
        // derives from a transform block. Reading the three components through
        // index-for-index transposes the spacing, and the round-trip suite
        // cannot see it: `write_mif` always emits a `transform:`, so this
        // branch is reached only by files this crate did not write.
        (
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([vox_sizes[2], vox_sizes[1], vox_sizes[0]]),
            Direction::identity(),
        )
    };

    // ── Binary data ──────────────────────────────────────────────────────
    let voxels_per_volume = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .ok_or(MifError::VoxelCountOverflow { nx, ny, nz })?;
    let total_voxels = voxels_per_volume
        .checked_mul(nframes)
        .ok_or(MifError::SeriesCountOverflow)?;
    let expected_bytes =
        total_voxels
            .checked_mul(elem_size)
            .ok_or(MifError::ByteCountOverflow {
                count: total_voxels,
                elem_size,
            })?;

    // `dim` and `datatype` are claims about the file, so the sizes they imply
    // are bounded before any payload byte is read.
    budget.check_len(ImageReadResource::SeriesVolumes, nframes)?;
    budget.check_len(ImageReadResource::EncodedBytes, expected_bytes)?;
    budget.check_len(
        ImageReadResource::DecodedBytes,
        total_voxels.saturating_mul(size_of::<f32>()),
    )?;

    let raw_bytes: Vec<u8> = match header.entries.get("file") {
        // Inline data right after END, read from the current reader position.
        None => read_payload(reader, 0, expected_bytes, path)?,
        Some(file_val) => {
            // "file: . 2" means data starts at byte offset 2 after the header.
            // "file: sub-01_dwi.mif.dat 0" means a detached file.
            let file_spec = file_val.as_line();
            let parts: Vec<&str> = file_spec.split_whitespace().collect();
            let [file_name, offset, ..] = parts.as_slice() else {
                return Err(MifError::InvalidFileKey {
                    spec: file_spec.to_owned(),
                });
            };
            let offset: u64 = offset
                .parse()
                .map_err(|source| MifError::InvalidFileOffset { source })?;
            if *file_name == "." {
                read_payload(reader, offset, expected_bytes, path)?
            } else {
                let data_path = path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(file_name);
                let data_file =
                    std::fs::File::open(&data_path).map_err(|source| MifError::PayloadRead {
                        path: data_path.clone(),
                        source,
                    })?;
                read_payload(
                    BufReader::new(data_file),
                    offset,
                    expected_bytes,
                    &data_path,
                )?
            }
        }
    };

    // ── Decode to f32 ────────────────────────────────────────────────────
    let f32_data = decode::decode_bytes(&raw_bytes, elem_size, total_voxels, is_big_endian)?;

    // ── De-interleave frames ─────────────────────────────────────────────
    // MRtrix data is stored with the fastest-varying axis determined by
    // the layout.  For a contiguous 4-D file with layout +0,+1,+2,+3
    // this means axis 3 varies fastest (interleaved by frame).
    let mut volume_data: Vec<Vec<f32>> = Vec::with_capacity(nframes);
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
        for chunk in f32_data.chunks(nframes) {
            for (fi, &val) in chunk.iter().enumerate() {
                volume_data[fi].push(val);
            }
        }
    } else {
        // Single frame or frames-outermost: contiguous volumes.
        for (fi, chunk) in f32_data.chunks(voxels_per_volume).enumerate() {
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

fn require_entry<'h>(
    header: &'h MifHeader,
    field: &'static str,
) -> Result<&'h HeaderValue, MifError> {
    header
        .entries
        .get(field)
        .ok_or(MifError::MissingField { field })
}

/// Skips `offset` bytes of `source`, then reads exactly `expected_bytes`.
///
/// The offset is a number on a header line, not a fact about the file, so it
/// is discarded through a sink instead of sizing a buffer: `file: . 4000000000`
/// in a 300-byte file once zero-filled 4 GB and succeeded. Copying is bounded
/// by the bytes that arrive, so an overstated offset fails as
/// [`MifError::OffsetOverrun`]. Reading stops at `expected_bytes`, which has
/// already passed the read budget.
fn read_payload(
    mut source: impl Read,
    offset: u64,
    expected_bytes: usize,
    path: &Path,
) -> Result<Vec<u8>, MifError> {
    let read_error = |source| MifError::PayloadRead {
        path: path.to_path_buf(),
        source,
    };
    let skipped = std::io::copy(&mut source.by_ref().take(offset), &mut std::io::sink())
        .map_err(read_error)?;
    if skipped != offset {
        return Err(MifError::OffsetOverrun {
            offset,
            available: skipped,
        });
    }
    let mut bytes = Vec::new();
    source
        .take(u64::try_from(expected_bytes).unwrap_or(u64::MAX))
        .read_to_end(&mut bytes)
        .map_err(read_error)?;
    if bytes.len() < expected_bytes {
        return Err(MifError::PayloadTooShort {
            actual: bytes.len(),
            expected: expected_bytes,
        });
    }
    Ok(bytes)
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
    /// Read a `.mif` file at `path` into an [`Image`] on `backend`.
    pub fn read<B: ComputeBackend, P: AsRef<Path>>(
        &self,
        path: P,
        backend: &B,
        budget: ImageReadBudget,
    ) -> Result<Image<f32, B, 3>, MifError> {
        read_mif(path, backend, budget)
    }
}

#[cfg(test)]
#[path = "reader_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "reader_budget_tests.rs"]
mod budget_tests;
