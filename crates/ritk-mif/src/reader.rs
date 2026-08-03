//! MRtrix `.mif` image reader.
//!
//! Reads the MRtrix3 `.mif` container format — a text header followed by raw
//! binary voxel data.  Both inline (single-file) and detached (`file:` key
//! pointing to `.mif.dat`) layouts are supported.

use crate::decode;
use crate::header::{
    parse_datatype, parse_dim, parse_f64_vec, parse_layout, parse_mif_header_from_path,
    parse_transform,
};
use anyhow::{anyhow, Context, Result};
use coeus_core::ComputeBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing, Vector};
use std::io::Read;
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
    fn into_single_volume(mut self) -> Result<DecodedMif> {
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
pub fn read_mif<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<Image<f32, B, 3>> {
    let decoded = decode_mif(path)?.into_single_volume()?;
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
/// Every returned image shares the file's single spatial grid.
pub fn read_mif_series<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<Vec<Image<f32, B, 3>>> {
    let DecodedMif {
        volumes,
        dims,
        origin,
        spacing,
        direction,
    } = decode_mif(path)?;

    volumes
        .into_iter()
        .map(|data| Image::from_flat_on(data, dims, origin, spacing, direction, backend))
        .collect()
}

// ── Internal decode ─────────────────────────────────────────────────────

fn decode_mif<P: AsRef<Path>>(path: P) -> Result<DecodedMif> {
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

    if nframes == 0 {
        return Err(anyhow!(".mif 'dim' must declare at least 1 frame"));
    }

    // ── Datatype ──────────────────────────────────────────────────────────
    let dt_str = header
        .entries
        .get("datatype")
        .ok_or_else(|| anyhow!("Missing 'datatype' in .mif header"))?
        .as_line();
    let (elem_size, _signed, _float) = parse_datatype(dt_str)?;

    // Determine endianness from the datatype string suffix.
    let is_big_endian = dt_str.trim().to_lowercase().ends_with("be");

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
        let v = parse_f64_vec(vox_val.as_line())?;
        if v.len() < 3 {
            return Err(anyhow!(
                ".mif 'vox' expected at least 3 spatial sizes, got {}",
                v.len()
            ));
        }
        v
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
    let expected_bytes = total_voxels
        .checked_mul(elem_size)
        .ok_or_else(|| anyhow!(".mif byte count overflows usize"))?;

    // Handle detached data file.
    let raw_bytes: Vec<u8> = if let Some(file_val) = header.entries.get("file") {
        let file_spec = file_val.as_line();
        // "file: . 2" means data starts at byte offset 2 in the same file.
        // "file: sub-01_dwi.mif.dat 0" means a detached file.
        let parts: Vec<&str> = file_spec.split_whitespace().collect();
        if parts.len() >= 2 {
            let fname = parts[0];
            let offset: u64 = parts[1]
                .parse()
                .context("Invalid offset in .mif 'file' key")?;
            if fname == "." {
                // Inline data at offset — consume from the current reader.
                // We already read the header, so we need to account for that.
                // Actually, parse_mif_header_from_path leaves the reader
                // positioned right after END\n, so offset 0 here means
                // "start now".  >0 means skip additional bytes.
                if offset > 0 {
                    let mut skip_buf = vec![0u8; offset as usize];
                    std::io::Read::read_exact(&mut reader, &mut skip_buf)
                        .context("Failed to skip inline data offset in .mif")?;
                }
                let mut bytes = Vec::new();
                reader
                    .read_to_end(&mut bytes)
                    .context("Failed to read inline .mif binary data")?;
                bytes
            } else {
                let data_path = path.parent().unwrap_or_else(|| Path::new(".")).join(fname);
                let mut bytes = std::fs::read(&data_path)
                    .with_context(|| format!("Cannot read .mif data file {:?}", data_path))?;
                if offset > 0 {
                    bytes.drain(..(offset as usize).min(bytes.len()));
                }
                bytes
            }
        } else {
            return Err(anyhow!(
                "Invalid 'file' key in .mif header: '{}'",
                file_spec
            ));
        }
    } else {
        // Inline data right after END — read from the current reader position.
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .context("Failed to read inline .mif binary data")?;
        bytes
    };

    if raw_bytes.len() < expected_bytes {
        return Err(anyhow!(
            ".mif payload has {} bytes but dim requires {expected_bytes}",
            raw_bytes.len()
        ));
    }
    let raw_bytes = &raw_bytes[..expected_bytes];

    // ── Decode to f32 ────────────────────────────────────────────────────
    let f32_data = decode::decode_bytes(raw_bytes, elem_size, total_voxels, is_big_endian)?;

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
        if sz > 0.0 { sz } else { vox_sizes[2] },
        if sy > 0.0 { sy } else { vox_sizes[1] },
        if sx > 0.0 { sx } else { vox_sizes[0] },
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
    ) -> Result<Image<f32, B, 3>> {
        read_mif(path, backend)
    }
}

#[cfg(test)]
#[path = "reader_tests.rs"]
mod tests;
