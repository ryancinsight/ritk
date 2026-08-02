//! Analyze 7.5 writer — produces a `.hdr` header file and a `.img` raw-data file.
//!
//! # Format Overview
//!
//! Analyze 7.5 (Mayo Clinic, 1989) stores a 3-D volume as two files sharing the
//! same base name:
//!
//! * `<name>.hdr` — 348-byte binary header (little-endian).
//! * `<name>.img` — raw IEEE-754 single-precision voxel values (little-endian).
//!
//! # Axis Convention
//!
//! The Analyze format stores voxels with X varying fastest and Z varying slowest
//! (column-major for the [X, Y, Z] axis order):
//!
//! ```text
//!   flat_index(ix, iy, iz) = ix + nx·iy + nx·ny·iz
//! ```
//!
//! RITK stores tensors with shape `[nz, ny, nx]` using Z-major order:
//!
//! ```text
//!   flat_index(iz, iy, ix) = iz·ny·nx + iy·nx + ix
//! ```
//!
//! Both layouts produce the **same byte sequence** for equal (nx, ny, nz), so
//! no axis permutation is required for the raw data.  The header fields are
//! set accordingly: `dim[1]=nx`, `dim[2]=ny`, `dim[3]=nz`.
//!
//! # Spatial Metadata
//!
//! RITK's core `spacing` is per tensor axis `[z, y, x]`, while Analyze `pixdim`
//! is file-axis `[x, y, z]`; the writer reverses the columns
//! (`pixdim[1]=sx=spacing[2]`, `pixdim[2]=sy=spacing[1]`, `pixdim[3]=sz=spacing[0]`).
//! The core `origin` is already a world-space `[x, y, z]` point and is written
//! to the `originator` field as five little-endian `i16` values encoding voxel
//! coordinates `(round(ox/sx), round(oy/sy), round(oz/sz), 0, 0)`.

use anyhow::{Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_spatial::{Point, Spacing};

use crate::codec::{write_le, DT_FLOAT, EXTENTS, HDR_SIZE};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

// ── Public API ────────────────────────────────────────────────────────────────

/// Write a 3-D image to an Analyze 7.5 `.hdr` + `.img` file pair.
///
/// `path` must have a `.hdr` extension (or any other extension); the `.img`
/// sibling file is derived by replacing the extension with `.img`.  An existing
/// `.img` file at the derived path is overwritten.
///
/// # Errors
/// Returns an error if:
/// - `path`'s parent directory does not exist.
/// - Any dimension is zero or exceeds `i16::MAX` (32 767).
/// - The image storage length does not match its shape.
/// - Spacing cannot be represented as a positive finite header `f32`, or any
///   spatial metadata is non-finite.
/// - The rounded origin voxel coordinate exceeds the format's `i16` field.
/// - Writing the header or data file fails.
pub fn write_analyze<B, P>(path: P, image: &ritk_image::Image<f32, B, 3>, backend: &B) -> Result<()>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
    let vals = image.data_cow_on(backend);
    write_analyze_flat(
        path.as_ref(),
        image.shape(),
        image.spacing(),
        image.origin(),
        &vals,
    )
}

/// Substrate-agnostic Analyze serialization core. Takes flat `[Z, Y, X]` voxels plus the
/// (backend-independent) spatial metadata so header layout and byte order live
/// in exactly one place. Analyze 7.5 has no direction field (identity implied).
fn write_analyze_flat(
    path: &Path,
    shape: [usize; 3],
    spacing: &Spacing<3>,
    origin: &Point<3>,
    vals: &[f32],
) -> Result<()> {
    // Derive sibling paths (<base>.hdr, <base>.img).
    let hdr_path = path.with_extension("hdr");
    let img_path = path.with_extension("img");

    // Spatial metadata.  RITK shape = [nz, ny, nx]; spacing/origin in XYZ order.
    let [nz, ny, nx] = shape;
    let sp = spacing; // tensor-axis order [sz, sy, sx]
    let orig = origin; // world-space [ox, oy, oz]
                       // File-axis spacing [sx, sy, sz] is the reverse of core [sz, sy, sx].
    let (sx, sy, sz) = (sp[2], sp[1], sp[0]);

    // Validate the complete logical input before creating either file.
    for (name, &val) in [("nx", &nx), ("ny", &ny), ("nz", &nz)].iter() {
        if val == 0 {
            anyhow::bail!("Analyze: dimension {name} must be positive");
        }
        if val > i16::MAX as usize {
            anyhow::bail!(
                "Analyze: dimension {name}={val} exceeds i16::MAX ({})",
                i16::MAX
            );
        }
    }
    let voxel_count = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .context("Analyze voxel count overflows usize")?;
    if vals.len() != voxel_count {
        anyhow::bail!(
            "Analyze: image storage length {} does not match shape {:?} ({voxel_count} voxels)",
            vals.len(),
            shape
        );
    }
    voxel_count
        .checked_mul(size_of::<f32>())
        .context("Analyze payload byte count overflows usize")?;
    let sx_header = header_spacing("x", sx)?;
    let sy_header = header_spacing("y", sy)?;
    let sz_header = header_spacing("z", sz)?;
    for (axis, value) in [("x", orig[0]), ("y", orig[1]), ("z", orig[2])] {
        if !value.is_finite() {
            anyhow::bail!("Analyze: origin[{axis}] must be finite, found {value}");
        }
    }

    // ── Build 348-byte header ─────────────────────────────────────────────────
    let mut hdr = [0u8; HDR_SIZE];

    write_le::<i32>(&mut hdr, 0, HDR_SIZE as i32); // sizeof_hdr
    write_le::<i32>(&mut hdr, 32, EXTENTS); // extents
    hdr[38] = b'r'; // regular

    // image_dimension — dim[8] at offset 40
    write_le::<i16>(&mut hdr, 40, 4); // dim[0] = num dimensions
    write_le::<i16>(&mut hdr, 42, nx as i16); // dim[1] = X
    write_le::<i16>(&mut hdr, 44, ny as i16); // dim[2] = Y
    write_le::<i16>(&mut hdr, 46, nz as i16); // dim[3] = Z
    write_le::<i16>(&mut hdr, 48, 1); // dim[4] = time (1 volume)

    write_le::<i16>(&mut hdr, 70, DT_FLOAT); // datatype = DT_FLOAT (16)
    write_le::<i16>(&mut hdr, 72, 32); // bitpix   = 32 bits

    // pixdim[8] at offset 76
    write_le::<f32>(&mut hdr, 76, 4.0_f32); // pixdim[0] = number of dims
    write_le::<f32>(&mut hdr, 80, sx_header); // pixdim[1] = sx
    write_le::<f32>(&mut hdr, 84, sy_header); // pixdim[2] = sy
    write_le::<f32>(&mut hdr, 88, sz_header); // pixdim[3] = sz
    write_le::<f32>(&mut hdr, 92, 1.0_f32); // pixdim[4] = TR (unused)

    write_le::<f32>(&mut hdr, 108, 0.0_f32); // vox_offset
    write_le::<f32>(&mut hdr, 112, 1.0_f32); // funused1 = scale factor (1 = no scaling)

    // data_history — descrip[80] at offset 148
    let descrip = b"RITK";
    hdr[148..148 + descrip.len()].copy_from_slice(descrip);

    // originator[10] at offset 253 — voxel-space origin (5 × i16)
    let ox_vox = vox_coord("x", orig[0], f64::from(sx_header))?;
    let oy_vox = vox_coord("y", orig[1], f64::from(sy_header))?;
    let oz_vox = vox_coord("z", orig[2], f64::from(sz_header))?;
    write_le::<i16>(&mut hdr, 253, ox_vox); // originator[0] = x voxel
    write_le::<i16>(&mut hdr, 255, oy_vox); // originator[1] = y voxel
    write_le::<i16>(&mut hdr, 257, oz_vox); // originator[2] = z voxel

    // ── Write .img (raw f32 little-endian, same memory order as RITK) ─────────
    // RITK layout: flat[iz*ny*nx + iy*nx + ix] — identical to Analyze X-fastest.
    let img_file = File::create(&img_path).context("Failed to create Analyze data file")?;
    let mut img_data = BufWriter::with_capacity(8 * 1024, img_file);
    for v in vals {
        img_data
            .write_all(&v.to_le_bytes())
            .context("Failed to write Analyze voxel data")?;
    }
    img_data.flush().context("Failed to flush Analyze data")?;

    // Publish the header only after the complete voxel payload was written.
    std::fs::write(&hdr_path, hdr).context("Failed to write Analyze header")?;

    tracing::debug!(
        shape = ?shape,
        "write_analyze: complete"
    );

    Ok(())
}

// ── Analyze writer wrapper type ───────────────────────────────────────────────

/// Write-side type implementing the `ImageWriter` domain trait.
pub struct AnalyzeWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> AnalyzeWriter<B> {
    /// Construct a new writer.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Write an Analyze image through the bound backend.
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &ritk_image::Image<f32, B, 3>) -> Result<()>
    where
        B: Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        write_analyze(path, image, &self.backend)
    }
}

fn header_spacing(axis: &str, spacing_mm: f64) -> Result<f32> {
    let encoded = spacing_mm as f32;
    if !encoded.is_finite() || encoded <= 0.0 {
        anyhow::bail!(
            "Analyze: spacing[{axis}]={spacing_mm} is not representable as a positive finite f32 header value"
        );
    }
    Ok(encoded)
}

/// Convert a physical origin coordinate to the format's rounded voxel index.
#[inline]
fn vox_coord(axis: &str, origin_mm: f64, spacing_mm: f64) -> Result<i16> {
    let voxel = (origin_mm / spacing_mm).round();
    if !voxel.is_finite() || voxel < f64::from(i16::MIN) || voxel > f64::from(i16::MAX) {
        anyhow::bail!(
            "Analyze: origin[{axis}]={origin_mm} maps to voxel coordinate {voxel}, outside the i16 header range"
        );
    }
    Ok(voxel as i16)
}

#[cfg(test)]
mod tests {
    use super::write_analyze_flat;
    use anyhow::Result;
    use ritk_spatial::{Point, Spacing};
    use tempfile::tempdir;

    #[test]
    fn writer_rejects_invalid_input_before_creating_files() -> Result<()> {
        let directory = tempdir()?;
        let path = directory.path().join("invalid.hdr");

        let error = write_analyze_flat(
            &path,
            [1, 1, 2],
            &Spacing::new([1.0; 3]),
            &Point::new([0.0; 3]),
            &[1.0],
        )
        .expect_err("storage shorter than shape must be rejected");
        assert!(
            error.to_string().contains("storage length 1"),
            "unexpected error: {error:#}"
        );
        assert!(!path.exists());
        assert!(!path.with_extension("img").exists());

        let error = write_analyze_flat(
            &path,
            [1, 0, 1],
            &Spacing::new([1.0; 3]),
            &Point::new([0.0; 3]),
            &[],
        )
        .expect_err("zero dimensions must be rejected");
        assert!(
            error.to_string().contains("dimension ny"),
            "unexpected error: {error:#}"
        );
        assert!(!path.exists());
        assert!(!path.with_extension("img").exists());

        let error = write_analyze_flat(
            &path,
            [1, 1, 1],
            &Spacing::new([1.0; 3]),
            &Point::new([0.0, f64::INFINITY, 0.0]),
            &[1.0],
        )
        .expect_err("non-finite origin must be rejected");
        assert!(
            error.to_string().contains("origin[y]"),
            "unexpected error: {error:#}"
        );
        assert!(!path.exists());
        assert!(!path.with_extension("img").exists());

        let error = write_analyze_flat(
            &path,
            [1, 1, 1],
            &Spacing::new([1.0, f64::MAX, 1.0]),
            &Point::new([0.0; 3]),
            &[1.0],
        )
        .expect_err("spacing outside the header f32 range must be rejected");
        assert!(
            error.to_string().contains("not representable"),
            "unexpected error: {error:#}"
        );
        assert!(!path.exists());
        assert!(!path.with_extension("img").exists());

        let error = write_analyze_flat(
            &path,
            [1, 1, 1],
            &Spacing::new([1.0; 3]),
            &Point::new([0.0, f64::from(i16::MAX) + 1.0, 0.0]),
            &[1.0],
        )
        .expect_err("origin outside the header voxel range must be rejected");
        assert!(
            error.to_string().contains("outside the i16 header range"),
            "unexpected error: {error:#}"
        );
        assert!(!path.exists());
        assert!(!path.with_extension("img").exists());

        Ok(())
    }
}
