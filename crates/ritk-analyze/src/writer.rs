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
use ritk_spatial::{Direction, Point, Spacing};

use crate::header::{self, AnalyzeDatatype, AnalyzeHeaderFields};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::path::Path;

/// Provenance string the native `f32` writer records in `descrip`.
const NATIVE_DESCRIPTION: &[u8] = b"RITK";

// ── Public API ────────────────────────────────────────────────────────────────

/// Write a 3-D image to an Analyze 7.5 `.hdr` + `.img` file pair.
///
/// `path` must have a `.hdr` extension (or any other extension); the `.img`
/// sibling file is derived by replacing the extension with `.img`.  An existing
/// `.img` file at the derived path is overwritten.
///
/// # Pair atomicity
///
/// The `.img` payload is written and flushed before the `.hdr` is published, so
/// the header is the commit marker: within one process the write order is
/// fixed, so a failure between the two writes leaves an `.img` with no matching
/// header rather than a header describing a payload that was never written.
/// The orphan is inert — the next successful write replaces it — and the
/// recovery is to re-run the write.  The two files are not staged through
/// temporary names, so a write that fails partway still leaves the partial
/// `.img`; it is never paired with a stale `.hdr`, because the header is only
/// written after the payload succeeds.
///
/// This is **process-crash** ordering, not durability: neither file is
/// `fsync`ed, so a power loss or kernel panic can reorder the two at the
/// storage layer and leave a header whose payload did not survive.  A caller
/// that needs the pair to outlive power loss must `fsync` both paths itself.
///
/// # Errors
/// Returns an error if:
/// - `path`'s parent directory does not exist.
/// - The image has a non-identity `direction`: Analyze 7.5 has no direction
///   field, so writing one would silently drop it.
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
    // Analyze 7.5 has no direction field, so a non-identity direction is
    // unrepresentable. Reject before touching the file system rather than
    // writing a dataset whose geometry silently disagrees with the source.
    anyhow::ensure!(
        image.direction() == &Direction::identity(),
        "Analyze 7.5 has no direction field; the image's non-identity direction \
         is unrepresentable and cannot be written without silently dropping it"
    );
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

    // RITK shape is [nz, ny, nx]; spacing and origin are already in the
    // tensor-axis and world-space orders the header encoder expects.
    let [nz, ny, nx] = shape;
    let voxel_count = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .context("Analyze voxel count overflows usize")?;

    // The complete logical input is validated here, before either file exists.
    let hdr = header::encode(&AnalyzeHeaderFields {
        shape,
        spacing,
        origin,
        datatype: AnalyzeDatatype::Float,
        scale: 1.0,
        description: NATIVE_DESCRIPTION,
    })?;
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

#[cfg(test)]
mod tests {
    use super::write_analyze_flat;
    use anyhow::Result;
    use ritk_spatial::{Direction, Point, Spacing};
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

    /// A non-identity direction is rejected before either file is created.
    ///
    /// Analyze 7.5 has no direction field. Writing such an image would produce a
    /// dataset whose geometry silently disagrees with the source, so the writer
    /// refuses instead. This is the case the shared round-trip harness cannot
    /// reach: it writes an identity direction for every `SpacingAndOrigin`
    /// codec, so only a direct test can pin the rejection.
    #[test]
    fn writer_rejects_a_non_identity_direction_before_creating_files() -> Result<()> {
        use coeus_core::SequentialBackend;
        use ritk_image::Image;

        let directory = tempdir()?;
        let path = directory.path().join("oblique.hdr");
        let image = Image::from_flat_on(
            vec![1.0_f32; 2],
            [1, 1, 2],
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            &SequentialBackend,
        )
        .expect("fixture image");

        let error = super::write_analyze(&path, &image, &SequentialBackend)
            .expect_err("a non-identity direction is unrepresentable in Analyze 7.5");
        assert!(
            error.to_string().contains("no direction field"),
            "unexpected error: {error:#}"
        );
        assert!(!path.exists(), "the header must not be created");
        assert!(
            !path.with_extension("img").exists(),
            "the payload must not be created"
        );

        Ok(())
    }
}
