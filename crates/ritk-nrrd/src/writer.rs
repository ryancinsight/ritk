use anyhow::{Context, Result};
use ritk_image::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_image::HostExtract;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::spatial::file_space_directions_from_internal;

/// Write a 3-D `Image` to a NRRD (Nearly Raw Raster Data) file.
///
/// # Format
/// Writes NRRD version 4 (`NRRD0004`) with `encoding: raw` and
/// `endian: little`.  The file is self-contained (inline data).
///
/// # Axis convention
/// RITK stores voxels in `[Z, Y, X]` order. NRRD stores raw data with X as
/// the fastest-varying axis. These flat orders are identical, so voxel bytes
/// are written directly while the `sizes` header is emitted as `nx ny nz`
/// (`shape()[2] shape()[1] shape()[0]` of the RITK image).
///
/// # Spatial metadata
/// * `space directions` — NRRD file-axis vectors `[x,y,z]` are emitted from
///   RITK metadata columns `[col,row,depth]`, each scaled by its matching
///   spacing.
/// * `space origin` — the image origin in physical `[X, Y, Z]` space.
///
/// # Binary payload
/// Voxel values are written as 32-bit IEEE 754 floats in little-endian byte
/// order, immediately after a blank header-terminator line.
pub fn write_nrrd<B: HostExtract, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    // RITK [Z,Y,X] flat layout is already NRRD X-fastest raw order.  Extract via
    // the backend's fast host path to avoid the `into_data()` materialization.
    let f32_vec = image.data_vec_fast();
    write_nrrd_flat(
        path.as_ref(),
        image.shape(),
        image.spacing(),
        image.origin(),
        image.direction(),
        &f32_vec,
    )
}

/// Like [`write_nrrd`] but uses caller-provided voxel data.
///
/// `image` supplies only spatial metadata; the binary payload comes from
/// `f32_slice`.  This lets a caller that already holds a fast (e.g. zero-copy
/// NdArray) slice skip the generic `into_data()` materialization that dominates
/// write time for large volumes.  `f32_slice.len()` must equal the voxel count.
pub fn write_nrrd_with_data<B: Backend, P: AsRef<Path>>(
    path: P,
    image: &Image<B, 3>,
    f32_slice: &[f32],
) -> Result<()> {
    write_nrrd_flat(
        path.as_ref(),
        image.shape(),
        image.spacing(),
        image.origin(),
        image.direction(),
        f32_slice,
    )
}

/// Substrate-agnostic NRRD serialization core: the shared SSOT the Burn and
/// Atlas-native writers both wrap. Takes flat `[Z, Y, X]` voxels plus the
/// (backend-independent) spatial metadata so header emission and byte layout
/// live in exactly one place. `f32_slice.len()` must equal the voxel count.
fn write_nrrd_flat(
    path: &Path,
    shape: [usize; 3],
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    f32_slice: &[f32],
) -> Result<()> {
    // shape is [nz, ny, nx] in RITK convention.
    let nz = shape[0];
    let ny = shape[1];
    let nx = shape[2];

    // ── Spatial metadata ──────────────────────────────────────────────────
    let dir = direction.0;

    let file_directions = file_space_directions_from_internal(
        [spacing[0], spacing[1], spacing[2]],
        [
            dir[(0, 0)],
            dir[(0, 1)],
            dir[(0, 2)],
            dir[(1, 0)],
            dir[(1, 1)],
            dir[(1, 2)],
            dir[(2, 0)],
            dir[(2, 1)],
            dir[(2, 2)],
        ],
    );
    let sd0 = format_nrrd_vector(file_directions[0]);
    let sd1 = format_nrrd_vector(file_directions[1]);
    let sd2 = format_nrrd_vector(file_directions[2]);

    let space_origin = format!("({},{},{})", origin[0], origin[1], origin[2]);

    // ── File I/O ──────────────────────────────────────────────────────────
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create NRRD file {:?}", path))?;
    let mut writer = BufWriter::new(file);

    // Header — field order matches the ITK NrrdIO convention.
    writeln!(writer, "NRRD0004")?;
    writeln!(writer, "# Complete NRRD file written by ritk")?;
    writeln!(writer, "type: float")?;
    writeln!(writer, "dimension: 3")?;
    // ITK/SimpleITK and ritk's own reader work in LPS: the reader stores the
    // `space origin` / `space directions` verbatim (no space conversion), and ITK
    // NRRDs are written LPS. Declaring RAS here made SimpleITK reinterpret the
    // LPS-valued origin/directions and negate the x and y (R↔L, A↔P) components on
    // read, corrupting the origin of an anisotropic-origin volume on round-trip.
    writeln!(writer, "space: left-posterior-superior")?;
    // sizes is in NRRD [X, Y, Z] order.
    writeln!(writer, "sizes: {} {} {}", nx, ny, nz)?;
    writeln!(writer, "space directions: {} {} {}", sd0, sd1, sd2)?;
    writeln!(writer, "kinds: domain domain domain")?;
    writeln!(writer, "endian: little")?;
    writeln!(writer, "encoding: raw")?;
    writeln!(writer, "space origin: {}", space_origin)?;
    // Blank line terminates the header; binary data follows immediately.
    writeln!(writer)?;

    // Binary payload — little-endian f32, written in a single bulk call.
    // On little-endian targets the f32 slice reinterprets to bytes with no copy
    // (the on-disk encoding is little-endian); a per-element `write_all` loop is
    // ~10× slower from the per-call overhead across millions of voxels.
    #[cfg(target_endian = "little")]
    writer.write_all(bytemuck::cast_slice(f32_slice))?;
    #[cfg(target_endian = "big")]
    {
        let mut bytes = Vec::with_capacity(f32_slice.len() * 4);
        for &v in f32_slice {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        writer.write_all(&bytes)?;
    }

    writer.flush().context("Failed to flush NRRD output file")?;

    Ok(())
}

fn format_nrrd_vector(vector: [f64; 3]) -> String {
    format!("({},{},{})", vector[0], vector[1], vector[2])
}

// ── Public writer struct ──────────────────────────────────────────────────────

/// Thin writer struct for NRRD files.
///
/// The backend `B` is supplied per-call so a single `NrrdWriter` instance can
/// write images from different backends.
pub struct NrrdWriter;

impl NrrdWriter {
    /// Write `image` to the NRRD file at `path`.
    pub fn write<B: HostExtract, P: AsRef<Path>>(
        &self,
        path: P,
        image: &Image<B, 3>,
    ) -> Result<()> {
        write_nrrd(path, image)
    }
}

/// Atlas-native-substrate NRRD writers (plain end-state names, disambiguated
/// from the Burn functions by module path only; folds away when the Burn path
/// is deleted — ADR 0002 A1).
pub mod native {
    use super::write_nrrd_flat;
    use anyhow::Result;
    use std::path::Path;

    /// Write an Atlas-native 3-D image to a NRRD file.
    ///
    /// Host data is extracted layout-independently via `data_cow_on`, then
    /// serialized through the same [`write_nrrd_flat`](super::write_nrrd_flat)
    /// core as the Burn [`write_nrrd`](super::write_nrrd) — byte-identical
    /// output for the same logical image.
    pub fn write_nrrd<B, P>(
        path: P,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> Result<()>
    where
        B: coeus_core::ComputeBackend + Default,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
        P: AsRef<Path>,
    {
        let voxels = image.data_cow_on(backend);
        write_nrrd_flat(
            path.as_ref(),
            image.shape(),
            image.spacing(),
            image.origin(),
            image.direction(),
            &voxels,
        )
    }

    /// Stateless Atlas-native writer for NRRD files.
    pub struct NrrdWriter;

    impl NrrdWriter {
        /// Write an Atlas-native `image` to the NRRD file at `path`.
        pub fn write<B, P>(
            &self,
            path: P,
            image: &ritk_image::native::Image<f32, B, 3>,
            backend: &B,
        ) -> Result<()>
        where
            B: coeus_core::ComputeBackend + Default,
            B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
            P: AsRef<Path>,
        {
            write_nrrd(path, image, backend)
        }
    }
}
