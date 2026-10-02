//! MRtrix `.mif` image writer.
//!
//! Writes the MRtrix3 `.mif` container format: text header (key: value)
//! terminated by `END`, followed by raw binary voxel data in the declared
//! datatype and layout.

use crate::header::datatype_name;
use anyhow::{anyhow, Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use consus_core::ByteOrder;
use ritk_codecs::sample::{write_samples, Sample};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::fmt::Write as _;
use std::io::{BufWriter, Write};
use std::path::Path;

/// The byte order the writer stores multi-byte samples in.
const STORED_BYTE_ORDER: ByteOrder = ByteOrder::LittleEndian;

/// Write a 3‑D [`Image`] of `T` to a `.mif` file storing `T`'s samples.
///
/// MRtrix stores every sample type RITK has, so the writer never converts: the
/// `datatype` key names the type of `T` and the voxels are written as `T`,
/// little-endian.
///
/// # Format
///
/// - `mrtrix image: version 3.0` magic
/// - `dim: nx ny nz` (X, Y, Z order — MRtrix convention)
/// - `vox: sx sy sz` (voxel sizes in mm)
/// - `layout: +0,+1,+2` (contiguous)
/// - `datatype:` the type of `T` (`Int8`, `UInt8`, `Int16LE`, …, `Float64LE`)
/// - `transform:` followed by 4 matrix rows
/// - `file: . N` where `N` is the byte offset of the first voxel from the start
///   of the file, a multiple of 4 (MRtrix aligns inline data to 4 bytes)
/// - `END\n`, zero padding up to offset `N`, then raw binary
///
/// # Spatial metadata
///
/// The `.mif` `transform` is assembled from RITK `origin` + `spacing` +
/// `direction` as the voxel→scanner affine.  Columns are reordered from
/// internal ZYX to file XYZ order.
///
/// # Errors
///
/// Returns an error when creating or writing the file fails.
pub fn write_mif<T, B, P>(path: P, image: &Image<T, B, 3>, backend: &B) -> Result<()>
where
    T: Sample,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    P: AsRef<Path>,
{
    let voxels = image.data_cow_on(backend);
    write_mif_flat(
        path.as_ref(),
        image.shape(),
        image.spacing(),
        image.origin(),
        image.direction(),
        &[voxels],
    )
}

/// Write an acquisition series to a `.mif` file.
///
/// Multi‑frame `.mif` files carry a fourth axis (`dim` axis 3) with one
/// frame per volume.  Frames are interleaved (fastest-varying axis 3 for
/// contiguous layout), which matches MRtrix3's default output.
///
/// A single‑volume series writes as a rank‑3 file identical to
/// [`write_mif`].
///
/// # Errors
///
/// Returns an error when `volumes` is empty, when any volume's grid differs
/// from the first, or when writing fails.
pub fn write_mif_series<T, B, P>(path: P, volumes: &[Image<T, B, 3>], backend: &B) -> Result<()>
where
    T: Sample,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    P: AsRef<Path>,
{
    let Some((first, rest)) = volumes.split_first() else {
        return Err(anyhow!(
            "write_mif_series: a series requires at least one volume"
        ));
    };

    let shape = first.shape();
    for (index, volume) in rest.iter().enumerate() {
        let position = index + 1;
        if volume.shape() != shape {
            return Err(anyhow!(
                "write_mif_series: volume {position} shape {:?} differs from volume 0 \
                 {shape:?}; a .mif series has one spatial grid",
                volume.shape()
            ));
        }
        if volume.origin() != first.origin() || volume.spacing() != first.spacing() {
            return Err(anyhow!(
                "write_mif_series: volume {position} origin or spacing differs from \
                 volume 0; a .mif series has one spatial grid"
            ));
        }
    }

    let payloads: Vec<_> = volumes.iter().map(|v| v.data_cow_on(backend)).collect();
    write_mif_flat(
        path.as_ref(),
        shape,
        first.spacing(),
        first.origin(),
        first.direction(),
        &payloads,
    )
}

// ── Core serialisation ───────────────────────────────────────────────────

/// Length in bytes of `file: . ` and of the `\nEND\n` that closes the header,
/// the part of the `file` line that does not depend on the offset.
const FILE_LINE_FRAME_LEN: usize = "file: . ".len() + "\nEND\n".len();

/// The byte offset of the first voxel of an inline `.mif` whose header, up to
/// but excluding the `file` line, is `preceding_len` bytes long.
///
/// MRtrix defines the offset as bytes from the start of the file, rounded up
/// to a multiple of 4 beyond the `END` line (`core/formats/mrtrix.cpp`,
/// `MRtrix::create`). The `file` line spells the offset in decimal, so the
/// line's own length depends on the offset it carries; the smallest digit
/// count whose aligned offset has that many digits is the fixed point.
fn inline_data_offset(preceding_len: usize) -> usize {
    let mut digits = 1;
    loop {
        let end = preceding_len + FILE_LINE_FRAME_LEN + digits;
        let offset = end.next_multiple_of(4);
        let needed = offset.ilog10() as usize + 1;
        if needed <= digits {
            return offset;
        }
        digits = needed;
    }
}

fn write_mif_flat<T: Sample>(
    path: &Path,
    shape: [usize; 3],
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    payloads: &[impl std::ops::Deref<Target = [T]>],
) -> Result<()> {
    let [nz, ny, nx] = shape;
    let nframes = payloads.len();

    let voxels_per_volume = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .ok_or_else(|| anyhow!(".mif shape [{nz},{ny},{nx}] voxel count overflows usize"))?;

    for (position, payload) in payloads.iter().enumerate() {
        if payload.len() != voxels_per_volume {
            return Err(anyhow!(
                "write_mif_flat: volume {position} has {} voxels but shape \
                 [{nz},{ny},{nx}] requires {voxels_per_volume}",
                payload.len()
            ));
        }
    }

    // ── Header ───────────────────────────────────────────────────────────

    let mut header = String::new();

    // Magic.
    writeln!(header, "mrtrix image: version 3.0")?;
    writeln!(header, "# Written by ritk-mif")?;

    // Dimensions: X, Y, Z [, frames]
    if nframes > 1 {
        writeln!(header, "dim: {nx} {ny} {nz} {nframes}")?;
        writeln!(header, "layout: +0,+1,+2,+3")?;
    } else {
        writeln!(header, "dim: {nx} {ny} {nz}")?;
        writeln!(header, "layout: +0,+1,+2")?;
    }

    // Voxel sizes: spatial only, in X,Y,Z order.
    let vx = spacing[2];
    let vy = spacing[1];
    let vz = spacing[0];
    writeln!(header, "vox: {vx} {vy} {vz}")?;

    writeln!(
        header,
        "datatype: {}",
        datatype_name(T::TYPE, STORED_BYTE_ORDER)
    )?;

    // Transform: 4×4 voxel→scanner affine (standard MRtrix multi-line).
    let transform = build_transform(origin, spacing, direction);
    writeln!(header, "transform:")?;
    for row in &transform {
        writeln!(
            header,
            "{:.6} {:.6} {:.6} {:.6}",
            row[0], row[1], row[2], row[3]
        )?;
    }

    // Inline data: `file: . N`, N the byte offset from the start of the file.
    let offset = inline_data_offset(header.len());
    writeln!(header, "file: . {offset}")?;
    writeln!(header, "END")?;

    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create .mif file {:?}", path))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(header.as_bytes())
        .context("Failed to write .mif header")?;
    // Zero padding from the END line to the aligned data offset, as MRtrix writes.
    let padding = offset - header.len();
    writer
        .write_all(&vec![0_u8; padding])
        .context("Failed to write .mif header padding")?;

    // ── Binary data ──────────────────────────────────────────────────────
    // Interleave frames: axis 3 varies fastest (per MRtrix convention).
    if nframes > 1 {
        let total = nframes * voxels_per_volume;
        let mut interleaved = Vec::with_capacity(total);
        for voxel in 0..voxels_per_volume {
            for payload in payloads {
                interleaved.push(payload[voxel]);
            }
        }
        write_samples(&interleaved, STORED_BYTE_ORDER, &mut writer)
            .context("Failed to write .mif voxel data")?;
    } else {
        write_samples(&payloads[0], STORED_BYTE_ORDER, &mut writer)
            .context("Failed to write .mif voxel data")?;
    }

    writer.flush().context("Failed to flush .mif output file")?;
    Ok(())
}

// ── Transform builder ────────────────────────────────────────────────────

/// Build a 4×4 voxel→scanner affine `[row][col]` from RITK metadata.
///
/// RITK stores `direction` columns as `(dz, dy, dx)` in scanner coords
/// with spacing applied.  The `.mif` transform expects columns in voxel
/// axis order (X, Y, Z).  Translation maps voxel (0,0,0) to the corner.
fn build_transform(
    origin: &Point<3>,
    spacing: &Spacing<3>,
    direction: &Direction<3>,
) -> [[f64; 4]; 4] {
    let d = direction.0;

    // RITK direction columns: col 0 = Z, col 1 = Y, col 2 = X.
    // Apply spacing.
    let dz = [
        d[(0, 0)] * spacing[0],
        d[(1, 0)] * spacing[0],
        d[(2, 0)] * spacing[0],
    ];
    let dy = [
        d[(0, 1)] * spacing[1],
        d[(1, 1)] * spacing[1],
        d[(2, 1)] * spacing[1],
    ];
    let dx = [
        d[(0, 2)] * spacing[2],
        d[(1, 2)] * spacing[2],
        d[(2, 2)] * spacing[2],
    ];

    // .mif transform: row 0-2 = scanner-x, scanner-y, scanner-z
    // columns 0,1,2 = voxel-x, voxel-y, voxel-z
    // Translation is the corner position (origin).
    [
        [dx[0], dy[0], dz[0], origin[0]],
        [dx[1], dy[1], dz[1], origin[1]],
        [dx[2], dy[2], dz[2], origin[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

// ── Public writer struct ─────────────────────────────────────────────────────

/// Thin writer struct for `.mif` files.
pub struct MifWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> MifWriter<B> {
    /// Creates a writer that extracts image storage through `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B: ComputeBackend + Default> MifWriter<B> {
    /// Write `image` to the `.mif` file at `path`, storing `T`'s samples.
    ///
    /// # Errors
    ///
    /// Returns the error of [`write_mif`].
    pub fn write<T, P>(&self, path: P, image: &Image<T, B, 3>) -> Result<()>
    where
        T: Sample,
        B::DeviceBuffer<T>: CpuAddressableStorage<T>,
        P: AsRef<Path>,
    {
        write_mif(path, image, &self.backend)
    }
}

#[cfg(test)]
#[path = "tests_writer.rs"]
mod tests;
