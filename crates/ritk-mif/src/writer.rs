//! MRtrix `.mif` image writer.
//!
//! Writes the MRtrix3 `.mif` container format: text header (key: value)
//! terminated by `END`, followed by raw binary voxel data in the declared
//! datatype and layout.

use anyhow::{anyhow, Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write a 3‑D [`Image`] to a `.mif` file.
///
/// # Format
///
/// - `mrtrix image: version 3.0` magic
/// - `dim: nx ny nz` (X, Y, Z order — MRtrix convention)
/// - `vox: sx sy sz` (voxel sizes in mm)
/// - `layout: +0,+1,+2` (contiguous)
/// - `datatype: Float32LE`
/// - `transform:` followed by 4 matrix rows
/// - `END\n` then raw binary
///
/// # Spatial metadata
///
/// The `.mif` `transform` is assembled from RITK `origin` + `spacing` +
/// `direction` as the voxel→scanner affine.  Columns are reordered from
/// internal ZYX to file XYZ order.
pub fn write_mif<B, P>(path: P, image: &Image<f32, B, 3>, backend: &B) -> Result<()>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
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
pub fn write_mif_series<B, P>(path: P, volumes: &[Image<f32, B, 3>], backend: &B) -> Result<()>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
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

fn write_mif_flat(
    path: &Path,
    shape: [usize; 3],
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    payloads: &[impl std::ops::Deref<Target = [f32]>],
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

    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create .mif file {:?}", path))?;
    let mut writer = BufWriter::new(file);

    // ── Header ───────────────────────────────────────────────────────────

    // Magic.
    writeln!(writer, "mrtrix image: version 3.0")?;
    writeln!(writer, "# Written by ritk-mif")?;

    // Dimensions: X, Y, Z [, frames]
    if nframes > 1 {
        writeln!(writer, "dim: {nx} {ny} {nz} {nframes}")?;
        writeln!(writer, "layout: +0,+1,+2,+3")?;
    } else {
        writeln!(writer, "dim: {nx} {ny} {nz}")?;
        writeln!(writer, "layout: +0,+1,+2")?;
    }

    // Voxel sizes: spatial only, in X,Y,Z order.
    let vx = spacing[2];
    let vy = spacing[1];
    let vz = spacing[0];
    writeln!(writer, "vox: {vx} {vy} {vz}")?;

    writeln!(writer, "datatype: Float32LE")?;

    // Transform: 4×4 voxel→scanner affine (standard MRtrix multi-line).
    let transform = build_transform(origin, spacing, direction);
    writeln!(writer, "transform:")?;
    for row in &transform {
        writeln!(
            writer,
            "{:.6} {:.6} {:.6} {:.6}",
            row[0], row[1], row[2], row[3]
        )?;
    }

    // Reference to the data file.  Inline: "." with offset 0.
    writeln!(writer, "file: . 0")?;

    // END marker.
    writeln!(writer, "END")?;

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
        write_le_f32(&mut writer, &interleaved)?;
    } else {
        write_le_f32(&mut writer, &payloads[0])?;
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

// ── Byte writing ─────────────────────────────────────────────────────────

fn write_le_f32(writer: &mut impl Write, values: &[f32]) -> Result<()> {
    #[cfg(target_endian = "little")]
    writer.write_all(bytemuck::cast_slice(values))?;
    #[cfg(target_endian = "big")]
    {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        writer.write_all(&bytes)?;
    }
    Ok(())
}

// ── Public writer struct ─────────────────────────────────────────────────────

/// Thin writer struct for `.mif` files.
pub struct MifWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> MifWriter<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B> MifWriter<B>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> Result<()> {
        write_mif(path, image, &self.backend)
    }
}

#[cfg(test)]
#[path = "writer_tests.rs"]
mod tests;
