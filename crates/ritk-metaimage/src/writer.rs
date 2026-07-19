use crate::spatial::file_spatial_fields_from_internal;
use anyhow::{anyhow, Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write a 3-D `Image` to a `.mha` (MetaImage single-file) format.
///
/// # Axis convention
/// RITK stores voxels in `[Z, Y, X]` order. Its row-major flat payload is
/// X-fastest, which is the MetaImage file layout. The writer emits tensor data
/// directly and writes `DimSize = nx ny nz`.
///
/// # Spatial metadata
/// `origin` is written in physical coordinate order. `spacing` and `direction`
/// columns are converted from RITK `[Z,Y,X]` image-axis order into MetaImage
/// `[X,Y,Z]` file-axis order.
///
/// # Binary payload
/// Voxel values are written as 32-bit IEEE 754 floats in little-endian byte
/// order immediately after the `ElementDataFile = LOCAL` header line.
pub fn write_metaimage<B, P>(path: P, image: &Image<f32, B, 3>, backend: &B) -> Result<()>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
    let voxels = image.data_cow_on(backend);
    write_metaimage_with_data(path, image, &voxels)
}

/// Like [`write_metaimage`] but uses caller-provided voxel data.
///
/// `image` supplies only spatial metadata (shape, spacing, origin, direction);
/// the binary payload comes from `f32_slice`.  This lets a caller that already
/// holds a fast (e.g. zero-copy NdArray) slice skip the generic
/// `into_data()` materialization, which dominates write time for large volumes.
/// `f32_slice.len()` must equal the image voxel count.
pub fn write_metaimage_with_data<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    image: &Image<f32, B, 3>,
    f32_slice: &[f32],
) -> Result<()> {
    write_metaimage_flat(
        path.as_ref(),
        image.shape(),
        image.spacing(),
        image.origin(),
        image.direction(),
        f32_slice,
    )
}

/// MetaImage serialization core. Takes flat `[Z, Y, X]` voxels plus the
/// (backend-independent) spatial metadata so header emission and byte layout
/// live in exactly one place. `f32_slice.len()` must equal the voxel count.
fn write_metaimage_flat(
    path: &Path,
    shape: [usize; 3],
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    f32_slice: &[f32],
) -> Result<()> {
    // shape is [nz, ny, nx] in RITK convention.
    // MetaImage DimSize is written in [nx, ny, nz] file-axis order.
    let nz = shape[0];
    let ny = shape[1];
    let nx = shape[2];
    let voxel_count = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .ok_or_else(|| anyhow!("MetaImage shape [{nz}, {ny}, {nx}] voxel count overflows usize"))?;
    if f32_slice.len() != voxel_count {
        return Err(anyhow!(
            "MetaImage payload has {} voxels but shape [{nz}, {ny}, {nx}] requires {voxel_count}",
            f32_slice.len()
        ));
    }

    // â”€â”€ Spatial metadata â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let dir = direction.0;
    let spatial_fields = file_spatial_fields_from_internal(
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
    let tm = spatial_fields.transform_matrix_row_major;

    // â”€â”€ File I/O â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create MetaImage file {:?}", path))?;
    let mut writer = BufWriter::new(file);

    // Header â€” field order matches the ITK MetaImageIO convention.
    writeln!(writer, "ObjectType = Image")?;
    writeln!(writer, "NDims = 3")?;
    writeln!(writer, "BinaryData = True")?;
    writeln!(writer, "BinaryDataByteOrderMSB = False")?;
    writeln!(writer, "CompressedData = False")?;
    writeln!(
        writer,
        "TransformMatrix = {} {} {} {} {} {} {} {} {}",
        tm[0], tm[1], tm[2], tm[3], tm[4], tm[5], tm[6], tm[7], tm[8]
    )?;
    writeln!(writer, "Offset = {} {} {}", origin[0], origin[1], origin[2])?;
    writeln!(writer, "CenterOfRotation = 0 0 0")?;
    writeln!(
        writer,
        "ElementSpacing = {} {} {}",
        spatial_fields.element_spacing[0],
        spatial_fields.element_spacing[1],
        spatial_fields.element_spacing[2]
    )?;
    // DimSize is in MetaImage [X, Y, Z] order.
    writeln!(writer, "DimSize = {} {} {}", nx, ny, nz)?;
    writeln!(writer, "ElementType = MET_FLOAT")?;
    // LOCAL signals that binary data follows immediately.
    writeln!(writer, "ElementDataFile = LOCAL")?;

    // Binary payload â€” little-endian f32, written in a single bulk call.
    // On little-endian targets the f32 slice reinterprets to bytes with no copy
    // (BinaryDataByteOrderMSB = False); a per-element `write_all` loop is ~10Ã—
    // slower from the per-call overhead across millions of voxels.
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

    writer
        .flush()
        .context("Failed to flush MetaImage output file")?;

    Ok(())
}

// â”€â”€ Public writer struct â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Thin writer struct for MetaImage files.
///
/// The backend `B` is supplied per-call so a single `MetaImageWriter`
/// instance can write images from different backends.
pub struct MetaImageWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> MetaImageWriter<B> {
    /// Creates a writer that extracts image storage through `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B> MetaImageWriter<B>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    /// Write `image` to the MetaImage file at `path`.
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> Result<()> {
        write_metaimage(path, image, &self.backend)
    }
}
