use crate::element_type::element_type_name;
use crate::spatial::file_spatial_fields_from_internal;
use anyhow::{anyhow, Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use consus_core::ByteOrder;
use ritk_codecs::sample::{write_samples, Sample};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write a 3-D `Image` of `T` to a `.mha` (MetaImage single-file) format.
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
/// Voxel values are written as `T` samples in little-endian byte order,
/// uncompressed, immediately after the `ElementDataFile = LOCAL` header line,
/// and `ElementType` names `T`: `MET_CHAR`, `MET_UCHAR`, `MET_SHORT`,
/// `MET_USHORT`, `MET_INT`, `MET_UINT`, `MET_LONG_LONG`, `MET_ULONG_LONG`,
/// `MET_FLOAT`, or `MET_DOUBLE` for `i8`, `u8`, `i16`, `u16`, `i32`, `u32`,
/// `i64`, `u64`, `f32`, or `f64`. Every sample round-trips bit for bit.
///
/// # Errors
///
/// Returns an error when the image shape overflows `usize`, or when the file
/// cannot be created or written.
pub fn write_metaimage<T, B, P>(path: P, image: &Image<T, B, 3>, backend: &B) -> Result<()>
where
    T: Sample,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    P: AsRef<Path>,
{
    let voxels = image.data_cow_on(backend);
    write_metaimage_with_data(path, image, &voxels)
}

/// Like [`write_metaimage`] but uses caller-provided voxel data.
///
/// `image` supplies only spatial metadata (shape, spacing, origin, direction);
/// the binary payload comes from `values`.  This lets a caller that already
/// holds a fast (e.g. zero-copy NdArray) slice skip the generic
/// `into_data()` materialization, which dominates write time for large volumes.
///
/// # Errors
///
/// Returns an error, before creating any file, when `values.len()` differs from
/// the image voxel count; otherwise as [`write_metaimage`].
pub fn write_metaimage_with_data<T: Sample, B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    image: &Image<T, B, 3>,
    values: &[T],
) -> Result<()> {
    write_metaimage_flat(
        path.as_ref(),
        image.shape(),
        image.spacing(),
        image.origin(),
        image.direction(),
        values,
    )
}

/// MetaImage serialization core. Takes flat `[Z, Y, X]` voxels plus the
/// (backend-independent) spatial metadata so header emission and byte layout
/// live in exactly one place. `values.len()` must equal the voxel count.
fn write_metaimage_flat<T: Sample>(
    path: &Path,
    shape: [usize; 3],
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    values: &[T],
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
    if values.len() != voxel_count {
        return Err(anyhow!(
            "MetaImage payload has {} voxels but shape [{nz}, {ny}, {nx}] requires {voxel_count}",
            values.len()
        ));
    }

    // ── Spatial metadata ──────────────────────────────────────────────────
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

    // ── File I/O ──────────────────────────────────────────────────────────
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create MetaImage file {:?}", path))?;
    let mut writer = BufWriter::new(file);

    // Header — field order matches the ITK MetaImageIO convention.
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
    writeln!(writer, "ElementType = {}", element_type_name(T::TYPE))?;
    // LOCAL signals that binary data follows immediately.
    writeln!(writer, "ElementDataFile = LOCAL")?;

    // Binary payload: `T` samples little-endian (`BinaryDataByteOrderMSB = False`),
    // encoded in bounded blocks so no second copy of the volume is held.
    write_samples(values, ByteOrder::LittleEndian, &mut writer)
        .context("Failed to write MetaImage payload")?;

    writer
        .flush()
        .context("Failed to flush MetaImage output file")?;

    Ok(())
}

// ── Public writer struct ──────────────────────────────────────────────────────

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
{
    /// Write `image` to the MetaImage file at `path`, storing its samples as `T`.
    ///
    /// # Errors
    ///
    /// See [`write_metaimage`].
    pub fn write<T, P>(&self, path: P, image: &Image<T, B, 3>) -> Result<()>
    where
        T: Sample,
        B::DeviceBuffer<T>: CpuAddressableStorage<T>,
        P: AsRef<Path>,
    {
        write_metaimage(path, image, &self.backend)
    }
}
