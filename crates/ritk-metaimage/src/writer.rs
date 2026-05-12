use crate::spatial::file_spatial_fields_from_internal;
use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
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
pub fn write_metaimage<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    let path = path.as_ref();

    // ── Voxel data ────────────────────────────────────────────────────────
    let tensor_data = image.data().clone().to_data();
    let f32_slice = match tensor_data.as_slice::<f32>() {
        Ok(s) => s,
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to extract f32 slice from tensor data: {:?}",
                e
            ))
        }
    };

    // image.shape() is [nz, ny, nx] in RITK convention.
    // MetaImage DimSize is written in [nx, ny, nz] file-axis order.
    let shape = image.shape();
    let nz = shape[0];
    let ny = shape[1];
    let nx = shape[2];

    // ── Spatial metadata ──────────────────────────────────────────────────
    let origin = image.origin();
    let dir = image.direction().0;
    let spatial_fields = file_spatial_fields_from_internal(
        [image.spacing()[0], image.spacing()[1], image.spacing()[2]],
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
    writeln!(writer, "ElementType = MET_FLOAT")?;
    // LOCAL signals that binary data follows immediately.
    writeln!(writer, "ElementDataFile = LOCAL")?;

    // Binary payload — little-endian f32.
    for &v in f32_slice {
        writer.write_all(&v.to_le_bytes())?;
    }

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
pub struct MetaImageWriter;

impl MetaImageWriter {
    /// Write `image` to the MetaImage file at `path`.
    pub fn write<B: Backend, P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> Result<()> {
        write_metaimage(path, image)
    }
}
